import json

from dm_agent.core.plan_progress import (
    PlanProgressTracker,
    summarize_plan,
)
from dm_agent.core.planner import PlanStep
from dm_agent.core.replan import FailureContext, ReplanCoordinator, failure_disposition
from dm_agent.tools.base import ToolResult


def _step(number: int, phase: str, tool: str, evidence: str) -> PlanStep:
    return PlanStep(
        step_number=number,
        phase=phase,
        goal=f"complete {phase}",
        preferred_tools=(tool,),
        completion_evidence=evidence,
    )


def test_runtime_fact_advances_phase_without_exact_tool_match() -> None:
    plan = [_step(1, "inspect", "read_file", "source_observed")]
    tracker = PlanProgressTracker()

    changes = tracker.observe(
        plan,
        action="inspect_python_symbol",
        result=ToolResult("success", "signature"),
        observation="signature",
        step_number=3,
    )

    assert plan[0].status == "satisfied"
    assert changes[0].evidence == "source_observed"
    assert plan[0].evidence_refs == ("runtime:3:inspect_python_symbol",)


def test_zero_result_search_does_not_satisfy_locate_phase() -> None:
    plan = [_step(1, "locate", "search_code", "candidate_observed")]

    changes = PlanProgressTracker().observe(
        plan,
        action="search_code",
        result=ToolResult("success", "search completed"),
        observation=json.dumps({"match_count": 0, "matches": []}),
        step_number=2,
    )

    assert plan[0].status == "in_progress"
    assert changes[0].reason == "repository search returned no candidates"


def test_nonempty_search_satisfies_locate_phase() -> None:
    plan = [_step(1, "locate", "find_files", "candidate_observed")]

    PlanProgressTracker().observe(
        plan,
        action="find_files",
        result=ToolResult("success", "search completed"),
        observation=json.dumps({"match_count": 1, "matches": ["service.py"]}),
        step_number=2,
    )

    assert plan[0].status == "satisfied"


def test_reverted_edit_invalidates_change_phase(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("value = 1\n", encoding="utf-8")
    tracker = PlanProgressTracker()
    tracker.begin(tmp_path)
    plan = [_step(1, "change", "edit_file", "workspace_changed")]

    source.write_text("value = 2\n", encoding="utf-8")
    tracker.observe(
        plan,
        action="edit_file",
        result=ToolResult("success", "changed", changed_files=(str(source),)),
        observation="changed",
        step_number=3,
    )
    assert plan[0].status == "satisfied"

    source.write_text("value = 1\n", encoding="utf-8")
    changes = tracker.observe(
        plan,
        action="edit_file",
        result=ToolResult("success", "changed", changed_files=(str(source),)),
        observation="changed",
        step_number=4,
    )

    assert plan[0].status == "in_progress"
    assert any(change.reason == "workspace returned to the run baseline" for change in changes)


def test_failed_verification_is_attempted_not_satisfied() -> None:
    plan = [_step(1, "validate", "run_tests", "verification_passed")]

    PlanProgressTracker().observe(
        plan,
        action="run_tests",
        result=ToolResult(
            "failed",
            "1 failed",
            metadata={"verification": {"outcome": "failed"}},
        ),
        observation="1 failed",
        no_progress=True,
        step_number=4,
    )

    assert plan[0].status == "in_progress"
    assert plan[0].completed is False


def test_write_after_validation_invalidates_validation_phase() -> None:
    validate = _step(1, "validate", "run_tests", "verification_passed")
    validate.set_status("satisfied", result="passed")
    change = _step(2, "change", "edit_file", "workspace_changed")
    plan = [validate, change]

    changes = PlanProgressTracker().observe(
        plan,
        action="edit_file",
        result=ToolResult("success", "changed", changed_files=("service.py",)),
        observation="changed",
        step_number=5,
    )

    assert validate.status == "in_progress"
    assert change.status == "satisfied"
    assert {item.phase for item in changes} == {"validate", "change"}
    assert summarize_plan(plan)["in_progress"] == 1


def test_targeted_assertion_failure_is_a_patch_contradiction() -> None:
    failure = FailureContext(
        observation="assertion failed",
        action="run_tests",
        verification={
            "execution_status": "completed",
            "outcome": "failed",
            "failure_kind": "assertion_failure",
            "scope": ["tests/test_service.py::test_behavior"],
        },
    )

    assert failure_disposition(failure) == "contradicted"


def test_environment_and_broad_suite_failures_do_not_contradict_patch() -> None:
    unavailable = FailureContext(
        observation="dependency missing",
        action="run_tests",
        verification={
            "execution_status": "unavailable",
            "outcome": "unknown",
            "failure_kind": "dependency_unavailable",
            "scope": ["tests/test_service.py"],
        },
    )
    broad = FailureContext(
        observation="unrelated failures",
        action="run_tests",
        verification={
            "execution_status": "completed",
            "outcome": "failed",
            "failure_kind": "assertion_failure",
            "scope": ["."],
        },
    )

    assert failure_disposition(unavailable) == "unavailable"
    assert failure_disposition(broad) == "inconclusive"


def test_default_replanner_accepts_only_one_targeted_contradiction() -> None:
    class Planner:
        calls = 0

        def replan(self, _task, plan, _error, **_kwargs):
            self.calls += 1
            return list(plan)

    planner = Planner()
    coordinator = ReplanCoordinator(planner=planner, policy=None)
    metadata = {
        "replan_count": 0,
        "replan_skipped_count": 0,
        "replan_suppressed_count": 0,
        "replan_budget_exhausted_count": 0,
        "failure_disposition_counts": {},
        "repeated_failure_count": 0,
        "repeated_failures": [],
    }
    failure = FailureContext(
        observation="1 failed",
        action="run_tests",
        error_kind="assertion_failure",
        verification={
            "execution_status": "completed",
            "outcome": "failed",
            "failure_kind": "assertion_failure",
            "scope": ["tests/test_service.py"],
        },
    )
    plan = [_step(1, "validate", "run_tests", "verification_passed")]

    coordinator.try_replan("fix", plan, failure, metadata, default_budget=5)
    coordinator.try_replan("fix", plan, failure, metadata, default_budget=5)

    assert planner.calls == 1
    assert metadata["replan_count"] == 1
    assert metadata["replan_suppressed_count"] == 0
    assert metadata["replan_budget_exhausted_count"] == 1
