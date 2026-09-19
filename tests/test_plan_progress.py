from dm_agent.core.plan_progress import PlanProgressTracker, summarize_plan
from dm_agent.core.planner import PlanStep
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
