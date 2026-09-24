from __future__ import annotations

import json
from typing import Any

from dm_agent.core import ReactAgent
from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.checkpoint import RunCheckpoint
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeLLMRequestEvent,
    BeforeToolCallEvent,
    EventBus,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.core.evidence_policy import EvidenceCompletionPolicy
from dm_agent.extensions.capabilities import EvidenceGraphCapability
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing.evidence import analyze_evidence_events, rebuild_evidence_graph


class _Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, payload: dict[str, Any]) -> str:
        self.events.append({"event": event, "payload": payload})
        return f"entry-{len(self.events)}"


class _ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def respond(self, messages, **extra):
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.pop(0)


def _action(action: str, action_input: Any) -> str:
    return json.dumps(
        {"thought": "test step", "action": action, "action_input": action_input},
        ensure_ascii=False,
    )


def _plan() -> list[dict[str, Any]]:
    return [
        {
            "step_number": 1,
            "action": "read_file",
            "reason": "Inspect the target implementation.",
            "completed": False,
        },
        {
            "step_number": 2,
            "action": "edit_file",
            "reason": "Apply the minimal fix.",
            "completed": False,
        },
    ]


def _test_result(
    *,
    outcome: str,
    scope: tuple[str, ...] = ("tests/test_service.py",),
    execution_status: str = "completed",
    failure_kind: str = "assertion_failure",
) -> ToolResult:
    passed = outcome == "passed"
    return ToolResult(
        "success" if passed else "failed",
        "1 passed" if passed else "1 failed",
        exit_code=0 if passed else 1,
        check_scope=scope,
        metadata={
            "verification": {
                "execution_status": execution_status,
                "outcome": outcome,
                "framework": "pytest",
                "scope": list(scope),
                "collected": 1 if outcome in {"passed", "failed"} else 0,
                "passed": 1 if passed else 0,
                "failed": 1 if outcome == "failed" else 0,
                "errors": 1 if outcome == "error" else 0,
                "skipped": 0,
                "failure_kind": "" if passed else failure_kind,
                "scope_level": (
                    "direct" if scope and all("::" in item for item in scope) else "related"
                ),
            }
        },
    )


def test_evidence_graph_builds_a_verified_chain_and_round_trips() -> None:
    graph = EvidenceGraph("Normalize imported users")
    graph.sync_plan(_plan())
    graph.add_observation(tool="read_file", path="users.py", step_number=1, succeeded=True)
    graph.add_change(tool="edit_file", path="users.py", step_number=2)
    graph.add_verification(
        tool="run_tests",
        step_number=3,
        passed=True,
        target_change_ids=("change-1",),
        direct=True,
    )
    conclusion, _ = graph.add_conclusion(text="Implemented and tested.", step_number=4)

    assert graph.status() == "verified"
    assert graph.audit()["counts"]["change"] == 1
    assert graph.audit()["edge_confidence_counts"] == {
        "deterministic": 3,
        "claimed": 4,
        "inferred": 5,
    }
    assert any(edge.relation == "motivated_by" for edge in graph.edges)
    assert any(edge.relation == "verifies" for edge in graph.edges)
    assert {
        edge.confidence for edge in graph.edges
    } == {"deterministic", "claimed", "inferred"}
    assert conclusion[0].metadata["evidence_status"] == graph.status()

    restored = EvidenceGraph.from_dict(graph.to_dict())
    assert restored.audit() == graph.audit()
    assert restored.prompt_summary(max_chars=200).startswith("[Task Evidence]")
    assert len(restored.prompt_summary(max_chars=200)) <= 200


def test_runtime_fact_links_to_the_matching_plan_phase() -> None:
    graph = EvidenceGraph("Inspect before changing")
    graph.sync_plan(
        [
            {"step_number": 1, "phase": "locate", "goal": "Locate implementation"},
            {"step_number": 2, "phase": "inspect", "goal": "Inspect implementation"},
        ]
    )

    nodes, edges = graph.add_observation(
        tool="read_file",
        path="service.py",
        step_number=1,
        succeeded=True,
        phase="inspect",
    )

    inspect_plan = next(
        node
        for node in graph.nodes.values()
        if node.kind == "plan_step" and node.metadata.get("phase") == "inspect"
    )
    assert any(
        edge.source_id == inspect_plan.node_id
        and edge.target_id == nodes[0].node_id
        and edge.confidence == "deterministic"
        for edge in edges
    )


def test_latest_verification_supersedes_an_older_result_for_the_same_scope() -> None:
    graph = EvidenceGraph("Fix service")
    graph.workspace_version = "v1"
    graph.add_verification(
        tool="run_tests",
        step_number=1,
        passed=False,
        workspace_version="v1",
        check='{"verbose": false}',
        scope=("tests/test_service.py",),
        outcome="failed",
        failure_kind="assertion_failure",
        blocking=False,
        scope_level="related",
    )
    graph.add_verification(
        tool="run_tests",
        step_number=2,
        passed=True,
        workspace_version="v1",
        check='{"verbose": true}',
        scope=("tests/test_service.py",),
        outcome="passed",
        blocking=False,
        scope_level="related",
    )

    current = graph.current_verifications()
    assert len(current) == 1
    assert current[0].step_number == 2
    assert current[0].metadata["passed"] is True


def test_legacy_edge_confidence_is_migrated_to_the_three_source_levels() -> None:
    graph = EvidenceGraph.from_dict(
        {
            "task": "legacy",
            "nodes": [
                {"node_id": "requirement-1", "kind": "requirement", "title": "legacy"},
                {"node_id": "change-1", "kind": "change", "title": "changed"},
            ],
            "edges": [
                {
                    "source_id": "change-1",
                    "target_id": "requirement-1",
                    "relation": "old-direct",
                    "confidence": "direct",
                },
                {
                    "source_id": "requirement-1",
                    "target_id": "change-1",
                    "relation": "old-indirect",
                    "confidence": "indirect",
                },
            ],
        }
    )

    assert [edge.confidence for edge in graph.edges] == ["deterministic", "inferred"]


def test_failed_verification_contradicts_a_change() -> None:
    graph = EvidenceGraph("Fix cache expiration")
    graph.add_change(tool="edit_file", path="cache.py", step_number=1)
    graph.add_verification(
        tool="run_tests",
        step_number=2,
        passed=False,
        target_change_ids=("change-1",),
        direct=True,
    )

    assert graph.status() == "contradicted"
    assert graph.audit()["failed_verifications"] == 1


def test_change_evidence_requires_matching_read_and_current_direct_verification() -> None:
    graph = EvidenceGraph("Fix cache expiration")
    graph.workspace_version = "v1"
    graph.add_observation(
        tool="read_file",
        path="other.py",
        step_number=1,
        succeeded=True,
        workspace_version="v1",
    )
    graph.add_change(
        tool="edit_file",
        path="cache.py",
        step_number=2,
        before_version="v1",
        after_version="v2",
    )

    assert graph.change_states() == {"change-1": "missing_read_basis"}

    graph.add_observation(
        tool="read_file",
        path="cache.py",
        step_number=3,
        succeeded=True,
        workspace_version="v2",
    )
    graph.add_change(
        tool="edit_file",
        path="cache.py",
        step_number=4,
        before_version="v2",
        after_version="v3",
    )
    graph.add_verification(
        tool="run_linter",
        step_number=5,
        passed=True,
        workspace_version="v3",
        target_change_ids=("change-2",),
        direct=False,
    )

    assert graph.change_states()["change-2"] == "indirectly_checked"

    graph.add_verification(
        tool="run_tests",
        step_number=6,
        passed=True,
        workspace_version="v3",
        target_change_ids=("change-2",),
        direct=True,
    )

    assert graph.change_states()["change-2"] == "verified"
    assert graph.change_states()["change-1"] == "missing_read_basis"


def test_tool_validated_write_basis_avoids_false_missing_read_gap() -> None:
    for index, basis_kind in enumerate(("content_anchor", "expected_hash"), start=1):
        graph = EvidenceGraph(f"Apply {basis_kind} edit")
        graph.add_change(
            tool="edit_file" if basis_kind == "content_anchor" else "edit_python_symbol",
            path="service.py",
            step_number=index,
            before_version=f"v{index}",
            after_version=f"v{index + 1}",
            requires_read_basis=False,
            basis_kind=basis_kind,
        )
        assert graph.change_states()["change-1"] == "unverified"


def test_capability_records_content_anchor_as_tool_validated_basis(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Update service", 1, "run", metadata=metadata))
    arguments = {"path": "service.py", "old_string": "value = 1", "new_string": "value = 2"}
    bus.emit_before_tool_call(
        BeforeToolCallEvent(
            "edit_file", arguments, 1, "run", metadata, content_anchor_safe=True
        )
    )
    target.write_text("value = 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            arguments,
            "updated",
            1,
            "run",
            True,
            metadata,
            result=ToolResult("success", "updated", changed_files=(str(target),)),
        )
    )

    change = capability.graph.nodes["change-1"]
    assert change.metadata["basis_kind"] == "content_anchor"
    assert capability.graph.change_states()["change-1"] == "unverified"


def test_new_change_makes_prior_verification_stale() -> None:
    graph = EvidenceGraph("Fix parser")
    graph.workspace_version = "v1"
    graph.add_observation(
        tool="read_file",
        path="parser.py",
        step_number=1,
        succeeded=True,
        workspace_version="v1",
    )
    graph.add_change(
        tool="edit_file",
        path="parser.py",
        step_number=2,
        before_version="v1",
        after_version="v2",
    )
    graph.add_verification(
        tool="run_tests",
        step_number=3,
        passed=True,
        workspace_version="v2",
        target_change_ids=("change-1",),
        direct=True,
    )
    assert graph.change_states()["change-1"] == "verified"

    graph.add_observation(
        tool="read_file",
        path="parser.py",
        step_number=4,
        succeeded=True,
        workspace_version="v2",
    )
    graph.add_change(
        tool="edit_file",
        path="parser.py",
        step_number=5,
        before_version="v2",
        after_version="v3",
    )

    assert set(graph.change_states().values()) == {"unverified"}


def test_targeted_verification_does_not_verify_or_contradict_other_changes() -> None:
    graph = EvidenceGraph("Update two modules")
    for index, path in enumerate(("a.py", "b.py"), start=1):
        graph.add_observation(
            tool="read_file",
            path=path,
            step_number=index * 2 - 1,
            succeeded=True,
            workspace_version=f"v{index}",
        )
        graph.add_change(
            tool="edit_file",
            path=path,
            step_number=index * 2,
            before_version=f"v{index}",
            after_version=f"v{index + 1}",
        )
    graph.add_verification(
        tool="run_tests",
        step_number=5,
        passed=True,
        workspace_version="v3",
        target_change_ids=("change-1",),
        direct=True,
    )

    assert graph.change_states()["change-1"] == "verified"
    assert graph.change_states()["change-2"] == "unverified"

    graph.add_verification(
        tool="run_tests",
        step_number=6,
        passed=False,
        workspace_version="v3",
        target_change_ids=("change-1",),
        direct=True,
    )
    assert graph.change_states()["change-1"] == "contradicted"
    assert graph.change_states()["change-2"] == "unverified"


def test_untargeted_verification_is_transaction_evidence_not_file_evidence() -> None:
    graph = EvidenceGraph("Update two modules")
    graph.add_change(tool="edit_file", path="a.py", step_number=1)
    graph.add_change(tool="edit_file", path="b.py", step_number=2)
    graph.add_verification(tool="run_tests", step_number=3, passed=True, direct=True)

    assert set(graph.change_states().values()) == {"unverified"}
    assert not any(edge.relation == "verifies" for edge in graph.edges)


def test_start_resets_workspace_version_and_verification_only_tasks_have_status() -> None:
    graph = EvidenceGraph("Old task")
    graph.workspace_version = "old-version"
    graph.start("Run checks")
    assert graph.workspace_version == ""

    graph.add_verification(tool="run_tests", step_number=1, passed=True, direct=True)
    conclusion, _ = graph.add_conclusion(text="Checks passed", step_number=2)
    assert graph.status() == "verified"
    assert conclusion[0].metadata["evidence_status"] == "verified"

    failed = EvidenceGraph("Run checks")
    failed.add_verification(tool="run_tests", step_number=1, passed=False, direct=True)
    failed.add_conclusion(text="Checks failed", step_number=2)
    assert failed.status() == "contradicted"


def test_completion_policy_uses_transaction_level_test_for_source_changes() -> None:
    graph = EvidenceGraph("Update a dependency and its caller")
    graph.workspace_version = "v1"
    for step, path in enumerate(("dependency.py", "caller.py"), start=1):
        graph.add_observation(
            tool="read_file",
            path=path,
            step_number=step * 2 - 1,
            succeeded=True,
            workspace_version=graph.workspace_version,
        )
        graph.add_change(
            tool="edit_file",
            path=path,
            step_number=step * 2,
            before_version=graph.workspace_version,
            after_version=f"v{step + 1}",
        )
    graph.add_verification(
        tool="run_tests",
        step_number=5,
        passed=True,
        workspace_version=graph.workspace_version,
        scope=("tests/test_caller.py",),
    )

    decision = EvidenceCompletionPolicy().evaluate(graph)

    assert decision.decision == "allow"
    assert decision.issues == ()


def test_completion_policy_marks_unchecked_changes_not_run() -> None:
    docs = EvidenceGraph("Update documentation")
    docs.add_change(
        tool="edit_file",
        path="README.md",
        step_number=1,
        requires_read_basis=False,
    )
    config = EvidenceGraph("Update configuration")
    config.add_change(
        tool="edit_file",
        path="pyproject.toml",
        step_number=1,
        requires_read_basis=False,
    )

    docs_decision = EvidenceCompletionPolicy().evaluate(docs)
    assert docs_decision.decision == "warn"
    assert docs_decision.verification_state == "not_run"
    config_decision = EvidenceCompletionPolicy().evaluate(config)
    assert config_decision.decision == "warn"
    assert config_decision.verification_state == "not_run"
    assert config_decision.warnings[0]["status"] == "verification_not_run"


def test_completion_policy_still_blocks_failed_checks_and_missing_read_basis() -> None:
    graph = EvidenceGraph("Fix service")
    graph.workspace_version = "v1"
    graph.add_observation(
        tool="read_file",
        path="service.py",
        step_number=1,
        succeeded=True,
        workspace_version="v1",
    )
    graph.add_change(
        tool="edit_file",
        path="service.py",
        step_number=2,
        before_version="v1",
        after_version="v2",
    )
    graph.add_verification(
        tool="run_tests",
        step_number=3,
        passed=False,
        workspace_version="v2",
    )

    decision = EvidenceCompletionPolicy().evaluate(graph)

    assert decision.decision == "block"
    assert {item["status"] for item in decision.issues} == {"confirmed_test_failure"}


def test_completion_policy_warns_instead_of_blocking_for_missing_read_basis() -> None:
    graph = EvidenceGraph("Update service")
    graph.add_change(
        tool="edit_file",
        path="service.py",
        step_number=1,
        before_version="v1",
        after_version="v2",
    )

    decision = EvidenceCompletionPolicy().evaluate(graph)

    assert decision.decision == "warn"
    assert decision.verification_state == "not_run"
    assert decision.issues == ()
    assert {item["status"] for item in decision.warnings} == {"verification_not_run"}


def test_evidence_capability_never_injects_status_and_rejects_repeated_contradiction() -> None:
    bus = EventBus()
    trace = _Trace()
    state = {"plan": _plan()}
    capability = EvidenceGraphCapability(
        summary_chars=300,
        repeated_contradiction="critic_rejected",
    )
    capability.install(CapabilityContext(bus, lambda phase: None, trace, lambda: state))
    metadata: dict[str, Any] = {}

    bus.emit_run_start(RunStartEvent("Fix users", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "users.py"},
            "updated",
            1,
            "run-1",
            True,
            metadata,
        )
    )
    request = BeforeLLMRequestEvent(
        [{"role": "user", "content": "continue"}],
        2,
        "run-1",
        "agent",
        metadata,
    )
    outgoing = bus.emit_before_llm_request(request)
    assert outgoing == [{"role": "user", "content": "continue"}]
    metadata["memory_compression_count"] = 1
    compressed = bus.emit_before_llm_request(
        BeforeLLMRequestEvent(
            [{"role": "user", "content": "continue compressed"}],
            2,
            "run-1",
            "agent",
            metadata,
        )
    )
    assert compressed == [{"role": "user", "content": "continue compressed"}]

    # A duplicate pending signal without any operational evidence change does
    # not append the same evidence summary again.
    duplicate = bus.emit_before_llm_request(
        BeforeLLMRequestEvent(
            [{"role": "user", "content": "continue"}], 2, "run-1", "agent", metadata
        )
    )
    assert duplicate == [{"role": "user", "content": "continue"}]

    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {"targets": ["tests/test_users.py"]},
            "1 failed\nreturncode: 1",
            2,
            "run-1",
            False,
            metadata,
            result=_test_result(
                outcome="failed", scope=("tests/test_users.py::test_update",)
            ),
        )
    )
    finish = BeforeFinishEvent("Fix users", "finish", "done", [], 3, "run-1", metadata)
    first = bus.emit_before_finish(finish)
    second = bus.emit_before_finish(finish)
    assert first is not None and "Latest failed test" in first["reason"]
    assert second is not None and "Completion rejected" in second["reason"]
    assert metadata["evidence_contradiction_block_count"] == 2
    assert metadata["evidence_completion_block_count"] == 2
    assert metadata["evidence_intervention_prompt_count"] == 1
    assert metadata["evidence_terminal_rejection_count"] == 1
    conclusions = [node for node in capability.graph.nodes.values() if node.kind == "conclusion"]
    assert conclusions and all(node.metadata["accepted"] is False for node in conclusions)
    assert capability.graph.audit()["has_conclusion"] is False
    assert any(edge.relation == "attempts_to_conclude" for edge in capability.graph.edges)


def test_benchmark_policy_marks_repeated_unchanged_contradiction_terminal() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability(repeated_contradiction="critic_rejected")
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix users", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {"targets": ["tests/test_users.py"]},
            "1 failed\nreturncode: 1",
            1,
            "run-1",
            False,
            metadata,
            result=_test_result(
                outcome="failed", scope=("tests/test_users.py::test_update",)
            ),
        )
    )
    finish = BeforeFinishEvent("Fix users", "finish", "done", [], 2, "run-1", metadata)

    first = bus.emit_before_finish(finish)
    second = bus.emit_before_finish(finish)

    assert first is not None
    assert metadata["evidence_terminal_completion_rejection"] is True
    assert second is not None and "after repeated attempts" in second["reason"]
    assert metadata["evidence_completion_block_count"] == 2
    assert metadata["evidence_terminal_rejection_count"] == 1


def test_completion_pauses_once_when_historical_writes_leave_no_net_code_change(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    bus = EventBus()
    trace = _Trace()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None, trace))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run-1", metadata=metadata))

    for step_number, content in ((1, "value = 2\n"), (2, "value = 1\n")):
        arguments = {"path": "service.py"}
        bus.emit_before_tool_call(
            BeforeToolCallEvent("edit_file", arguments, step_number, "run-1", metadata)
        )
        target.write_text(content, encoding="utf-8")
        bus.emit_after_tool_result(
            AfterToolResultEvent(
                "edit_file",
                arguments,
                "updated",
                step_number,
                "run-1",
                True,
                metadata,
                result=ToolResult(
                    "success", "updated", changed_files=(str(target),)
                ),
            )
        )

    test_arguments = {"targets": ["tests/test_service.py"]}
    bus.emit_before_tool_call(
        BeforeToolCallEvent("run_tests", test_arguments, 3, "run-1", metadata)
    )
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            test_arguments,
            "1 passed",
            3,
            "run-1",
            True,
            metadata,
            result=_test_result(outcome="passed"),
        )
    )

    finish = BeforeFinishEvent("Fix service", "finish", "done", [], 4, "run-1", metadata)
    first = bus.emit_before_finish(finish)
    second = bus.emit_before_finish(finish)

    assert first is not None and "no net workspace change remains" in first["reason"]
    assert second is None
    assert metadata["evidence_no_net_change_block_count"] == 1
    assert metadata["evidence_completion_block_count"] == 1
    assert any(
        event["event"] == "evidence_no_net_change_blocked" for event in trace.events
    )
    conclusions = [node for node in capability.graph.nodes.values() if node.kind == "conclusion"]
    assert [node.metadata["accepted"] for node in conclusions[-2:]] == [False, True]


def test_invalid_test_invocation_warns_without_blocking_completion() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability(repeated_contradiction="critic_rejected")
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix users", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {"targets": ["tests/missing.py"]},
            "test target missing",
            1,
            "run-1",
            False,
            metadata,
            result=_test_result(
                outcome="unknown",
                scope=("tests/missing.py",),
                execution_status="invalid",
                failure_kind="invalid_target",
            ),
        )
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix users", "finish", "done", [], 2, "run-1", metadata)
    )

    assert block is None
    assert metadata["evidence_completion_decision"] == "warn"
    assert metadata["evidence_verification_state"] == "unavailable"
    assert metadata["evidence_completion_block_count"] == 0


def test_declared_shell_verification_can_mark_current_workspace_tested() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    arguments = {"command": "pytest tests/test_service.py", "purpose": "verification"}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run-1", metadata=metadata))
    bus.emit_before_tool_call(BeforeToolCallEvent("run_shell", arguments, 1, "run-1", metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_shell",
            arguments,
            "1 passed\nreturncode: 0",
            1,
            "run-1",
            True,
            metadata,
            result=ToolResult(
                "success",
                "1 passed\nreturncode: 0",
                exit_code=0,
                check_scope=(arguments["command"],),
                metadata={
                    "verification": {
                        "execution_status": "completed",
                        "outcome": "passed",
                        "failure_kind": "",
                        "scope": [arguments["command"]],
                    }
                },
            ),
        )
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 2, "run-1", metadata)
    )

    assert block is None
    assert metadata["evidence_verification_state"] == "tested"


def test_direct_shell_assertion_failure_blocks_but_ordinary_shell_does_not() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability(repeated_contradiction="critic_rejected")
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run-1", metadata=metadata))

    ordinary = {"command": "pytest tests/test_service.py"}
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_shell", ordinary, "FAILED - AssertionError", 1, "run-1", False, metadata
        )
    )
    assert not capability.graph.current_verifications()

    declared = {
        "command": "pytest tests/test_service.py::test_update",
        "purpose": "verification",
        "verification_scope": "direct",
        "expected_exit_code": 0,
    }
    bus.emit_before_tool_call(BeforeToolCallEvent("run_shell", declared, 2, "run-1", metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_shell",
            declared,
            "FAILED tests/test_service.py::test_update - AssertionError\nreturncode: 1",
            2,
            "run-1",
            False,
            metadata,
            result=ToolResult(
                "failed",
                "FAILED tests/test_service.py::test_update - AssertionError\nreturncode: 1",
                error_code="assertion_failure",
                exit_code=1,
                check_scope=(declared["command"],),
                metadata={
                    "verification": {
                        "execution_status": "completed",
                        "outcome": "failed",
                        "failure_kind": "assertion_failure",
                        "scope": [declared["command"]],
                        "scope_level": "direct",
                    }
                },
            ),
        )
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 3, "run-1", metadata)
    )

    assert block is not None
    assert "Latest failed verification" in block["reason"]
    assert "AssertionError" in block["reason"]
    assert metadata["evidence_verification_state"] == "contradicted"


def test_related_shell_failure_is_recorded_without_blocking_completion() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    arguments = {"command": "pytest --runxfail test_xf.py", "purpose": "verification"}
    bus.emit_run_start(RunStartEvent("Fix xfail reporting", 1, "run-1", metadata=metadata))
    bus.emit_before_tool_call(BeforeToolCallEvent("run_shell", arguments, 1, "run-1", metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_shell",
            arguments,
            "AssertionError\nreturncode: 1",
            1,
            "run-1",
            False,
            metadata,
            result=ToolResult(
                "failed",
                "AssertionError\nreturncode: 1",
                error_code="assertion_failure",
                exit_code=1,
                check_scope=(arguments["command"],),
                metadata={
                    "verification": {
                        "execution_status": "completed",
                        "outcome": "failed",
                        "failure_kind": "assertion_failure",
                        "scope": [arguments["command"]],
                        "scope_level": "related",
                    }
                },
            ),
        )
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix xfail reporting", "finish", "done", [], 2, "run-1", metadata)
    )

    assert block is None
    assert metadata["evidence_completion_decision"] == "warn"
    assert metadata["evidence_verification_state"] == "unavailable"


def test_full_suite_failure_is_a_warning_not_a_completion_block() -> None:
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix users", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {"test_path": "tests"},
            "1 failed",
            1,
            "run-1",
            False,
            metadata,
            result=_test_result(outcome="failed", scope=("tests",)),
        )
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix users", "finish", "done", [], 2, "run-1", metadata)
    )

    assert block is None
    assert metadata["evidence_completion_block_count"] == 0
    assert metadata["evidence_verification_state"] == "unavailable"


def test_new_edit_makes_prior_confirmed_failure_stale_and_allows_unverified_finish(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    bus = EventBus()
    capability = EvidenceGraphCapability(repeated_contradiction="critic_rejected")
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {"targets": ["tests/test_service.py"]},
            "1 failed",
            1,
            "run-1",
            False,
            metadata,
            result=_test_result(
                outcome="failed", scope=("tests/test_service.py::test_update",)
            ),
        )
    )
    first = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 2, "run-1", metadata)
    )
    assert first is not None
    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": "service.py"}, 3, "run-1", metadata)
    )
    target.write_text("value = 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "service.py"},
            "updated",
            3,
            "run-1",
            True,
            metadata,
            result=ToolResult("success", "updated", changed_files=(str(target),)),
        )
    )

    first_unverified = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 4, "run-1", metadata)
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 5, "run-1", metadata)
    )
    assert first_unverified is not None
    assert block is None
    assert metadata["evidence_verification_state"] == "not_run"
    assert metadata["evidence_completion_pause_count"] == 1
    assert metadata["evidence_completion_status"] == "unverified"


def test_syntax_error_in_current_changed_file_blocks_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run-1", metadata=metadata))
    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": "service.py"}, 1, "run-1", metadata)
    )
    target.write_text("def broken(:\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "service.py"},
            "updated",
            1,
            "run-1",
            True,
            metadata,
            result=ToolResult("success", "updated", changed_files=(str(target),)),
        )
    )
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_python",
            {"path": "service.py"},
            f'File "{target}", line 1\nSyntaxError: invalid syntax',
            2,
            "run-1",
            False,
            metadata,
            result=ToolResult("failed", "SyntaxError", exit_code=1),
        )
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("Fix service", "finish", "done", [], 3, "run-1", metadata)
    )

    assert block is not None
    assert metadata["evidence_completion_decision"] == "block"
    assert metadata["evidence_policy_issues"][0]["status"] == "confirmed_code_error"


def test_react_agent_returns_critic_rejected_after_repeated_contradiction(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def failed_test(arguments: dict[str, Any]) -> ToolResult:
        return _test_result(
            outcome="failed", scope=("tests/test_service.py::test_update",)
        )

    agent = ReactAgent(
        _ScriptedClient(
            [
                _action("run_tests", {"targets": ["tests/test_service.py"]}),
                _action("finish", {"answer": "done"}),
                _action("finish", {"answer": "done again"}),
            ]
        ),
        [Tool("run_tests", "test", lambda arguments: "", result_runner=failed_test)],
        enable_planning=False,
        enable_compression=False,
        capabilities=[
            EvidenceGraphCapability(repeated_contradiction="critic_rejected")
        ],
    )

    result = agent.run("Fix the failing test", max_steps=3)

    assert result["metadata"]["status"] == "critic_rejected"
    assert result["metadata"]["evidence_completion_block_count"] == 2
    assert result["metadata"]["evidence_terminal_rejection_count"] == 1
    assert result["steps"][-1]["action"] == "finish"


def test_react_agent_allows_unchecked_finish_with_explicit_status(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")

    def read_result(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult("success", target.read_text(encoding="utf-8"))

    def edit_result(arguments: dict[str, Any]) -> ToolResult:
        target.write_text("value = 2\n", encoding="utf-8")
        return ToolResult("success", "updated", changed_files=(str(target),))

    agent = ReactAgent(
        _ScriptedClient(
            [
                _action("read_file", {"path": "service.py"}),
                _action("edit_file", {"path": "service.py"}),
                _action("finish", {"answer": "done"}),
                _action("finish", {"answer": "done without local verification"}),
            ]
        ),
        [
            Tool("read_file", "read", lambda arguments: "", result_runner=read_result),
            Tool("edit_file", "edit", lambda arguments: "", result_runner=edit_result),
        ],
        enable_planning=False,
        enable_compression=False,
        capabilities=[EvidenceGraphCapability()],
    )

    result = agent.run("Update service.py", max_steps=4)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["evidence_completion_decision"] == "warn"
    assert result["metadata"]["evidence_verification_state"] == "not_run"
    assert result["metadata"]["evidence_completion_block_count"] == 1
    assert result["metadata"]["evidence_completion_pause_count"] == 1
    assert result["metadata"]["evidence_completion_status"] == "unverified"
    assert result["steps"][-1]["observation"] == "<finished>"
    assert result["metadata"]["completion_summary"].endswith("本轮未运行本地验证。")


def test_react_agent_marks_tested_finish_after_tests(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")

    def read_result(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult("success", target.read_text(encoding="utf-8"))

    def edit_result(arguments: dict[str, Any]) -> ToolResult:
        target.write_text("value = 2\n", encoding="utf-8")
        return ToolResult("success", "updated", changed_files=(str(target),))

    def test_result(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult("success", "1 passed", exit_code=0, check_scope=("tests",))

    capability = EvidenceGraphCapability()
    agent = ReactAgent(
        _ScriptedClient(
            [
                _action("read_file", {"path": "service.py"}),
                _action("edit_file", {"path": "service.py"}),
                _action("run_tests", {"test_path": "tests"}),
                _action("finish", {"answer": "implemented and tested"}),
            ]
        ),
        [
            Tool("read_file", "read", lambda arguments: "", result_runner=read_result),
            Tool("edit_file", "edit", lambda arguments: "", result_runner=edit_result),
            Tool("run_tests", "test", lambda arguments: "", result_runner=test_result),
        ],
        enable_planning=False,
        enable_compression=False,
        capabilities=[capability],
    )

    result = agent.run("Update service.py and verify it", max_steps=4)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["evidence_completion_block_count"] == 0
    assert result["steps"][3]["observation"] == "<finished>"
    assert capability.graph.status() == "implemented"
    assert result["metadata"]["evidence_completion_decision"] == "allow"
    assert result["metadata"]["evidence_verification_state"] == "tested"


def test_evidence_gate_accepts_prior_verified_edit_transaction(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8")
    bus = EventBus()
    capability = EvidenceGraphCapability()
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("Update service", 1, "run", metadata=metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "read_file", {"path": "service.py"}, "value = 1", 1, "run", True, metadata
        )
    )
    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": "service.py"}, 2, "run", metadata)
    )
    target.write_text("value = 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "service.py"},
            "updated",
            2,
            "run",
            True,
            metadata,
            result=ToolResult("success", "updated", changed_files=(str(target),)),
        )
    )
    metadata["edit_transaction_status"] = "committed"

    block = bus.emit_before_finish(
        BeforeFinishEvent("Update service", "finish", "done", [], 3, "run", metadata)
    )

    assert block is None
    # VerifiedEdit owns transaction validation; the graph deliberately records
    # no fabricated run_tests node merely to make its factual status verified.
    assert capability.graph.status() == "implemented"
    assert metadata["evidence_completion_decision"] == "allow"


def test_checkpoint_and_trace_rebuild_preserve_evidence_state() -> None:
    graph = EvidenceGraph("Fix parser")
    nodes, edges = graph.add_change(tool="edit_file", path="parser.py", step_number=2)
    checkpoint = RunCheckpoint(
        task="Fix parser",
        step_count=2,
        capability_state={"evidence_graph": graph.to_dict()},
    )
    restored_checkpoint = RunCheckpoint.from_dict(checkpoint.to_dict())
    restored = EvidenceGraph.from_dict(restored_checkpoint.capability_state["evidence_graph"])
    assert restored.status() == "implemented"

    events = [
        *({"event": "evidence_node", "payload": node.to_dict()} for node in nodes),
        *({"event": "evidence_edge", "payload": edge.to_dict()} for edge in edges),
    ]
    rebuilt = rebuild_evidence_graph(events)
    analysis = analyze_evidence_events(events)
    assert rebuilt.status() == "implemented"
    assert analysis["enabled"] is True
    assert analysis["unverified_changes"][0]["path"] == "parser.py"
