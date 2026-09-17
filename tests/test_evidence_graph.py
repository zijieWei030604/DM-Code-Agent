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
    assert any(edge.relation == "motivated_by" for edge in graph.edges)
    assert any(edge.relation == "verifies" for edge in graph.edges)
    assert conclusion[0].metadata["evidence_status"] == graph.status()

    restored = EvidenceGraph.from_dict(graph.to_dict())
    assert restored.audit() == graph.audit()
    assert restored.prompt_summary(max_chars=200).startswith("[Task Evidence]")
    assert len(restored.prompt_summary(max_chars=200)) <= 200


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


def test_completion_policy_allows_docs_and_warns_for_unchecked_config() -> None:
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

    assert EvidenceCompletionPolicy().evaluate(docs).decision == "allow"
    config_decision = EvidenceCompletionPolicy().evaluate(config)
    assert config_decision.decision == "warn"
    assert config_decision.warnings[0]["status"] == "configuration_unchecked"


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
    assert {item["status"] for item in decision.issues} == {"transaction_test_failed"}


def test_evidence_capability_injects_changed_state_once_and_bounds_repeated_finishes() -> None:
    bus = EventBus()
    trace = _Trace()
    state = {"plan": _plan()}
    capability = EvidenceGraphCapability(summary_chars=300)
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
    assert outgoing[-1]["role"] == "system"
    assert outgoing[-1]["content"].startswith("[Task Evidence]")
    assert metadata["evidence_summary_injection_count"] == 1

    # A duplicate pending signal without any operational evidence change does
    # not append the same evidence summary again.
    capability.graph.summary_pending = True
    duplicate = bus.emit_before_llm_request(
        BeforeLLMRequestEvent(
            [{"role": "user", "content": "continue"}], 2, "run-1", "agent", metadata
        )
    )
    assert duplicate == [{"role": "user", "content": "continue"}]

    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            {},
            "1 failed\nreturncode: 1",
            2,
            "run-1",
            True,
            metadata,
        )
    )
    finish = BeforeFinishEvent("Fix users", "finish", "done", [], 3, "run-1", metadata)
    first = bus.emit_before_finish(finish)
    second = bus.emit_before_finish(finish)
    third = bus.emit_before_finish(finish)
    assert first is not None and "Latest failed test" in first["reason"]
    assert second is not None and "verification state has not changed" in second["reason"]
    assert third is not None and "after repeated attempts" in third["reason"]
    assert metadata["evidence_contradiction_block_count"] == 0
    assert metadata["evidence_completion_block_count"] == 3
    assert metadata["evidence_recovery_prompt_count"] == 1
    assert metadata["evidence_terminal_rejection_count"] == 1
    conclusions = [node for node in capability.graph.nodes.values() if node.kind == "conclusion"]
    assert conclusions and all(node.metadata["accepted"] is False for node in conclusions)
    assert capability.graph.audit()["has_conclusion"] is False
    assert any(edge.relation == "attempts_to_conclude" for edge in capability.graph.edges)


def test_react_agent_allows_unverified_finish_with_explicit_status(tmp_path, monkeypatch) -> None:
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

    result = agent.run("Update service.py", max_steps=3)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["evidence_completion_decision"] == "warn"
    assert result["metadata"]["evidence_completion_status"] == "unverified"
    assert result["metadata"]["evidence_completion_block_count"] == 0
    assert result["steps"][-1]["observation"] == "<finished>"
    assert result["metadata"]["completion_summary"].endswith("未获得成功的本地验证记录。")


def test_react_agent_marks_verified_finish_after_tests(
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
    assert result["metadata"]["evidence_completion_status"] == "verified"


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
