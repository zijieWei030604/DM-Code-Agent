from __future__ import annotations

from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.checkpoint import RunCheckpoint
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeLLMRequestEvent,
    EventBus,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.extensions.capabilities import EvidenceGraphCapability
from dm_agent.tracing.evidence import analyze_evidence_events, rebuild_evidence_graph


class _Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, payload: dict[str, Any]) -> str:
        self.events.append({"event": event, "payload": payload})
        return f"entry-{len(self.events)}"


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
    graph.add_verification(tool="run_tests", step_number=3, passed=True)
    graph.add_conclusion(text="Implemented and tested.", step_number=4)

    assert graph.status() == "verified"
    assert graph.audit()["counts"]["change"] == 1
    assert any(edge.relation == "motivated_by" for edge in graph.edges)
    assert any(edge.relation == "verifies" for edge in graph.edges)

    restored = EvidenceGraph.from_dict(graph.to_dict())
    assert restored.audit() == graph.audit()
    assert restored.prompt_summary(max_chars=200).startswith("[Task Evidence]")
    assert len(restored.prompt_summary(max_chars=200)) <= 200


def test_failed_verification_contradicts_a_change() -> None:
    graph = EvidenceGraph("Fix cache expiration")
    graph.add_change(tool="edit_file", path="cache.py", step_number=1)
    graph.add_verification(tool="run_tests", step_number=2, passed=False)

    assert graph.status() == "contradicted"
    assert graph.audit()["failed_verifications"] == 1


def test_evidence_capability_injects_bounded_summary_and_blocks_once() -> None:
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
    assert len(outgoing[-1]["content"]) <= 300

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
    assert bus.emit_before_finish(finish) == {
        "block": True,
        "reason": (
            "Completion evidence is contradicted by a failing verification. Address or explain "
            "the failing check before claiming success. This evidence gate blocks only once."
        ),
    }
    assert bus.emit_before_finish(finish) is None
    assert metadata["evidence_contradiction_block_count"] == 1


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
