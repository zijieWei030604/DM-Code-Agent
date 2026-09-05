"""Low-intervention decision evidence graph capability."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeLLMRequestEvent,
    BeforeToolCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceEdge, EvidenceGraph, EvidenceNode
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.core.observation import is_failure_observation
from dm_agent.core.workspace_version import workspace_version

READ_ACTIONS = frozenset({"read_file", "search_in_file"})
VERIFICATION_ACTIONS = frozenset({"run_python", "run_tests", "run_linter"})


class EvidenceGraphCapability:
    """Collect provenance and provide bounded progress feedback to the model.

    It never starts tools, replans, or blocks exploration. The only hard gate is
    a single rejection when a known failing verification contradicts a success
    claim; a later completion request is allowed and remains visibly marked as
    contradicted in the audit report.
    """

    checkpoint_key = "evidence_graph"

    def __init__(self, *, summary_chars: int = 800, inject_summaries: bool = True) -> None:
        if summary_chars < 200:
            raise ValueError("summary_chars must be at least 200.")
        self.summary_chars = summary_chars
        self.inject_summaries = inject_summaries
        self.graph = EvidenceGraph()
        self._trace_writer: Any | None = None
        self._get_run_state: Any = None
        self._workspace_root = Path.cwd()
        self._verification_versions: dict[int, str] = {}

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        self._get_run_state = context.get_run_state
        context.event_bus.on("on_run_start", self._on_run_start, name="evidence.run_start")
        context.event_bus.on(
            "before_tool_call",
            self._before_tool_call,
            name="evidence.check_version",
            kind="observer",
        )
        context.event_bus.on(
            "after_tool_result", self._after_tool_result, name="evidence.after_tool"
        )
        context.event_bus.on(
            "before_llm_request", self._before_llm_request, name="evidence.before_llm"
        )
        context.event_bus.on("before_finish", self._before_finish, name="evidence.before_finish")
        context.event_bus.on("on_run_end", self._on_run_end, name="evidence.run_end")

    def export_state(self) -> dict[str, Any]:
        return self.graph.to_dict()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.graph = EvidenceGraph.from_dict(state)

    def _on_run_start(self, event: RunStartEvent) -> None:
        nodes, edges = self.graph.start(event.task)
        self._workspace_root = Path.cwd()
        self._record(nodes, edges)
        event.metadata.update(
            {
                "evidence_graph_enabled": True,
                "evidence_summary_injection_enabled": self.inject_summaries,
                "evidence_summary_injection_count": 0,
                "evidence_contradiction_block_count": 0,
            }
        )

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.tool_name in VERIFICATION_ACTIONS:
            self._verification_versions[event.step_number] = workspace_version(self._workspace_root)

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        self._sync_plan()
        path_value = event.arguments.get("path")
        path = path_value if isinstance(path_value, str) else ""
        nodes: list[EvidenceNode] = []
        edges: list[EvidenceEdge] = []
        if event.tool_name in READ_ACTIONS:
            if event.tool_succeeded:
                nodes, edges = self.graph.add_observation(
                    tool=event.tool_name,
                    path=path,
                    step_number=event.step_number,
                    succeeded=True,
                )
        elif event.tool_name in WRITE_ACTIONS:
            if event.has_effect and not event.no_change:
                nodes, edges = self.graph.add_change(
                    tool=event.tool_name,
                    path=path,
                    step_number=event.step_number,
                )
        elif event.tool_name in VERIFICATION_ACTIONS:
            if event.result is not None and event.result.status in {
                "unavailable",
                "cancelled",
                "unknown",
            }:
                self._record_event(
                    "evidence_check_unavailable",
                    {"tool": event.tool_name, "status": event.result.status},
                )
                return
            self.graph.workspace_version = workspace_version(self._workspace_root)
            before = self._verification_versions.pop(
                event.step_number, self.graph.workspace_version
            )
            if before != self.graph.workspace_version:
                self._record_event(
                    "evidence_check_unavailable",
                    {"tool": event.tool_name, "status": "workspace_changed_during_check"},
                )
                return
            passed = (
                event.result.status == "success"
                if event.result is not None
                else event.tool_succeeded
                and not is_failure_observation(event.observation, action=event.tool_name)
            )
            nodes, edges = self.graph.add_verification(
                tool=event.tool_name,
                step_number=event.step_number,
                passed=passed,
                workspace_version=self.graph.workspace_version,
                check=json.dumps(
                    {"tool": event.tool_name, "arguments": event.arguments}, sort_keys=True
                ),
                scope=(
                    event.result.check_scope
                    if event.result
                    else (
                        str(
                            event.arguments.get(
                                "test_path", event.arguments.get("path", "unspecified")
                            )
                        ),
                    )
                ),
            )
        if nodes or edges:
            self._record(nodes, edges)
            self._update_metadata(event.metadata)

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        self._sync_plan()
        if not self.inject_summaries or event.phase != "agent" or not self.graph.summary_pending:
            return
        summary = self.graph.prompt_summary(max_chars=self.summary_chars)
        if summary:
            event.messages.append({"role": "system", "content": summary})
            event.metadata["evidence_summary_injection_count"] = (
                int(event.metadata.get("evidence_summary_injection_count", 0)) + 1
            )
            self._record_event(
                "evidence_summary_injected",
                {
                    "step_number": event.step_number,
                    "chars": len(summary),
                    "status": self.graph.status(),
                },
            )
        self.graph.summary_pending = False

    def _before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        self.graph.workspace_version = workspace_version(self._workspace_root)
        self._sync_plan()
        nodes, edges = self.graph.add_conclusion(
            text=event.completion_text,
            step_number=event.step_number,
        )
        self._record(nodes, edges)
        self._update_metadata(event.metadata)
        if self.graph.status() != "contradicted" or self.graph.contradiction_blocked_once:
            return None
        self.graph.contradiction_blocked_once = True
        event.metadata["evidence_contradiction_block_count"] = (
            int(event.metadata.get("evidence_contradiction_block_count", 0)) + 1
        )
        reason = (
            "Completion evidence is contradicted by a failing verification. Address or explain "
            "the failing check before claiming success. This evidence gate blocks only once."
        )
        self._record_event(
            "evidence_completion_contradicted",
            {"step_number": event.step_number, "reason": reason},
        )
        return {"block": True, "reason": reason}

    def _on_run_end(self, event: RunEndEvent) -> None:
        self._sync_plan()
        audit = self.graph.audit()
        self._update_metadata(event.metadata)
        self._record_event("evidence_summary", audit)

    def _sync_plan(self) -> None:
        if not callable(self._get_run_state):
            return
        state = self._get_run_state()
        if not isinstance(state, Mapping):
            return
        raw_plan = state.get("plan")
        if not isinstance(raw_plan, list):
            return
        plan = [item for item in raw_plan if isinstance(item, Mapping)]
        nodes, edges = self.graph.sync_plan(plan)
        self._record(nodes, edges)

    def _update_metadata(self, metadata: dict[str, Any]) -> None:
        audit = self.graph.audit()
        metadata.update(
            {
                "evidence_status": audit["status"],
                "evidence_node_count": audit["node_count"],
                "evidence_edge_count": audit["edge_count"],
                "evidence_failed_verifications": audit["failed_verifications"],
            }
        )

    def _record(self, nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> None:
        for node in nodes:
            self._record_event("evidence_node", node.to_dict())
        for edge in edges:
            self._record_event("evidence_edge", edge.to_dict())

    def _record_event(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)
