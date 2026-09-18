"""Low-intervention decision evidence graph capability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

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
from dm_agent.core.evidence_policy import EvidenceCompletionPolicy, EvidenceCompletionResult
from dm_agent.core.guards import READ_ACTIONS, WRITE_ACTIONS
from dm_agent.core.observation import is_failure_observation
from dm_agent.core.workspace_version import workspace_version

VERIFICATION_ACTIONS = frozenset({"run_python", "run_tests", "run_linter"})
EvidenceEnforcement = Literal["observe", "warn", "strict"]
RepeatedContradiction = Literal["continue", "critic_rejected"]


class EvidenceGraphCapability:
    """Collect provenance and enforce evidence requirements at completion."""

    checkpoint_key = "evidence_graph"

    def __init__(
        self,
        *,
        summary_chars: int = 800,
        inject_summaries: bool = True,
        enforcement: EvidenceEnforcement = "strict",
        max_stalled_completion_attempts: int = 2,
        repeated_contradiction: RepeatedContradiction = "continue",
        max_recovery_tool_steps: int = 8,
    ) -> None:
        if summary_chars < 200:
            raise ValueError("summary_chars must be at least 200.")
        if enforcement not in {"observe", "warn", "strict"}:
            raise ValueError("enforcement must be 'observe', 'warn', or 'strict'.")
        if max_stalled_completion_attempts < 1:
            raise ValueError("max_stalled_completion_attempts must be at least 1.")
        if repeated_contradiction not in {"continue", "critic_rejected"}:
            raise ValueError("repeated_contradiction must be 'continue' or 'critic_rejected'.")
        if max_recovery_tool_steps < 1:
            raise ValueError("max_recovery_tool_steps must be at least 1.")
        self.summary_chars = summary_chars
        self.inject_summaries = inject_summaries
        self.enforcement = enforcement
        self.max_stalled_completion_attempts = max_stalled_completion_attempts
        self.repeated_contradiction = repeated_contradiction
        self.max_recovery_tool_steps = max_recovery_tool_steps
        self.graph = EvidenceGraph()
        self.completion_policy = EvidenceCompletionPolicy()
        self._trace_writer: Any | None = None
        self._get_run_state: Any = None
        self._workspace_root = Path.cwd()
        self._verification_versions: dict[int, str] = {}
        self._write_versions: dict[int, str] = {}
        self._write_basis_kinds: dict[int, str] = {}
        self._recovery_signature = ""
        self._stalled_completion_attempts = 0
        self._last_compression_count = 0
        self._recovery_active = False
        self._recovery_tool_calls = 0
        self._recovery_progress_events = 0

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
        context.event_bus.on(
            "before_finish",
            self._before_finish,
            name="evidence.before_finish",
            kind="policy",
        )
        context.event_bus.on("on_run_end", self._on_run_end, name="evidence.run_end")

    def export_state(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "recovery_signature": self._recovery_signature,
            "stalled_completion_attempts": self._stalled_completion_attempts,
            "last_compression_count": self._last_compression_count,
            "recovery_active": self._recovery_active,
            "recovery_tool_calls": self._recovery_tool_calls,
            "recovery_progress_events": self._recovery_progress_events,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        graph_state = state.get("graph") if isinstance(state.get("graph"), Mapping) else state
        self.graph = EvidenceGraph.from_dict(graph_state)
        self._recovery_signature = str(state.get("recovery_signature", ""))
        self._stalled_completion_attempts = int(state.get("stalled_completion_attempts", 0))
        self._last_compression_count = int(state.get("last_compression_count", 0))
        self._recovery_active = bool(state.get("recovery_active", False))
        self._recovery_tool_calls = int(state.get("recovery_tool_calls", 0))
        self._recovery_progress_events = int(state.get("recovery_progress_events", 0))

    def _on_run_start(self, event: RunStartEvent) -> None:
        nodes, edges = self.graph.start(event.task)
        self._workspace_root = Path.cwd()
        self.graph.workspace_version = workspace_version(self._workspace_root)
        self._verification_versions.clear()
        self._write_versions.clear()
        self._write_basis_kinds.clear()
        self._recovery_signature = ""
        self._stalled_completion_attempts = 0
        self._last_compression_count = int(event.metadata.get("memory_compression_count", 0))
        self._recovery_active = False
        self._recovery_tool_calls = 0
        self._recovery_progress_events = 0
        self._record(nodes, edges)
        event.metadata.update(
            {
                "evidence_graph_enabled": True,
                "evidence_summary_injection_enabled": self.inject_summaries,
                "evidence_summary_injection_count": 0,
                "evidence_contradiction_block_count": 0,
                "evidence_completion_block_count": 0,
                "evidence_recovery_prompt_count": 0,
                "evidence_stalled_completion_count": 0,
                "evidence_terminal_rejection_count": 0,
                "evidence_enforcement": self.enforcement,
                "evidence_recovery_active": False,
                "evidence_recovery_tool_calls": 0,
                "evidence_recovery_progress_events": 0,
                "evidence_recovery_budget": self.max_recovery_tool_steps,
                "evidence_recovery_budget_exhausted": False,
                "evidence_recovered_after_block": False,
            }
        )

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if event.tool_name in VERIFICATION_ACTIONS:
            self._verification_versions[event.step_number] = workspace_version(self._workspace_root)
        if event.tool_name in WRITE_ACTIONS:
            self._write_versions[event.step_number] = workspace_version(self._workspace_root)
            self._write_basis_kinds[event.step_number] = _write_basis_kind(event)

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        self._sync_plan()
        path = self._normalize_path(event.arguments.get("path"))
        nodes: list[EvidenceNode] = []
        edges: list[EvidenceEdge] = []
        if event.tool_name in READ_ACTIONS:
            if event.tool_succeeded:
                nodes, edges = self.graph.add_observation(
                    tool=event.tool_name,
                    path=path,
                    step_number=event.step_number,
                    succeeded=True,
                    workspace_version=workspace_version(self._workspace_root),
                )
        elif event.tool_name in WRITE_ACTIONS:
            if event.has_effect and not event.no_change:
                before_version = self._write_versions.pop(event.step_number, "")
                basis_kind = self._write_basis_kinds.pop(event.step_number, "read")
                after_version = workspace_version(self._workspace_root)
                changed_paths = {
                    self._normalize_path(item)
                    for item in (event.result.changed_files if event.result else ())
                    if item
                }
                if path:
                    changed_paths.add(path)
                for changed_path in sorted(item for item in changed_paths if item):
                    added_nodes, added_edges = self.graph.add_change(
                        tool=event.tool_name,
                        path=changed_path,
                        step_number=event.step_number,
                        before_version=before_version,
                        after_version=after_version,
                        requires_read_basis=basis_kind == "read",
                        basis_kind=basis_kind,
                    )
                    nodes.extend(added_nodes)
                    edges.extend(added_edges)
            else:
                self._write_versions.pop(event.step_number, None)
                self._write_basis_kinds.pop(event.step_number, None)
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
                details=event.observation,
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
                # Generic tool results do not reliably identify which exact
                # changes a test covers. Keep this transaction-level fact out
                # of file-level verification edges.
                direct=False,
            )
        if nodes or edges:
            self._record(nodes, edges)
            self._update_metadata(event.metadata)
        self._advance_recovery(event)

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        self._sync_plan()
        compression_count = int(event.metadata.get("memory_compression_count", 0))
        if (
            not self.inject_summaries
            or event.phase != "agent"
            or compression_count <= self._last_compression_count
        ):
            return
        self._last_compression_count = compression_count
        summary = self.graph.prompt_summary(max_chars=self.summary_chars)
        fingerprint = hashlib.sha256(summary.encode("utf-8")).hexdigest() if summary else ""
        if summary and fingerprint != self.graph.last_summary_fingerprint:
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
                    "fingerprint": fingerprint[:16],
                    "reason": "context_compressed",
                },
            )
            self.graph.last_summary_fingerprint = fingerprint
        self.graph.summary_pending = False

    def _before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        self.graph.workspace_version = workspace_version(self._workspace_root)
        self._sync_plan()
        decision = self.completion_policy.evaluate(
            self.graph,
            verified_transaction=str(
                event.metadata.get("edit_transaction_status", "")
            ).startswith("committed"),
        )
        if self._recovery_active and decision.decision == "warn":
            decision = EvidenceCompletionResult(
                "block",
                (
                    {
                        "node_id": "evidence-recovery",
                        "path": "<workspace>",
                        "status": "recovery_verification_required",
                    },
                ),
                decision.warnings,
            )
        issues = list(decision.issues)
        warnings = list(decision.warnings)
        should_block = self.enforcement == "strict" and decision.decision == "block"
        if self.enforcement == "warn":
            should_block = any(item["status"] == "contradicted" for item in issues)
        nodes, edges = self.graph.add_conclusion(
            text=event.completion_text,
            step_number=event.step_number,
            accepted=not should_block,
        )
        self._record(nodes, edges)
        self._update_metadata(event.metadata, decision=decision)
        if not should_block:
            if self._recovery_active:
                self._finish_recovery(event.metadata, succeeded=True)
            if decision.decision == "warn":
                event.metadata["evidence_completion_status"] = "unverified"
            else:
                event.metadata["evidence_completion_status"] = "verified"
            return None
        event.metadata["evidence_completion_block_count"] = (
            int(event.metadata.get("evidence_completion_block_count", 0)) + 1
        )
        if any(
            item["status"] in {"contradicted", "transaction_test_failed"}
            for item in issues
        ):
            event.metadata["evidence_contradiction_block_count"] = (
                int(event.metadata.get("evidence_contradiction_block_count", 0)) + 1
            )
        signature = self._completion_signature(decision)
        detailed = signature != self._recovery_signature
        if detailed:
            self._recovery_signature = signature
            self._stalled_completion_attempts = 0
            reason = _completion_recovery_reason(self.graph, issues)
            event.metadata["evidence_recovery_prompt_count"] = (
                int(event.metadata.get("evidence_recovery_prompt_count", 0)) + 1
            )
            if not self._recovery_active:
                self._start_recovery(event.metadata)
        else:
            self._stalled_completion_attempts += 1
            event.metadata["evidence_stalled_completion_count"] = (
                int(event.metadata.get("evidence_stalled_completion_count", 0)) + 1
            )
            reason = (
                "Completion blocked by evidence policy: verification state has not changed. "
                "Make a new relevant edit or run a relevant test before trying to finish again."
            )
        repeated_contradiction = detailed is False and any(
            item["status"] in {"contradicted", "transaction_test_failed"}
            for item in issues
        )
        repeated_recovery_stall = detailed is False and self._recovery_active
        terminal = (
            self.repeated_contradiction == "critic_rejected"
            and (repeated_contradiction or repeated_recovery_stall)
        ) or self._stalled_completion_attempts >= self.max_stalled_completion_attempts
        if terminal:
            event.metadata["evidence_terminal_completion_rejection"] = True
            event.metadata["evidence_terminal_rejection_count"] = (
                int(event.metadata.get("evidence_terminal_rejection_count", 0)) + 1
            )
            reason = (
                "Completion rejected by evidence policy after repeated attempts without new "
                "verification evidence."
            )
        self._record_event(
            "evidence_completion_blocked",
            {
                "step_number": event.step_number,
                "reason": reason,
                "issues": issues,
                "warnings": warnings,
                "decision": decision.decision,
                "detailed": detailed,
                "terminal": terminal,
                "signature": signature[:16],
            },
        )
        return {"block": True, "reason": reason}

    def _start_recovery(self, metadata: dict[str, Any]) -> None:
        self._recovery_active = True
        self._recovery_tool_calls = 0
        self._recovery_progress_events = 0
        metadata.update(
            {
                "evidence_recovery_active": True,
                "evidence_recovery_tool_calls": 0,
                "evidence_recovery_progress_events": 0,
                "evidence_recovery_budget_exhausted": False,
            }
        )
        self._record_event(
            "evidence_recovery_started",
            {"budget": self.max_recovery_tool_steps},
        )

    def _advance_recovery(self, event: AfterToolResultEvent) -> None:
        if not self._recovery_active:
            return
        self._recovery_tool_calls += 1
        progressed = False
        if event.tool_name in WRITE_ACTIONS:
            progressed = bool(event.has_effect and not event.no_change)
        elif event.tool_name in VERIFICATION_ACTIONS:
            progressed = event.result is None or event.result.status not in {
                "unavailable",
                "cancelled",
                "unknown",
            }
        if progressed:
            self._recovery_progress_events += 1

        event.metadata.update(
            {
                "evidence_recovery_active": True,
                "evidence_recovery_tool_calls": self._recovery_tool_calls,
                "evidence_recovery_progress_events": self._recovery_progress_events,
            }
        )
        decision = self.completion_policy.evaluate(
            self.graph,
            verified_transaction=str(
                event.metadata.get("edit_transaction_status", "")
            ).startswith("committed"),
        )
        if decision.decision == "allow":
            self._finish_recovery(event.metadata, succeeded=True)
            return
        if self._recovery_tool_calls < self.max_recovery_tool_steps:
            return

        event.metadata["evidence_recovery_budget_exhausted"] = True
        event.metadata["evidence_terminal_completion_rejection"] = True
        event.metadata["evidence_terminal_rejection_count"] = (
            int(event.metadata.get("evidence_terminal_rejection_count", 0)) + 1
        )
        self._record_event(
            "evidence_recovery_exhausted",
            {
                "tool_calls": self._recovery_tool_calls,
                "progress_events": self._recovery_progress_events,
                "budget": self.max_recovery_tool_steps,
            },
        )

    def _finish_recovery(self, metadata: dict[str, Any], *, succeeded: bool) -> None:
        if not self._recovery_active:
            return
        metadata["evidence_recovery_active"] = False
        metadata["evidence_recovered_after_block"] = succeeded
        self._record_event(
            "evidence_recovery_finished",
            {
                "succeeded": succeeded,
                "tool_calls": self._recovery_tool_calls,
                "progress_events": self._recovery_progress_events,
            },
        )
        self._recovery_active = False

    def _completion_signature(self, decision: EvidenceCompletionResult) -> str:
        current_checks = [
            {
                "tool": str(node.metadata.get("tool", "")),
                "passed": bool(node.metadata.get("passed")),
                "check": str(node.metadata.get("check", "")),
                "scope": list(node.metadata.get("scope") or ()),
            }
            for node in self.graph.current_verifications()
        ]
        payload = {
            "workspace_version": self.graph.workspace_version,
            "issues": sorted(decision.issues, key=lambda item: (item["path"], item["status"])),
            "checks": current_checks,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _on_run_end(self, event: RunEndEvent) -> None:
        self._sync_plan()
        audit = self.graph.audit()
        self._update_metadata(event.metadata)
        event.metadata.update(
            {
                "evidence_completion_attempts": audit["completion_attempts"],
                "evidence_rejected_completion_attempts": audit[
                    "rejected_completion_attempts"
                ],
            }
        )
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

    def _update_metadata(
        self,
        metadata: dict[str, Any],
        *,
        decision: EvidenceCompletionResult | None = None,
    ) -> None:
        audit = self.graph.audit()
        completion_issues = (
            list(decision.issues)
            if decision is not None
            else list(metadata.get("evidence_completion_issues", audit["completion_issues"]))
        )
        metadata.update(
            {
                "evidence_status": audit["status"],
                "evidence_node_count": audit["node_count"],
                "evidence_edge_count": audit["edge_count"],
                "evidence_failed_verifications": audit["failed_verifications"],
                "evidence_change_evidence_gaps": audit["completion_issues"],
                "evidence_completion_issues": completion_issues,
                "evidence_completion_status": metadata.get(
                    "evidence_completion_status", "pending"
                ),
            }
        )
        if decision is not None:
            metadata.update(
                {
                    "evidence_completion_decision": decision.decision,
                    "evidence_policy_issues": list(decision.issues),
                    "evidence_policy_warnings": list(decision.warnings),
                }
            )

    def _normalize_path(self, value: Any) -> str:
        if not isinstance(value, (str, Path)) or not str(value):
            return ""
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        try:
            return candidate.resolve().relative_to(self._workspace_root.resolve()).as_posix()
        except ValueError:
            return candidate.resolve().as_posix()

    def _record(self, nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> None:
        for node in nodes:
            self._record_event("evidence_node", node.to_dict())
        for edge in edges:
            self._record_event("evidence_edge", edge.to_dict())

    def _record_event(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)


def _completion_block_reason(issues: list[dict[str, str]]) -> str:
    labels = {
        "missing_read_basis": "missing a valid pre-edit read",
        "unverified": "has not been verified",
        "indirectly_checked": "has only indirect checks",
        "contradicted": "has a failing verification",
        "source_unverified": "has no successful test for the current change transaction",
        "source_indirect_only": "has only indirect checks for the current change transaction",
        "transaction_test_failed": "has a failing test for the current change transaction",
        "recovery_verification_required": "requires a successful current-version verification",
    }
    detail = "; ".join(
        f"{item['path']}: {labels.get(item['status'], item['status'])}" for item in issues[:5]
    )
    suffix = f"; and {len(issues) - 5} more" if len(issues) > 5 else ""
    return f"Completion blocked by evidence policy: {detail}{suffix}."


def _completion_recovery_reason(graph: EvidenceGraph, issues: list[dict[str, str]]) -> str:
    paths = list(dict.fromkeys(item["path"] for item in issues if item["path"]))
    failed_tests = [
        node
        for node in graph.current_verifications()
        if node.metadata.get("tool") == "run_tests" and not bool(node.metadata.get("passed"))
    ]
    lines = ["Completion blocked by evidence policy."]
    if paths:
        lines.append("Changed files requiring attention: " + ", ".join(paths[:5]) + ".")
    if failed_tests:
        latest = failed_tests[-1]
        lines.append("Latest failed test: " + str(latest.metadata.get("check") or "run_tests") + ".")
        details = str(latest.metadata.get("details") or "").strip()
        if details:
            lines.append("Failure excerpt: " + details[-600:])
    else:
        lines.append(_completion_block_reason(issues))
    lines.append(
        "Next: inspect the failure, make a relevant correction, and run a relevant test before finishing."
    )
    return "\n".join(lines)


def _write_basis_kind(event: BeforeToolCallEvent) -> str:
    """Classify write preconditions already enforced by the underlying tool."""
    if event.tool_name == "create_file":
        return "new_file"
    old_string = event.arguments.get("old_string")
    if event.content_anchor_safe and isinstance(old_string, str) and old_string:
        return "content_anchor"
    expected_hash = event.arguments.get("expected_hash")
    if event.tool_name == "edit_python_symbol" and isinstance(expected_hash, str) and expected_hash:
        return "expected_hash"
    return "read"
