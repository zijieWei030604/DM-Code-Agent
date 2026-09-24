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
    BeforeToolCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceEdge, EvidenceGraph, EvidenceNode
from dm_agent.core.evidence_policy import EvidenceCompletionPolicy, EvidenceCompletionResult
from dm_agent.core.guards import READ_ACTIONS, WRITE_ACTIONS
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
        enforcement: EvidenceEnforcement = "strict",
        repeated_contradiction: RepeatedContradiction = "critic_rejected",
    ) -> None:
        if summary_chars < 200:
            raise ValueError("summary_chars must be at least 200.")
        if enforcement not in {"observe", "warn", "strict"}:
            raise ValueError("enforcement must be 'observe', 'warn', or 'strict'.")
        if repeated_contradiction not in {"continue", "critic_rejected"}:
            raise ValueError("repeated_contradiction must be 'continue' or 'critic_rejected'.")
        self.summary_chars = summary_chars
        self.enforcement = enforcement
        self.repeated_contradiction = repeated_contradiction
        self.graph = EvidenceGraph()
        self.completion_policy = EvidenceCompletionPolicy()
        self._trace_writer: Any | None = None
        self._get_run_state: Any = None
        self._workspace_root = Path.cwd()
        self._verification_versions: dict[int, str] = {}
        self._write_versions: dict[int, str] = {}
        self._write_basis_kinds: dict[int, str] = {}
        self._initial_workspace_version = ""
        self._last_contradiction_signature = ""
        self._last_unverified_pause_signature = ""
        self._last_no_net_change_signature = ""

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
            "before_finish",
            self._before_finish,
            name="evidence.before_finish",
            kind="policy",
        )
        context.event_bus.on("on_run_end", self._on_run_end, name="evidence.run_end")

    def export_state(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "last_contradiction_signature": self._last_contradiction_signature,
            "last_unverified_pause_signature": self._last_unverified_pause_signature,
            "initial_workspace_version": self._initial_workspace_version,
            "last_no_net_change_signature": self._last_no_net_change_signature,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        graph_state = state.get("graph") if isinstance(state.get("graph"), Mapping) else state
        self.graph = EvidenceGraph.from_dict(graph_state)
        self._last_contradiction_signature = str(
            state.get("last_contradiction_signature", "")
        )
        self._last_unverified_pause_signature = str(
            state.get("last_unverified_pause_signature", "")
        )
        self._initial_workspace_version = str(state.get("initial_workspace_version", ""))
        self._last_no_net_change_signature = str(
            state.get("last_no_net_change_signature", "")
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        nodes, edges = self.graph.start(event.task)
        self._workspace_root = Path.cwd()
        self.graph.workspace_version = workspace_version(self._workspace_root)
        self._initial_workspace_version = self.graph.workspace_version
        self._verification_versions.clear()
        self._write_versions.clear()
        self._write_basis_kinds.clear()
        self._last_contradiction_signature = ""
        self._last_unverified_pause_signature = ""
        self._last_no_net_change_signature = ""
        self._record(nodes, edges)
        event.metadata.update(
            {
                "evidence_graph_enabled": True,
                "evidence_contradiction_block_count": 0,
                "evidence_completion_block_count": 0,
                "evidence_completion_pause_count": 0,
                "evidence_no_net_change_block_count": 0,
                "evidence_unverified_completion_count": 0,
                "evidence_recovered_after_pause": 0,
                "evidence_intervention_prompt_count": 0,
                "evidence_terminal_rejection_count": 0,
                "evidence_enforcement": self.enforcement,
                "evidence_schema_version": 3,
                "evidence_verification_state": "not_run",
            }
        )

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        if _is_verification_action(event.tool_name, event.arguments):
            self._verification_versions[event.step_number] = workspace_version(self._workspace_root)
        if event.tool_name in WRITE_ACTIONS:
            self._write_versions[event.step_number] = workspace_version(self._workspace_root)
            self._write_basis_kinds[event.step_number] = _write_basis_kind(event)

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        self._sync_plan()
        fact = event.execution_fact
        phase = fact.phase if fact is not None else ""
        if fact is not None and fact.fact_type != "other":
            self._record_event(fact.fact_type, fact.to_dict())
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
                    phase=phase,
                )
        elif event.tool_name in WRITE_ACTIONS:
            if event.has_effect and not event.no_change:
                prior_checks = bool(self.graph.current_verifications())
                self.graph.change_revision += 1
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
                        change_revision=self.graph.change_revision,
                        phase=phase,
                    )
                    nodes.extend(added_nodes)
                    edges.extend(added_edges)
                if prior_checks:
                    self._record_event(
                        "verification_invalidated",
                        {"step_number": event.step_number, "workspace_version": after_version},
                    )
            else:
                self._write_versions.pop(event.step_number, None)
                self._write_basis_kinds.pop(event.step_number, None)
        elif _is_verification_action(event.tool_name, event.arguments):
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
            verification = _verification_result(event)
            passed = verification["outcome"] == "passed"
            blocking, failure_kind = self._blocking_verification(
                verification,
                tool=event.tool_name,
                details=event.observation,
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
                execution_status=str(verification["execution_status"]),
                outcome=str(verification["outcome"]),
                failure_kind=failure_kind,
                blocking=blocking,
                change_revision=self.graph.change_revision,
                phase=phase,
                scope_level=str(verification["scope_level"]),
            )
        if nodes or edges:
            self._record(nodes, edges)
            self._update_metadata(event.metadata)

    def _before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        self.graph.workspace_version = workspace_version(self._workspace_root)
        self._sync_plan()
        decision = self.completion_policy.evaluate(
            self.graph,
            verified_transaction=str(
                event.metadata.get("edit_transaction_status", "")
            ).startswith("committed"),
        )
        no_net_change_signature = self._no_net_change_signature()
        if (
            decision.decision != "block"
            and no_net_change_signature
            and no_net_change_signature != self._last_no_net_change_signature
        ):
            self._last_no_net_change_signature = no_net_change_signature
            reason = (
                "Completion paused once: code or configuration was edited during this run, "
                "but no net workspace change remains. Re-check the requested behavior and "
                "leave the intended change in the workspace, or finish again only if no "
                "change is genuinely required."
            )
            nodes, edges = self.graph.add_conclusion(
                text=event.completion_text,
                step_number=event.step_number,
                accepted=False,
                evidence_status="no_net_change",
            )
            self._record(nodes, edges)
            event.metadata["evidence_completion_block_count"] = (
                int(event.metadata.get("evidence_completion_block_count", 0)) + 1
            )
            event.metadata["evidence_no_net_change_block_count"] = (
                int(event.metadata.get("evidence_no_net_change_block_count", 0)) + 1
            )
            event.metadata["evidence_intervention_prompt_count"] = (
                int(event.metadata.get("evidence_intervention_prompt_count", 0)) + 1
            )
            event.metadata["evidence_completion_status"] = "no_net_change"
            self._update_metadata(event.metadata)
            self._record_event(
                "evidence_no_net_change_blocked",
                {
                    "step_number": event.step_number,
                    "reason": reason,
                    "change_revision": self.graph.change_revision,
                    "workspace_version": self.graph.workspace_version,
                },
            )
            return {"block": True, "reason": reason}
        issues = list(decision.issues)
        warnings = list(decision.warnings)
        should_block = self.enforcement == "strict" and decision.decision == "block"
        pause_unverified = False
        completion_signature = self._completion_signature(decision)
        if (
            self.enforcement == "strict"
            and decision.decision == "warn"
            and self._requires_verification()
            and completion_signature != self._last_unverified_pause_signature
        ):
            should_block = True
            pause_unverified = True
        if self.enforcement == "warn":
            should_block = any(item["status"] == "contradicted" for item in issues)
        nodes, edges = self.graph.add_conclusion(
            text=event.completion_text,
            step_number=event.step_number,
            accepted=not should_block,
            evidence_status=(
                "unverified"
                if not should_block and decision.decision == "warn"
                else decision.verification_state
            ),
        )
        self._record(nodes, edges)
        self._update_metadata(event.metadata, decision=decision)
        if not should_block:
            event.metadata["evidence_verification_state"] = decision.verification_state
            if decision.decision == "warn" and self._requires_verification():
                event.metadata["evidence_unverified_completion_count"] = (
                    int(event.metadata.get("evidence_unverified_completion_count", 0)) + 1
                )
                event.metadata["evidence_completion_status"] = "unverified"
            else:
                event.metadata["evidence_completion_status"] = decision.verification_state
            if self._last_unverified_pause_signature and decision.verification_state == "tested":
                event.metadata["evidence_recovered_after_pause"] = (
                    int(event.metadata.get("evidence_recovered_after_pause", 0)) + 1
                )
            self._record_event(
                "completion_accepted",
                {
                    "step_number": event.step_number,
                    "verification_state": decision.verification_state,
                    "status": event.metadata["evidence_completion_status"],
                },
            )
            return None
        event.metadata["evidence_completion_block_count"] = (
            int(event.metadata.get("evidence_completion_block_count", 0)) + 1
        )
        if any(
            item["status"] in {"contradicted", "confirmed_test_failure", "confirmed_code_error"}
            for item in issues
        ):
            event.metadata["evidence_contradiction_block_count"] = (
                int(event.metadata.get("evidence_contradiction_block_count", 0)) + 1
            )
        signature = completion_signature
        if pause_unverified:
            self._last_unverified_pause_signature = signature
            event.metadata["evidence_completion_pause_count"] = (
                int(event.metadata.get("evidence_completion_pause_count", 0)) + 1
            )
            event.metadata["evidence_intervention_prompt_count"] = (
                int(event.metadata.get("evidence_intervention_prompt_count", 0)) + 1
            )
            reason = (
                "Completion paused once: code or configuration changed, but the current "
                "workspace has no successful verification. Run a relevant check if available; "
                "otherwise you may finish again and the result will be recorded as unverified."
            )
            self._record_event(
                "completion_paused",
                {
                    "step_number": event.step_number,
                    "reason": reason,
                    "verification_state": decision.verification_state,
                    "signature": signature[:16],
                },
            )
            return {"block": True, "reason": reason}
        repeated = signature == self._last_contradiction_signature
        if not repeated:
            self._last_contradiction_signature = signature
            reason = _completion_intervention_reason(self.graph, issues)
            event.metadata["evidence_intervention_prompt_count"] = (
                int(event.metadata.get("evidence_intervention_prompt_count", 0)) + 1
            )
        else:
            reason = (
                "Completion rejected: the same explicit verification failure is still present."
            )
        terminal = (
            self.repeated_contradiction == "critic_rejected"
            and repeated
        )
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
                "detailed": not repeated,
                "terminal": terminal,
                "signature": signature[:16],
            },
        )
        return {"block": True, "reason": reason}

    def _no_net_change_signature(self) -> str:
        changed_paths = [
            path for path in self.graph.changed_paths() if _path_requires_verification(path)
        ]
        if (
            not self._initial_workspace_version
            or not changed_paths
            or self.graph.workspace_version != self._initial_workspace_version
        ):
            return ""
        payload = {
            "initial_workspace_version": self._initial_workspace_version,
            "workspace_version": self.graph.workspace_version,
            "change_revision": self.graph.change_revision,
            "changed_paths": sorted(changed_paths),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _requires_verification(self) -> bool:
        return any(_path_requires_verification(path) for path in self.graph.changed_paths())

    def _blocking_verification(
        self,
        verification: Mapping[str, Any],
        *,
        tool: str,
        details: str,
    ) -> tuple[bool, str]:
        outcome = str(verification.get("outcome", "unknown"))
        failure_kind = str(verification.get("failure_kind", ""))
        scope_level = str(verification.get("scope_level", "related"))

        if _current_change_has_syntax_error(self.graph, details):
            return True, "syntax_error"
        if (
            tool not in {"run_tests", "run_shell"}
            or outcome != "failed"
            or failure_kind != "assertion_failure"
        ):
            return False, failure_kind
        if scope_level == "direct":
            return True, failure_kind
        return False, "suite_failure"

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
                "evidence_verification_state": metadata.get(
                    "evidence_verification_state", _verification_state(self.graph)
                ),
            }
        )
        if decision is not None:
            metadata.update(
                {
                    "evidence_completion_decision": decision.decision,
                    "evidence_verification_state": decision.verification_state,
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
        "confirmed_test_failure": "has a confirmed failing targeted test",
        "confirmed_code_error": "has a syntax error in a changed file",
    }
    detail = "; ".join(
        f"{item['path']}: {labels.get(item['status'], item['status'])}" for item in issues[:5]
    )
    suffix = f"; and {len(issues) - 5} more" if len(issues) > 5 else ""
    return f"Completion blocked by evidence policy: {detail}{suffix}."


def _verification_state(graph: EvidenceGraph) -> str:
    checks = graph.current_verifications()
    if any(
        not bool(node.metadata.get("passed"))
        and bool(node.metadata.get("blocking", False))
        for node in checks
    ):
        return "contradicted"
    if any(bool(node.metadata.get("passed")) for node in checks):
        return "tested"
    if checks:
        return "unavailable"
    return "not_run"


def _completion_intervention_reason(graph: EvidenceGraph, issues: list[dict[str, str]]) -> str:
    paths = list(dict.fromkeys(item["path"] for item in issues if item["path"]))
    failed_checks = [
        node
        for node in graph.current_verifications()
        if not bool(node.metadata.get("passed"))
        and bool(node.metadata.get("blocking", False))
    ]
    lines = ["Completion blocked by evidence policy."]
    if paths:
        lines.append("Changed files requiring attention: " + ", ".join(paths[:5]) + ".")
    if failed_checks:
        latest = failed_checks[-1]
        label = (
            "Latest failed test"
            if latest.metadata.get("tool") == "run_tests"
            else "Latest failed verification"
        )
        lines.append(
            label
            + ": "
            + str(latest.metadata.get("check") or latest.metadata.get("tool") or "check")
            + "."
        )
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


def _verification_result(event: AfterToolResultEvent) -> dict[str, Any]:
    metadata = event.result.metadata if event.result is not None else {}
    raw = metadata.get("verification") if isinstance(metadata, Mapping) else None
    if isinstance(raw, Mapping):
        return {
            "execution_status": str(raw.get("execution_status", "unavailable")),
            "outcome": str(raw.get("outcome", "unknown")),
            "failure_kind": str(raw.get("failure_kind", "")),
            "scope": list(raw.get("scope") or (event.result.check_scope if event.result else ())),
            "scope_level": str(raw.get("scope_level", "related")),
        }

    passed = (
        event.result.status == "success" if event.result is not None else event.tool_succeeded
    )
    return {
        "execution_status": "completed" if passed else "unavailable",
        "outcome": "passed" if passed else "unknown",
        "failure_kind": "" if passed else str(event.result.error_code if event.result else ""),
        "scope": list(event.result.check_scope if event.result else ()),
        "scope_level": "related",
    }


def _is_verification_action(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    if tool_name in VERIFICATION_ACTIONS:
        return True
    return tool_name == "run_shell" and arguments.get("purpose") == "verification"


def _is_targeted_test_scope(scope: tuple[str, ...]) -> bool:
    if not scope:
        return False
    for target in scope:
        path_part = target.split("::", 1)[0].replace("\\", "/")
        if "::" not in target and Path(path_part).suffix.lower() != ".py":
            return False
    return True


def _has_prior_passing_scope(graph: EvidenceGraph, scope: tuple[str, ...]) -> bool:
    if not scope:
        return False
    normalized = tuple(sorted(item.replace("\\", "/") for item in scope))
    for node in graph.nodes.values():
        if node.kind != "verification" or not bool(node.metadata.get("passed")):
            continue
        prior_scope = tuple(
            sorted(str(item).replace("\\", "/") for item in node.metadata.get("scope") or ())
        )
        if prior_scope == normalized:
            return True
    return False


def _current_change_has_syntax_error(graph: EvidenceGraph, details: str) -> bool:
    if "SyntaxError" not in details and "IndentationError" not in details:
        return False
    normalized = details.replace("\\", "/").lower()
    current_paths = [
        str(node.metadata.get("path", "")).replace("\\", "/")
        for node in graph.nodes.values()
        if node.kind == "change"
        and int(node.metadata.get("change_revision", -1)) == graph.change_revision
    ]
    return any(
        path and (path.lower() in normalized or Path(path).name.lower() in normalized)
        for path in current_paths
    )


def _path_requires_verification(path: str) -> bool:
    """Classify files whose behavior can change; docs and metadata remain advisory."""
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    if Path(normalized).suffix in {
        ".md", ".rst", ".txt", ".adoc", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    }:
        return False
    if name in {
        "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "package.json",
        "dockerfile", "makefile", "requirements.txt",
    }:
        return True
    return Path(normalized).suffix in {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go",
        ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
        ".swift", ".scala", ".sh", ".ps1", ".toml", ".yaml", ".yml", ".json",
        ".ini", ".cfg", ".xml",
    }
