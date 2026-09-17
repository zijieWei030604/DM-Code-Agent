"""Task evidence graph primitives for explainable Agent runs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

EvidenceKind = Literal[
    "requirement",
    "plan_step",
    "observation",
    "change",
    "verification",
    "conclusion",
]
EvidenceStatus = Literal[
    "unaddressed",
    "implemented",
    "partially_verified",
    "verified",
    "contradicted",
]
ChangeEvidenceStatus = Literal[
    "missing_read_basis",
    "unverified",
    "indirectly_checked",
    "verified",
    "contradicted",
]


@dataclass(frozen=True)
class EvidenceNode:
    """One auditable fact or claim in an Agent run."""

    node_id: str
    kind: EvidenceKind
    title: str
    step_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceNode:
        return cls(
            node_id=str(data.get("node_id", "")),
            kind=str(data.get("kind", "observation")),  # type: ignore[arg-type]
            title=str(data.get("title", "")),
            step_number=(int(data["step_number"]) if data.get("step_number") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceEdge:
    """A directed, typed relationship between two evidence nodes."""

    source_id: str
    target_id: str
    relation: str
    confidence: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceEdge:
        return cls(
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            relation=str(data.get("relation", "")),
            confidence=str(data.get("confidence", "deterministic")),
        )


class EvidenceGraph:
    """Small in-memory graph backed by trace/checkpoint serialization.

    The graph records provenance. It deliberately does not infer that every
    passing command proves every requirement: only a passing targeted test is
    treated as direct verification; other checks remain supporting evidence.
    """

    ROOT_REQUIREMENT_ID = "requirement-1"

    def __init__(self, task: str = "") -> None:
        self.task = task
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._counters: dict[str, int] = {}
        self.current_plan_id = ""
        self.workspace_version = ""
        self.summary_pending = False
        self.last_summary_fingerprint = ""
        self.contradiction_blocked_once = False
        if task:
            self.start(task)

    def start(self, task: str) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        """Reset for a new task and create its root requirement."""
        self.task = task
        self.nodes.clear()
        self.edges.clear()
        self._edge_keys.clear()
        self._counters.clear()
        self.current_plan_id = ""
        self.workspace_version = ""
        self.summary_pending = False
        self.last_summary_fingerprint = ""
        self.contradiction_blocked_once = False
        node = EvidenceNode(
            self.ROOT_REQUIREMENT_ID,
            "requirement",
            _compact_task(task),
            metadata={"source": "task"},
        )
        self.nodes[node.node_id] = node
        return [node], []

    def sync_plan(
        self, plan: Sequence[Mapping[str, Any]]
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        """Mirror the current plan without asking another model to classify it."""
        added_nodes: list[EvidenceNode] = []
        added_edges: list[EvidenceEdge] = []
        current = ""
        for index, raw in enumerate(plan, start=1):
            step_number = int(raw.get("step_number", raw.get("step", index)))
            action = str(raw.get("action", ""))
            reason = str(raw.get("reason", ""))
            signature = hashlib.sha256(f"{step_number}\0{action}\0{reason}".encode()).hexdigest()[
                :8
            ]
            node_id = f"plan-{step_number}-{signature}"
            completed = bool(raw.get("completed", False))
            existing = self.nodes.get(node_id)
            metadata = {"action": action, "completed": completed}
            if existing is None:
                node = EvidenceNode(node_id, "plan_step", reason or action, step_number, metadata)
                self.nodes[node_id] = node
                added_nodes.append(node)
                edge = self._add_edge(node_id, self.ROOT_REQUIREMENT_ID, "decomposes")
                if edge:
                    added_edges.append(edge)
            elif existing.metadata.get("completed") != completed:
                self.nodes[node_id] = EvidenceNode(
                    existing.node_id,
                    existing.kind,
                    existing.title,
                    existing.step_number,
                    metadata,
                )
            if not completed and not current:
                current = node_id
        self.current_plan_id = current
        return added_nodes, added_edges

    def add_observation(
        self,
        *,
        tool: str,
        path: str,
        step_number: int,
        succeeded: bool,
        workspace_version: str = "",
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        node = self._new_node(
            "observation",
            _tool_title(tool, path, succeeded),
            step_number,
            {
                "tool": tool,
                "path": _normalize_path(path),
                "succeeded": succeeded,
                "workspace_version": workspace_version,
            },
        )
        edges = self._link_current_plan(node.node_id, "supported_by")
        return [node], edges

    def add_change(
        self,
        *,
        tool: str,
        path: str,
        step_number: int,
        before_version: str = "",
        after_version: str = "",
        requires_read_basis: bool | None = None,
        basis_kind: str = "read",
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        if after_version:
            self.workspace_version = after_version
        elif self.workspace_version:
            self.workspace_version = f"pending-change-{step_number}"
        require_basis = (
            bool(before_version or after_version)
            if requires_read_basis is None
            else requires_read_basis
        )
        node = self._new_node(
            "change",
            f"Changed {path or '<workspace>'}",
            step_number,
            {
                "tool": tool,
                "path": _normalize_path(path),
                "before_version": before_version,
                "after_version": after_version,
                "requires_read_basis": require_basis,
                "basis_kind": basis_kind,
            },
        )
        edges = self._link_current_plan(node.node_id, "implements")
        edge = self._add_edge(
            node.node_id,
            self.ROOT_REQUIREMENT_ID,
            "claims_to_address",
            confidence="claimed",
        )
        if edge:
            edges.append(edge)
        observations = [
            item
            for item in self.nodes.values()
            if item.kind == "observation"
            and item.step_number is not None
            and item.step_number < step_number
            and bool(item.metadata.get("succeeded"))
            and _normalize_path(str(item.metadata.get("path", ""))) == _normalize_path(path)
            and str(item.metadata.get("workspace_version", "")) == before_version
        ]
        for observation in sorted(observations, key=lambda item: item.step_number or 0)[-1:]:
            edge = self._add_edge(
                node.node_id, observation.node_id, "motivated_by", confidence="direct"
            )
            if edge:
                edges.append(edge)
        self.summary_pending = True
        return [node], edges

    def add_verification(
        self,
        *,
        tool: str,
        step_number: int,
        passed: bool,
        workspace_version: str = "",
        check: str = "",
        scope: Sequence[str] = (),
        target_change_ids: Sequence[str] = (),
        direct: bool | None = None,
        details: str = "",
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        is_direct = False if direct is None else direct
        targets = tuple(
            change_id
            for change_id in dict.fromkeys(str(item) for item in target_change_ids)
            if self.nodes.get(change_id) is not None
            and self.nodes[change_id].kind == "change"
        )
        node = self._new_node(
            "verification",
            f"{tool} {'passed' if passed else 'failed'}",
            step_number,
            {
                "tool": tool,
                "passed": passed,
                "direct": is_direct,
                "workspace_version": workspace_version,
                "check": check or tool,
                "details": _shorten(details, 800),
                "scope": list(scope),
                "target_change_ids": list(targets),
            },
        )
        edges: list[EvidenceEdge] = []
        for change_id in targets:
            edge = self._add_edge(
                node.node_id,
                change_id,
                "verifies" if passed else "contradicts",
                confidence="direct" if is_direct else "indirect",
            )
            if edge:
                edges.append(edge)
        edge = self._add_edge(
            node.node_id,
            self.ROOT_REQUIREMENT_ID,
            "verifies" if passed and is_direct and targets else ("supports" if passed else "contradicts"),
            confidence="direct" if is_direct else "indirect",
        )
        if edge:
            edges.append(edge)
        self.summary_pending = True
        return [node], edges

    def add_conclusion(
        self,
        *,
        text: str,
        step_number: int,
        accepted: bool = True,
        evidence_status: str = "",
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        states = list(self.change_states().values())
        conclusion_status = evidence_status or (
            "verified"
            if accepted and states and all(item == "verified" for item in states)
            else self.status()
        )
        node = self._new_node(
            "conclusion",
            _shorten(text.strip() or "Task completion requested", 240),
            step_number,
            {
                "claimed_success": True,
                "accepted": accepted,
                "evidence_status": conclusion_status,
            },
        )
        edges: list[EvidenceEdge] = []
        supporting = [
            item
            for item in self.nodes.values()
            if item.kind in {"verification", "change", "observation"}
        ]
        for item in sorted(
            supporting, key=lambda candidate: candidate.step_number or 0, reverse=True
        )[:5]:
            edge = self._add_edge(node.node_id, item.node_id, "supported_by")
            if edge:
                edges.append(edge)
        edge = self._add_edge(
            node.node_id,
            self.ROOT_REQUIREMENT_ID,
            "concludes" if accepted else "attempts_to_conclude",
        )
        if edge:
            edges.append(edge)
        final_status = evidence_status or self.status()
        self.nodes[node.node_id] = EvidenceNode(
            node.node_id,
            node.kind,
            node.title,
            node.step_number,
            {**node.metadata, "evidence_status": final_status},
        )
        return [self.nodes[node.node_id]], edges

    def change_states(self) -> dict[str, ChangeEvidenceStatus]:
        """Return the current evidence state for every recorded change."""
        result: dict[str, ChangeEvidenceStatus] = {}
        for change in self._nodes_of_kind("change"):
            checks = self._current_checks_for_change(change.node_id)
            if any(not bool(node.metadata.get("passed")) for node in checks):
                result[change.node_id] = "contradicted"
                continue
            legacy_change = "before_version" not in change.metadata
            tool_validated_basis = str(change.metadata.get("basis_kind", "")) in {
                "content_anchor",
                "expected_hash",
            }
            has_read_basis = any(
                edge.source_id == change.node_id
                and edge.relation == "motivated_by"
                and (edge.confidence == "direct" or legacy_change)
                for edge in self.edges
            )
            if (
                bool(change.metadata.get("requires_read_basis", True))
                and not has_read_basis
                and not tool_validated_basis
            ):
                result[change.node_id] = "missing_read_basis"
                continue
            if any(
                bool(node.metadata.get("passed")) and bool(node.metadata.get("direct"))
                for node in checks
            ):
                result[change.node_id] = "verified"
            elif any(bool(node.metadata.get("passed")) for node in checks):
                result[change.node_id] = "indirectly_checked"
            else:
                result[change.node_id] = "unverified"
        return result

    def completion_issues(self) -> list[dict[str, str]]:
        """Describe per-change evidence gaps for audit, not policy enforcement."""
        issues = []
        for node_id, status in self.change_states().items():
            if status == "verified":
                continue
            node = self.nodes[node_id]
            issues.append(
                {
                    "node_id": node_id,
                    "path": str(node.metadata.get("path") or "<workspace>"),
                    "status": status,
                }
            )
        return issues

    def current_verifications(self) -> list[EvidenceNode]:
        """Return verification facts that apply to the current workspace version."""
        result = []
        for node in self._nodes_of_kind("verification"):
            version = str(node.metadata.get("workspace_version", ""))
            if self.workspace_version and version != self.workspace_version:
                continue
            result.append(node)
        return _latest_verifications(result)

    def status(self) -> EvidenceStatus:
        observations = self._nodes_of_kind("observation")
        conclusions = [
            node
            for node in self._nodes_of_kind("conclusion")
            if bool(node.metadata.get("accepted", True))
        ]
        states = list(self.change_states().values())
        if "contradicted" in states:
            return "contradicted"
        if states:
            if conclusions and all(state == "verified" for state in states):
                return "verified"
            if any(state in {"verified", "indirectly_checked"} for state in states):
                return "partially_verified"
            return "implemented"
        checks = self.current_verifications()
        if any(not bool(node.metadata.get("passed")) for node in checks):
            return "contradicted"
        if conclusions and any(
            bool(node.metadata.get("passed")) and bool(node.metadata.get("direct"))
            for node in checks
        ):
            return "verified"
        if conclusions and (observations or any(bool(node.metadata.get("passed")) for node in checks)):
            return "partially_verified"
        return "unaddressed"

    def _current_checks_for_change(self, change_id: str) -> list[EvidenceNode]:
        checks: list[EvidenceNode] = []
        for edge in self.edges:
            if edge.target_id != change_id or edge.relation not in {"verifies", "contradicts"}:
                continue
            node = self.nodes.get(edge.source_id)
            if node is None or node.kind != "verification":
                continue
            version = str(node.metadata.get("workspace_version", ""))
            if self.workspace_version and version != self.workspace_version:
                continue
            checks.append(node)
        return _latest_verifications(checks)

    def audit(self) -> dict[str, Any]:
        counts = {
            kind: len(self._nodes_of_kind(kind))
            for kind in (
                "requirement",
                "plan_step",
                "observation",
                "change",
                "verification",
                "conclusion",
            )
        }
        failed = sum(
            1
            for node in self._nodes_of_kind("verification")
            if not bool(node.metadata.get("passed"))
        )
        change_states = self.change_states()
        state_counts = {
            status: sum(1 for value in change_states.values() if value == status)
            for status in (
                "missing_read_basis",
                "unverified",
                "indirectly_checked",
                "verified",
                "contradicted",
            )
        }
        conclusions = self._nodes_of_kind("conclusion")
        accepted_conclusions = [
            node for node in conclusions if bool(node.metadata.get("accepted", True))
        ]
        return {
            "status": self.status(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "counts": counts,
            "failed_verifications": failed,
            "has_conclusion": bool(accepted_conclusions),
            "completion_attempts": len(conclusions),
            "rejected_completion_attempts": len(conclusions) - len(accepted_conclusions),
            "change_states": change_states,
            "change_status_counts": state_counts,
            "completion_issues": self.completion_issues(),
        }

    def prompt_summary(self, *, max_chars: int = 800) -> str:
        audit = self.audit()
        changes = self._nodes_of_kind("change")
        verifications = [
            node
            for node in self._nodes_of_kind("verification")
            if not self.workspace_version
            or node.metadata.get("workspace_version") == self.workspace_version
        ]
        lines = ["[Task Evidence]", f"Status: {audit['status']}"]
        if changes:
            paths = [str(node.metadata.get("path") or "<workspace>") for node in changes[-3:]]
            lines.append("Implemented: " + ", ".join(dict.fromkeys(paths)))
        passed = [node for node in verifications if bool(node.metadata.get("passed"))]
        failed = [node for node in verifications if not bool(node.metadata.get("passed"))]
        if passed:
            lines.append(
                "Verified/supporting checks: " + ", ".join(node.title for node in passed[-3:])
            )
        if failed:
            lines.append("Contradicting checks: " + ", ".join(node.title for node in failed[-3:]))
        issues = self.completion_issues()
        if issues:
            detail = ", ".join(f"{item['path']} ({item['status']})" for item in issues[:4])
            lines.append(f"Completion gaps: {detail}")
        if audit["status"] == "contradicted":
            lines.append(
                "Do not claim success until the failing verification is addressed or explained."
            )
        return _shorten("\n".join(lines), max_chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "workspace_version": self.workspace_version,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "counters": dict(self._counters),
            "current_plan_id": self.current_plan_id,
            "summary_pending": self.summary_pending,
            "last_summary_fingerprint": self.last_summary_fingerprint,
            "contradiction_blocked_once": self.contradiction_blocked_once,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceGraph:
        graph = cls()
        graph.task = str(data.get("task", ""))
        graph.workspace_version = str(data.get("workspace_version", ""))
        for raw in data.get("nodes") or []:
            if not isinstance(raw, Mapping):
                continue
            node = EvidenceNode.from_dict(raw)
            if node.node_id:
                graph.nodes[node.node_id] = node
        for raw in data.get("edges") or []:
            if not isinstance(raw, Mapping):
                continue
            edge = EvidenceEdge.from_dict(raw)
            if edge.source_id and edge.target_id and edge.relation:
                graph.edges.append(edge)
                graph._edge_keys.add((edge.source_id, edge.target_id, edge.relation))
        graph._counters = {
            str(key): int(value) for key, value in dict(data.get("counters") or {}).items()
        }
        graph.current_plan_id = str(data.get("current_plan_id", ""))
        graph.summary_pending = bool(data.get("summary_pending", False))
        graph.last_summary_fingerprint = str(data.get("last_summary_fingerprint", ""))
        graph.contradiction_blocked_once = bool(data.get("contradiction_blocked_once", False))
        return graph

    def _new_node(
        self,
        kind: EvidenceKind,
        title: str,
        step_number: int | None,
        metadata: dict[str, Any],
    ) -> EvidenceNode:
        number = self._counters.get(kind, 0) + 1
        self._counters[kind] = number
        node = EvidenceNode(f"{kind}-{number}", kind, title, step_number, metadata)
        self.nodes[node.node_id] = node
        return node

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        confidence: str = "deterministic",
    ) -> EvidenceEdge | None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        key = (source_id, target_id, relation)
        if key in self._edge_keys:
            return None
        edge = EvidenceEdge(source_id, target_id, relation, confidence)
        self.edges.append(edge)
        self._edge_keys.add(key)
        return edge

    def _link_current_plan(self, node_id: str, relation: str) -> list[EvidenceEdge]:
        if not self.current_plan_id:
            return []
        edge = self._add_edge(self.current_plan_id, node_id, relation)
        return [edge] if edge else []

    def _nodes_of_kind(self, kind: EvidenceKind) -> list[EvidenceNode]:
        return [node for node in self.nodes.values() if node.kind == kind]


def plan_snapshot(plan: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert PlanStep-like values into a capability-safe read-only view."""
    result = []
    for index, step in enumerate(plan, start=1):
        result.append(
            {
                "step_number": int(getattr(step, "step_number", index)),
                "action": str(getattr(step, "action", "")),
                "reason": str(getattr(step, "reason", "")),
                "completed": bool(getattr(step, "completed", False)),
            }
        )
    return result


def _compact_task(task: str) -> str:
    compact = " ".join(task.split())
    return _shorten(compact or "Complete the requested task", 320)


def _tool_title(tool: str, path: str, succeeded: bool) -> str:
    target = path or "<workspace>"
    return f"{tool} {'observed' if succeeded else 'failed on'} {target}"


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _normalize_path(path: str) -> str:
    normalized = PurePosixPath(str(path or "").replace("\\", "/")).as_posix()
    return normalized[2:] if normalized.startswith("./") else normalized


def _latest_verifications(nodes: Sequence[EvidenceNode]) -> list[EvidenceNode]:
    latest: dict[tuple[str, tuple[str, ...]], EvidenceNode] = {}
    for node in nodes:
        key = (
            str(node.metadata.get("check", node.metadata.get("tool", ""))),
            tuple(str(item) for item in node.metadata.get("scope", [])),
        )
        current = latest.get(key)
        if current is None or (node.step_number or 0) >= (current.step_number or 0):
            latest[key] = node
    return list(latest.values())
