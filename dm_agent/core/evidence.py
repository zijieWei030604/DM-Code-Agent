"""Task evidence graph primitives for explainable Agent runs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
        self.summary_pending = False
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
        self.summary_pending = False
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
        self, *, tool: str, path: str, step_number: int, succeeded: bool
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        node = self._new_node(
            "observation",
            _tool_title(tool, path, succeeded),
            step_number,
            {"tool": tool, "path": path, "succeeded": succeeded},
        )
        edges = self._link_current_plan(node.node_id, "supported_by")
        return [node], edges

    def add_change(
        self, *, tool: str, path: str, step_number: int
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        node = self._new_node(
            "change",
            f"Changed {path or '<workspace>'}",
            step_number,
            {"tool": tool, "path": path},
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
        ]
        for observation in sorted(
            observations, key=lambda item: item.step_number or 0, reverse=True
        )[:3]:
            edge = self._add_edge(node.node_id, observation.node_id, "motivated_by")
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
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        direct = tool == "run_tests"
        node = self._new_node(
            "verification",
            f"{tool} {'passed' if passed else 'failed'}",
            step_number,
            {"tool": tool, "passed": passed, "direct": direct},
        )
        edges: list[EvidenceEdge] = []
        changes = [
            item
            for item in self.nodes.values()
            if item.kind == "change"
            and item.step_number is not None
            and item.step_number <= step_number
        ]
        latest_verification_step = max(
            (
                item.step_number or 0
                for item in self.nodes.values()
                if item.kind == "verification" and item.node_id != node.node_id
            ),
            default=0,
        )
        for change in changes:
            if (change.step_number or 0) <= latest_verification_step:
                continue
            edge = self._add_edge(
                node.node_id,
                change.node_id,
                "verifies" if passed else "contradicts",
                confidence="direct" if direct else "indirect",
            )
            if edge:
                edges.append(edge)
        edge = self._add_edge(
            node.node_id,
            self.ROOT_REQUIREMENT_ID,
            "verifies" if passed and direct else ("supports" if passed else "contradicts"),
            confidence="direct" if direct else "indirect",
        )
        if edge:
            edges.append(edge)
        self.summary_pending = True
        return [node], edges

    def add_conclusion(
        self, *, text: str, step_number: int
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        node = self._new_node(
            "conclusion",
            _shorten(text.strip() or "Task completion requested", 240),
            step_number,
            {"claimed_success": True},
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
        edge = self._add_edge(node.node_id, self.ROOT_REQUIREMENT_ID, "concludes")
        if edge:
            edges.append(edge)
        return [node], edges

    def status(self) -> EvidenceStatus:
        changes = self._nodes_of_kind("change")
        observations = self._nodes_of_kind("observation")
        conclusions = self._nodes_of_kind("conclusion")
        verifications = self._nodes_of_kind("verification")
        latest_change = max((node.step_number or 0 for node in changes), default=0)
        relevant = [
            node
            for node in verifications
            if (node.step_number or 0) >= latest_change and latest_change > 0
        ]
        if any(not bool(node.metadata.get("passed")) for node in relevant):
            return "contradicted"
        passed = [node for node in relevant if bool(node.metadata.get("passed"))]
        if passed:
            if conclusions and any(bool(node.metadata.get("direct")) for node in passed):
                return "verified"
            return "partially_verified"
        if changes:
            return "implemented"
        if conclusions and observations:
            return "partially_verified"
        return "unaddressed"

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
        return {
            "status": self.status(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "counts": counts,
            "failed_verifications": failed,
            "has_conclusion": bool(counts["conclusion"]),
        }

    def prompt_summary(self, *, max_chars: int = 800) -> str:
        audit = self.audit()
        changes = self._nodes_of_kind("change")
        verifications = self._nodes_of_kind("verification")
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
        if changes and not verifications:
            lines.append("Missing evidence: the current changes have not been checked yet.")
        if audit["status"] == "contradicted":
            lines.append(
                "Do not claim success until the failing verification is addressed or explained."
            )
        return _shorten("\n".join(lines), max_chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "counters": dict(self._counters),
            "current_plan_id": self.current_plan_id,
            "summary_pending": self.summary_pending,
            "contradiction_blocked_once": self.contradiction_blocked_once,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceGraph:
        graph = cls()
        graph.task = str(data.get("task", ""))
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
