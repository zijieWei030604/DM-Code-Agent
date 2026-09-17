"""Read-side reconstruction and audit of decision evidence trace events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dm_agent.core.evidence import EvidenceGraph


def rebuild_evidence_graph(events: Sequence[Mapping[str, Any]]) -> EvidenceGraph:
    """Rebuild the latest evidence graph from append-only trace events."""
    nodes = []
    edges = []
    task = ""
    for event in events:
        name = event.get("event")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if name == "run_start":
            task = str(payload.get("task", task))
        elif name == "evidence_node":
            nodes.append(dict(payload))
        elif name == "evidence_edge":
            edges.append(dict(payload))
    graph = EvidenceGraph.from_dict({"task": task, "nodes": nodes, "edges": edges})
    versioned = [
        node
        for node in graph.nodes.values()
        if node.kind in {"change", "verification"} and node.step_number is not None
    ]
    if versioned:
        latest = max(versioned, key=lambda node: node.step_number or 0)
        graph.workspace_version = str(
            latest.metadata.get("after_version") or latest.metadata.get("workspace_version") or ""
        )
    return graph


def analyze_evidence_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a compact, deterministic evidence coverage report."""
    graph = rebuild_evidence_graph(events)
    enabled = bool(graph.nodes)
    if not enabled:
        return {
            "enabled": False,
            "status": "unmeasured",
            "node_count": 0,
            "edge_count": 0,
            "counts": {},
            "failed_verifications": 0,
            "unverified_changes": [],
        }
    audit = graph.audit()
    unverified_changes = [
        {
            "node_id": issue["node_id"],
            "path": issue["path"],
            "step_number": graph.nodes[issue["node_id"]].step_number,
            "status": issue["status"],
        }
        for issue in graph.completion_issues()
    ]
    return {
        "enabled": True,
        **audit,
        "unverified_changes": unverified_changes,
    }
