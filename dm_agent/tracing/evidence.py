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
    return EvidenceGraph.from_dict({"task": task, "nodes": nodes, "edges": edges})


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
    verified_change_ids = {
        edge.target_id
        for edge in graph.edges
        if edge.relation == "verifies"
        and graph.nodes.get(edge.source_id) is not None
        and graph.nodes[edge.source_id].kind == "verification"
        and bool(graph.nodes[edge.source_id].metadata.get("passed"))
    }
    unverified_changes = [
        {
            "node_id": node.node_id,
            "path": str(node.metadata.get("path") or ""),
            "step_number": node.step_number,
        }
        for node in graph.nodes.values()
        if node.kind == "change" and node.node_id not in verified_change_ids
    ]
    return {
        "enabled": True,
        **audit,
        "unverified_changes": unverified_changes,
    }
