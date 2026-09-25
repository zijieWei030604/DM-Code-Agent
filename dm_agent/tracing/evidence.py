"""Read-side reconstruction and audit of decision evidence trace events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dm_agent.core.evidence import EvidenceGraph


def rebuild_evidence_graph(events: Sequence[Mapping[str, Any]]) -> EvidenceGraph:
    """Rebuild the latest evidence graph from append-only trace events."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    task = ""
    snapshot_version = ""
    for event in events:
        name = event.get("event")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if name == "evidence_graph_started":
            nodes = []
            edges = []
            snapshot_version = str(payload.get("workspace_version", ""))
            task = str(payload.get("task", ""))
        elif name == "evidence_snapshot":
            nodes = list(payload.get("nodes", []))
            edges = list(payload.get("edges", []))
            task = str(payload.get("task", ""))
            snapshot_version = str(payload.get("workspace_version", ""))
        elif name == "run_start":
            task = str(payload.get("task", task))
        elif name == "evidence_node":
            nodes.append(dict(payload))
            if payload.get("kind") in {"change", "verification", "conclusion"}:
                metadata = payload.get("metadata", {})
                version = metadata.get("after_version") or metadata.get("workspace_version")
                if version:
                    snapshot_version = str(version)
        elif name == "evidence_check_unavailable" and payload.get("workspace_version"):
            snapshot_version = str(payload["workspace_version"])
        elif name == "evidence_edge":
            edges.append(dict(payload))
    graph = EvidenceGraph.from_dict({"task": task, "nodes": nodes, "edges": edges})
    graph.workspace_version = snapshot_version
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
