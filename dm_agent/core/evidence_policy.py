"""Completion policy built on top of recorded decision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from .evidence import EvidenceGraph, EvidenceNode

CompletionDecision = Literal["allow", "warn", "block"]

_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_CONFIG_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".lock", ".toml", ".xml", ".yaml", ".yml"}
)
_DOCUMENT_SUFFIXES = frozenset({".md", ".rst", ".txt"})


@dataclass(frozen=True)
class EvidenceCompletionResult:
    """A policy decision without mutating the underlying evidence graph."""

    decision: CompletionDecision
    issues: tuple[dict[str, str], ...] = ()
    warnings: tuple[dict[str, str], ...] = ()


class EvidenceCompletionPolicy:
    """Evaluate a change transaction instead of demanding one test per file."""

    def evaluate(
        self,
        graph: EvidenceGraph,
        *,
        verified_transaction: bool = False,
    ) -> EvidenceCompletionResult:
        states = graph.change_states()
        changes = [node for node in graph.nodes.values() if node.kind == "change"]
        issues: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        for change in changes:
            status = states.get(change.node_id, "unverified")
            if status == "contradicted":
                issues.append(_issue(change, "contradicted"))
            elif status == "missing_read_basis":
                issues.append(_issue(change, "missing_read_basis"))

        checks = graph.current_verifications()
        passed_checks = [node for node in checks if bool(node.metadata.get("passed"))]
        failed_tests = [
            node
            for node in checks
            if node.metadata.get("tool") == "run_tests" and not bool(node.metadata.get("passed"))
        ]
        passed_tests = [
            node
            for node in passed_checks
            if node.metadata.get("tool") == "run_tests"
        ]

        if failed_tests:
            issues[0:0] = [_issue(change, "transaction_test_failed") for change in changes]
            return EvidenceCompletionResult("block", tuple(issues), tuple(warnings))

        if issues:
            return EvidenceCompletionResult("block", tuple(issues), tuple(warnings))

        if verified_transaction:
            return EvidenceCompletionResult("allow")

        source_changes = [node for node in changes if _change_category(node) == "source"]
        if source_changes and not passed_tests:
            # A missing successful local test is important evidence for the
            # caller, but it is not proof that the patch is wrong.  The Agent
            # must not present this as verified completion, while an offline
            # evaluator may still establish that the patch is correct.
            status = "source_indirect_only" if passed_checks else "source_unverified"
            warnings.extend(_issue(node, status) for node in source_changes)

        for change in changes:
            category = _change_category(change)
            if category == "config" and not passed_checks:
                warnings.append(_issue(change, "configuration_unchecked"))
            elif category == "other" and not passed_checks:
                warnings.append(_issue(change, "change_unchecked"))

        if issues:
            return EvidenceCompletionResult("block", tuple(issues), tuple(warnings))
        if warnings:
            return EvidenceCompletionResult("warn", warnings=tuple(warnings))
        return EvidenceCompletionResult("allow")


def _change_category(node: EvidenceNode) -> str:
    path = str(node.metadata.get("path") or "")
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    if suffix in _SOURCE_SUFFIXES:
        return "source"
    if suffix in _CONFIG_SUFFIXES:
        return "config"
    if suffix in _DOCUMENT_SUFFIXES:
        return "document"
    return "other"


def _issue(node: EvidenceNode, status: str) -> dict[str, str]:
    return {
        "node_id": node.node_id,
        "path": str(node.metadata.get("path") or "<workspace>"),
        "status": status,
    }


__all__ = ["EvidenceCompletionPolicy", "EvidenceCompletionResult"]
