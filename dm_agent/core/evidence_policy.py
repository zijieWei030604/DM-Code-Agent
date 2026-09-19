"""Small, execution-grounded completion policy for decision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .evidence import EvidenceGraph, EvidenceNode

CompletionDecision = Literal["allow", "warn", "block"]
VerificationState = Literal["tested", "contradicted", "unavailable", "not_run"]


@dataclass(frozen=True)
class EvidenceCompletionResult:
    """A completion decision derived only from current execution facts."""

    decision: CompletionDecision
    verification_state: VerificationState
    issues: tuple[dict[str, str], ...] = ()
    warnings: tuple[dict[str, str], ...] = ()


class EvidenceCompletionPolicy:
    """Block only explicit current-version contradictions.

    Repository CI or an external harness remains the correctness authority.
    This policy only distinguishes usable local test feedback from unavailable
    or missing verification.
    """

    def evaluate(
        self,
        graph: EvidenceGraph,
        *,
        verified_transaction: bool = False,
    ) -> EvidenceCompletionResult:
        checks = graph.current_verifications()
        contradictions = [
            node
            for node in checks
            if not bool(node.metadata.get("passed"))
            and bool(node.metadata.get("blocking", False))
        ]
        if contradictions:
            latest = contradictions[-1]
            issue = _verification_issue(latest, _blocking_status(latest))
            return EvidenceCompletionResult("block", "contradicted", (issue,))

        if verified_transaction or any(bool(node.metadata.get("passed")) for node in checks):
            return EvidenceCompletionResult("allow", "tested")

        if checks:
            latest = checks[-1]
            warning = _verification_issue(latest, "verification_unavailable")
            return EvidenceCompletionResult("warn", "unavailable", warnings=(warning,))

        warning = {
            "node_id": "",
            "path": "<workspace>",
            "status": "verification_not_run",
        }
        return EvidenceCompletionResult("warn", "not_run", warnings=(warning,))


def _verification_issue(node: EvidenceNode, status: str) -> dict[str, str]:
    scope = [str(item) for item in node.metadata.get("scope") or () if item]
    return {
        "node_id": node.node_id,
        "path": scope[0] if scope else "<workspace>",
        "status": status,
    }


def _blocking_status(node: EvidenceNode) -> str:
    if str(node.metadata.get("failure_kind", "")) == "syntax_error":
        return "confirmed_code_error"
    return "confirmed_test_failure"


__all__ = [
    "EvidenceCompletionPolicy",
    "EvidenceCompletionResult",
    "VerificationState",
]
