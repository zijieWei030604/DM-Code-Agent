"""Project runtime facts onto intent-level planner phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from dm_agent.tools.base import ToolResult

from .planner import PlanStep


LOCATE_TOOLS = frozenset({"search_code", "search_symbol", "find_files", "list_directory"})
INSPECT_TOOLS = frozenset(
    {
        "read_file",
        "search_in_file",
        "parse_ast",
        "inspect_python_symbol",
        "get_function_signature",
        "find_dependencies",
        "dependency_graph",
        "inspect_change_impact",
    }
)
VERIFY_TOOLS = frozenset({"run_tests", "run_linter", "run_python"})


@dataclass(frozen=True)
class PlanProgressChange:
    step_number: int
    phase: str
    previous_status: str
    status: str
    evidence: str
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "phase": self.phase,
            "previous_status": self.previous_status,
            "status": self.status,
            "evidence": self.evidence,
            "action": self.action,
            "reason": self.reason,
        }


class PlanProgressTracker:
    """Maintain plan status from structured tool outcomes, independent of EvidenceGraph."""

    def observe(
        self,
        plan: list[PlanStep],
        *,
        action: str,
        result: ToolResult | None,
        observation: str,
        accepted_completion: bool = False,
        no_progress: bool = False,
        tool_succeeded: bool = False,
        step_number: int = 0,
    ) -> list[PlanProgressChange]:
        if not plan:
            return []
        evidence, successful, reason = _classify_event(
            action,
            result,
            observation,
            accepted_completion=accepted_completion,
            no_progress=no_progress,
            tool_succeeded=tool_succeeded,
        )
        if evidence is None:
            return []

        changes: list[PlanProgressChange] = []
        if evidence == "workspace_changed":
            changes.extend(self._invalidate_validation(plan, action, step_number))

        target = _target_step(plan, evidence)
        if target is None:
            return changes
        previous = target.status
        status = "satisfied" if successful else "in_progress"
        target.set_status(
            status,
            result=_compact(observation),
            reason=reason,
            evidence_ref=f"runtime:{step_number}:{action}",
        )
        if previous != target.status or target.result:
            changes.append(
                PlanProgressChange(
                    step_number=target.step_number,
                    phase=target.phase,
                    previous_status=previous,
                    status=target.status,
                    evidence=evidence,
                    action=action,
                    reason=reason,
                )
            )
        return changes

    @staticmethod
    def _invalidate_validation(
        plan: Iterable[PlanStep], action: str, runtime_step: int
    ) -> list[PlanProgressChange]:
        changes: list[PlanProgressChange] = []
        for step in plan:
            if step.phase != "validate" or step.status != "satisfied":
                continue
            previous = step.status
            step.set_status(
                "in_progress",
                reason="workspace changed after the last successful verification",
                evidence_ref=f"runtime:{runtime_step}:{action}:verification_stale",
            )
            changes.append(
                PlanProgressChange(
                    step.step_number,
                    step.phase,
                    previous,
                    step.status,
                    "workspace_changed",
                    action,
                    step.status_reason,
                )
            )
        return changes


def summarize_plan(plan: Iterable[PlanStep]) -> dict[str, Any]:
    steps = list(plan)
    counts = {status: sum(step.status == status for step in steps) for status in (
        "pending",
        "in_progress",
        "satisfied",
    )}
    return {
        "steps": len(steps),
        **counts,
        "phases": [step.phase for step in steps],
    }


def _target_step(plan: Iterable[PlanStep], evidence: str) -> PlanStep | None:
    candidates = [step for step in plan if step.completion_evidence == evidence]
    return next((step for step in candidates if step.status != "satisfied"), None)


def _classify_event(
    action: str,
    result: ToolResult | None,
    observation: str,
    *,
    accepted_completion: bool,
    no_progress: bool,
    tool_succeeded: bool,
) -> tuple[str | None, bool, str]:
    succeeded = (tool_succeeded or bool(result and result.status == "success")) and not no_progress
    if accepted_completion and action in {"finish", "task_complete"}:
        return "completion_accepted", True, "completion gate accepted the result"
    if result and result.changed_files:
        return "workspace_changed", result.status == "success", "tool changed workspace files"
    if action in VERIFY_TOOLS or (
        action == "run_shell" and _purpose(result) == "verification"
    ):
        outcome = _verification_outcome(result)
        return "verification_passed", outcome == "passed", f"verification outcome: {outcome}"
    if action in LOCATE_TOOLS:
        return "candidate_observed", succeeded, "repository candidate search completed"
    if action in INSPECT_TOOLS:
        return "source_observed", succeeded, "source evidence was inspected"
    return None, False, ""


def _purpose(result: ToolResult | None) -> str:
    if result is None or not isinstance(result.metadata, Mapping):
        return ""
    verification = result.metadata.get("verification")
    if isinstance(verification, Mapping):
        return "verification"
    return ""


def _verification_outcome(result: ToolResult | None) -> str:
    if result is None:
        return "unknown"
    verification = result.metadata.get("verification")
    if isinstance(verification, Mapping):
        return str(verification.get("outcome") or "unknown")
    return "passed" if result.status == "success" else "failed"


def _compact(text: str, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."
