"""Project runtime facts onto intent-level planner phases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.tools.base import ToolResult

from .execution_facts import ExecutionFact, build_execution_fact
from .planner import PlanStep
from .workspace_version import workspace_version

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

    def __init__(self) -> None:
        self._workspace_root: Path | None = None
        self._baseline_workspace_version = ""

    def begin(self, workspace_root: Path) -> None:
        """Capture the run baseline used to distinguish durable edits from reverted ones."""
        self._workspace_root = workspace_root.resolve()
        self._baseline_workspace_version = workspace_version(self._workspace_root)

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
        fact: ExecutionFact | None = None,
    ) -> list[PlanProgressChange]:
        if not plan:
            return []
        if accepted_completion and action in {"finish", "task_complete"}:
            evidence, successful, reason = (
                "completion_accepted", True, "completion gate accepted the result"
            )
        else:
            fact = fact or build_execution_fact(
                action, result, observation, step_number=step_number,
                tool_succeeded=tool_succeeded, no_progress=no_progress,
            )
            evidence, successful, reason = fact.evidence or None, fact.succeeded, fact.reason
        if evidence is None:
            return []

        changes: list[PlanProgressChange] = []
        if evidence == "workspace_changed":
            changes.extend(self._invalidate_validation(plan, action, step_number))
            if not self._has_net_workspace_change():
                successful = False
                reason = "workspace returned to the run baseline"
                changes.extend(self._invalidate_change(plan, action, step_number, reason))

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

    def _has_net_workspace_change(self) -> bool:
        if self._workspace_root is None or not self._baseline_workspace_version:
            return True
        return workspace_version(self._workspace_root) != self._baseline_workspace_version

    @staticmethod
    def _invalidate_change(
        plan: Iterable[PlanStep], action: str, runtime_step: int, reason: str
    ) -> list[PlanProgressChange]:
        changes: list[PlanProgressChange] = []
        for step in plan:
            if step.phase != "change" or step.status != "satisfied":
                continue
            previous = step.status
            step.set_status(
                "in_progress",
                reason=reason,
                evidence_ref=f"runtime:{runtime_step}:{action}:change_reverted",
            )
            changes.append(
                PlanProgressChange(
                    step.step_number,
                    step.phase,
                    previous,
                    step.status,
                    "workspace_changed",
                    action,
                    reason,
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


def _compact(text: str, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."
