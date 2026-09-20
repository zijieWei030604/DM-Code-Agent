"""Shared, execution-grounded facts produced by the tool runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from dm_agent.tools.base import ToolResult

FactType = Literal[
    "candidate_observed",
    "source_observed",
    "workspace_changed",
    "verification_recorded",
    "other",
]

LOCATE_TOOLS = frozenset({"search_code", "search_symbol", "find_files", "list_directory"})
INSPECT_TOOLS = frozenset(
    {
        "read_file", "search_in_file", "parse_ast", "inspect_python_symbol",
        "get_function_signature", "find_dependencies", "dependency_graph",
        "inspect_change_impact",
    }
)
VERIFY_TOOLS = frozenset({"run_tests", "run_linter", "run_python"})


@dataclass(frozen=True)
class ExecutionFact:
    """Canonical interpretation of one completed runtime action."""

    fact_type: FactType
    phase: str
    evidence: str
    tool_name: str
    step_number: int
    succeeded: bool
    reason: str
    changed_paths: tuple[str, ...] = ()
    workspace_version: str = ""
    verification_outcome: str = ""
    verification_scope: tuple[str, ...] = ()
    verification_scope_level: str = ""
    failure_kind: str = ""
    execution_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_fact(
    action: str,
    result: ToolResult | None,
    observation: str,
    *,
    step_number: int,
    tool_succeeded: bool,
    no_progress: bool = False,
    workspace_version: str = "",
) -> ExecutionFact:
    succeeded = (tool_succeeded or bool(result and result.status == "success")) and not no_progress
    changed_paths = tuple(dict.fromkeys(str(path) for path in (result.changed_files if result else ()) if path))
    if changed_paths:
        return ExecutionFact(
            "workspace_changed", "change", "workspace_changed", action, step_number,
            bool(result and result.status == "success"), "tool changed workspace files",
            changed_paths, workspace_version,
        )
    if action in VERIFY_TOOLS or (action == "run_shell" and _verification(result) is not None):
        verification = _verification(result) or {}
        outcome = str(verification.get("outcome") or ("passed" if succeeded else "unknown"))
        scope = tuple(str(item) for item in verification.get("scope") or (result.check_scope if result else ()))
        return ExecutionFact(
            "verification_recorded", "validate", "verification_passed", action,
            step_number, outcome == "passed", f"verification outcome: {outcome}",
            workspace_version=workspace_version, verification_outcome=outcome,
            verification_scope=scope,
            verification_scope_level=str(verification.get("scope_level") or "related"),
            failure_kind=str(verification.get("failure_kind") or (result.error_code if result else "")),
            execution_status=str(verification.get("execution_status") or ""),
        )
    if action in LOCATE_TOOLS:
        count = _candidate_count(action, observation)
        found = succeeded if count is None else succeeded and count > 0
        reason = (
            "repository candidate search completed"
            if count is None
            else (
                "repository search returned no candidates"
                if count == 0
                else f"repository search returned {count} candidate(s)"
            )
        )
        return ExecutionFact("candidate_observed", "locate", "candidate_observed", action, step_number, found, reason)
    if action in INSPECT_TOOLS:
        return ExecutionFact("source_observed", "inspect", "source_observed", action, step_number, succeeded, "source evidence was inspected")
    return ExecutionFact("other", "", "", action, step_number, succeeded, "")


def _verification(result: ToolResult | None) -> Mapping[str, Any] | None:
    value = result.metadata.get("verification") if result and isinstance(result.metadata, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _candidate_count(action: str, observation: str) -> int | None:
    if action in {"search_code", "search_symbol", "find_files"}:
        try:
            payload = json.loads(observation)
        except (TypeError, ValueError):
            return None
        count = payload.get("match_count") if isinstance(payload, Mapping) else None
        return count if isinstance(count, int) and not isinstance(count, bool) else None
    if action == "list_directory":
        value = observation.strip()
        return 0 if value == "<空>" else len(value.splitlines()) if value else 0
    return None


__all__ = ["ExecutionFact", "build_execution_fact"]
