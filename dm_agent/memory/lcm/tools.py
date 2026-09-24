"""Only local search, description and bounded source expansion are exposed."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from dm_agent.tools.base import Tool, ToolResult

from .dag import SummaryDAG
from .store import LCMStore


def build_lcm_tools(
    store: LCMStore | Callable[[], LCMStore],
    get_branch: Callable[[], str],
    resolve_artifact: Callable[[dict[str, Any]], str] | None = None,
) -> list[Tool]:
    def execute(name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            branch = get_branch()
            active_store = store() if callable(store) else store
            dag = SummaryDAG(active_store, branch, resolve_artifact)
            if name == "lcm_grep":
                rows = active_store.search(
                    branch, str(arguments["query"]), limit=int(arguments.get("limit", 10))
                )
                result: Any = [
                    {"id": row["id"], "kind": row["kind"], "preview": row["body"][:400]}
                    for row in rows
                ]
            elif name == "lcm_describe":
                result = dag.describe(str(arguments["record_id"]))
            else:
                result = dag.expand(
                    str(arguments["record_id"]),
                    offset=int(arguments.get("offset", 0)),
                    source_offset=int(arguments.get("source_offset", 0)),
                    limit=int(arguments.get("limit", 4000)),
                )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return ToolResult(status="failed", message=str(exc), error_code="lcm_invalid_request")
        return ToolResult(status="success", message=json.dumps(result, ensure_ascii=False))

    result = []
    for name, description, required, properties in (
        (
            "lcm_grep",
            "Search visible SQLite historical messages and summaries; no Trace search.",
            ["query"],
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
        ),
        (
            "lcm_describe",
            "Describe a visible record and list its immediate source IDs.",
            ["record_id"],
            {"record_id": {"type": "string"}},
        ),
        (
            "lcm_expand",
            "Read one record page; follow returned source IDs for original details.",
            ["record_id"],
            {
                "record_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "source_offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12000},
            },
        ),
    ):

        def runner(arguments: dict[str, Any], tool_name: str = name) -> ToolResult:
            return execute(tool_name, arguments)

        result.append(
            Tool(
                name=name,
                description=description,
                runner=runner,
                read_only=True,
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            )
        )
    return result
