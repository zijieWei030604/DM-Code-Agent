"""Agent tool adapters for the LSP impact service."""

from __future__ import annotations

import json
from typing import Any

from dm_agent.lsp_impact.service import LspImpactService

from .base import ToolResult


def analyze_lsp_impact_result(arguments: dict[str, Any], *, service: LspImpactService | None = None) -> ToolResult:
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return ToolResult("failed", "path is required", error_code="invalid_arguments")
    if service is None:
        return ToolResult("unavailable", "LSP impact analysis is disabled for this run.", error_code="disabled")
    data = service.tool_report(path)
    report = data["report"]
    return ToolResult(
        "success" if report["status"] == "ok" else "unavailable",
        json.dumps(data, ensure_ascii=False),
        metadata={"lsp_impact_report": report},
    )


def analyze_lsp_impact(arguments: dict[str, Any], *, service: LspImpactService | None = None) -> str:
    return analyze_lsp_impact_result(arguments, service=service).message


def lsp_query_result(arguments: dict[str, Any], *, service: LspImpactService | None = None) -> ToolResult:
    path = arguments.get("path")
    action = arguments.get("action", "symbols")
    if not isinstance(path, str) or not path or not isinstance(action, str):
        return ToolResult("failed", "path and action are required", error_code="invalid_arguments")
    if service is None:
        return ToolResult("unavailable", "LSP impact analysis is disabled for this run.", error_code="disabled")
    line = arguments.get("line", 0)
    character = arguments.get("character", 0)
    data = service.query(path, action, int(line), int(character))
    return ToolResult("success" if data["status"] == "ok" else "unavailable", json.dumps(data, ensure_ascii=False))


def lsp_query(arguments: dict[str, Any], *, service: LspImpactService | None = None) -> str:
    return lsp_query_result(arguments, service=service).message
