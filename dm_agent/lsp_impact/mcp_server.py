"""Minimal stdio MCP adapter for the LSP impact service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .service import LspImpactService

TOOLS = [
    {"name": "analyze_lsp_impact", "description": "Analyze Python LSP impact and diagnostics.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "lsp_query", "description": "Query LSP symbols, references, definition, or diagnostics.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "action": {"type": "string", "enum": ["symbols", "references", "definition", "diagnostics"]}, "line": {"type": "integer"}, "character": {"type": "integer"}}, "required": ["path", "action"]}},
]


def main() -> int:
    service = LspImpactService(Path.cwd())
    service.start("mcp")
    try:
        while message := _read_message():
            response = _handle(message, service)
            if response is not None:
                _write_message(response)
    finally:
        service.close()
    return 0


def _handle(message: dict[str, Any], service: LspImpactService) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "dm-agent-lsp-impact", "version": "1"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params") or {}
        name, arguments = params.get("name"), params.get("arguments") or {}
        if name == "analyze_lsp_impact":
            data = service.tool_report(str(arguments.get("path", "")))
        elif name == "lsp_query":
            data = service.query(str(arguments.get("path", "")), str(arguments.get("action", "symbols")), int(arguments.get("line", 0)), int(arguments.get("character", 0)))
        else:
            return _error(request_id, -32602, f"Unknown tool: {name}")
        return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}}
    return _error(request_id, -32601, f"Unknown method: {method}")


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while line := sys.stdin.buffer.readline():
        value = line.decode("ascii").strip()
        if not value:
            break
        key, _, content = value.partition(":")
        headers[key.lower()] = content.strip()
    if not headers:
        return None
    length = int(headers.get("content-length", "0"))
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8")) if length > 0 else None


def _write_message(message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
