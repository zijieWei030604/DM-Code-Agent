"""Print one real DeepSeek streamed tool call and verify its execution order.

Run from the repository root with ``DEEPSEEK_API_KEY`` configured. The generated
trace is deliberately stored below ``traces/`` (which is gitignored): it includes
full model I/O only for local debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _clip(value: str, limit: int = 160) -> str:
    return value if len(value) <= limit else f"{value[:limit]}…"


def main() -> int:
    from dm_agent.clients.deepseek_client import DeepSeekClient
    from dm_agent.clients.streaming import StreamEvent
    from dm_agent.core import ReactAgent
    from dm_agent.paths import load_env_files
    from dm_agent.tools.base import Tool, ToolResult
    from dm_agent.tracing.writer import TraceWriter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "traces" / "deepseek-stream-debug.jsonl",
        help="Private local trace path. It contains full LLM I/O for this inspection.",
    )
    args = parser.parse_args()

    load_env_files(cwd=ROOT)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Missing DEEPSEEK_API_KEY. Add it to .env or set it in this CMD session.")
        return 2

    sequence: list[tuple[float, str, str]] = []
    started_at = time.perf_counter()

    def mark(kind: str, detail: str) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        sequence.append((elapsed_ms, kind, detail))
        print(f"[{elapsed_ms:9.1f} ms] {kind:<18} {detail}")

    def show_event(event: StreamEvent) -> None:
        details = [f"type={event.type}"]
        if event.tool_index is not None:
            details.append(f"index={event.tool_index}")
        if event.tool_name:
            details.append(f"tool={event.tool_name}")
        if event.delta:
            details.append(f"delta={_clip(event.delta)!r}")
        if event.arguments_preview:
            details.append(
                "preview=" + _clip(json.dumps(event.arguments_preview, ensure_ascii=False))
            )
        if event.reason:
            details.append(f"reason={event.reason!r}")
        mark("STREAM_EVENT", " ".join(details))

    def inspect_stream_source(_: dict[str, Any]) -> ToolResult:
        completed = any(
            kind == "STREAM_EVENT"
            and "type=toolcall_end" in detail
            and "inspect_stream_source" in detail
            for _, kind, detail in sequence
        )
        if not completed:
            raise RuntimeError("Tool execution began before its streamed arguments completed.")
        mark("TOOL_EXECUTE", "inspect_stream_source: full parameters already assembled")
        content = (ROOT / "dm_agent" / "clients" / "streaming.py").read_text(encoding="utf-8")
        return ToolResult("success", content[:700], check_scope=("streaming.py",))

    def task_complete(arguments: dict[str, Any]) -> ToolResult:
        message = str(arguments.get("message", "stream demo complete"))
        mark("TOOL_EXECUTE", f"task_complete: {_clip(message)!r}")
        return ToolResult("success", message)

    tools = [
        Tool(
            "inspect_stream_source",
            "Read the first part of the local streaming implementation. Call this first.",
            inspect_stream_source,
            result_runner=inspect_stream_source,
            read_only=True,
            input_schema=_schema({}),
        ),
        Tool(
            "task_complete",
            "Finish only after inspect_stream_source returns successfully.",
            task_complete,
            result_runner=task_complete,
            input_schema=_schema({"message": {"type": "string"}}, ["message"]),
            completes_task=True,
        ),
    ]

    trace = TraceWriter(args.trace, capture_llm_io=True)
    client = DeepSeekClient(api_key, model=args.model)
    agent = ReactAgent(
        client,
        tools,
        enable_planning=False,
        enable_compression=False,
        max_steps=2,
        trace_writer=trace,
        stream_callback=show_event,
    )
    try:
        result = agent.run(
            "First call inspect_stream_source exactly once. After you receive its result, "
            "call task_complete with a concise message. Do not call multiple tools in one turn."
        )
    finally:
        agent.close()

    print("\n--- Run result ---")
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    print(f"\nTrace: {args.trace.resolve()}")
    print("\nExpected ordering: toolcall_delta → toolcall_end → stream_done → TOOL_EXECUTE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
