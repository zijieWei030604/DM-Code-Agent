"""Isolated Explore process. Never loads project extensions or executable skills."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from dm_agent.clients import LLMError, create_llm_client
from dm_agent.core.agent import ReactAgent
from dm_agent.core.events import EventBus
from dm_agent.core.persistence import load_resume_state
from dm_agent.memory.context_budget import estimate_tokens_from_chars
from dm_agent.tools import default_tools
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing import TraceWriter

from .delivery import DeliveryExhausted, DeliveryPolicy
from .schema import DEFAULT_REPORT, check_schema, validate

READ_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "find_files",
        "search_code",
        "search_in_file",
    }
)
DEFAULT_MAX_SOURCE_CALLS = 6
MAX_READ_LINES = 120
INLINE_OUTPUT_THRESHOLD = 5_000


def scoped_tools(workspace: Path) -> list[Tool]:
    """Only audited builtins. Resolve explicit paths and refuse escaping links in scans."""
    root = workspace.resolve()
    result = []
    for tool in default_tools():
        if tool.name not in READ_TOOLS:
            continue

        def run(arguments: dict[str, Any], selected: Tool = tool) -> str | ToolResult:
            args = dict(arguments)
            key = "root" if selected.name in {"find_files", "search_code"} else "path"
            value = args.get(key, ".")
            if not isinstance(value, str):
                raise ValueError("path must be a string")
            candidate = (root / value).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("path escapes workspace")
            if candidate.is_dir():
                for current, directories, files in os.walk(candidate, followlinks=False):
                    directories[:] = [
                        d for d in directories if d not in {".git", ".venv", "node_modules"}
                    ]
                    for name in directories + files:
                        if not (Path(current) / name).resolve().is_relative_to(root):
                            raise ValueError("scan contains a link outside workspace")
            if selected.name == "read_file" and candidate.is_file():
                line_start = args.get("line_start")
                line_end = args.get("line_end")
                line_count = candidate.read_text(encoding="utf-8").count("\n") + 1
                if line_start is None and line_end is None and line_count > MAX_READ_LINES:
                    return ToolResult(
                        "failed",
                        (
                            f"{candidate.name} has {line_count} lines. Use search_in_file first, then "
                            f"read_file with a range of at most {MAX_READ_LINES} lines."
                        ),
                        error_code="read_range_required",
                    )
                if (
                    isinstance(line_start, int)
                    and isinstance(line_end, int)
                    and line_end - line_start + 1 > MAX_READ_LINES
                ):
                    return ToolResult(
                        "failed",
                        f"Read ranges are limited to {MAX_READ_LINES} lines; narrow the range first.",
                        error_code="read_range_too_wide",
                    )
            args[key] = str(candidate)
            return selected.execute(args)

        result.append(
            Tool(tool.name, tool.description, run, read_only=True, input_schema=tool.input_schema)
        )
    return result


def parse_report(text: str, schema: dict[str, Any] | None = None) -> Any:
    """Validate the requested contract before marking the task successful."""
    schema = schema if schema is not None else DEFAULT_REPORT
    check_schema(schema)
    value = json.loads(text)
    validate(value, schema)
    return value


def result_output_preview(text: str) -> tuple[str, bool]:
    """Return model-visible child output without forcing a second read for short reports."""
    if len(text) <= INLINE_OUTPUT_THRESHOLD:
        return text, False
    head = text[:INLINE_OUTPUT_THRESHOLD]
    last_newline = head.rfind("\n")
    if last_newline >= INLINE_OUTPUT_THRESHOLD // 2:
        head = head[:last_newline]
    return head, True


def exploration_result_tool(schema: dict[str, Any], submitted: dict[str, Any]) -> Tool:
    """Create the single terminal handoff from an Explore worker to its parent.

    The report is deliberately wrapped in ``report`` because provider function
    parameters must be an object even when a caller requests an array/scalar
    output schema. The same runner validates native tool-call arguments and
    prompt-JSON fallback action_input.
    """

    check_schema(schema)

    def submit(arguments: dict[str, Any]) -> ToolResult:
        if set(arguments) != {"report"}:
            raise ValueError("submit_exploration_result requires only report")
        report = arguments["report"]
        validate(report, schema)
        submitted["report"] = report
        return ToolResult(
            "success",
            json.dumps(report, ensure_ascii=False),
            metadata={"structured_completion": True},
        )

    return Tool(
        "submit_exploration_result",
        "Submit the final structured Explore report. Call this exactly once when investigation is complete.",
        submit,
        read_only=True,
        input_schema={
            "type": "object",
            "properties": {"report": schema},
            "required": ["report"],
            "additionalProperties": False,
        },
        completes_task=True,
    )


class Meter:
    """Count model-visible text estimates; do not label them provider billing tokens."""

    def __init__(self, client: Any, usage_path: Path, delivery: DeliveryPolicy) -> None:
        self.client = client
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_path = usage_path
        self.delivery = delivery

    def record(self, complete: bool) -> None:
        with self.usage_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "estimated_input_tokens": self.input_tokens,
                        "estimated_output_tokens": self.output_tokens,
                        "model_calls": self.calls,
                        "usage_complete": complete,
                    }
                )
                + "\n"
            )
            stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def respond(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.delivery.ensure_request_allowed()
        estimated_next = estimate_tokens_from_chars(
            sum(len(m.get("content", "")) for m in messages)
        )
        if self.input_tokens + self.output_tokens + estimated_next > 32000:
            raise LLMError("subagent estimated text token budget exceeded")
        self.calls += 1
        self.input_tokens += estimate_tokens_from_chars(
            sum(len(m.get("content", "")) for m in messages)
        )
        self.record(False)
        response = self.client.respond(messages, **kwargs)
        self.output_tokens += estimate_tokens_from_chars(len(response))
        self.record(True)
        return response


def execute(request: dict[str, Any], client: Any = None) -> dict[str, Any]:
    settings = request["settings"]
    attempt = Path(request["attempt"])
    attempt.mkdir(parents=True, exist_ok=True)
    workspace = Path(settings["workspace"]).resolve()
    output_schema = request.get("output_schema", DEFAULT_REPORT)
    check_schema(output_schema)
    delivery = DeliveryPolicy(output_schema, attempt / "rejected_submissions.jsonl")
    trace = TraceWriter(attempt / "trace.jsonl")
    client = Meter(
        client
        or create_llm_client(
            provider=settings["provider"],
            api_key=settings["api_key"],
            model=settings["model"],
            base_url=settings["base_url"],
            timeout=30,
            respond_retries=0,
        ),
        attempt / "usage.jsonl",
        delivery,
    )
    bus = EventBus()
    delivery.install(bus)
    evidence: list[dict[str, Any]] = []
    submitted: dict[str, Any] = {}
    source_calls = 0
    max_source_calls = max(1, int(settings.get("max_source_calls", DEFAULT_MAX_SOURCE_CALLS)))

    def limit_source_calls(event: Any) -> dict[str, Any] | None:
        """Reserve a final turn for structured delivery instead of exhausting context."""
        nonlocal source_calls
        if event.tool_name not in READ_TOOLS:
            return None
        if source_calls >= max_source_calls:
            return {
                "block": True,
                "reason": (
                    "Explore source-call budget reached. Use submit_exploration_result now and "
                    "state any remaining uncertainty."
                ),
            }
        source_calls += 1
        return None

    def observe(event: Any) -> None:
        if event.tool_name == "submit_exploration_result":
            return
        evidence.append(
            {
                "tool": event.tool_name,
                "arguments": event.arguments,
                "step": event.step_number,
                "succeeded": event.tool_succeeded,
                "run_id": event.run_id,
            }
        )

    bus.on("before_tool_call", limit_source_calls, name="explore.source_budget")
    bus.on("after_tool_result", observe, name="explore.sources")
    tools = [*scoped_tools(workspace), exploration_result_tool(output_schema, submitted)]
    resources: list[Any] = []
    if settings.get("lsp"):
        from dm_agent.lsp_impact.service import LspImpactService
        from dm_agent.tools.lsp_impact_tools import lsp_query_result

        service = LspImpactService(workspace, command=settings["lsp_command"], timeout_seconds=5)
        # Child workers do not install LspImpactCapability, so they must start
        # their own local LSP session explicitly before exposing lsp_query.
        service.start(f"subagent-{attempt.name}")
        resources.append(service)

        def query(arguments: dict[str, Any]) -> ToolResult:
            if arguments.get("action", "symbols") not in {
                "symbols",
                "references",
                "definition",
                "diagnostics",
            }:
                raise ValueError("only read-only LSP queries are allowed")
            path = (workspace / str(arguments.get("path", ""))).resolve()
            if not path.is_relative_to(workspace):
                raise ValueError("path escapes workspace")
            return lsp_query_result({**arguments, "path": str(path)}, service=service)

        tools.append(
            Tool(
                "lsp_query",
                "Read-only LSP query: path, action (symbols/references/definition/diagnostics), line, character.",
                query,
                read_only=True,
            )
        )
    agent = ReactAgent(
        client,
        tools,
        max_steps=settings.get("steps", 20),
        temperature=settings.get("temperature", 0),
        enable_planning=False,
        enable_compression=False,
        enable_edit_guard=False,
        event_bus=bus,
        step_callback=delivery.observe_step,
        trace_writer=trace,
        owned_resources=resources,
        include_workspace_version=False,
        # Explore workers retain full source in their append-only trace but send a
        # compact view back to the model, leaving room for several investigation
        # steps and the final structured submission.
        max_observation_chars=4000,
    )
    task = (
        "You are a read-only Explore agent. Search and read code; do not execute code, modify files, "
        "or delegate. Distinguish facts from hypotheses. For files longer than 120 lines, use "
        "search_in_file to locate symbols before reading one non-overlapping range of at most 120 lines. "
        "Use at most "
        + str(max_source_calls)
        + " source lookup tools; prefer precise line ranges and do not reread the same range. "
        "When investigation is complete, call "
        "submit_exploration_result with the report object matching the supplied output schema. "
        "After an invalid delivery, at most two correction attempts are allowed within the existing budgets. "
        "Do not put the report in a finish message. Include file/line references in findings. "
        "Tool provenance is recorded independently.\n"
        + request["context"]
        + "\nAssignment:\n"
        + request["instruction"]
        + "\nOutput schema:\n"
        + json.dumps(output_schema)
    )
    previous = request.get("previous")
    if previous:
        old = Path(previous) / "checkpoint.jsonl"
        if old.exists():
            agent.conversation_history = load_resume_state(old).conversation_history
            task += "\nThis is a follow-up attempt. Prior observations may predate parent edits; re-read relevant code."
    try:
        trace.record("delegated_from", {"parent_run_id": request.get("parent_run_id", "")})
        error = ""
        result: dict[str, Any] = {}
        try:
            result = agent.run(task, checkpoint_path=attempt / "checkpoint.jsonl")
            answer = result["final_answer"]
            status = "succeeded" if result["metadata"].get("status") == "success" else "failed"
        except DeliveryExhausted:
            answer = ""
            status = "failed"
            error = "report_corrections_exhausted"
        except Exception as exc:
            answer = ""
            status = "failed"
            error = type(exc).__name__
        if (
            status == "succeeded"
            and result["metadata"].get("completion_action") == "submit_exploration_result"
            and "report" in submitted
        ):
            report = submitted["report"]
            answer = json.dumps(report, ensure_ascii=False)
            completion_protocol = "structured_tool"
        elif status == "succeeded":
            # Compatibility path for models that ignore the requested terminal
            # tool and return a JSON report through the legacy finish action.
            try:
                report = parse_report(answer, output_schema)
                completion_protocol = "legacy_finish_json"
            except (ValueError, TypeError):
                report = None
                status = "failed"
                completion_protocol = "invalid"
        else:
            report = None
            completion_protocol = "invalid"
        if status == "failed" and delivery.exhausted:
            error = "report_corrections_exhausted"
        (attempt / "answer.txt").write_text(answer, encoding="utf-8")
        trace.record(
            "exploration_result_submitted",
            {"protocol": completion_protocol, "accepted": status == "succeeded"},
        )
        output, output_truncated = result_output_preview(answer)
        payload = {
            "status": status,
            # ``summary`` stays a concise semantic conclusion when the caller's
            # schema supplies one. ``output`` carries the full short report, or
            # a clearly marked preview of a longer one, for the parent task tool.
            "summary": (
                str(report.get("summary", ""))
                if status == "succeeded" and isinstance(report, dict)
                else ""
            ),
            "output": output if status == "succeeded" else "",
            "output_char_count": len(answer),
            "output_truncated": output_truncated if status == "succeeded" else False,
            "report": report,
            "error": error or ("invalid_or_incomplete_report" if status == "failed" else ""),
            "evidence": evidence,
            "trace_ref": str(attempt / "trace.jsonl"),
            "estimated_input_tokens": client.input_tokens,
            "estimated_output_tokens": client.output_tokens,
            "model_calls": client.calls,
            "completion_protocol": completion_protocol,
            "schema_valid": status == "succeeded",
            "rejected_delivery_count": len(delivery.rejections),
            "validation_errors": [entry["error"][:1000] for entry in delivery.rejections],
            "rejections_ref": str(delivery.journal) if delivery.rejections else "",
        }
        (attempt / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return payload
    finally:
        agent.close()
        trace.close()


def main() -> None:
    request = json.loads(sys.stdin.readline())

    def watch_owner() -> None:
        sys.stdin.read()
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            posix: Any = os
            signals: Any = signal
            posix.killpg(posix.getpgrp(), signals.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch_owner, daemon=True).start()
    execute(request)


if __name__ == "__main__":
    main()
