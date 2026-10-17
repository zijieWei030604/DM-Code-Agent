"""Bounded shared execution with exclusive barriers and ordered commits.

Preparation and yielding results run on the coordinator. Only tool bodies run
on workers. The generator never advances across a barrier until the caller has
committed the preceding group's observations and lifecycle effects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Event
from typing import Any

from dm_agent.tools.base import Tool, ToolResult

from .run_state import RunContext
from .tool_invoker import ExecutedInvocation, PreparedInvocation, ToolInvocation, ToolInvoker


class BatchCancelled(BaseException):
    pass


@dataclass
class DispatchGroup:
    executions: list[ExecutedInvocation]
    interruption: BaseException | None = None


def concurrency_mode(tool: Tool, arguments: dict[str, Any]) -> str:
    if not tool.read_only or tool.completes_task:
        return "exclusive"
    try:
        mode = tool.concurrency(arguments) if callable(tool.concurrency) else tool.concurrency
    except Exception:
        return "exclusive"
    return "shared" if mode == "shared" else "exclusive"


def cancelled_execution(prepared: PreparedInvocation, error: BaseException) -> ExecutedInvocation:
    return ExecutedInvocation(
        prepared,
        ToolResult(
            "cancelled",
            "Tool execution was interrupted; effects may need reconciliation.",
            error_code="interrupted",
        ),
        "Tool execution was interrupted; effects may need reconciliation.",
        "interrupted",
        exception=error,
    )


def dispatch_groups(
    calls: list[dict[str, Any]],
    *,
    tools: Callable[[], dict[str, Tool]],
    invoker: ToolInvoker,
    context: RunContext,
    start: int,
    cancelled: Event,
    workers: int = 4,
    on_started: Callable[[PreparedInvocation], None] | None = None,
    on_settled: Callable[[ExecutedInvocation], None] | None = None,
) -> Iterator[DispatchGroup]:
    def prepare(call: dict[str, Any], offset: int) -> PreparedInvocation:
        child = RunContext(
            run_id=context.run_id, step_number=start + offset, metadata=context.metadata
        )
        tool = tools().get(call["action"])
        failure = call.get("argument_error", "")
        if failure:
            context.metadata["argument_error_count"] += 1
        if tool is None:
            context.metadata["unknown_tool_count"] += 1
            failure = f"Unknown tool '{call['action']}'."
            tool = Tool(call["action"], "Unavailable tool", lambda args: "")
        if failure:
            result = ToolResult(
                "failed",
                failure,
                error_code="invalid_arguments" if call.get("argument_error") else "unknown_tool",
            )
            return PreparedInvocation(
                tool,
                call["action"],
                call["action_input"],
                child,
                immediate=ToolInvocation(
                    call["action_input"], failure, result.error_code, result=result
                ),
            )
        return invoker.prepare(
            tool, action=call["action"], action_input=call["action_input"], context=child
        )

    def run(prepared: PreparedInvocation) -> ExecutedInvocation:
        if cancelled.is_set():
            return cancelled_execution(prepared, BatchCancelled())
        try:
            return invoker.execute(prepared, cancelled=cancelled)
        except BaseException as exc:
            cancelled.set()
            return cancelled_execution(prepared, exc)

    offset = 0
    while offset < len(calls):
        if cancelled.is_set():
            yield DispatchGroup([], BatchCancelled())
            return
        try:
            first = prepare(calls[offset], offset)
            prepared_calls = [first]
            offset += 1
            if (
                first.immediate is None
                and concurrency_mode(first.tool, first.arguments) == "shared"
            ):
                while offset < len(calls) and len(prepared_calls) < workers:
                    tool = tools().get(calls[offset]["action"])
                    # Resolve callable policies only after earlier observations commit.
                    if (
                        tool is None
                        or callable(tool.concurrency)
                        or concurrency_mode(tool, calls[offset]["action_input"]) != "shared"
                    ):
                        break
                    prepared = prepare(calls[offset], offset)
                    prepared_calls.append(prepared)
                    offset += 1
                    if (
                        prepared.immediate is None
                        and concurrency_mode(prepared.tool, prepared.arguments) != "shared"
                    ):
                        break
        except BaseException as exc:
            # No bodies have started. Keep all call ids available for synthetic results.
            cancelled.set()
            yield DispatchGroup([], exc)
            return
        # An argument-revised exclusive call waits for all preceding shared calls.
        deferred = None
        if (
            len(prepared_calls) > 1
            and concurrency_mode(prepared_calls[-1].tool, prepared_calls[-1].arguments) != "shared"
        ):
            deferred = prepared_calls.pop()
        interruption = None
        if len(prepared_calls) == 1:
            if on_started:
                on_started(prepared_calls[0])
            executions = [run(prepared_calls[0])]
            if on_settled:
                on_settled(executions[0])
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool-batch") as pool:
                futures = []
                for prepared in prepared_calls:
                    if on_started:
                        on_started(prepared)
                    futures.append(pool.submit(run, prepared))
                try:
                    for future in as_completed(futures):
                        execution = future.result()
                        if on_settled:
                            on_settled(execution)
                    executions = [future.result() for future in futures]
                except BaseException as exc:
                    cancelled.set()
                    interruption = exc
                    # Drain before returning; completed siblings retain real outcomes.
                    executions = [future.result() for future in futures]
        interruption = interruption or next((e.exception for e in executions if e.exception), None)
        yield DispatchGroup(executions, interruption)
        if interruption is not None or cancelled.is_set():
            return
        if deferred is not None:
            if on_started:
                on_started(deferred)
            execution = run(deferred)
            if on_settled:
                on_settled(execution)
            yield DispatchGroup([execution], execution.exception)
            if execution.exception is not None:
                return
