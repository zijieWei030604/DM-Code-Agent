"""Commit native tool batches through the ordinary lifecycle and persistence chain."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from dm_agent.tools.base import Tool, ToolResult

from .completion import CompletionGate
from .run_state import RunContext, Step
from .tool_dispatch import BatchCancelled, dispatch_groups
from .tool_invoker import ExecutedInvocation, PreparedInvocation, ToolInvocation, ToolInvoker


class BatchHost(Protocol):
    _run_context: RunContext
    _tool_invoker: ToolInvoker
    _completion_gate: CompletionGate
    _cancel_event: Event
    trace_writer: Any
    tools: dict[str, Tool]

    def _append_history(
        self,
        role: str,
        content: str,
        *,
        kind: str,
        context_content: str | None = None,
        message_fields: dict[str, Any] | None = None,
    ) -> None: ...
    def _publish_step(self, step: Step, step_num: int) -> None: ...
    def _save_checkpoint_snapshot(
        self,
        path: Path,
        *,
        task: str,
        step_count: int,
        steps: list[Step],
        metadata: dict[str, Any],
        limit: int,
    ) -> None: ...


def execute_tool_batch(
    host: BatchHost, task: str, steps: list[Step], checkpoint_path: Path | None, limit: int
) -> tuple[bool, str]:
    context = host._run_context
    metadata = context.metadata
    pending = context.pending_tool_calls
    start = context.step_number
    committed_step = start - 1

    def save() -> None:
        if checkpoint_path is not None:
            host._save_checkpoint_snapshot(
                checkpoint_path,
                task=task,
                step_count=committed_step,
                steps=steps,
                metadata=metadata,
                limit=limit,
            )

    def record_result(
        call: dict[str, Any],
        observation: str,
        status: str,
        *,
        synthetic: bool = False,
        result: ToolResult | None = None,
    ) -> None:
        fields = {
            "tool_call_id": call["call_id"],
            "tool_name": call["action"],
            "status": status,
            "is_error": status != "success",
            "synthetic": synthetic,
        }
        if result is not None:
            fields.update(
                json.loads(
                    json.dumps(
                        {
                            "error_code": result.error_code,
                            "exit_code": result.exit_code,
                            "changed_files": result.changed_files,
                            "check_scope": result.check_scope,
                            "result_metadata": result.metadata,
                        },
                        default=str,
                    )
                )
            )
        host._append_history("tool", observation, kind="tool_result", message_fields=fields)
        if host.trace_writer:
            host.trace_writer.record(
                "tool_batch_call",
                {
                    **{key: call[key] for key in ("batch_id", "call_id", "index")},
                    **(
                        {"provider_call_id": call["provider_call_id"]}
                        if call.get("provider_call_id")
                        else {}
                    ),
                    "step_number": context.step_number,
                    "status": status,
                    "synthetic": synthetic,
                },
            )

    metadata["tool_batch_active"] = True
    save()
    allowed = list(pending[: max(0, limit - start + 1)])

    def progress(
        event: str, prepared: PreparedInvocation, execution: ExecutedInvocation | None = None
    ) -> None:
        if host.trace_writer:
            call = allowed[prepared.context.step_number - start]
            result = execution.result if execution else None
            if result is None and prepared.immediate is not None:
                result = prepared.immediate.result
            status: str = result.status if result is not None else "pending"
            if execution is not None and result is None:
                status = (
                    "blocked" if prepared.immediate and prepared.immediate.blocked else "success"
                )
            host.trace_writer.record(
                event,
                {
                    "batch_id": call["batch_id"],
                    "call_id": call["call_id"],
                    "tool_name": call["action"],
                    "step_number": prepared.context.step_number,
                    "status": status,
                },
            )

    interruption: BaseException | None = None
    terminal = False
    answer = ""
    groups = dispatch_groups(
        allowed,
        tools=lambda: host.tools,
        invoker=host._tool_invoker,
        context=context,
        start=start,
        cancelled=host._cancel_event,
        on_started=lambda prepared: progress("tool_execution_start", prepared),
        on_settled=lambda execution: progress("tool_execution_end", execution.prepared, execution),
    )
    for group in groups:
        for execution in group.executions:
            call = pending.pop(0)
            context.step_number = execution.prepared.context.step_number
            metadata["current_tool_call"] = {
                key: call[key] for key in ("batch_id", "call_id", "index")
            }
            try:
                invocation = host._tool_invoker.commit(execution)
            except BaseException as exc:
                # A cancellation hook must not erase an already finished tool's result.
                interruption = exc
                host._cancel_event.set()
                invocation = ToolInvocation(
                    execution.prepared.arguments,
                    execution.observation,
                    execution.error_kind,
                    tool_succeeded=execution.succeeded,
                    result=execution.result,
                )
            observation = invocation.observation
            tool = execution.prepared.tool
            accepted = False
            is_completion = call["action"] in {"finish", "task_complete"} or tool.completes_task
            if is_completion and invocation.tool_succeeded:
                try:
                    accepted, observation = host._completion_gate.review(
                        task=task,
                        action=call["action"],
                        completion_text=observation,
                        steps=steps,
                        context=context,
                    )
                except BaseException as exc:
                    interruption = exc
                    host._cancel_event.set()
            status = (
                invocation.result.status
                if invocation.result is not None
                else "success" if invocation.tool_succeeded else "failed"
            )
            if invocation.blocked or (is_completion and not accepted):
                status = "failed"
            step = Step(
                str(call.get("thought", "")),
                call["action"],
                invocation.arguments,
                observation,
                json.dumps(call, ensure_ascii=False),
            )
            steps.append(step)
            reported_result = invocation.result
            if invocation.blocked:
                reported_result = ToolResult("failed", observation, error_code="policy_blocked")
            record_result(call, observation, status, result=reported_result)
            if host.trace_writer:
                host.trace_writer.record_tool_call(
                    step_number=context.step_number,
                    action=call["action"],
                    action_input=invocation.arguments,
                    observation=observation,
                    failed=status != "success",
                )
            host._publish_step(step, context.step_number)
            committed_step = context.step_number
            if accepted:
                metadata["status"] = "success"
                metadata["completion_action"] = call["action"]
                metadata["failure_reason"] = ""
                terminal, answer = True, observation
            elif metadata.get("evidence_terminal_completion_rejection"):
                metadata["status"] = "critic_rejected"
                terminal = True
            save()
        interruption = interruption or group.interruption
        if interruption is not None or host._cancel_event.is_set() or terminal:
            break
    if interruption is not None or host._cancel_event.is_set():
        for call in list(pending):
            record_result(
                call,
                "Tool was not executed because the run was interrupted.",
                "cancelled",
                synthetic=True,
            )
            metadata["skipped_tool_call_count"] += 1
        pending.clear()
        metadata["status"] = str(getattr(interruption, "status", "cancelled"))
        metadata["failure_reason"] = "Tool batch interrupted; completed results were preserved."
        save()
        if interruption is not None and not isinstance(interruption, BatchCancelled):
            raise interruption
        terminal = True
    if interruption is None and not host._cancel_event.is_set() and not pending:
        metadata.pop("tool_batch_active", None)
        save()
    return terminal, answer
