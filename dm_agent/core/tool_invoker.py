"""一次工具调用的完整链路：入参校验 -> 前置钩子 -> 写前备份 -> 执行 -> 截断 -> 后置钩子。

内核护栏（观察截断、写前备份）与扩展钩子（``before_tool_call`` /
``after_tool_result``）的相对次序是有意为之，也是这个模块存在的理由：

- 备份发生在 ``before_tool_call`` 放行之后、真正执行之前——被拦下的调用不备份；
- 截断发生在 ``after_tool_result`` 链之前——处理器看到的就是最终写进 step、
  对话历史与 trace 的那份文本。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tools.file_tools import edit_file as builtin_edit_file

from .events import AfterToolResultEvent, BeforeToolCallEvent, EventBus, HookErrorHandler
from .execution_facts import ExecutionFact, build_execution_fact
from .guards import (
    WRITE_ACTIONS,
    is_identity_content_edit,
    observation_reports_missing_path,
)
from .observation import ObservationBounder, is_failure_observation
from .persistence import RunPersistence
from .run_state import RunContext
from .workspace_version import workspace_version


@dataclass
class ToolInvocation:
    """一次工具调用的结果。

    ``arguments`` 可能被 ``before_tool_call`` 处理器就地改过，调用方要用返回的这份
    （step、对话历史与 trace 都以它为准）。

    ``no_change`` 表示工具的预期效果没有发生，本次调用不应推进 planner；它不表示
    “这是一个只读工具”。
    """

    arguments: Any
    observation: str
    error_kind: str = ""
    blocked: bool = False
    tool_succeeded: bool = False
    no_change: bool = False
    result: ToolResult | None = None
    fact: ExecutionFact | None = None


def coerce_task_complete_arguments(action_input: Any) -> dict[str, Any]:
    """``task_complete`` 允许模型传字符串或干脆不传参数。"""
    if action_input is None:
        return {}
    if isinstance(action_input, str):
        return {"message": action_input}
    if not isinstance(action_input, dict):
        return {}
    return action_input


def validate_tool_arguments(action_input: Any) -> tuple[str, str] | None:
    """校验普通工具的入参，返回 (failure_reason, observation)；合法时返回 None。"""
    if action_input is None:
        return "Tool arguments missing", "Tool arguments missing: action_input is null."
    if not isinstance(action_input, dict):
        return "Tool arguments must be a JSON object", "Tool arguments must be a JSON object."
    return None


class ToolInvoker:
    """按固定次序把一次工具调用跑完，并把过程记进 metadata。"""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        bounder: ObservationBounder,
        persistence: RunPersistence,
        on_error: HookErrorHandler | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.bounder = bounder
        self.persistence = persistence
        self.on_error = on_error

    def invoke(
        self,
        tool: Tool,
        *,
        action: str,
        action_input: Any,
        context: RunContext,
    ) -> ToolInvocation:
        """执行一次工具调用；入参非法或被钩子拦下时不会真正调用 runner。"""
        metadata = context.metadata
        result: ToolResult | None
        if action == "task_complete":
            action_input = coerce_task_complete_arguments(action_input)
        else:
            invalid = validate_tool_arguments(action_input)
            if invalid is not None:
                failure_reason, observation = invalid
                metadata["argument_error_count"] += 1
                metadata["failure_reason"] = failure_reason
                result = ToolResult(
                    "failed",
                    observation,
                    error_code="invalid_arguments",
                )
                return ToolInvocation(
                    arguments=action_input,
                    observation=observation,
                    error_kind="invalid_arguments",
                    result=result,
                )

        before_event = BeforeToolCallEvent(
            tool_name=action,
            arguments=cast(dict[str, Any], action_input),
            step_number=context.step_number,
            run_id=context.run_id,
            metadata=metadata,
            content_anchor_safe=(action == "edit_file" and tool.runner is builtin_edit_file),
        )
        block = self.event_bus.emit_before_tool_call(before_event, on_error=self.on_error)
        action_input = before_event.arguments
        if validate_tool_arguments(action_input) is not None:
            observation = "Tool arguments must be a JSON object."
            result = ToolResult(
                "failed",
                observation,
                error_code="invalid_arguments",
            )
            return ToolInvocation(
                arguments=action_input,
                observation=observation,
                error_kind="invalid_arguments",
                result=result,
            )
        if block is not None:
            # 被拦下的调用不计入计划完成，也不备份。
            return ToolInvocation(
                arguments=action_input,
                observation=str(block["reason"]),
                blocked=True,
            )

        identity_edit_noop = (
            action == "edit_file"
            and tool.runner is builtin_edit_file
            and is_identity_content_edit(action_input)
        )
        if action in WRITE_ACTIONS and not identity_edit_noop:
            self.persistence.backup_before_write(action_input, context)

        error_kind = ""
        tool_succeeded = False
        result = None
        journal = self.persistence.call_journal
        call_id = (
            journal.begin(action, context.step_number, context.run_id, read_only=tool.read_only)
            if journal
            else ""
        )
        try:
            output = (
                tool.result_runner(action_input)
                if tool.result_runner
                else tool.execute(action_input)
            )
            result = output if isinstance(output, ToolResult) else None
            raw_observation = result.message if result is not None else str(output)
        except Exception as exc:
            metadata["tool_error_count"] += 1
            metadata["failure_reason"] = str(exc)
            raw_observation = f"Tool execution failed: {exc}"
            error_kind = "tool_error"
            result = ToolResult(
                "failed",
                raw_observation,
                error_code=error_kind,
            )
        else:
            tool_succeeded = (
                result.status == "success"
                if result is not None
                else not is_failure_observation(raw_observation, action=action)
            )
            if result is not None and not tool_succeeded:
                error_kind = result.error_code

        bounded_observation = self.bounder.bound(
            raw_observation,
            action=action,
            action_input=action_input,
            context=context,
        )
        confirmed_no_change = (
            tool_succeeded
            and identity_edit_noop
            and not observation_reports_missing_path(bounded_observation)
        )
        fact = build_execution_fact(
            action,
            result,
            bounded_observation,
            step_number=context.step_number,
            tool_succeeded=tool_succeeded,
            no_progress=confirmed_no_change,
            workspace_version=workspace_version(Path.cwd()),
        )
        after_event = AfterToolResultEvent(
            tool_name=action,
            arguments=action_input,
            observation=bounded_observation,
            step_number=context.step_number,
            run_id=context.run_id,
            tool_succeeded=tool_succeeded,
            no_change=confirmed_no_change,
            no_change_reason="identical_content" if confirmed_no_change else "",
            metadata=metadata,
            result=result,
            execution_fact=fact,
        )
        observation = self.event_bus.emit_after_tool_result(after_event, on_error=self.on_error)
        if journal:
            journal.finish(call_id)
        return ToolInvocation(
            arguments=action_input,
            observation=observation,
            error_kind=error_kind,
            tool_succeeded=tool_succeeded,
            no_change=after_event.no_change,
            result=result,
            fact=fact,
        )
