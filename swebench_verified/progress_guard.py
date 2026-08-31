"""SWE-bench 预测专用的确定性进度环守卫。

它只在独立预测子系统中装配，处理两种已有真实轨迹证据的机械固定点：

- 连续重复、参数/工作区状态/输出均不变的只读或执行工具调用；
- 内容锚定编辑撤销后，再次试图进入刚刚访问过的文件内容状态。

这不是内核通用熔断器，也不按失败次数禁用工具。重复工具允许一次真实复查，只有
第三次仍得到相同结果才复用缓存并拦截；文件工具以目标文件内容指纹为准；
内容锚定编辑允许一次撤销，只有随后再次进入旧状态时才在执行前拦截。行号编辑与
覆盖既有文件的 ``create_file`` 只在执行后记录回访，不承诺预拦。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, BeforeToolCallEvent, RunStartEvent
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.core.observation import is_failure_observation

_READ_ACTION = "read_file"
_SEARCH_ACTION = "search_in_file"
_REPEATABLE_TOOLS = frozenset(
    {
        _READ_ACTION,
        _SEARCH_ACTION,
        "run_python",
        "run_shell",
        "run_tests",
        "run_linter",
    }
)
_EXECUTION_TOOLS = _REPEATABLE_TOOLS - {_READ_ACTION, _SEARCH_ACTION}
_IGNORED_STATE_DIRS = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)


@dataclass(frozen=True)
class _ToolRecord:
    action: str
    signature: str
    state_fingerprint: str
    first_success_step: int
    observation: str
    observation_fingerprint: str
    stable_repeats: int = 0


@dataclass(frozen=True)
class _PendingWrite:
    path_key: str
    display_path: str
    before_fingerprint: str
    content_edit_preblock_eligible: bool


class SWEProgressLoopGuard:
    """切断 SWE-bench 预测里的搜索固定点与内容锚定编辑状态二周期。"""

    def __init__(self) -> None:
        self._last_tool: _ToolRecord | None = None
        self._pending_tool_states: dict[tuple[str, int], str] = {}
        self._file_states: dict[str, dict[str, str]] = {}
        self._state_revisit_counts: dict[str, int] = {}
        self._pending_writes: dict[tuple[str, int], _PendingWrite] = {}
        self._trace_writer: Any | None = None

    def install(self, context: CapabilityContext) -> None:
        """把守卫挂到 SWE-bench Agent 的生命周期事件总线上。"""
        self._trace_writer = context.trace_writer
        context.event_bus.on(
            "on_run_start",
            self._on_run_start,
            name="swebench.progress_loop.run_start",
        )
        context.event_bus.on(
            "before_tool_call",
            self._before_tool_call,
            name="swebench.progress_loop.before_tool_call",
        )
        context.event_bus.on(
            "after_tool_result",
            self._after_tool_result,
            name="swebench.progress_loop.after_tool_result",
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._reset_repeatable_sequence()
        self._pending_tool_states.clear()
        self._file_states.clear()
        self._state_revisit_counts.clear()
        self._pending_writes.clear()
        event.metadata["progress_loop_guard_enabled"] = True
        event.metadata["repeat_tool_block_count"] = 0
        event.metadata["repeat_search_block_count"] = 0
        event.metadata["repeat_read_block_count"] = 0
        event.metadata["edit_state_revisit_count"] = 0
        event.metadata["edit_cycle_block_count"] = 0

    def _before_tool_call(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name in _REPEATABLE_TOOLS:
            return self._before_repeatable_tool(event)
        self._reset_repeatable_sequence()
        if event.tool_name not in WRITE_ACTIONS:
            return None

        state = _read_path_state(event.arguments)
        if state is None:
            return None
        path_key, display_path, current_text, current_fingerprint = state
        predicted_fingerprint = _predicted_content_edit_fingerprint(
            event,
            current_text=current_text,
        )
        pending_key = (event.run_id, event.step_number)
        self._pending_writes[pending_key] = _PendingWrite(
            path_key=path_key,
            display_path=display_path,
            before_fingerprint=current_fingerprint,
            content_edit_preblock_eligible=predicted_fingerprint is not None,
        )
        states = self._file_states.setdefault(path_key, {})
        states.setdefault(current_fingerprint, f"before step {event.step_number}")

        if predicted_fingerprint is None or predicted_fingerprint == current_fingerprint:
            return None
        previous_label = states.get(predicted_fingerprint)
        if previous_label is None or self._state_revisit_counts.get(path_key, 0) < 1:
            return None

        self._pending_writes.pop(pending_key, None)
        block_count = int(event.metadata.get("edit_cycle_block_count", 0)) + 1
        reason = (
            f"Skipped edit-state cycle #{block_count}: this edit would return {display_path} "
            f"to content already seen {previous_label}. One undo was allowed, but repeating the "
            "same edit/revert path would only erase progress. Choose a materially different "
            "change instead of reapplying the same transition."
        )
        self._record_trace(
            "swebench_edit_cycle_block",
            {
                "step_number": event.step_number,
                "path": display_path,
                "previous_state": previous_label,
                "block_count": block_count,
            },
        )
        event.metadata["edit_cycle_block_count"] = block_count
        return {"block": True, "reason": reason}

    def _before_repeatable_tool(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        signature = _arguments_signature(event.arguments)
        state_fingerprint = _tool_state_fingerprint(event.tool_name, event.arguments)
        self._pending_tool_states[(event.run_id, event.step_number)] = state_fingerprint
        record = self._last_tool
        if (
            record is None
            or record.action != event.tool_name
            or record.signature != signature
            or record.state_fingerprint != state_fingerprint
        ):
            self._reset_repeatable_sequence()
            return None
        if record.stable_repeats < 1:
            return None

        block_count = int(event.metadata.get("repeat_tool_block_count", 0)) + 1
        noun = (
            "read"
            if event.tool_name == _READ_ACTION
            else ("search" if event.tool_name == _SEARCH_ACTION else "tool call")
        )
        reason = (
            f"Skipped exact duplicate {noun} #{block_count}: the same call was executed twice "
            f"with unchanged state and output, first at step {record.first_success_step}. "
            "Reuse the cached result, then change the arguments, inspect different evidence, "
            "edit the workspace, or finish the task.\n"
            f"Cached result:\n{_quote_observation(record.observation)}"
        )
        payload = {
            "step_number": event.step_number,
            "tool_name": event.tool_name,
            "arguments": dict(event.arguments),
            "first_success_step": record.first_success_step,
            "block_count": block_count,
        }
        self._record_trace("swebench_repeat_tool_block", payload)
        legacy_key = None
        legacy_event = None
        if event.tool_name == _READ_ACTION:
            legacy_key = "repeat_read_block_count"
            legacy_event = "swebench_repeat_read_block"
        elif event.tool_name == _SEARCH_ACTION:
            legacy_key = "repeat_search_block_count"
            legacy_event = "swebench_repeat_search_block"
        if legacy_key and legacy_event:
            event.metadata[legacy_key] = int(event.metadata.get(legacy_key, 0)) + 1
            self._record_trace(legacy_event, payload)
        event.metadata["repeat_tool_block_count"] = block_count
        return {"block": True, "reason": reason}

    def _after_tool_result(self, event: AfterToolResultEvent) -> str | None:
        if event.tool_name in _REPEATABLE_TOOLS:
            self._remember_repeatable_tool(event)
            return None
        if event.tool_name not in WRITE_ACTIONS:
            return None

        pending = self._pending_writes.pop((event.run_id, event.step_number), None)
        if pending is None:
            return None
        state = _read_path_state({"path": pending.display_path})
        if state is None:
            return None
        _, _, _, after_fingerprint = state
        if after_fingerprint == pending.before_fingerprint:
            return None

        states = self._file_states.setdefault(pending.path_key, {})
        previous_label = states.get(after_fingerprint)
        if previous_label is None:
            states[after_fingerprint] = f"after step {event.step_number}"
            self._state_revisit_counts[pending.path_key] = 0
            return None

        revisit_count = self._state_revisit_counts.get(pending.path_key, 0) + 1
        self._state_revisit_counts[pending.path_key] = revisit_count
        event.metadata["edit_state_revisit_count"] = (
            int(event.metadata.get("edit_state_revisit_count", 0)) + 1
        )
        self._record_trace(
            "swebench_edit_state_revisit",
            {
                "step_number": event.step_number,
                "path": pending.display_path,
                "previous_state": previous_label,
                "revisit_count": revisit_count,
                "content_edit_preblock_eligible": pending.content_edit_preblock_eligible,
            },
        )
        if pending.content_edit_preblock_eligible:
            guidance = (
                "This state revisit was applied. A later canonical old_string/new_string edit "
                "that exactly re-enters a seen state will be skipped before execution. Choose "
                "a materially different fix before editing it again."
            )
        else:
            guidance = (
                "This state revisit was recorded for diagnostics only. This write mode is not "
                "predicted or blocked before execution; re-read the current content before the "
                "next change."
            )
        return (
            f"{event.observation}\n\n"
            f"Edit-state revisit #{revisit_count}: {pending.display_path} now matches content "
            f"already seen {previous_label}. {guidance}"
        )

    def _remember_repeatable_tool(self, event: AfterToolResultEvent) -> None:
        before_state = self._pending_tool_states.pop(
            (event.run_id, event.step_number),
            _tool_state_fingerprint(event.tool_name, event.arguments),
        )
        if not event.tool_succeeded or is_failure_observation(
            event.observation, action=event.tool_name
        ):
            self._reset_repeatable_sequence()
            return
        signature = _arguments_signature(event.arguments)
        state_fingerprint = _tool_state_fingerprint(event.tool_name, event.arguments)
        if event.tool_name in _EXECUTION_TOOLS and state_fingerprint != before_state:
            self._reset_repeatable_sequence()
            return
        observation_fingerprint = _text_fingerprint(event.observation)
        record = self._last_tool
        if (
            record is not None
            and record.action == event.tool_name
            and record.signature == signature
            and record.state_fingerprint == state_fingerprint
            and record.observation_fingerprint == observation_fingerprint
        ):
            self._last_tool = _ToolRecord(
                action=record.action,
                signature=record.signature,
                state_fingerprint=record.state_fingerprint,
                first_success_step=record.first_success_step,
                observation=event.observation,
                observation_fingerprint=observation_fingerprint,
                stable_repeats=record.stable_repeats + 1,
            )
            return
        self._last_tool = _ToolRecord(
            action=event.tool_name,
            signature=signature,
            state_fingerprint=state_fingerprint,
            first_success_step=event.step_number,
            observation=event.observation,
            observation_fingerprint=observation_fingerprint,
        )

    def _reset_repeatable_sequence(self) -> None:
        self._last_tool = None

    def _record_trace(self, event: str, payload: dict[str, Any]) -> None:
        if not self._trace_writer:
            return
        try:
            self._trace_writer.record(event, payload)
        except OSError:
            # trace 可用性不能改变守卫的放行/拦截决策。
            return


def _arguments_signature(arguments: dict[str, Any]) -> str:
    """参数顺序无关、跨进程稳定的工具调用签名。"""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_state_fingerprint(action: str, arguments: dict[str, Any]) -> str:
    """返回与调用相关的最小状态；文件内容变化会立即打断重复序列。"""
    if action in {_READ_ACTION, _SEARCH_ACTION}:
        state = _read_path_state(arguments)
        return state[3] if state is not None else "missing-or-unreadable"
    return _workspace_state_fingerprint(Path.cwd())


def _workspace_state_fingerprint(root: Path) -> str:
    """用路径、大小和修改时间检测执行工具是否改变了工作区。"""
    digest = hashlib.sha256()
    try:
        for current_root, directories, files in os.walk(root):
            directories[:] = sorted(name for name in directories if name not in _IGNORED_STATE_DIRS)
            current = Path(current_root)
            for name in sorted(files):
                path = current / name
                try:
                    stat = path.stat()
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                digest.update(relative.encode("utf-8", errors="surrogatepass"))
                digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    except OSError:
        return "workspace-unreadable"
    return digest.hexdigest()


def _read_path_state(arguments: dict[str, Any]) -> tuple[str, str, str, str] | None:
    path_value = arguments.get("path")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    path_key = os.path.normcase(os.path.abspath(path))
    return path_key, path_value, text, _text_fingerprint(text)


def _predicted_content_edit_fingerprint(
    event: BeforeToolCallEvent,
    *,
    current_text: str,
) -> str | None:
    if event.tool_name != "edit_file" or not event.content_anchor_safe:
        return None
    arguments = event.arguments
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string", "")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return None
    if any(arguments.get(name) is not None for name in ("operation", "line_start", "line_end")):
        return None
    if not old_string or old_string == new_string or current_text.count(old_string) != 1:
        return None
    return _text_fingerprint(current_text.replace(old_string, new_string, 1))


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quote_observation(observation: str) -> str:
    # observation 已经由内核的 ObservationBounder 限制过；这里必须完整回放，
    # 否则上下文折叠后模型会失去重新取得后半段匹配结果的唯一通道。
    quoted = "\n".join(f"> {line}" for line in observation.splitlines())
    return quoted or "> (empty result)"
