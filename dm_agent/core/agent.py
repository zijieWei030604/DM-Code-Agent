"""由 LLM API 驱动的 ReAct 风格智能体。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from dm_agent.clients.base_client import BaseLLMClient
from dm_agent.memory.lcm.compaction import ContextOverflow
from dm_agent.memory.repo_map import RepoMapResult, RepositoryMap
from dm_agent.prompts import build_code_agent_prompt
from dm_agent.tools.base import Tool
from dm_agent.tracing.session import parse_failed_response_placeholder
from dm_agent.tracing.writer import SessionWriter

from .capabilities import AgentCapability, CapabilityContext
from .checkpoint import RunCheckpoint
from .completion import CompletionGate, build_run_result, format_final_answer
from .events import (
    EventBus,
    HookFailure,
    LLMRequestClient,
    RunEndEvent,
    RunStartEvent,
)
from .guards import ReadBeforeEditGuard
from .lcm_context_window import LCMContextWindow
from .lcm_memory import LCM_RECALL_GUIDANCE, LCMMemory
from .observation import ObservationBounder, is_failure_observation
from .persistence import (
    RunPersistence,
    agent_config_snapshot,
    json_safe_metadata,
    metadata_from_checkpoint,
    steps_from_checkpoint,
    warn_on_config_mismatch,
)
from .planner import AdaptiveReplanPolicy
from .prompting import build_user_prompt
from .response_parser import normalize_action, parse_agent_response
from .run_state import RunContext, Step, initial_run_metadata
from .task_plan import TaskPlan
from .tool_invoker import ToolInvoker

__all__ = ["ReactAgent", "Step"]


def _required_tool_choice_unsupported(exc: Exception) -> bool:
    """Return true only for explicit provider rejection of required tool choice."""
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(str(current).lower())
        current = current.__cause__
    text = " ".join(chain)
    mentions_option = "tool_choice" in text or "tool choice" in text
    mentions_required = "required" in text
    explicit_rejection = any(
        token in text
        for token in (
            "unsupported",
            "not supported",
            "invalid",
            "not allowed",
            "unknown parameter",
            "unrecognized",
        )
    )
    return mentions_option and mentions_required and explicit_rejection


class ReactAgent:
    """ReAct（推理 + 行动）循环的内核。

    职责只剩三件：装配协作者、跑主循环、维护对话历史。其余能力各自住在
    独立模块里，本类通过它们完成一步的各个环节：

    - ``core.context_window``：构造发给 LLM 的消息（按需压缩旧上下文）
    - ``core.response_parser``：容错解析模型响应
    - ``core.tool_invoker``：工具调用链（校验 → 钩子 → 备份 → 执行 → 截断）
    - ``core.completion``：完成判定与结果格式化
    - ``core.task_plan``：模型通过工具维护的任务清单
    - ``core.persistence``：checkpoint 存取与写前备份

    可选能力（熔断等）不住在这里，而是通过 ``core.events``
    的生命周期钩子接入，见 ``core.capabilities``。
    """

    def __init__(
        self,
        client: BaseLLMClient,
        tools: list[Tool],
        *,
        max_steps: int = 200,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        step_callback: Callable[[int, Step], None] | None = None,  # 步骤回调函数
        enable_planning: bool = True,  # 是否启用规划
        enable_compression: bool = True,  # 是否启用上下文压缩
        skill_manager: Any | None = None,  # 技能管理器
        trace_writer: Any | None = None,
        capabilities: Sequence[AgentCapability] = (),
        enable_adaptive_replanning: bool = False,
        replan_policy: AdaptiveReplanPolicy | None = None,
        max_replans: int = -1,
        max_observation_chars: int = 8000,
        context_token_budget: int = 24000,
        enable_edit_guard: bool = True,
        enable_semantic_workspace: bool = False,
        enable_repo_map: bool = False,
        repository_map: RepositoryMap | None = None,
        event_bus: EventBus | None = None,
        owned_resources: Sequence[Any] = (),
    ) -> None:
        """初始化 ReactAgent。

        ``tools`` 之外的参数都有可用的默认值。带 ``enable_`` 前缀的行为类开关默认
        关闭，基础设施护栏（观察截断、token 预算、read-before-edit 守卫）默认开启；
        取值口径与 ``dm-agent`` CLI 的同名参数一致。

        Raises:
            ValueError: 工具列表为空
        """
        if not tools:
            raise ValueError("必须为 ReactAgent 提供至少一个工具。")
        self.client = client
        self._owned_resources = list(owned_resources)
        self._closed = False
        self.trace_writer = (
            trace_writer if isinstance(trace_writer, SessionWriter) else SessionWriter(trace_writer)
        )
        self.event_bus = event_bus or EventBus()
        self._run_context = RunContext()
        request_client = LLMRequestClient(
            client,
            self.event_bus,
            self._run_context.as_event_context,
            self._record_hook_error,
        )

        def client_for(phase: str) -> BaseLLMClient:
            return cast(BaseLLMClient, request_client.with_phase(phase))

        self._request_client = client_for("agent")
        self._completion_gate = CompletionGate(self.event_bus, on_error=self._record_hook_error)

        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.native_tool_calling = bool(getattr(client, "supports_tool_calling", False))
        self.system_prompt = system_prompt or build_code_agent_prompt(
            tools, native_tool_calling=self.native_tool_calling
        )
        self.step_callback = step_callback
        # 多轮对话历史记录
        self.conversation_history: list[dict[str, str]] = []

        # Task-scoped checklist, updated by the same ReAct loop.
        self.enable_planning = enable_planning
        self.task_plan = TaskPlan(self.trace_writer)
        if enable_planning:
            if "update_plan" in self.tools:
                raise ValueError("update_plan is reserved for the task checklist.")
            self.tools["update_plan"] = self.task_plan.tool()

        # LCM owns history storage and tool-free summary calls; zero disables compaction.
        self.enable_compression = enable_compression
        self.context_token_budget = max(0, int(context_token_budget))
        self.compressor = (
            LCMMemory(
                client,
                token_budget=self.context_token_budget,
                trace_writer=self.trace_writer,
            )
            if enable_compression
            else None
        )
        if self.compressor:
            memory_tools = self.compressor.tools()
            self.tools.update({tool.name: tool for tool in memory_tools})
            self.tools_list = list(self.tools.values())
            if system_prompt is None:
                self.system_prompt = build_code_agent_prompt(
                    self.tools_list, native_tool_calling=self.native_tool_calling
                )
            self.system_prompt += LCM_RECALL_GUIDANCE
        self._context_window = LCMContextWindow(
            compressor=self.compressor,
            enabled=enable_compression,
            trace_writer=self.trace_writer,
        )
        # 单条工具观察的字符上限；0 表示不截断。
        self.max_observation_chars = max(0, int(max_observation_chars))
        self._observation_bounder = ObservationBounder(
            max_chars=self.max_observation_chars,
            trace_writer=self.trace_writer,
            preserve_output=self.compressor.preserve_output if self.compressor else None,
        )
        self._persistence = RunPersistence(trace_writer=self.trace_writer)
        self._tool_invoker = ToolInvoker(
            event_bus=self.event_bus,
            bounder=self._observation_bounder,
            persistence=self._persistence,
            on_error=self._record_hook_error,
        )
        # read-before-edit 守卫：首次编辑前必须读过目标文件；依赖行号的连续编辑
        # 在写后需重读，内容锚定编辑由唯一精确匹配保证当前性。
        self.enable_edit_guard = enable_edit_guard
        self.enable_semantic_workspace = enable_semantic_workspace
        self._edit_guard = ReadBeforeEditGuard(
            enabled=enable_edit_guard, trace_writer=self.trace_writer
        )
        self.enable_repo_map = enable_repo_map
        self._repo_map = (repository_map or RepositoryMap()) if enable_repo_map else None
        # 技能管理器
        self.skill_manager = skill_manager
        self.tools_list = list(self.tools.values())
        if enable_planning:
            self.system_prompt += (
                "\nUse update_plan for complex tasks when useful. Update progress yourself; "
                "tool success does not automatically complete a goal. Plan completion does "
                "not establish correctness. Follow actual observations and update the plan "
                "when the route changes. Previous tasks' plans are historical only.\n"
            )
            if not self.native_tool_calling and (system_prompt is not None or not self.compressor):
                self.system_prompt += json.dumps(
                    self.task_plan.tool().function_definition(), ensure_ascii=False
                )
        self._base_system_prompt = self.system_prompt
        self._base_tools = dict(self.tools)
        # Legacy constructor arguments remain accepted, but never start another LLM loop.
        self.enable_adaptive_replanning = False
        self.max_replans = 0
        self._legacy_replan_options = (
            enable_adaptive_replanning or replan_policy is not None or max_replans != -1
        )

        # 可选能力装配：注册顺序即钩子执行顺序，能力先于内核内置守卫注册。
        self.capabilities: list[AgentCapability] = list(capabilities)
        capability_context = CapabilityContext(
            event_bus=self.event_bus,
            client_for=client_for,
            trace_writer=self.trace_writer,
            get_run_state=self._capability_run_state,
        )
        for capability in self.capabilities:
            capability.install(capability_context)

        self.event_bus.on(
            "before_tool_call",
            self._edit_guard.before_tool_call,
            name="builtin.read_before_edit_guard",
            kind="policy",
        )
        self.event_bus.on(
            "after_tool_result",
            self._edit_guard.after_tool_result,
            name="builtin.read_before_edit_ledger",
        )

    def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        checkpoint_path: Path | None = None,
        resume_state: RunCheckpoint | None = None,
    ) -> dict[str, Any]:
        """跑一次任务，并让 ``on_run_end`` 处理器决定是否重试。

        重试时对话历史恢复到调用前的快照，因此每次尝试都是干净的一轮；
        ``on_run_start`` 处理器可以借 ``prompt_suffix`` 把上一轮的经验带进来。
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("任务必须是非空字符串。")
        if (
            checkpoint_path is not None or resume_state is not None
        ) and self.event_bus.has_retry_handlers():
            raise ValueError("checkpoint/resume 暂不支持与 on_run_end 重试同时使用。")

        initial_history = [dict(message) for message in self.conversation_history]
        initial_compressor_state = (
            self.compressor.snapshot_runtime_state() if self.compressor else None
        )
        attempt = 1
        while True:
            if attempt > 1:
                self.conversation_history = [dict(message) for message in initial_history]
                if self.compressor and initial_compressor_state is not None:
                    self.compressor.restore_runtime_state(initial_compressor_state)
            result = self._run_once(
                task,
                max_steps=max_steps,
                attempt=attempt,
                checkpoint_path=checkpoint_path if attempt == 1 else None,
                resume_state=resume_state if attempt == 1 else None,
            )
            end_event = RunEndEvent(
                task=task,
                attempt=attempt,
                run_id=self._run_context.run_id,
                result=result,
                metadata=result.get("metadata", {}),
            )
            decision = self.event_bus.emit_run_end(end_event, on_error=self._record_hook_error)
            if decision is None or not decision.get("retry"):
                return result
            attempt += 1

    def _record_hook_error(self, failure: HookFailure) -> None:
        if self.trace_writer:
            self.trace_writer.record("hook_error", failure.to_trace_payload())

    def _append_history(
        self,
        role: str,
        content: str,
        *,
        kind: str,
        context_content: str | None = None,
    ) -> None:
        """把一条消息追加进对话历史，同时在会话日志里落一条 ``message`` 条目。

        ``conversation_history`` 与 ``RunContext.history_entry_ids`` 必须逐位对应——
        压缩条目的 ``first_kept_entry_id`` 靠这个映射把「第几条消息」翻译成 entry id。
        没有 trace_writer 时补一个空 id 占位，保证下标始终对齐。

        ``context_content`` 只改变后续发给模型的派生视图；会话日志仍记录 ``content``
        原文。解析失败响应用这条路径保留审计证据，同时不再污染上下文。
        """
        self.conversation_history.append(
            {"role": role, "content": content if context_content is None else context_content}
        )
        if self.compressor:
            self.compressor.adopt(self.conversation_history[:-1])
            self.compressor.append(role, content, kind=kind, context_content=context_content)
        entry_id = ""
        if self.trace_writer:
            entry_id = self.trace_writer.record_message(
                role=role,
                content=content,
                step_number=self._run_context.step_number,
                kind=kind,
            )
        self._run_context.history_entry_ids.append(entry_id)

    def _adopt_existing_history(self, *, kind: str) -> None:
        """让会话日志补齐「这一轮开始前就在历史里」的消息。

        ``conversation_history`` 可能在 ``_run_once`` 之前就非空（交互式多轮、
        on_run_end 重试恢复的快照、resume 恢复的历史）。会话日志要能独立复现上下文，
        所以把这些消息补记成 ``message`` 条目，顺带把 ``history_entry_ids``
        对齐回 ``len(conversation_history)``。历史为空时这是个空操作。
        """
        pending = self.conversation_history[len(self._run_context.history_entry_ids) :]
        for message in pending:
            entry_id = ""
            if self.trace_writer:
                entry_id = self.trace_writer.record_message(
                    role=str(message.get("role", "")),
                    content=str(message.get("content", "")),
                    step_number=self._run_context.step_number,
                    kind=kind,
                )
            self._run_context.history_entry_ids.append(entry_id)

    def _note_observation(self, observation: str) -> None:
        """把一条观察作为 user 消息投递回对话历史。"""
        self._append_history("user", f"观察：{observation}", kind="observation")

    def _publish_step(self, step: Step, step_num: int) -> None:
        """一步收尾：实时回调 + trace 落步。"""
        if self.step_callback:
            self.step_callback(step_num, step)
        if self.trace_writer:
            self.trace_writer.record_step(step_number=step_num, step=step)

    def _run_once(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        attempt: int = 1,
        checkpoint_path: Path | None = None,
        resume_state: RunCheckpoint | None = None,
    ) -> dict[str, Any]:
        """跑完整的一轮 ReAct 循环：规划 → 推理 → 行动 → 观察，直到完成或步数耗尽。

        ``run`` 负责多次尝试之间的编排，本方法只管一次尝试。

        Returns:
            含 ``final_answer`` / ``steps`` / ``metadata`` 三个键的结果字典

        Raises:
            ValueError: 任务不是非空字符串
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("任务必须是非空字符串。")

        # 每次尝试都从基础 prompt/工具重新出发，避免上一次 run（技能、prompt 追加）的残留。
        self.system_prompt = self._base_system_prompt
        self.tools = dict(self._base_tools)

        started_at = time.perf_counter()
        steps: list[Step] = []
        limit = max_steps or self.max_steps
        self._persistence.call_journal = None
        if checkpoint_path is not None:
            self._persistence.prepare_session_checkpoint(checkpoint_path)
        self._edit_guard.reset()
        self.task_plan.reset()
        run_token = getattr(self.trace_writer, "run_id", "") or uuid.uuid4().hex[:12]
        retry_baseline = getattr(self.client, "total_respond_retries", 0)
        self._context_window.reset()
        metadata: dict[str, Any] = initial_run_metadata(
            attempt=attempt,
            planning_enabled=self.enable_planning,
            compression_enabled=self.enable_compression,
            skills_enabled=bool(self.skill_manager),
            edit_guard_enabled=self.enable_edit_guard,
            adaptive_replanning_enabled=self.enable_adaptive_replanning,
            max_replans=self.max_replans,
        )
        metadata.update(
            {
                "planning_mode": "model_checklist" if self.enable_planning else "disabled",
                "legacy_replan_options_ignored": self._legacy_replan_options,
                "repo_map_enabled": self.enable_repo_map,
                "repo_map_files": 0,
                "repo_map_chars": 0,
                "repo_map_cache_hits": 0,
                "repo_map_parse_errors": 0,
                "repo_map_truncated": False,
                "repo_map_error": "",
            }
        )
        self._run_context.begin(run_id=run_token, metadata=metadata)
        start_event = RunStartEvent(
            task=task,
            attempt=attempt,
            run_id=run_token,
            metadata=metadata,
        )
        prompt_suffix = self.event_bus.emit_run_start(start_event, on_error=self._record_hook_error)

        # resume 必须先恢复持久化状态，再落 trace 的 run_start。
        resume_from = 0
        if resume_state is not None:
            resume_from = self._restore_from_checkpoint(resume_state, steps, metadata)

        if self.trace_writer:
            self.trace_writer.start_run(
                task,
                metadata={
                    "max_steps": limit,
                    "temperature": self.temperature,
                    "planning_enabled": self.enable_planning,
                    "planning_mode": "model_checklist" if self.enable_planning else "disabled",
                    "plan": self.task_plan.items,
                    "plan_scope": self.task_plan.scope,
                    "plan_revision": self.task_plan.revision,
                    "compression_enabled": self.enable_compression,
                    "max_observation_chars": self.max_observation_chars,
                    "context_token_budget": self.context_token_budget,
                    "edit_guard_enabled": self.enable_edit_guard,
                    "repo_map_enabled": self.enable_repo_map,
                    "skills_enabled": bool(self.skill_manager),
                    "adaptive_replanning_enabled": self.enable_adaptive_replanning,
                    "max_replans": self.max_replans,
                    "trial": metadata["trial"],
                    "tools": [
                        {"name": tool.name, "description": tool.description}
                        for tool in self.tools_list
                    ],
                },
            )

        def finish_result(final_answer: str) -> dict[str, Any]:
            metadata["plan_revision"] = self.task_plan.revision
            metadata["plan_scope"] = self.task_plan.scope
            metadata["plan"] = self.task_plan.items
            metadata["plan_status_source"] = "model_reported"
            metadata["llm_retry_count"] = (
                getattr(self.client, "total_respond_retries", 0) - retry_baseline
            )
            if metadata.get("backup_count"):
                print(f"[backup] 修改前的原文件备份目录：{metadata['backup_dir']}")
            result = build_run_result(final_answer, steps, metadata)
            if self.trace_writer:
                self.trace_writer.finish_run(result)
            return result

        # 技能自动选择
        if self.skill_manager:
            metadata["activated_skills"] = self._apply_skills_for_task(task)
            if self.trace_writer:
                self.trace_writer.record_skills(metadata["activated_skills"])
        if prompt_suffix:
            self.system_prompt += "\n\n" + prompt_suffix

        repo_map_result = RepoMapResult("", 0, 0, 0, 0, False, "")
        if resume_state is None and self._repo_map is not None:
            try:
                repo_map_result = self._repo_map.build(task, Path.cwd())
            except (OSError, ValueError) as exc:
                metadata["repo_map_error"] = str(exc)
                print(f"[warn] Repo Map 生成失败：{exc}")
            else:
                metadata["repo_map_files"] = repo_map_result.included_files
                metadata["repo_map_chars"] = len(repo_map_result.content)
                metadata["repo_map_cache_hits"] = repo_map_result.cache_hits
                metadata["repo_map_parse_errors"] = repo_map_result.parse_errors
                metadata["repo_map_truncated"] = repo_map_result.truncated
                if self.trace_writer:
                    self.trace_writer.record(
                        "repo_map",
                        {
                            "scanned_files": repo_map_result.scanned_files,
                            "included_files": repo_map_result.included_files,
                            "chars": len(repo_map_result.content),
                            "cache_hits": repo_map_result.cache_hits,
                            "parse_errors": repo_map_result.parse_errors,
                            "truncated": repo_map_result.truncated,
                            "fingerprint": repo_map_result.fingerprint,
                        },
                    )

        # Resume restores the checklist; new tasks let ReAct plan through update_plan.
        if resume_state is not None:
            self._adopt_existing_history(kind="resumed")
            if self.trace_writer:
                self.trace_writer.record(
                    "run_resumed",
                    {"from_step": resume_from, "history_messages": len(self.conversation_history)},
                )
            print(f"[resume] 已恢复 checkpoint，从第 {resume_from + 1} 步继续执行")
        else:
            self._adopt_existing_history(kind="carried")
            # 添加新任务到对话历史
            task_prompt: str = build_user_prompt(
                task,
                [],
                repository_map=repo_map_result.content,
                native_tool_calling=self.native_tool_calling,
            )
            self._append_history("user", task_prompt, kind="task")

        native_tool_retry_budget = 3
        for step_num in range(resume_from + 1, limit + 1):
            self._run_context.step_number = step_num
            # 每步开始前落盘上一步完成后的快照（若启用 checkpoint）。
            if checkpoint_path is not None:
                self._save_checkpoint_snapshot(
                    checkpoint_path,
                    task=task,
                    step_count=step_num - 1,
                    steps=steps,
                    metadata=metadata,
                    limit=limit,
                )
            # 第二步：整理旧上下文为本地记忆（如果需要）
            tool_definitions = (
                [tool.function_definition() for tool in self.tools.values()]
                if self.native_tool_calling
                else []
            )
            try:
                messages_to_send = self._context_window.build_messages(
                    self.system_prompt + (self.task_plan.context() if self.enable_planning else ""),
                    self.conversation_history,
                    context=self._run_context,
                    tool_definitions=tool_definitions,
                )
            except ContextOverflow as exc:
                metadata["status"] = "context_overflow"
                metadata["failure_reason"] = str(exc)
                metadata["duration_seconds"] = time.perf_counter() - started_at
                if self.trace_writer:
                    self.trace_writer.record(
                        "context_budget",
                        {
                            "phase": "overflow_stopped",
                            "step_number": step_num,
                            "reason": str(exc),
                        },
                    )
                return finish_result("")

            # 获取 AI 响应
            request_messages = messages_to_send
            try:
                request_options: dict[str, Any] = {"temperature": self.temperature}
                if getattr(self._request_client, "supports_tool_calling", False):
                    request_options["tool_definitions"] = tool_definitions
                    request_options["tool_choice"] = "auto"
                raw = self._request_client.respond(request_messages, **request_options)

                response_mode = str(
                    getattr(self._request_client, "last_response_mode", "")
                    or ("json_fallback" if self.native_tool_calling else "prompt_json")
                )
                if self.native_tool_calling and response_mode != "native_tool_call":
                    metadata["native_tool_call_missing_count"] += 1
                    try:
                        parse_agent_response(raw)
                    except ValueError:
                        needs_retry = True
                    else:
                        needs_retry = False
                    can_retry = (
                        needs_retry
                        and metadata["native_tool_retry_count"] < native_tool_retry_budget
                    )
                    if self.trace_writer:
                        self.trace_writer.record(
                            "native_tool_call_missing",
                            {
                                "step_number": step_num,
                                "response_mode": response_mode,
                                "valid_json_fallback": not needs_retry,
                                "retry": can_retry,
                            },
                        )
                    if can_retry:
                        metadata["native_tool_retry_count"] += 1
                        request_messages = [
                            *messages_to_send,
                            {
                                "role": "system",
                                "content": (
                                    "The previous response did not contain a native tool call. "
                                    "Continue by calling exactly one available tool; call "
                                    "task_complete when the task is finished."
                                ),
                            },
                        ]
                        if self.trace_writer:
                            self.trace_writer.record(
                                "native_tool_retry",
                                {
                                    "step_number": step_num,
                                    "attempt": metadata["native_tool_retry_count"],
                                    "task_budget": native_tool_retry_budget,
                                },
                            )
                        retry_options = {**request_options, "tool_choice": "required"}
                        try:
                            raw = self._request_client.respond(request_messages, **retry_options)
                        except Exception as exc:
                            if not _required_tool_choice_unsupported(exc):
                                raise
                            metadata["protocol_downgrade_count"] += 1
                            if self.trace_writer:
                                self.trace_writer.record(
                                    "tool_protocol_downgraded",
                                    {
                                        "step_number": step_num,
                                        "from": "required",
                                        "to": "auto",
                                        "reason": str(exc),
                                    },
                                )
                            raw = self._request_client.respond(request_messages, **request_options)
            except Exception as exc:
                if self.trace_writer:
                    self.trace_writer.record(
                        "llm_error",
                        {"step_number": step_num, "error": str(exc)},
                    )
                raise
            if self.trace_writer:
                response_mode = str(
                    getattr(self._request_client, "last_response_mode", "")
                    or ("json_fallback" if self.native_tool_calling else "prompt_json")
                )
                tool_call_count = int(getattr(self._request_client, "last_tool_call_count", 0) or 0)
                selected_tool = str(getattr(self._request_client, "last_selected_tool", "") or "")
                self.trace_writer.record_llm_call(
                    step_number=step_num,
                    messages=request_messages,
                    temperature=self.temperature,
                    raw_response=raw,
                    response_mode=response_mode,
                    tool_call_count=tool_call_count,
                    selected_tool=selected_tool,
                    budget_breakdown=(
                        self._context_window.last_budget_breakdown.to_dict()
                        if self._context_window.last_budget_breakdown
                        else None
                    ),
                )
            response_mode = str(
                getattr(self._request_client, "last_response_mode", "")
                or ("json_fallback" if self.native_tool_calling else "prompt_json")
            )
            if response_mode == "native_tool_call":
                metadata["native_tool_call_count"] += 1
                metadata["discarded_tool_call_count"] += max(
                    int(getattr(self._request_client, "last_tool_call_count", 0) or 0) - 1,
                    0,
                )
            elif response_mode == "json_fallback":
                metadata["json_fallback_count"] += 1

            try:
                parsed_response = parse_agent_response(raw)
            except ValueError as exc:
                context_replacement = parse_failed_response_placeholder(len(raw))
                self._append_history(
                    "assistant",
                    raw,
                    kind="model_response",
                    context_content=context_replacement,
                )
                metadata["parse_error_count"] += 1
                metadata["parse_error_context_omitted_count"] = (
                    int(metadata.get("parse_error_context_omitted_count", 0)) + 1
                )
                metadata["parse_error_context_omitted_chars"] = int(
                    metadata.get("parse_error_context_omitted_chars", 0)
                ) + len(raw)
                metadata["failure_reason"] = str(exc)
                observation = f"Agent response parse failed: {exc}"
                if self.trace_writer:
                    self.trace_writer.record_parse_error(
                        step_number=step_num,
                        raw_response=raw,
                        error=str(exc),
                        context_replacement=context_replacement,
                    )
                step = Step(
                    thought="",
                    action="error",
                    action_input={},
                    observation=observation,
                    raw=raw,
                )
                steps.append(step)

                # 将错误观察添加到历史记录
                self._note_observation(observation)
                self._publish_step(step, step_num)
                continue
            self._append_history("assistant", raw, kind="model_response")
            parsed = parsed_response.data
            if parsed_response.repaired:
                metadata["parse_repair_count"] += 1

            # 获取动作、thought 和输入
            raw_action_value = parsed.get("action", "")
            raw_action = "" if raw_action_value is None else str(raw_action_value).strip()
            action = normalize_action(raw_action)
            if action != raw_action:
                metadata["terminal_action_alias_count"] += 1
                metadata["terminal_action_aliases"].append(
                    {
                        "step_number": step_num,
                        "raw": raw_action,
                        "normalized": action,
                    }
                )
            thought = parsed.get("thought", "").strip()
            action_input = parsed.get("action_input")

            # 检查是否完成
            if action == "finish":
                final = format_final_answer(action_input)
                accepted, observation = self._completion_gate.review(
                    task=task,
                    action=action,
                    completion_text=final,
                    steps=steps,
                    context=self._run_context,
                )
                step = Step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation="<finished>" if accepted else observation,
                    raw=raw,
                )
                steps.append(step)

                if accepted:
                    metadata["status"] = "success"
                    metadata["failure_reason"] = ""
                    metadata["duration_seconds"] = time.perf_counter() - started_at
                    # 添加完成标记到历史记录
                    self._append_history("user", f"任务完成：{final}", kind="completion")
                else:
                    self._note_observation(observation)

                self._publish_step(step, step_num)
                if accepted:
                    return finish_result(final)
                if metadata.get("evidence_terminal_completion_rejection"):
                    metadata["status"] = "critic_rejected"
                    metadata["failure_reason"] = observation
                    metadata["duration_seconds"] = time.perf_counter() - started_at
                    return finish_result("")
                continue

            # 检查工具
            tool = self.tools.get(action)
            if tool is None:
                metadata["unknown_tool_count"] += 1
                metadata["failure_reason"] = f"Unknown tool: {action}"
                observation = f"Unknown tool '{action}'."
                step = Step(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw=raw,
                )
                steps.append(step)

                # 将观察结果添加到历史记录
                self._note_observation(observation)

                if self.step_callback:
                    self.step_callback(step_num, step)
                if self.trace_writer:
                    self.trace_writer.record_tool_call(
                        step_number=step_num,
                        action=action,
                        action_input=action_input,
                        observation=observation,
                        failed=True,
                    )
                    self.trace_writer.record_step(step_number=step_num, step=step)
                continue

            invocation = self._tool_invoker.invoke(
                tool,
                action=action,
                action_input=action_input,
                context=self._run_context,
            )
            action_input = invocation.arguments
            observation = invocation.observation
            invocation_failed = (
                invocation.result.status == "failed"
                if invocation.result is not None
                else self._is_failure_observation(observation, action=action)
            )
            accepted = False
            if action == "task_complete" and invocation.tool_succeeded:
                accepted, observation = self._completion_gate.review(
                    task=task,
                    action=action,
                    completion_text=str(observation),
                    steps=steps,
                    context=self._run_context,
                )
                if not accepted:
                    invocation_failed = True

            step = Step(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                raw=raw,
            )
            steps.append(step)
            if self.trace_writer:
                self.trace_writer.record_tool_call(
                    step_number=step_num,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    failed=invocation_failed,
                )

            # 将工具执行结果添加到历史记录
            tool_info = f"执行工具 {action}，输入：{json.dumps(action_input, ensure_ascii=False)}\n观察：{observation}"
            self._append_history("user", tool_info, kind="tool_result")

            # 调用回调函数实时输出步骤
            self._publish_step(step, step_num)

            if not accepted and metadata.get("evidence_terminal_completion_rejection"):
                metadata["status"] = "critic_rejected"
                metadata["failure_reason"] = observation
                metadata["duration_seconds"] = time.perf_counter() - started_at
                return finish_result("")

            # 检查是否调用了 task_complete 工具
            if (
                action == "task_complete"
                and accepted
                and not self._is_failure_observation(observation, action=action)
            ):
                metadata["status"] = "success"
                metadata["failure_reason"] = ""
                metadata["duration_seconds"] = time.perf_counter() - started_at
                return finish_result(observation)

        metadata["status"] = "max_steps_exceeded"
        metadata["duration_seconds"] = time.perf_counter() - started_at
        metadata["failure_reason"] = metadata["failure_reason"] or "Max steps exceeded"
        if checkpoint_path is not None:
            # 步数耗尽也落一份终态快照：换更大的 --max-steps 即可 resume 续跑。
            self._save_checkpoint_snapshot(
                checkpoint_path,
                task=task,
                step_count=limit,
                steps=steps,
                metadata=metadata,
                limit=limit,
            )
        return finish_result("Reached step limit without completion.")

    def _apply_skills_for_task(self, task: str) -> list[str]:
        """根据任务自动选择技能，并把增量合并进本轮的 prompt 与工具表。"""
        # 调用点（_run_once）已用 `if self.skill_manager:` 守卫，此处的 None 分支不可达；
        # 取局部变量是为了让类型检查器完成收窄，同时避免管理器缺失时抛 AttributeError。
        manager = self.skill_manager
        if manager is None:
            return []

        # 恢复基础状态，避免上一次任务的技能残留
        self.system_prompt = self._base_system_prompt
        self.tools = dict(self._base_tools)

        from dm_agent.skills.runtime import prepare_skills

        activated: list[str] = []

        def record(kind: str, payload: dict[str, Any]) -> None:
            if self.trace_writer:
                self.trace_writer.record(kind, payload)

        self.system_prompt += prepare_skills(manager, task, self.tools, record, activated)
        return activated

    def _config_snapshot(self, *, max_steps: int | None = None) -> dict[str, Any]:
        """当前生效的配置快照：既用于落盘，也用于 resume 时的一致性比对。"""
        return agent_config_snapshot(
            temperature=self.temperature,
            model=getattr(self.client, "model", ""),
            enable_planning=self.enable_planning,
            enable_compression=self.enable_compression,
            enable_edit_guard=self.enable_edit_guard,
            enable_repo_map=self.enable_repo_map,
            enable_semantic_workspace=self.enable_semantic_workspace,
            max_observation_chars=self.max_observation_chars,
            context_token_budget=self.context_token_budget,
            max_steps=max_steps,
        )

    def _restore_from_checkpoint(
        self,
        resume_state: RunCheckpoint,
        steps: list[Step],
        metadata: dict[str, Any],
    ) -> int:
        """Restore checkpoint state and return the consumed step count."""
        resume_from = max(0, int(resume_state.step_count))
        self.conversation_history = [dict(message) for message in resume_state.conversation_history]
        steps.extend(steps_from_checkpoint(resume_state.steps))
        metadata.update(metadata_from_checkpoint(resume_state.metadata, resume_from=resume_from))

        if resume_state.plan_state is not None:
            self.task_plan.restore(resume_state.plan_state)
        elif resume_state.plan:
            raise ValueError("Legacy phase plans cannot be resumed as model-managed checklists.")
        if self.compressor:
            if resume_state.compressor_state is None:
                # 老 checkpoint 没有压缩器字段；checkpoint 是权威状态，不能沿用当前
                # agent 里上一段会话遗留的 memory / cadence / sticky 折叠。
                self.compressor.reset()
            else:
                self.compressor.restore_state(resume_state.compressor_state)
        self._restore_capability_state(resume_state.capability_state)
        warn_on_config_mismatch(resume_state.agent_config, self._config_snapshot())
        return resume_from

    def _save_checkpoint_snapshot(
        self,
        path: Path,
        *,
        task: str,
        step_count: int,
        steps: list[Step],
        metadata: dict[str, Any],
        limit: int,
    ) -> None:
        """把当前 run 的可恢复状态组装成快照并落盘。"""
        checkpoint = RunCheckpoint(
            task=task,
            step_count=step_count,
            conversation_history=[dict(message) for message in self.conversation_history],
            steps=[dict(step.__dict__) for step in steps],
            metadata=json_safe_metadata({**metadata, "run_id": self._run_context.run_id}),
            plan=self.task_plan.items,
            plan_state=self.task_plan.snapshot(),
            compressor_state=self.compressor.export_state() if self.compressor else None,
            capability_state=self._export_capability_state(),
            agent_config=self._config_snapshot(max_steps=limit),
            cwd=str(Path.cwd()),
        )
        self._persistence.save(path, checkpoint)

    def _capability_run_state(self) -> dict[str, Any]:
        """Return a narrow read-only state view for optional capabilities."""
        return {
            "plan": self.task_plan.items,
            "plan_scope": self.task_plan.scope,
            "plan_revision": self.task_plan.revision,
        }

    def _export_capability_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for capability in self.capabilities:
            key = getattr(capability, "checkpoint_key", "")
            export = getattr(capability, "export_state", None)
            if isinstance(key, str) and key and callable(export):
                state[key] = export()
        return state

    def _restore_capability_state(self, state: dict[str, Any]) -> None:
        for capability in self.capabilities:
            key = getattr(capability, "checkpoint_key", "")
            restore = getattr(capability, "restore_state", None)
            saved = state.get(key) if isinstance(key, str) else None
            if callable(restore) and isinstance(saved, dict):
                restore(saved)

    @staticmethod
    def _is_failure_observation(observation: str, *, action: str | None = None) -> bool:
        """委托到 ``core.observation``，让内核外的能力复用同一份失败判定。"""
        return is_failure_observation(observation, action=action)

    def reset_conversation(self) -> None:
        """重置对话历史

        清空所有对话历史记录，为新任务做准备。
        """
        self.conversation_history = []
        self.task_plan.reset()
        self._run_context.history_entry_ids.clear()
        if self.compressor:
            self.compressor.reset()

    def close(self) -> None:
        """Idempotently release resources owned by this Agent instance."""
        if self._closed:
            return
        self._closed = True
        resources = [self.compressor, self._repo_map, *reversed(self._owned_resources), self.client]
        seen: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                if self.trace_writer:
                    self.trace_writer.record(
                        "resource_close_error",
                        {"resource": type(resource).__name__, "message": str(exc)},
                    )

    def get_context_stats(self) -> dict[str, Any]:
        """Return current in-memory conversation and context-memory state."""
        return {
            "conversation_messages": len(self.conversation_history),
            "compression_enabled": self.enable_compression,
            "memory_items": self.compressor.memory_count if self.compressor else 0,
            "repo_map_enabled": self.enable_repo_map,
        }

    def get_conversation_history(self) -> list[dict[str, str]]:
        """获取对话历史

        Returns:
            conversation_history (List[Dict[str, str]]): 对话历史记录的副本
        """
        return self.conversation_history.copy()
