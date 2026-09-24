"""Build the active request from LCM summaries and complete recent decisions."""

from __future__ import annotations

from typing import Any

from dm_agent.memory.context_budget import (
    ContextBudgetBreakdown,
    build_context_budget,
    estimate_messages_tokens,
)
from dm_agent.memory.lcm.compaction import ContextOverflow

from .lcm_memory import LCMMemory
from .run_state import RunContext


class LCMContextWindow:
    def __init__(
        self,
        *,
        compressor: LCMMemory | None,
        enabled: bool,
        trace_writer: Any = None,
        output_token_reserve: int = 2048,
        safety_margin_tokens: int = 512,
    ) -> None:
        self.compressor = compressor
        self.enabled = enabled
        self.trace_writer = trace_writer
        self.output_token_reserve = output_token_reserve
        self.safety_margin_tokens = safety_margin_tokens
        self.last_budget_breakdown: ContextBudgetBreakdown | None = None
        self._recorded_summary = ""

    def reset(self) -> None:
        self.last_budget_breakdown = None
        self._recorded_summary = ""

    def build_messages(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        *,
        context: RunContext,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        memory = self.compressor
        effective = history
        if self.enabled and memory:
            memory.adopt(history)
            assert memory.compactor is not None
            effective = memory.compactor.render(memory.frontier)
        self.last_budget_breakdown = build_context_budget(
            total_budget=memory.token_budget if memory else 0,
            system_prompt=system_prompt,
            history=effective,
            tool_definitions=tool_definitions or (),
            output_reserve=self.output_token_reserve,
            safety_margin=self.safety_margin_tokens,
        )
        if not (self.enabled and memory):
            return [{"role": "system", "content": system_prompt}, *history]
        engine = memory.compactor
        assert engine is not None
        budget = self.last_budget_breakdown.available_history_tokens
        # Zero total budget explicitly disables automatic compaction.
        if memory.token_budget > 0:
            if budget <= 0:
                raise ContextOverflow("System prompt and tools exhaust the context budget")
            before_calls = len(engine.calls)

            def committed(frontier: list[str], node: str) -> None:
                context.metadata["memory_compression_count"] = (
                    int(context.metadata.get("memory_compression_count", 0)) + 1
                )
                context.metadata["budget_compression_count"] = (
                    int(context.metadata.get("budget_compression_count", 0)) + 1
                )
                self._record_compaction(memory, history, context, phase="accepted")

            try:
                engine.compact(memory.frontier, token_budget=budget, on_commit=committed)
            finally:
                for call in engine.calls[before_calls:]:
                    if self.trace_writer:
                        self.trace_writer.record(
                            "lcm_summary_call",
                            {"model": getattr(memory.client, "model", ""), **call},
                        )
                context.metadata["lcm_summary_calls"] = len(engine.calls)
                context.metadata["llm_summary_count"] = len(engine.calls)
                context.metadata["llm_summary_error_count"] = sum(
                    "error_type" in call for call in engine.calls
                )
                context.metadata["lcm_summary_input_tokens_estimated"] = sum(
                    call["estimated_input_tokens"] for call in engine.calls
                )
                context.metadata["lcm_summary_output_tokens_estimated"] = sum(
                    call.get("estimated_output_tokens", 0) for call in engine.calls
                )
        context.metadata["memory_items"] = memory.memory_count
        context.metadata["context_backend"] = "lcm"
        if (
            memory.frontier
            and memory.store.get(memory.branch, memory.frontier[0])["kind"] == "summary"
            and self._recorded_summary != memory.frontier[0]
        ):
            self._record_compaction(memory, history, context, phase="sticky_reuse")
        effective = engine.render(memory.frontier)
        self.last_budget_breakdown = build_context_budget(
            total_budget=memory.token_budget,
            system_prompt=system_prompt,
            history=effective,
            tool_definitions=tool_definitions or (),
            output_reserve=self.output_token_reserve,
            safety_margin=self.safety_margin_tokens,
        )
        return [{"role": "system", "content": system_prompt}, *effective]

    def _record_compaction(
        self, memory: LCMMemory, history: list[dict[str, str]], context: RunContext, *, phase: str
    ) -> None:
        engine = memory.compactor
        assert engine is not None
        node = memory.frontier[0]
        after = engine.tokens(memory.frontier)
        node_metadata = memory.store.get(memory.branch, node)["metadata"]
        before = (
            after + int(node_metadata["source_tokens"]) - engine.tokens([node])
            if phase == "accepted"
            else after
        )
        raw_ids = set(memory.history_ids)
        first = next(
            (memory.history_ids.index(item) for item in memory.frontier if item in raw_ids),
            len(history),
        )
        if self.trace_writer:
            self.trace_writer.record_compaction(
                {
                    "backend": "lcm",
                    "step_number": context.step_number,
                    "phase": phase,
                    "summary_id": node,
                    "summary": engine.render([node])[0]["content"],
                    "folded_message_count": first,
                    "kept_message_count": len(memory.frontier),
                    "original_message_count": len(history),
                    "estimated_tokens_before": before,
                    "estimated_tokens_after": after,
                    "estimated_original_history_tokens": estimate_messages_tokens(history),
                },
                first_kept_index=first,
                folded_indexes=range(first),
            )
        self._recorded_summary = node
