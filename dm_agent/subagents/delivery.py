"""Bounded output-contract corrections, installed only in Explore workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dm_agent.clients import LLMError
from dm_agent.core.events import BeforeFinishEvent, BeforeToolCallEvent, EventBus

from .schema import validate


class DeliveryExhausted(LLMError):
    """The initial delivery and both correction attempts were rejected."""


class DeliveryPolicy:
    """Share one correction budget across tool submissions and legacy finishes."""

    max_corrections = 2

    def __init__(self, schema: dict[str, Any], journal: Path) -> None:
        self.schema = schema
        self.journal = journal
        self.rejections: list[dict[str, Any]] = []
        self._rejected_steps: set[int] = set()

    @property
    def exhausted(self) -> bool:
        return len(self.rejections) > self.max_corrections

    def ensure_request_allowed(self) -> None:
        if self.exhausted:
            raise DeliveryExhausted("report correction budget exhausted")

    def reject(self, channel: str, candidate: Any, error: str, step: int) -> dict[str, Any]:
        if step not in self._rejected_steps:
            entry = {"channel": channel, "candidate": candidate, "error": error, "step": step}
            # Full candidates belong in the explicitly private attempt store, not
            # in the redacted public trace or the parent's lightweight result.
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.rejections.append(entry)
            self._rejected_steps.add(step)
        remaining = max(0, self.max_corrections + 1 - len(self.rejections))
        return {
            "block": True,
            "reason": (
                f"Report validation failed: {error}. "
                f"Correction attempts remaining: {remaining}. "
                "Resubmit using submit_exploration_result with a report matching output_schema."
                if remaining
                else f"Report validation failed: {error}. Report correction budget exhausted."
            ),
        }

    def before_tool(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name != "submit_exploration_result":
            return None
        try:
            if set(event.arguments) != {"report"}:
                raise ValueError("submit_exploration_result requires only report")
            validate(event.arguments["report"], self.schema)
        except (ValueError, TypeError) as exc:
            return self.reject("structured_tool", event.arguments, str(exc), event.step_number)
        return None

    def before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        if event.action == "submit_exploration_result":
            return None
        try:
            validate(json.loads(event.completion_text), self.schema)
        except (ValueError, TypeError) as exc:
            return self.reject(
                "legacy_finish_json", event.completion_text, str(exc), event.step_number
            )
        return None

    def observe_step(self, step_number: int, step: Any) -> None:
        # Non-object arguments are rejected by ToolInvoker before lifecycle
        # handlers run. Count and preserve these attempts as well.
        if step.action == "submit_exploration_result" and not isinstance(step.action_input, dict):
            self.reject(
                "structured_tool",
                step.action_input,
                "Tool arguments must be an object",
                step_number,
            )

    def install(self, bus: EventBus) -> None:
        bus.on("before_tool_call", self.before_tool, name="explore.delivery", kind="policy")
        bus.on("before_finish", self.before_finish, name="explore.delivery", kind="policy")
