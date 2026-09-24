"""Detect repeated no-progress tool calls and steer the next model request."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, BeforeLLMRequestEvent, RunStartEvent


@dataclass(frozen=True)
class _RepeatSignal:
    signature: str
    tool_name: str
    arguments: dict[str, Any]
    count: int


class RepeatCallRedirectCapability:
    """Steer, rather than block, consecutive identical calls with no new evidence."""

    def __init__(self, *, threshold: int = 4) -> None:
        if threshold < 2:
            raise ValueError("threshold must be at least 2.")
        self.threshold = threshold
        self._trace_writer: Any | None = None
        self._last_signature = ""
        self._last_result_fingerprint = ""
        self._last_workspace_version = ""
        self._consecutive_count = 0
        self._pending: _RepeatSignal | None = None
        self._last_injected: tuple[str, int] | None = None

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        context.event_bus.on("on_run_start", self._on_run_start, name="repeat_redirect.run_start")
        context.event_bus.on(
            "after_tool_result",
            self._after_tool_result,
            name="repeat_redirect.after_tool",
        )
        context.event_bus.on(
            "before_llm_request",
            self._before_llm_request,
            name="repeat_redirect.before_llm",
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._reset()
        event.metadata.update(
            {
                "repeat_call_redirect_enabled": True,
                "repeat_call_redirect_threshold": self.threshold,
                "repeat_call_redirect_detection_count": 0,
                "repeat_call_redirect_injection_count": 0,
            }
        )

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        signature = _call_signature(event.tool_name, event.arguments)
        result_fingerprint = _fingerprint(event.observation)
        workspace_version = (
            event.execution_fact.workspace_version if event.execution_fact else ""
        )
        unchanged = (
            signature == self._last_signature
            and result_fingerprint == self._last_result_fingerprint
            and workspace_version == self._last_workspace_version
        )
        if unchanged:
            self._consecutive_count += 1
        else:
            if self._consecutive_count:
                self._record(
                    "repeat_call_redirect_reset",
                    {
                        "step_number": event.step_number,
                        "previous_count": self._consecutive_count,
                        "reason": "tool, arguments, result, or workspace version changed",
                    },
                )
            self._consecutive_count = 1
            self._pending = None
            self._last_injected = None

        self._last_signature = signature
        self._last_result_fingerprint = result_fingerprint
        self._last_workspace_version = workspace_version
        if self._consecutive_count < self.threshold:
            return

        self._pending = _RepeatSignal(
            signature=signature,
            tool_name=event.tool_name,
            arguments=dict(event.arguments),
            count=self._consecutive_count,
        )
        event.metadata["repeat_call_redirect_detection_count"] = (
            int(event.metadata.get("repeat_call_redirect_detection_count", 0)) + 1
        )
        self._record(
            "repeat_call_redirect_detected",
            {
                "step_number": event.step_number,
                "tool_name": event.tool_name,
                "arguments": event.arguments,
                "signature": signature,
                "consecutive_count": self._consecutive_count,
                "workspace_version": workspace_version,
            },
        )

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        pending = self._pending
        if event.phase != "agent" or pending is None:
            return
        injection_key = (pending.signature, pending.count)
        if self._last_injected == injection_key:
            return
        event.messages.append(
            {
                "role": "system",
                "content": _render_redirect(pending),
            }
        )
        self._last_injected = injection_key
        event.metadata["repeat_call_redirect_injection_count"] = (
            int(event.metadata.get("repeat_call_redirect_injection_count", 0)) + 1
        )
        self._record(
            "repeat_call_redirect_injected",
            {
                "step_number": event.step_number,
                "tool_name": pending.tool_name,
                "arguments": pending.arguments,
                "signature": pending.signature,
                "consecutive_count": pending.count,
            },
        )

    def _reset(self) -> None:
        self._last_signature = ""
        self._last_result_fingerprint = ""
        self._last_workspace_version = ""
        self._consecutive_count = 0
        self._pending = None
        self._last_injected = None

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)


def _call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint(observation: str) -> str:
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def _render_redirect(signal: _RepeatSignal) -> str:
    arguments = json.dumps(signal.arguments, ensure_ascii=False, sort_keys=True)
    return (
        "[execution reminder]\n"
        f"You have called {signal.tool_name} with the same arguments {signal.count} "
        "times consecutively. The tool result and workspace version have not changed. "
        "Do not repeat this call unchanged. Use the evidence already available, choose "
        "a different tool or arguments, or proceed to an appropriate modification or verification.\n"
        f"tool: {signal.tool_name}\narguments: {arguments}"
    )


__all__ = ["RepeatCallRedirectCapability"]
