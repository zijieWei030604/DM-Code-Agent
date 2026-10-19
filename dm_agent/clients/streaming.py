"""Provider-neutral primitives for one streamed model turn.

The agent deliberately consumes these objects only after the provider has
reported a terminal event.  Partial tool arguments are useful for display and
for early *rejection*, but are never an execution input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .argument_json import PathFieldScanner, parse_final_arguments, preview_arguments

StreamEventType = Literal[
    "stream_start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "stream_done",
    "stream_error",
]


@dataclass(frozen=True)
class StreamEvent:
    """One immutable observation from a streamed provider response."""

    type: StreamEventType
    request_id: str
    delta: str = ""
    tool_index: int | None = None
    tool_call_id: str = ""
    tool_name: str = ""
    arguments_preview: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    safety_arguments: dict[str, Any] = field(default_factory=dict)
    arguments_error: str = ""


@dataclass
class StreamToolCall:
    """A provider tool call assembled from one or more argument chunks."""

    index: int
    call_id: str = ""
    name: str = ""
    raw_arguments: str = ""
    arguments: dict[str, Any] | None = None
    completed: bool = False
    _parsed_length: int = 0
    _preview: dict[str, Any] = field(default_factory=dict)
    _path_scanner: PathFieldScanner = field(default_factory=PathFieldScanner)

    def append(self, delta: str) -> None:
        self.raw_arguments += delta
        self._path_scanner.feed(delta)

    def preview(self) -> dict[str, Any]:
        length = len(self.raw_arguments)
        growth = max(256, length // 32)
        if not self._parsed_length or length - self._parsed_length >= growth:
            self._preview = preview_arguments(self.raw_arguments)
            self._parsed_length = length
        return self._preview

    def safety_arguments(self) -> dict[str, Any]:
        path = self._path_scanner.path
        return {"path": path} if path is not None else {}

    def finalize(self) -> dict[str, Any]:
        value = parse_final_arguments(self.raw_arguments)
        self.arguments = value
        self.completed = True
        return value


def parse_partial_json_object(text: str) -> dict[str, Any]:
    """Display-only nested preview; never use it to authorize or reject a call."""
    return preview_arguments(text)
