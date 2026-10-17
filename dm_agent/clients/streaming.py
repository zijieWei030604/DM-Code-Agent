"""Provider-neutral primitives for one streamed model turn.

The agent deliberately consumes these objects only after the provider has
reported a terminal event.  Partial tool arguments are useful for display and
for early *rejection*, but are never an execution input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

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


@dataclass
class StreamToolCall:
    """A provider tool call assembled from one or more argument chunks."""

    index: int
    call_id: str = ""
    name: str = ""
    raw_arguments: str = ""
    arguments: dict[str, Any] | None = None
    completed: bool = False

    def append(self, delta: str) -> None:
        self.raw_arguments += delta

    def preview(self) -> dict[str, Any]:
        return parse_partial_json_object(self.raw_arguments)

    def finalize(self) -> dict[str, Any]:
        value = json.loads(self.raw_arguments or "{}")
        if not isinstance(value, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        self.arguments = value
        self.completed = True
        return value


def parse_partial_json_object(text: str) -> dict[str, Any]:
    """Return only completed top-level scalar values from an object prefix.

    It intentionally does not guess unfinished strings or nested structures.
    The result can therefore be used to reject a completed dangerous path while
    remaining unsuitable as a substitute for final JSON schema validation.
    """

    preview: dict[str, Any] = {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"\s*:\s*', text):
        try:
            key = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
        start = match.end()
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
            preview[key] = value
    return preview
