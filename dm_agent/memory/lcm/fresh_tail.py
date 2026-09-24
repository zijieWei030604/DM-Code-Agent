"""Find safe boundaries using explicit tool call/result IDs, never text guesses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def safe_boundaries(messages: Sequence[dict[str, Any]]) -> list[int]:
    """Only split where no tool request remains unmatched.

    Accept native tool_calls/tool_call_id and normalized call_ids/result_ids.
    Runtime adapters must assign IDs for legacy JSON action responses.
    """
    pending: set[str] = set()
    seen: set[str] = set()
    boundaries = [0]
    for index, message in enumerate(messages):
        calls = list(message.get("call_ids", ()))
        calls.extend(str(call["id"]) for call in message.get("tool_calls", ()) if call.get("id"))
        for call in calls:
            if not isinstance(call, str) or not call or call in seen:
                raise ValueError("Tool call IDs must be nonempty and unique")
            pending.add(call)
            seen.add(call)
        results = list(message.get("result_ids", ()))
        if message.get("tool_call_id"):
            results.append(str(message["tool_call_id"]))
        if message.get("role") == "tool" and not results:
            raise ValueError("Tool result has no call ID")
        for result in results:
            if result not in pending:
                raise ValueError("Tool result does not match an open call")
            pending.remove(result)
        if not pending:
            boundaries.append(index + 1)
    return boundaries


def fresh_tail_start(messages: Sequence[dict[str, Any]], *, keep_recent: int = 8) -> int:
    target = max(0, len(messages) - max(1, keep_recent))
    return max(boundary for boundary in safe_boundaries(messages) if boundary <= target)
