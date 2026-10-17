"""Ordered model-turn calls, persisted at the same boundaries as ordinary steps.

Every call still goes through ToolInvoker. This module owns only the pending
queue and its audit identity; it never executes tools or bypasses policies.
"""

from __future__ import annotations

import uuid
from typing import Any

from .response_parser import normalize_action

MAX_BATCH_CALLS = 32


def prepare_batch(data: dict[str, Any], completion_tools: set[str]) -> list[dict[str, Any]]:
    """Validate the whole envelope before allowing its first side effect."""
    calls = data.get("tool_calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_BATCH_CALLS:
        raise ValueError(f"tool_calls must contain 1..{MAX_BATCH_CALLS} calls")
    batch_id = uuid.uuid4().hex
    pending = []
    ids: set[str] = set()
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be an object")
        action = call.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Each tool call requires an action")
        action = normalize_action(action.strip())
        argument_error = str(call.get("argument_error", ""))
        if not isinstance(call.get("action_input"), dict):
            argument_error = "Each tool call requires object action_input"
        if len(calls) > 1 and action in {"finish", "task_complete", *completion_tools}:
            raise ValueError("Completion must be submitted alone after observing tool results")
        provider_id = call.get("call_id", "")
        call_id = provider_id
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            call_id = f"call_{batch_id}_{index}"
        ids.add(call_id)
        pending.append(
            {
                "batch_id": batch_id,
                "call_id": call_id,
                **(
                    {"provider_call_id": provider_id}
                    if provider_id and provider_id != call_id
                    else {}
                ),
                "index": index,
                "action": action,
                "action_input": (
                    call["action_input"] if isinstance(call.get("action_input"), dict) else {}
                ),
                **({"argument_error": argument_error} if argument_error else {}),
                **(
                    {"raw_arguments": call["raw_arguments"]}
                    if isinstance(call.get("raw_arguments"), str)
                    else {}
                ),
                "thought": str(call.get("thought", data.get("thought", ""))),
            }
        )
    return pending


def validate_pending(pending: list[dict[str, Any]]) -> None:
    """Reject damaged checkpoint queues before consuming any restored call."""
    if len(pending) > MAX_BATCH_CALLS:
        raise ValueError("Checkpoint tool queue exceeds the batch limit")
    seen: set[str] = set()
    last_index = -1
    batch_id = ""
    for call in pending:
        if not isinstance(call, dict):
            raise ValueError("Invalid checkpoint tool queue entry")
        if not all(
            isinstance(call.get(key), str) for key in ("batch_id", "call_id", "action", "thought")
        ):
            raise ValueError("Invalid checkpoint tool identity")
        index = call.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index <= last_index:
            raise ValueError("Invalid checkpoint tool order")
        if not isinstance(call.get("action_input"), dict) or not call["action"].strip():
            raise ValueError("Invalid checkpoint tool arguments")
        if not call["batch_id"] or not call["call_id"] or call["call_id"] in seen:
            raise ValueError("Invalid checkpoint tool identity")
        if batch_id and call["batch_id"] != batch_id:
            raise ValueError("Checkpoint queue mixes different batches")
        seen.add(call["call_id"])
        batch_id, last_index = call["batch_id"], index
