"""Normalize every native call without dropping malformed siblings."""

from __future__ import annotations

import json
import uuid
from typing import Any


def serialize_tool_calls(calls: list[dict[str, Any]]) -> str:
    actions = []
    for call in calls:
        call = call if isinstance(call, dict) else {}
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        raw = function.get("arguments", "{}")
        error = str(call.get("argument_error", ""))
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Tool call requires a function name")
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
        except (TypeError, ValueError) as exc:
            arguments = {}
            error = str(exc)
        action = {
            "thought": "",
            "action": name if isinstance(name, str) and name else "invalid_tool",
            "action_input": arguments,
            "call_id": call.get("id") or uuid.uuid4().hex,
        }
        if error:
            action["argument_error"] = error
            action["raw_arguments"] = raw if isinstance(raw, str) else json.dumps(raw)
        actions.append(action)
    return json.dumps(
        actions[0] if len(actions) == 1 else {"tool_calls": actions}, ensure_ascii=False
    )
