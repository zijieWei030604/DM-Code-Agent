"""Project structured call/result history onto provider wire protocols."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_FIELDS = (
    "tool_calls",
    "tool_call_id",
    "tool_name",
    "status",
    "is_error",
    "synthetic",
    "error_code",
    "exit_code",
    "changed_files",
    "check_scope",
    "result_metadata",
)


def protocol_fields(message: dict[str, Any]) -> dict[str, Any]:
    return {key: message[key] for key in PROTOCOL_FIELDS if key in message}


def chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    pending: set[str] = set()
    for message in messages:
        role, content = message["role"], message.get("content", "")
        calls = message.get("tool_calls")
        if calls:
            if pending:
                raise ValueError("An assistant tool turn still has unpaired results")
            ids = [call["id"] for call in calls]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate tool call ids")
            pending.update(ids)
            result.append({"role": "assistant", "content": None, "tool_calls": calls})
        elif role == "tool":
            identity = message.get("tool_call_id")
            if identity not in pending:
                raise ValueError("Tool result has no matching assistant call")
            pending.remove(identity)
            if message.get("is_error"):
                content = json.dumps(
                    {
                        "status": message.get("status", "failed"),
                        "output": content,
                        "synthetic": message.get("synthetic", False),
                    },
                    ensure_ascii=False,
                )
            result.append({"role": "tool", "tool_call_id": identity, "content": content})
        else:
            if pending:
                raise ValueError("Message inserted before all tool results were paired")
            result.append({"role": role, "content": content})
    if pending:
        raise ValueError("Cannot request a model with unpaired tool calls")
    return result


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in chat_messages(messages):
        if message.get("tool_calls"):
            result.extend(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in message["tool_calls"]
            )
        elif message["role"] == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            result.append(message)
    return result


def text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Prompt-only providers receive explicit provenance, never an orphan tool role."""
    result: list[dict[str, Any]] = []
    for message in messages:
        role, content = str(message["role"]), str(message.get("content", ""))
        if message.get("tool_calls"):
            content = json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=False)
        if role == "tool":
            content = f"Tool result {message.get('tool_call_id')}: {content}"
            role = "user"
        result.append({"role": role, "content": content})
    return result
