"""Structured display previews and conservative final tool-argument decoding.

Display prefixes are never executable. Repairs are limited to single-quoted
strings and trailing commas; missing closers and duplicate keys are rejected.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


class _Incomplete(ValueError):
    pass


class _Parser:
    def __init__(self, text: str, partial: bool) -> None:
        self.text = text
        self.pos = 0
        self.partial = partial

    def space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def peek(self) -> str:
        self.space()
        return self.text[self.pos : self.pos + 1]

    def string(self) -> tuple[str, bool]:
        quote = self.text[self.pos]
        self.pos += 1
        chars: list[str] = []
        escapes = {
            '"': '"',
            "'": "'",
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        while self.pos < len(self.text):
            char = self.text[self.pos]
            self.pos += 1
            if char == quote:
                return "".join(chars), True
            if ord(char) < 32:
                raise ValueError("Unescaped control character")
            if char != "\\":
                chars.append(char)
                continue
            if self.pos == len(self.text):
                break
            escape = self.text[self.pos]
            self.pos += 1
            if escape == "u":
                digits = self.text[self.pos : self.pos + 4]
                if not re.fullmatch(r"[0-9a-fA-F]{0,4}", digits):
                    raise ValueError("Invalid Unicode escape")
                if len(digits) < 4:
                    self.pos = len(self.text)
                    break
                chars.append(chr(int(digits, 16)))
                self.pos += 4
            elif escape in escapes:
                chars.append(escapes[escape])
            else:
                raise ValueError("Invalid string escape")
        if not self.partial:
            raise _Incomplete("Unterminated string")
        return "".join(chars), False

    def value(self, depth: int = 0) -> Any:
        if depth > 128:
            raise ValueError("Tool arguments exceed nesting limit")
        char = self.peek()
        if not char:
            raise _Incomplete("Missing value")
        if char in "\"'":
            return self.string()[0]
        if char in "{[":
            return self.container(char == "{", depth + 1)
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",]} \t\r\n":
            self.pos += 1
        token = self.text[start : self.pos]
        try:
            value = json.loads(token)
        except ValueError as exc:
            if self.partial and self.pos == len(self.text):
                raise _Incomplete("Incomplete scalar") from exc
            raise ValueError("Invalid scalar") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite number")
        return value

    def container(self, object_mode: bool, depth: int) -> Any:
        self.pos += 1
        result: Any = {} if object_mode else []
        closer = "}" if object_mode else "]"
        while True:
            char = self.peek()
            if char == closer:
                self.pos += 1
                return result
            if not char:
                if self.partial:
                    return result
                raise _Incomplete("Unterminated container")
            key = ""
            if object_mode:
                if char not in "\"'":
                    raise ValueError("Object key must be quoted")
                key, complete = self.string()
                if not complete and self.partial:
                    return result
                if key in result:
                    raise ValueError("Duplicate object key")
                if not self.peek() and self.partial:
                    return result
                if self.peek() != ":":
                    raise ValueError("Expected colon")
                self.pos += 1
            try:
                value = self.value(depth)
            except _Incomplete:
                if self.partial:
                    return result
                raise
            if object_mode:
                result[key] = value
            else:
                result.append(value)
            char = self.peek()
            if char == closer:
                self.pos += 1
                return result
            if not char and self.partial:
                return result
            if char != ",":
                raise ValueError("Expected separator or closing bracket")
            self.pos += 1


def preview_arguments(text: str) -> dict[str, Any]:
    """Best-effort nested display value; numeric/string prefixes may change."""
    parser = _Parser(text, partial=True)
    try:
        result = parser.value()
        if parser.peek():
            return {}
        return result if isinstance(result, dict) else {}
    except (ValueError, RecursionError):
        return {}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Non-finite number: {value}")


def parse_final_arguments(text: str) -> dict[str, Any]:
    """Parse complete arguments, optionally repairing quotes/trailing commas."""
    text = text if text.strip() else "{}"
    try:
        result = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise ValueError("Tool arguments exceed nesting limit") from exc
    except json.JSONDecodeError:
        parser = _Parser(text, partial=False)
        result = parser.value()
        if parser.peek():
            raise ValueError("Trailing data after tool arguments") from None
    if not isinstance(result, dict):
        raise ValueError("Tool arguments must be a JSON object")
    pending: list[tuple[Any, int]] = [(result, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 128:
            raise ValueError("Tool arguments exceed nesting limit")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    # Also rejects overflow such as 1e999, which the standard decoder accepts.
    json.dumps(result, allow_nan=False)
    return result


class PathFieldScanner:
    """Incremental structural scanner for completed top-level string paths.

    Independent of display throttling. Invalid prefixes disable early checks;
    final parsing remains authoritative. Only standard JSON strings are used.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.phase = "root"
        self.key = ""
        self.string_role = ""
        self.escaped = False
        self.token: list[str] = []
        self.path: str | None = None
        self.invalid = False

    def feed(self, delta: str) -> None:
        for char in delta:
            if self.invalid:
                return
            if self.string_role:
                if self.string_role != "nested":
                    self.token.append(char)
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    role = self.string_role
                    self.string_role = ""
                    if role != "nested":
                        try:
                            value = json.loads("".join(self.token))
                        except ValueError:
                            self.invalid = True
                            self.path = None
                            return
                        if role == "key":
                            self.key, self.phase = value, "colon"
                        else:
                            if self.key == "path":
                                self.path = value
                            self.phase = "separator"
                    self.token = []
                continue
            if char in " \t\r\n":
                continue
            if self.depth > 1:
                if char == '"':
                    self.string_role = "nested"
                elif char in "{[":
                    self.depth += 1
                elif char in "}]":
                    self.depth -= 1
                    if self.depth == 1:
                        self.phase = "separator"
                continue
            if self.phase == "root" and char == "{":
                self.depth, self.phase = 1, "key"
            elif self.phase in {"key", "value"} and char == '"':
                self.string_role = self.phase
                self.token = ['"']
            elif self.phase == "colon" and char == ":":
                self.phase = "value"
            elif self.phase == "value" and char in "{[":
                self.depth += 1
            elif self.phase == "value" and char in "-0123456789tfn":
                self.phase = "atom"
            elif self.phase in {"separator", "atom"} and char == ",":
                self.phase = "key"
            elif self.phase in {"key", "separator", "atom"} and char == "}":
                self.depth, self.phase = 0, "done"
            elif self.phase == "atom" and char in "0123456789.eE+-truefalsn":
                continue
            else:
                self.invalid = True
                self.path = None
