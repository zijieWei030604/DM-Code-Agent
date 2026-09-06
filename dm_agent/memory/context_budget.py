"""Context budget helpers: token estimation, observation truncation, file ledger.

These utilities keep long tool outputs from silently flooding the conversation
history. They are deliberately dependency-free and deterministic:

- ``estimate_tokens`` uses the project-wide ``chars / 4`` heuristic (single
  source of truth; evals and benchmarks import it instead of re-deriving it).
- ``truncate_observation`` bounds one tool observation while keeping the head
  (imports, signatures) and tail (tracebacks live at the end) and inserting an
  explicit marker that tells the model how to page through the omitted range.
- ``FileLedger`` records which files were read/written at which step so the
  agent can require a fresh read before ``edit_file`` operates on line numbers.

Wording constraint: marker and guard texts produced here must never contain
failure markers ("error", "failed", "失败", "错误", "不存在", ...) because
``ReactAgent._is_failure_observation`` and the compressor's ``_ERROR_MARKERS``
would otherwise misclassify a healthy observation as a failure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Keep at least this many head characters when the cap allows it, so that
# failure-signature prefixes (observation[:160]) and file headers stay intact.
MIN_HEAD_CHARS = 500

# Share of the cap given to the head when the cap is large enough.
_HEAD_SHARE = 0.6

# Suggested page size (in lines) for read_file continuation hints.
_PAGE_LINES = 200


def estimate_tokens_from_chars(char_count: int) -> int:
    """Project-wide chars->tokens heuristic (~4 chars per token)."""
    return max(0, (int(char_count) + 3) // 4)


def estimate_tokens(text: str) -> int:
    return estimate_tokens_from_chars(len(str(text or "")))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Estimate the token footprint of a message list (content only)."""
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of bounding one observation."""

    text: str
    truncated: bool
    original_chars: int
    kept_chars: int
    original_lines: int
    head_end_line: int  # 1-based line (within the original text) where the head stops


@dataclass(frozen=True)
class ContextBudgetBreakdown:
    """Estimated token allocation for one complete provider request."""

    total_budget: int
    system_tokens: int
    history_tokens: int
    tool_schema_tokens: int
    output_reserve: int
    safety_margin: int
    available_history_tokens: int

    @property
    def estimated_total_tokens(self) -> int:
        return (
            self.system_tokens
            + self.history_tokens
            + self.tool_schema_tokens
            + self.output_reserve
            + self.safety_margin
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "total_budget": self.total_budget,
            "system_tokens": self.system_tokens,
            "history_tokens": self.history_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "output_reserve": self.output_reserve,
            "safety_margin": self.safety_margin,
            "available_history_tokens": self.available_history_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
        }


def estimate_json_tokens(value: Any) -> int:
    """Estimate a JSON request fragment using the same deterministic heuristic."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return estimate_tokens(text)


def build_context_budget(
    *,
    total_budget: int,
    system_prompt: str,
    history: Iterable[dict[str, Any]],
    tool_definitions: Iterable[dict[str, Any]] = (),
    output_reserve: int = 2048,
    safety_margin: int = 512,
) -> ContextBudgetBreakdown:
    """Allocate a total request budget and expose every estimated component."""
    total = max(0, int(total_budget))
    system_tokens = estimate_tokens(system_prompt)
    history_tokens = estimate_messages_tokens(history)
    tool_schema_tokens = estimate_json_tokens(list(tool_definitions))
    reserved = system_tokens + tool_schema_tokens + max(0, output_reserve) + max(0, safety_margin)
    available = max(total - reserved, 0) if total else 0
    return ContextBudgetBreakdown(
        total_budget=total,
        system_tokens=system_tokens,
        history_tokens=history_tokens,
        tool_schema_tokens=tool_schema_tokens,
        output_reserve=max(0, output_reserve),
        safety_margin=max(0, safety_margin),
        available_history_tokens=available,
    )


def _head_chars_for(max_chars: int) -> int:
    return max(min(max_chars, MIN_HEAD_CHARS), int(max_chars * _HEAD_SHARE))


def _read_file_hint(action_input: Any, head_end_line: int) -> str:
    """Build a concrete next-page suggestion for read_file observations."""
    if not isinstance(action_input, dict):
        return ""
    path = action_input.get("path")
    if not isinstance(path, str) or not path:
        return ""
    base_line = action_input.get("line_start")
    base = base_line if isinstance(base_line, int) and base_line >= 1 else 1
    next_start = base - 1 + head_end_line
    suggestion = {"path": path, "line_start": next_start, "line_end": next_start + _PAGE_LINES}
    return f" next page: read_file {json.dumps(suggestion, ensure_ascii=False)}"


def truncate_observation(
    text: str,
    *,
    max_chars: int,
    action: str = "",
    action_input: Any = None,
) -> TruncationResult:
    """Bound ``text`` to roughly ``max_chars`` visible characters.

    Keeps the head and tail and inserts an explicit ``[truncated: ...]`` marker
    with paging guidance. ``max_chars <= 0`` disables truncation.
    """
    text = str(text or "")
    original_chars = len(text)
    original_lines = text.count("\n") + 1 if text else 0
    if max_chars <= 0 or original_chars <= max_chars:
        return TruncationResult(
            text=text,
            truncated=False,
            original_chars=original_chars,
            kept_chars=original_chars,
            original_lines=original_lines,
            head_end_line=0,
        )

    head_chars = min(original_chars, _head_chars_for(max_chars))
    tail_chars = max(0, max_chars - head_chars)
    head = text[:head_chars]
    tail = text[original_chars - tail_chars :] if tail_chars else ""
    head_end_line = head.count("\n") + 1

    if action == "read_file":
        guidance = (
            "use read_file with line_start/line_end"
            + _read_file_hint(action_input, head_end_line)
            + " or search_in_file to view the omitted range"
        )
    else:
        guidance = (
            "narrow the command or use read_file with line_start/line_end "
            "on specific files to view the omitted range"
        )
    marker = (
        f"\n[truncated: showing first {head_chars} and last {tail_chars} of "
        f"{original_chars} chars ({original_lines} lines total); {guidance}]\n"
    )

    return TruncationResult(
        text=head + marker + tail,
        truncated=True,
        original_chars=original_chars,
        kept_chars=head_chars + tail_chars,
        original_lines=original_lines,
        head_end_line=head_end_line,
    )


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


class FileLedger:
    """Track per-run read/write steps for read-before-edit enforcement.

    The ledger is derived purely from the action history: it never touches the
    filesystem, so it behaves identically on all platforms and in replay.
    """

    def __init__(self) -> None:
        self._last_read: dict[str, int] = {}
        self._last_write: dict[str, int] = {}

    def reset(self) -> None:
        self._last_read.clear()
        self._last_write.clear()

    def note_read(self, path: str, step: int) -> None:
        if path:
            self._last_read[_normalize_path(path)] = step

    def note_write(self, path: str, step: int) -> None:
        if path:
            self._last_write[_normalize_path(path)] = step

    def check_edit(self, path: str) -> str | None:
        """Return a rejection reason for editing ``path``, or None if allowed."""
        key = _normalize_path(path)
        read_step = self._last_read.get(key)
        if read_step is None:
            return "never_read"
        write_step = self._last_write.get(key)
        if write_step is not None and write_step > read_step:
            return "stale_read"
        return None
