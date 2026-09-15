"""Local lexical retrieval primitives for compact agent memories.

The module deliberately keeps derived retrieval data in process memory.  A
checkpoint stores source memories only; token entries are rebuilt lazily after
restore so a tokenizer or domain dictionary upgrade cannot leave stale data.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]+")
_CHINESE_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")

# Keep multi-word concepts intact while jieba also contributes smaller search
# terms.  Code identifiers and paths continue through the rule-based branch.
_DOMAIN_TERMS = (
    "工具调用",
    "函数调用",
    "结构化输出",
    "上下文压缩",
    "语义工作区",
    "代码语义",
    "影响分析",
    "影响图",
    "依赖图",
    "证据图",
    "结构化编辑",
    "任务规划",
    "动态重规划",
    "断点恢复",
    "检查点",
    "会话记忆",
    "隐藏测试",
    "代码仓库",
)


class MemoryTokenizer:
    """Tokenize Chinese prose and code-oriented text without mixing their rules."""

    def __init__(self) -> None:
        self._segmenter = jieba.Tokenizer()
        for term in _DOMAIN_TERMS:
            self._segmenter.add_word(term)

    def tokenize(self, text: str) -> tuple[str, ...]:
        tokens: list[str] = []
        for match in _TOKEN_PATTERN.findall(str(text or "")):
            if _CHINESE_PATTERN.fullmatch(match):
                # Retain the exact phrase for precise matching, then add
                # search-mode segments for partial and reordered queries.
                tokens.append(match)
                tokens.extend(
                    token
                    for token in self._segmenter.lcut_for_search(match)
                    if token and not token.isspace()
                )
                continue
            for part in re.split(r"_+", match):
                tokens.extend(_split_camel_case(part))
        return tuple(token.lower() for token in tokens if token)


@dataclass(frozen=True)
class _CachedTokens:
    fingerprint: str
    tokens: tuple[str, ...]


class MemoryTokenCache:
    """Derived, non-persistent token cache keyed by memory id and content hash."""

    def __init__(self, tokenizer: MemoryTokenizer) -> None:
        self._tokenizer = tokenizer
        self._entries: dict[str, _CachedTokens] = {}

    def get(self, memory_id: str, text: str) -> tuple[str, ...]:
        fingerprint = _fingerprint(text)
        cached = self._entries.get(memory_id)
        if cached and cached.fingerprint == fingerprint:
            return cached.tokens
        tokens = self._tokenizer.tokenize(text)
        self._entries[memory_id] = _CachedTokens(fingerprint=fingerprint, tokens=tokens)
        return tokens

    def discard(self, memory_id: str) -> None:
        self._entries.pop(memory_id, None)

    def retain(self, memory_ids: set[str]) -> None:
        self._entries = {
            memory_id: entry
            for memory_id, entry in self._entries.items()
            if memory_id in memory_ids
        }

    def clear(self) -> None:
        self._entries.clear()


def bm25_scores(
    query_tokens: Sequence[str], documents: Sequence[Sequence[str]]
) -> list[float]:
    """Return normalized BM25 scores, with a deterministic empty-query result."""
    if not query_tokens or not documents:
        return [0.0] * len(documents)
    raw_scores = [float(score) for score in BM25Okapi(list(documents)).get_scores(query_tokens)]
    highest = max(raw_scores, default=0.0)
    if highest <= 0.0:
        return [0.0] * len(raw_scores)
    return [max(0.0, score) / highest for score in raw_scores]


def _fingerprint(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _split_camel_case(token: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).split()
    if len(parts) == 1:
        return parts
    return [*parts, token]
