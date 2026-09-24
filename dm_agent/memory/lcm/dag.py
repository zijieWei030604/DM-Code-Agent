"""Bounded views of summary provenance, not proof of current workspace state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .store import LCMStore


class SummaryDAG:
    def __init__(
        self,
        store: LCMStore,
        branch: str,
        resolve_artifact: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.store = store
        self.branch = branch
        self.resolve_artifact = resolve_artifact

    def describe(self, record_id: str) -> dict[str, Any]:
        row = self.store.get(self.branch, record_id)
        sources = self.store.sources(self.branch, record_id)
        return {
            "id": row["id"],
            "kind": row["kind"],
            "characters": len(row["body"]),
            "preview": row["body"][:400],
            "depth": row["metadata"].get("depth"),
            "source_count": len(sources),
            "sources": sources[:50],
            "next_source_offset": 50 if len(sources) > 50 else None,
        }

    def expand(
        self, record_id: str, *, offset: int = 0, limit: int = 4000, source_offset: int = 0
    ) -> dict[str, Any]:
        if offset < 0 or source_offset < 0:
            raise ValueError("offsets must be nonnegative")
        row = self.store.get(self.branch, record_id)
        size = max(1, min(limit, 12000))
        content = str(row["body"])
        reference = row["metadata"].get("trace_reference")
        if reference:
            if self.resolve_artifact is None:
                raise ValueError("Trace artifact resolver is unavailable")
            content = self.resolve_artifact(reference)
        end = min(len(content), offset + size)
        sources = self.store.sources(self.branch, record_id)
        source_end = source_offset + 50
        return {
            "id": record_id,
            "kind": row["kind"],
            "content": content[offset:end],
            "next_offset": end if end < len(content) else None,
            "sources": sources[source_offset:source_end],
            "next_source_offset": source_end if source_end < len(sources) else None,
        }
