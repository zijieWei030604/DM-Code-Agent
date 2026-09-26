"""Immutable SQLite records with branch-prefix visibility and local search."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .search_query import (
    escape_like,
    extract_search_terms,
    requires_like_fallback,
    sanitize_fts5_query,
    sanitize_like_query,
)


class LCMStore:
    """Compaction adds derived records, never overwrites its original sources."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        existing_records = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_records'"
        ).fetchone()
        existing_metadata = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_metadata'"
        ).fetchone()
        if existing_records and not existing_metadata:
            raise ValueError("Legacy LCM database is not compatible with schema lcm-2")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS lcm_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lcm_branches (
                id TEXT PRIMARY KEY,
                parent TEXT REFERENCES lcm_branches(id),
                parent_cutoff INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS lcm_records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                branch TEXT NOT NULL REFERENCES lcm_branches(id),
                event_key TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('message', 'summary', 'artifact')),
                body TEXT NOT NULL,
                metadata TEXT NOT NULL,
                UNIQUE(branch, event_key)
            );
            CREATE TABLE IF NOT EXISTS lcm_sources (
                summary_id TEXT NOT NULL REFERENCES lcm_records(id),
                source_id TEXT NOT NULL REFERENCES lcm_records(id),
                ordinal INTEGER NOT NULL,
                PRIMARY KEY(summary_id, ordinal),
                UNIQUE(summary_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS lcm_branch_seq ON lcm_records(branch, seq);
            CREATE TABLE IF NOT EXISTS lcm_branch_state (
                branch TEXT PRIMARY KEY REFERENCES lcm_branches(id),
                frontier TEXT NOT NULL
            );
            """)
        version = self.db.execute("SELECT value FROM lcm_metadata WHERE key='schema'").fetchone()
        if version is None:
            self.db.execute("INSERT INTO lcm_metadata VALUES ('schema', 'lcm-2')")
        elif version["value"] != "lcm-2":
            raise ValueError("Unsupported LCM database schema")
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS lcm_fts USING fts5(record_id UNINDEXED, body)"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError as exc:
            if "no such module" not in str(exc).lower():
                raise
            self.fts_enabled = False
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def create_branch(self, *, parent: str | None = None, cutoff: int | None = None) -> str:
        if (
            parent is not None
            and not self.db.execute("SELECT 1 FROM lcm_branches WHERE id=?", (parent,)).fetchone()
        ):
            raise ValueError("Unknown parent branch")
        head = self.head()
        boundary = head if cutoff is None else cutoff
        if boundary < 0 or boundary > head:
            raise ValueError("Invalid branch cutoff")
        branch = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO lcm_branches VALUES (?, ?, ?)", (branch, parent, boundary))
            self.db.execute("INSERT INTO lcm_branch_state VALUES (?, ?)", (branch, "[]"))
        return branch

    def head(self) -> int:
        return int(self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM lcm_records").fetchone()[0])

    def _visibility(self, branch: str) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        ceiling = self.head()
        visited = set()
        current: str | None = branch
        while current is not None:
            if current in visited:
                raise ValueError("Cyclic branch ancestry")
            visited.add(current)
            row = self.db.execute(
                "SELECT parent, parent_cutoff FROM lcm_branches WHERE id=?", (current,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown branch")
            clauses.append("(r.branch=? AND r.seq<=?)")
            args.extend((current, ceiling))
            ceiling = min(ceiling, row["parent_cutoff"])
            current = row["parent"]
        return "(" + " OR ".join(clauses) + ")", args

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def get(self, branch: str, record_id: str) -> dict[str, Any]:
        predicate, args = self._visibility(branch)
        row = self.db.execute(
            f"SELECT r.* FROM lcm_records r WHERE r.id=? AND {predicate}",
            [record_id, *args],
        ).fetchone()
        if row is None:
            raise ValueError("Record is not visible in this branch")
        return self._decode(row)

    def _insert(
        self, branch: str, event_key: str, kind: str, body: str, metadata: dict[str, Any]
    ) -> str:
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        existing = self.db.execute(
            "SELECT * FROM lcm_records WHERE branch=? AND event_key=?", (branch, event_key)
        ).fetchone()
        if existing is not None:
            if (existing["kind"], existing["body"], existing["metadata"]) != (kind, body, encoded):
                raise ValueError("An existing event cannot be overwritten")
            return str(existing["id"])
        record_id = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO lcm_records(id, branch, event_key, kind, body, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, branch, event_key, kind, body, encoded),
        )
        if self.fts_enabled:
            self.db.execute("INSERT INTO lcm_fts VALUES (?, ?)", (record_id, body))
        return record_id

    def append(
        self,
        branch: str,
        event_key: str,
        body: str,
        *,
        kind: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.db:
            return self._insert(branch, event_key, kind, body, metadata or {})

    def frontier(self, branch: str) -> list[str]:
        row = self.db.execute(
            "SELECT frontier FROM lcm_branch_state WHERE branch=?", (branch,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown branch state")
        frontier = json.loads(str(row["frontier"]))
        if not isinstance(frontier, list) or not all(isinstance(item, str) for item in frontier):
            raise ValueError("Invalid persisted frontier")
        for record_id in frontier:
            self.get(branch, record_id)
        return frontier

    def save_frontier(self, branch: str, frontier: Sequence[str]) -> None:
        values = list(frontier)
        if len(values) != len(set(values)):
            raise ValueError("Frontier cannot contain duplicate records")
        with self.db:
            for record_id in values:
                self.get(branch, record_id)
            self.db.execute(
                "UPDATE lcm_branch_state SET frontier=? WHERE branch=?",
                (json.dumps(values), branch),
            )

    def commit_summary(
        self,
        branch: str,
        body: str,
        sources: list[str],
        *,
        metadata: dict[str, Any],
        frontier: Sequence[str],
    ) -> tuple[str, list[str]]:
        """Atomically persist a derived node, its lineage, and active frontier."""
        if not body.strip() or not sources or len(sources) != len(set(sources)):
            raise ValueError("Summary must have text and distinct sources")
        original_frontier = list(frontier)
        if len(original_frontier) != len(set(original_frontier)):
            raise ValueError("Frontier cannot contain duplicate records")
        positions = [
            original_frontier.index(source) for source in sources if source in original_frontier
        ]
        if len(positions) != len(sources):
            raise ValueError("Summary sources must be active frontier records")
        with self.db:
            for source in sources:
                self.get(branch, source)
            node_id = self._insert(branch, uuid.uuid4().hex, "summary", body, metadata)
            self.db.executemany(
                "INSERT INTO lcm_sources VALUES (?, ?, ?)",
                [(node_id, source, index) for index, source in enumerate(sources)],
            )
            first = min(positions)
            source_set = set(sources)
            values = [
                *(
                    record_id
                    for record_id in original_frontier[:first]
                    if record_id not in source_set
                ),
                node_id,
                *(
                    record_id
                    for record_id in original_frontier[first:]
                    if record_id not in source_set
                ),
            ]
            for record_id in values:
                self.get(branch, record_id)
            self.db.execute(
                "UPDATE lcm_branch_state SET frontier=? WHERE branch=?",
                (json.dumps(values), branch),
            )
        return node_id, values

    def add_summary(
        self, branch: str, body: str, sources: list[str], *, metadata: dict[str, Any]
    ) -> str:
        if not body.strip() or not sources or len(sources) != len(set(sources)):
            raise ValueError("Summary must have text and distinct sources")
        with self.db:
            for source in sources:
                self.get(branch, source)
            node_id = self._insert(branch, uuid.uuid4().hex, "summary", body, metadata)
            self.db.executemany(
                "INSERT INTO lcm_sources VALUES (?, ?, ?)",
                [(node_id, source, index) for index, source in enumerate(sources)],
            )
        return node_id

    def sources(self, branch: str, node_id: str) -> list[str]:
        self.get(branch, node_id)
        ids = [
            str(row[0])
            for row in self.db.execute(
                "SELECT source_id FROM lcm_sources WHERE summary_id=? ORDER BY ordinal", (node_id,)
            )
        ]
        for source in ids:
            self.get(branch, source)
        return ids

    @staticmethod
    def record_type(row: dict[str, Any]) -> str:
        """Return the Runtime-facing type used for recall filtering."""
        if row["kind"] != "message":
            return str(row["kind"])
        return str(row["metadata"].get("kind", "message"))

    @staticmethod
    def match_context(body: str, query: str) -> dict[str, Any]:
        """Return a bounded preview centered on the earliest query-term hit."""
        terms = [query.strip(), *re.findall(r"\w+", query, re.UNICODE)]
        folded_body = body.casefold()
        matches = [
            (folded_body.find(term.casefold()), term)
            for term in dict.fromkeys(term for term in terms if term)
        ]
        position, term = min(
            ((position, term) for position, term in matches if position >= 0),
            default=(0, ""),
        )
        offset = max(0, position - 160)
        end = min(len(body), position + max(len(term), 1) + 240)
        preview = body[offset:end]
        if offset:
            preview = "..." + preview
        if end < len(body):
            preview += "..."
        return {
            "preview": preview,
            "offset": offset,
            "match_offset": position,
            "match_length": len(term),
        }

    def search(
        self,
        branch: str,
        query: str,
        *,
        limit: int = 10,
        record_types: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 30))
        query = query[:1000]
        safe_query = sanitize_fts5_query(query)
        requested_types = {str(item) for item in (record_types or ())}
        if any(not item for item in requested_types):
            raise ValueError("record_types must contain non-empty strings")
        predicate, args = self._visibility(branch)

        def collect(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
            # Filter before the result limit, including on SQLite without JSON1.
            results = []
            for raw in cursor:
                row = self._decode(raw)
                if requested_types and self.record_type(row) not in requested_types:
                    continue
                results.append(row)
                if len(results) == limit:
                    break
            return results

        if self.fts_enabled and not requires_like_fallback(query, safe_query):
            try:
                return collect(
                    self.db.execute(
                        "SELECT r.* FROM lcm_fts JOIN lcm_records r ON r.id=lcm_fts.record_id "
                        f"WHERE lcm_fts MATCH ? AND {predicate} ORDER BY r.seq DESC",
                        [safe_query, *args],
                    )
                )
            except sqlite3.OperationalError:
                # Upstream falls back on FTS errors, not on an empty hit list.
                pass
        terms = extract_search_terms(sanitize_like_query(query))
        if not terms:
            return []
        clauses = " OR ".join("r.body LIKE ? ESCAPE '\\'" for _ in terms)
        return collect(
            self.db.execute(
                f"SELECT r.* FROM lcm_records r WHERE {predicate} "
                f"AND ({clauses}) ORDER BY r.seq DESC",
                [*args, *(f"%{escape_like(term)}%" for term in terms)],
            )
        )
