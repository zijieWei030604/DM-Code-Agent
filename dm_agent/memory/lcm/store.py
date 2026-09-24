"""Immutable SQLite records with branch-prefix visibility and local search."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any


class LCMStore:
    """Compaction adds derived records, never overwrites its original sources."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
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
            """)
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

    def search(self, branch: str, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 30))
        terms = re.findall(r"\w+", query[:1000], re.UNICODE)[:20]
        if not terms:
            return []
        predicate, args = self._visibility(branch)
        rows: list[sqlite3.Row] = []
        if self.fts_enabled:
            expression = " OR ".join('"' + term + '"' for term in terms)
            rows = self.db.execute(
                "SELECT r.* FROM lcm_fts JOIN lcm_records r ON r.id=lcm_fts.record_id "
                f"WHERE lcm_fts MATCH ? AND {predicate} ORDER BY rank LIMIT ?",
                [expression, *args, limit],
            ).fetchall()
        if not rows:
            escaped = query[:1000].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self.db.execute(
                f"SELECT r.* FROM lcm_records r WHERE {predicate} "
                "AND r.body LIKE ? ESCAPE '\\' ORDER BY r.seq DESC LIMIT ?",
                [*args, f"%{escaped}%", limit],
            ).fetchall()
        return [self._decode(row) for row in rows]
