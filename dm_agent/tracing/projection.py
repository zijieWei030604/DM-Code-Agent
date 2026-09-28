"""Incrementally project append-only JSONL traces into a queryable SQLite index.

The JSONL files remain the source of truth.  SQLite contains only a rebuildable
projection used for cross-run queries; a failed projection never blocks an agent
from appending its trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECTION_VERSION = 1
_BOUNDARY_BYTES = 4096


class ProjectionError(RuntimeError):
    """A trace cannot safely be projected until an operator resolves it."""


@dataclass(frozen=True)
class ProjectionResult:
    path: Path
    projected_events: int = 0
    skipped_events: int = 0
    pending_tail: bool = False
    error: str = ""


class TraceProjection:
    """Single-writer JSONL-to-SQLite projection with resumable byte offsets."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def sync_directory(self, sessions_dir: str | Path) -> list[ProjectionResult]:
        root = Path(sessions_dir)
        if not root.is_dir():
            return []
        results: list[ProjectionResult] = []
        for path in sorted(root.rglob("*.jsonl")):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
                results.append(self.sync_file(path))
        return results

    def sync_file(self, path: str | Path) -> ProjectionResult:
        trace_path = Path(path).resolve()
        if not trace_path.is_file():
            return ProjectionResult(trace_path, error="trace file does not exist")

        connection = self._connect()
        try:
            state = self._offset_state(connection, trace_path)
            stat = trace_path.stat()
            identity = _file_identity(stat)
            offset = int(state["byte_offset"]) if state else 0
            if state is not None:
                error = self._validate_file_state(trace_path, stat, state, identity)
                if error:
                    self._record_error(connection, trace_path, error)
                    return ProjectionResult(trace_path, error=error)

            with trace_path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read()
            last_newline = data.rfind(b"\n")
            if last_newline < 0:
                return ProjectionResult(trace_path, pending_tail=bool(data))

            consumed = data[: last_newline + 1]
            entries = _parse_complete_entries(consumed, offset)
            new_offset = offset + len(consumed)
            boundary = _boundary_hash(trace_path, new_offset)
            projected, skipped = self._project_batch(
                connection,
                trace_path,
                entries,
                new_offset,
                identity,
                boundary,
            )
            return ProjectionResult(
                trace_path,
                projected_events=projected,
                skipped_events=skipped,
                pending_tail=len(consumed) != len(data),
            )
        except (OSError, ProjectionError) as exc:
            self._record_error(connection, trace_path, str(exc))
            return ProjectionResult(trace_path, error=str(exc))
        finally:
            connection.close()

    def list_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT run_id, trace_path, task, started_at, ended_at, status, duration_seconds
                FROM runs ORDER BY COALESCE(ended_at, started_at) DESC LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def tool_failure_counts(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT tool_name, COUNT(*) AS calls, SUM(failed) AS failures
                FROM trace_events
                WHERE event_type = 'tool_call' AND tool_name != ''
                GROUP BY tool_name ORDER BY failures DESC, calls DESC, tool_name
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    trace_path TEXT NOT NULL,
                    task TEXT NOT NULL DEFAULT '',
                    started_at TEXT,
                    ended_at TEXT,
                    status TEXT,
                    duration_seconds REAL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL DEFAULT '',
                    trace_path TEXT NOT NULL,
                    timestamp TEXT,
                    event_type TEXT NOT NULL,
                    step_number INTEGER,
                    tool_name TEXT NOT NULL DEFAULT '',
                    failed INTEGER NOT NULL DEFAULT 0,
                    content_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_events_run_time
                    ON trace_events(run_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_trace_events_tool_failed
                    ON trace_events(tool_name, failed);
                CREATE TABLE IF NOT EXISTS projection_offsets (
                    trace_path TEXT PRIMARY KEY,
                    file_identity TEXT NOT NULL,
                    byte_offset INTEGER NOT NULL,
                    boundary_sha256 TEXT NOT NULL DEFAULT '',
                    projection_version INTEGER NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _offset_state(connection: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM projection_offsets WHERE trace_path = ?", (str(path),)
        ).fetchone()

    @staticmethod
    def _validate_file_state(
        path: Path, stat: os.stat_result, state: sqlite3.Row, identity: str
    ) -> str:
        offset = int(state["byte_offset"])
        if stat.st_size < offset:
            return "trace file shrank after projection; manual reconciliation required"
        if str(state["file_identity"]) != identity:
            return "trace file identity changed after projection; manual reconciliation required"
        expected = str(state["boundary_sha256"])
        if expected and _boundary_hash(path, offset) != expected:
            return "trace content changed before projection offset; manual reconciliation required"
        if int(state["projection_version"]) != PROJECTION_VERSION:
            return "projection version changed; rebuild the SQLite projection"
        return ""

    @staticmethod
    def _record_error(connection: sqlite3.Connection, path: Path, error: str) -> None:
        connection.execute(
            "UPDATE projection_offsets SET last_error = ?, updated_at = CURRENT_TIMESTAMP WHERE trace_path = ?",
            (error, str(path)),
        )
        connection.commit()

    def _project_batch(
        self,
        connection: sqlite3.Connection,
        path: Path,
        entries: list[dict[str, Any]],
        offset: int,
        identity: str,
        boundary: str,
    ) -> tuple[int, int]:
        projected = 0
        skipped = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            for entry in entries:
                inserted = self._project_event(connection, path, entry)
                projected += int(inserted)
                skipped += int(not inserted)
            connection.execute(
                """
                INSERT INTO projection_offsets(
                    trace_path, file_identity, byte_offset, boundary_sha256,
                    projection_version, last_error
                ) VALUES (?, ?, ?, ?, ?, '')
                ON CONFLICT(trace_path) DO UPDATE SET
                    file_identity = excluded.file_identity,
                    byte_offset = excluded.byte_offset,
                    boundary_sha256 = excluded.boundary_sha256,
                    projection_version = excluded.projection_version,
                    last_error = '',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (str(path), identity, offset, boundary, PROJECTION_VERSION),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return projected, skipped

    @staticmethod
    def _project_event(connection: sqlite3.Connection, path: Path, entry: dict[str, Any]) -> bool:
        run_id = str(entry.get("run_id") or "")
        event_id = str(entry.get("id") or "")
        event_type = str(entry.get("event") or "")
        if not run_id or not event_id or not event_type:
            raise ProjectionError("complete JSONL entry is missing run_id, id, or event")
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            raise ProjectionError(f"entry {event_id} payload must be an object")
        canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        existing = connection.execute(
            "SELECT content_sha256 FROM trace_events WHERE run_id = ? AND event_id = ?",
            (run_id, event_id),
        ).fetchone()
        if existing is not None:
            if str(existing["content_sha256"]) != digest:
                raise ProjectionError(f"event conflict for ({run_id}, {event_id})")
            return False

        action = str(payload.get("action") or "") if event_type == "tool_call" else ""
        step = payload.get("step_number")
        connection.execute(
            """
            INSERT INTO trace_events(
                run_id, event_id, parent_id, trace_path, timestamp, event_type,
                step_number, tool_name, failed, content_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event_id,
                str(entry.get("parent_id") or ""),
                str(path),
                str(entry.get("timestamp") or ""),
                event_type,
                int(step) if isinstance(step, int) else None,
                action,
                int(bool(payload.get("failed", False))),
                digest,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        if event_type == "run_start":
            connection.execute(
                """
                INSERT INTO runs(run_id, trace_path, task, started_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    trace_path = excluded.trace_path,
                    task = excluded.task,
                    started_at = excluded.started_at
                """,
                (run_id, str(path), str(payload.get("task") or ""), str(entry.get("timestamp") or "")),
            )
        elif event_type == "run_end":
            connection.execute(
                """
                INSERT INTO runs(run_id, trace_path, ended_at, status, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    trace_path = excluded.trace_path,
                    ended_at = excluded.ended_at,
                    status = excluded.status,
                    duration_seconds = excluded.duration_seconds
                """,
                (
                    run_id,
                    str(path),
                    str(entry.get("timestamp") or ""),
                    str(payload.get("status") or ""),
                    _as_float(payload.get("duration_seconds")),
                ),
            )
        return True


def _parse_complete_entries(data: bytes, start_offset: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    cursor = start_offset
    for raw_line in data.splitlines(keepends=True):
        text = raw_line.strip()
        line_start = cursor
        cursor += len(raw_line)
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectionError(f"malformed complete JSONL record at byte {line_start}") from exc
        if not isinstance(value, dict):
            raise ProjectionError(f"JSONL record at byte {line_start} is not an object")
        entries.append(value)
    return entries


def _file_identity(stat: os.stat_result) -> str:
    return f"{stat.st_dev}:{stat.st_ino}"


def _boundary_hash(path: Path, offset: int) -> str:
    if offset <= 0:
        return ""
    start = max(0, offset - _BOUNDARY_BYTES)
    with path.open("rb") as handle:
        handle.seek(start)
        return hashlib.sha256(handle.read(offset - start)).hexdigest()


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
