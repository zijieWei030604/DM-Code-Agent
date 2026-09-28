"""Regression coverage for the JSONL-to-SQLite query projection."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dm_agent.tracing.projection import TraceProjection


def event(
    number: int,
    kind: str,
    payload: dict[str, object],
    *,
    run_id: str = "full-run-id",
) -> dict[str, object]:
    return {
        "id": f"run-000{number}",
        "parent_id": "" if number == 1 else f"run-000{number - 1}",
        "timestamp": f"2026-09-27T00:00:0{number}+00:00",
        "run_id": run_id,
        "event": kind,
        "payload": payload,
    }


def write_events(path: Path, *events: dict[str, object], newline: bool = True) -> None:
    text = "\n".join(json.dumps(value) for value in events)
    path.write_text(text + ("\n" if newline else ""), encoding="utf-8")


def test_projection_indexes_runs_events_and_tool_statistics(tmp_path: Path) -> None:
    trace = tmp_path / "one.jsonl"
    write_events(
        trace,
        event(1, "run_start", {"task": "fix retry"}),
        event(2, "tool_call", {"step_number": 1, "action": "read_file", "failed": False}),
        event(3, "tool_call", {"step_number": 2, "action": "run_tests", "failed": True}),
        event(4, "run_end", {"status": "failed", "duration_seconds": 1.5}),
    )
    projection = TraceProjection(tmp_path / "index.sqlite3")

    result = projection.sync_file(trace)

    assert result.projected_events == 4
    assert projection.list_runs() == [
        {
            "run_id": "full-run-id",
            "trace_path": str(trace.resolve()),
            "task": "fix retry",
            "started_at": "2026-09-27T00:00:01+00:00",
            "ended_at": "2026-09-27T00:00:04+00:00",
            "status": "failed",
            "duration_seconds": 1.5,
        }
    ]
    assert projection.tool_failure_counts() == [
        {"tool_name": "run_tests", "calls": 1, "failures": 1},
        {"tool_name": "read_file", "calls": 1, "failures": 0},
    ]
    assert projection.sync_file(trace).projected_events == 0


def test_projection_waits_for_an_unterminated_tail(tmp_path: Path) -> None:
    trace = tmp_path / "tail.jsonl"
    first = event(1, "run_start", {"task": "pending"})
    write_events(trace, first, newline=False)
    projection = TraceProjection(tmp_path / "index.sqlite3")

    assert projection.sync_file(trace).pending_tail is True
    assert projection.list_runs() == []

    with trace.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert projection.sync_file(trace).projected_events == 1
    assert projection.list_runs()[0]["task"] == "pending"


def test_projection_rejects_a_malformed_complete_record_without_advancing(tmp_path: Path) -> None:
    trace = tmp_path / "broken.jsonl"
    trace.write_text(json.dumps(event(1, "run_start", {"task": "x"})) + "\n{bad}\n", encoding="utf-8")
    projection = TraceProjection(tmp_path / "index.sqlite3")

    result = projection.sync_file(trace)

    assert "malformed complete JSONL" in result.error
    assert projection.list_runs() == []
    connection = sqlite3.connect(tmp_path / "index.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM projection_offsets").fetchone()[0] == 0
    finally:
        connection.close()


def test_projection_detects_event_conflicts_on_rebuild(tmp_path: Path) -> None:
    trace = tmp_path / "conflict.jsonl"
    write_events(trace, event(1, "run_start", {"task": "original"}))
    database = tmp_path / "index.sqlite3"
    projection = TraceProjection(database)
    assert projection.sync_file(trace).projected_events == 1

    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM projection_offsets")
        connection.commit()
    finally:
        connection.close()
    write_events(trace, event(1, "run_start", {"task": "changed"}))

    assert "event conflict" in projection.sync_file(trace).error


def test_projection_detects_truncated_trace_files(tmp_path: Path) -> None:
    trace = tmp_path / "truncated.jsonl"
    write_events(trace, event(1, "run_start", {"task": "x"}), event(2, "run_end", {"status": "success"}))
    projection = TraceProjection(tmp_path / "index.sqlite3")
    assert projection.sync_file(trace).projected_events == 2

    write_events(trace, event(1, "run_start", {"task": "x"}))
    assert "shrank" in projection.sync_file(trace).error


def test_trace_index_routes_read_the_background_projection(
    make_client, sessions_dir: Path
) -> None:
    trace = sessions_dir / "indexed.jsonl"
    write_events(
        trace,
        event(1, "run_start", {"task": "indexed"}),
        event(2, "tool_call", {"step_number": 1, "action": "read_file", "failed": False}),
        event(3, "run_end", {"status": "success", "duration_seconds": 0.5}),
    )

    with make_client() as client:
        runs = client.get("/api/trace-index/runs").json()["runs"]
        tools = client.get("/api/trace-index/tools").json()["tools"]

    assert any(row["task"] == "indexed" for row in runs)
    assert {row["tool_name"] for row in tools} >= {"read_file"}
