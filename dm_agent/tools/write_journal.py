"""Durable write intents and conservative recovery for local text writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def journal_directory(target: Path) -> Path:
    key = hashlib.sha256(str(target.resolve()).encode()).hexdigest()
    return Path(tempfile.gettempdir()) / "dm_agent_write_journal" / key


def recover_writes(target: Path) -> None:
    """Reconcile incomplete intents, without replaying any filesystem operation."""
    directory = journal_directory(target)
    if not directory.exists():
        return
    for journal in directory.glob("*.jsonl"):
        records = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Write journal requires inspection: {journal}") from exc
        if not records or records[-1].get("state") != "started":
            continue
        intent = records[0]
        current = fingerprint(target)
        if current == intent["after"]:
            state = "reconciled_completed"
        elif current == intent["before"]:
            state = "reconciled_not_applied"
        else:
            raise RuntimeError(f"Uncertain previous write; inspect {target} and {journal}")
        _append(journal, {"state": state, "call_id": intent["call_id"]})


def begin_write(target: Path, payload: bytes) -> Path:
    recover_writes(target)
    directory = journal_directory(target)
    directory.mkdir(parents=True, exist_ok=True)
    call_id = uuid.uuid4().hex
    journal = directory / f"{call_id}.jsonl"
    _append(
        journal,
        {
            "state": "started",
            "call_id": call_id,
            "path": str(target.resolve()),
            "before": fingerprint(target),
            "after": hashlib.sha256(payload).hexdigest(),
        },
    )
    return journal


def complete_write(journal: Path) -> None:
    _append(journal, {"state": "completed", "call_id": journal.stem})
