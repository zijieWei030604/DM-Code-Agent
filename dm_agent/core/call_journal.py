"""Checkpoint-side call intents: unresolved side effects block blind resume."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

_READ_ONLY = {
    "read_file",
    "list_directory",
    "search_in_file",
    "parse_ast",
    "inspect_python_symbol",
    "get_function_signature",
    "find_dependencies",
    "get_code_metrics",
    "search_symbol",
    "dependency_graph",
    "task_complete",
}


class CallJournal:
    def __init__(self, checkpoint: Path) -> None:
        self.path = checkpoint.with_name(checkpoint.name + ".calls.jsonl")

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def begin(self, action: str, step: int, run_id: str, *, read_only: bool = False) -> str:
        call_id = uuid.uuid4().hex
        self.append(
            {
                "state": "started",
                "call_id": call_id,
                "action": action,
                "step": step,
                "run_id": run_id,
                "read_only": read_only,
            }
        )
        return call_id

    def finish(self, call_id: str) -> None:
        self.append({"state": "completed", "call_id": call_id})

    def check_resume(self, step: int, run_id: str) -> None:
        if not self.path.exists():
            return
        calls: dict[str, dict[str, Any]] = {}
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                calls.setdefault(record["call_id"], {}).update(record)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Inspect incomplete call journal before resume: {self.path}") from exc
        for record in calls.values():
            if (
                record.get("run_id") != run_id
                or record.get("action") in _READ_ONLY
                or record.get("read_only")
            ):
                continue
            if record.get("state") == "started" or int(record.get("step", 0)) > step:
                raise ValueError(
                    f"Resume requires side-effect reconciliation: {record['action']} "
                    f"call {record['call_id']} is not covered by this checkpoint. "
                    f"Inspect {self.path}; do not blindly replay the operation."
                )
