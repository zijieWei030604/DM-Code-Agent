"""Bounded foreground batches with append-only attempt records.

Each worker owns an independent in-process session. Foreground batches drain all
workers before returning; credentials stay in memory, outside task records.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import uuid
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

from dm_agent.tools.base import Tool, ToolResult

from .control import ExplorationStopped, RunControl
from .schema import DEFAULT_REPORT, check_schema
from .worker import execute


class TaskManager:
    """One manager owns a session; batches serialize, workers run in parallel."""

    def __init__(
        self,
        root: Path,
        settings: dict[str, Any],
        *,
        workers: int = 3,
        timeout: float = 180,
    ) -> None:
        if not 1 <= workers <= 8 or timeout <= 0:
            raise ValueError("workers must be 1..8 and timeout must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._store_lock = (self.root / "owner.lock").open("a+b")
        self._store_lock.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(self._store_lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self._store_lock.close()
                raise ValueError("subagent store is already owned by another session") from None
        else:
            import fcntl

            locking: Any = fcntl
            locking.flock(self._store_lock, locking.LOCK_EX | locking.LOCK_NB)
        self.settings = deepcopy(settings)
        self.workers, self.timeout = workers, timeout
        self._lock = threading.RLock()
        self._batch_lock = threading.Lock()
        self._cancel = threading.Event()
        self._closed = False
        self.parent_run_id = ""
        self.step_number = 0
        self.trace: Any = None
        self.records: dict[str, dict[str, Any]] = {}
        self.journal = self.root / "tasks.jsonl"
        workspace_file = self.root / "workspace.json"
        workspace = settings.get("workspace", "")
        if workspace_file.exists() and json.loads(workspace_file.read_text()) != workspace:
            self._store_lock.close()
            raise ValueError("subagent store belongs to a different workspace")
        if not workspace_file.exists():
            workspace_file.write_text(json.dumps(workspace), encoding="utf-8")
        if self.journal.exists():
            lines = self.journal.read_text(encoding="utf-8").splitlines()
            recovered = set()
            for line in lines:
                with suppress(json.JSONDecodeError):
                    marker = json.loads(line)
                    if "recovery_of_line" in marker:
                        recovered.add(marker["recovery_of_line"])
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if index in recovered:
                        continue
                    if index != len(lines) - 1:
                        self._store_lock.close()
                        raise
                    # Preserve the broken suffix and delimit the next append.
                    with self.journal.open("a", encoding="utf-8") as stream:
                        prefix = "" if self.journal.read_bytes().endswith(b"\n") else "\n"
                        stream.write(prefix + json.dumps({"recovery_of_line": index}) + "\n")
                    continue
                if "recovery_of_line" in record:
                    continue
                self.records[record["id"]] = record
            for record in list(self.records.values()):
                if record["status"] in {"queued", "running"}:
                    self._update(record, status="interrupted")

    def _update(self, record: dict[str, Any], **changes: Any) -> None:
        with self._lock:
            record.update(changes)
            self.records[record["id"]] = record
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def cancel(self) -> None:
        """Host-side cancellation, including while the parent waits in task()."""
        self._cancel.set()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            self.cancel()
        with self._batch_lock:
            pass  # the batch drains every worker before releasing this lock
        self._store_lock.close()

    def batch(self, arguments: dict[str, Any]) -> ToolResult:
        context, tasks = arguments.get("context"), arguments.get("tasks")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be non-empty")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 16:
            raise ValueError("tasks must contain 1..16 independent tasks")
        seen: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("instruction"), str):
                raise ValueError("each task needs an instruction")
            if not task["instruction"].strip() or task.get("role", "explore") != "explore":
                raise ValueError("only non-empty explore tasks are supported")
            check_schema(task.get("output_schema", DEFAULT_REPORT))
            session = task.get("session_id")
            if session:
                if not isinstance(session, str) or session not in self.records:
                    raise ValueError("unknown or duplicate session_id")
                logical_id = self.records[session]["session_id"]
                if logical_id in seen:
                    raise ValueError("cannot run the same child session twice in a batch")
                seen.add(logical_id)
        with self._batch_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("manager is closed")
                self._cancel.clear()
            records = []
            for task in tasks:
                identity = uuid.uuid4().hex
                previous = self.records.get(task.get("session_id", ""))
                if previous:
                    previous = next(
                        r
                        for r in reversed(list(self.records.values()))
                        if r["session_id"] == previous["session_id"]
                    )
                record = {
                    "id": identity,
                    "session_id": previous["session_id"] if previous else identity,
                    "previous_attempt": previous["id"] if previous else None,
                    "parent_run_id": self.parent_run_id,
                    "parent_step": self.step_number,
                    "instruction": task["instruction"],
                    "context": context,
                    "status": "queued",
                    "delivery": "pending",
                    "output_schema": task.get("output_schema", DEFAULT_REPORT),
                }
                self._update(record)
                records.append(record)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._execute, record) for record in records]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    self.cancel()
                    concurrent.futures.wait(futures)
                    raise
            for record in records:
                self._update(record, delivery="returned")
            payload = {"results": [self._summary(r) for r in records]}
            if self.trace:
                self.trace.record("subagent_batch", payload)
            return ToolResult(
                "success",
                json.dumps(payload, ensure_ascii=False),
                metadata={"subagent_results": payload["results"]},
            )

    def _execute(self, record: dict[str, Any]) -> None:
        started = time.monotonic()
        control = RunControl(self._cancel, self.timeout)
        identity = record["id"]
        attempt = self.root / identity
        attempt.mkdir()
        try:
            if self._cancel.is_set():
                self._update(record, status="cancelled")
                return
            request = {
                "settings": deepcopy(self.settings),
                "instruction": record["instruction"],
                "context": record["context"],
                "attempt": str(attempt),
                "parent_run_id": record["parent_run_id"],
                "output_schema": record["output_schema"],
            }
            previous = record["previous_attempt"]
            while previous:
                if (self.root / previous / "checkpoint.jsonl").exists():
                    request["previous"] = str(self.root / previous)
                    break
                previous = self.records[previous].get("previous_attempt")
            self._update(record, status="running", execution_backend="in_process")
            result = execute(request, control=control)
            control.check()
            self._update(record, **result)
        except ExplorationStopped as exc:
            self._update(record, status=exc.status)
        except Exception as exc:
            self._update(record, status="failed", error=type(exc).__name__)
        finally:
            usage_path = attempt / "usage.jsonl"
            if usage_path.exists():
                for line in usage_path.read_text(encoding="utf-8").splitlines():
                    with suppress(json.JSONDecodeError):
                        record.update(json.loads(line))
            self._update(record, duration_seconds=round(time.monotonic() - started, 3))

    def _summary(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the parent-facing task envelope without leaking private inputs.

        Workers include ``output`` directly for reports up to 5,000 characters.
        Longer reports carry an ``output_truncated`` marker; callers can then use
        ``task_result`` with the attempt id to read the saved full answer.
        """
        return {
            key: value
            for key, value in record.items()
            if key not in {"context", "instruction", "pid", "report", "evidence", "output_schema"}
        }

    def list_results(self, arguments: dict[str, Any]) -> ToolResult:
        offset = arguments.get("offset", 0)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be non-negative")
        records = list(self.records.values())
        return ToolResult(
            "success",
            json.dumps(
                {
                    "results": [self._summary(r) for r in records[offset : offset + 20]],
                    "next_offset": offset + 20 if offset + 20 < len(records) else None,
                },
                ensure_ascii=False,
            ),
        )

    def result(self, arguments: dict[str, Any]) -> ToolResult:
        identity = arguments.get("task_id", "")
        if identity not in self.records:
            raise ValueError("unknown task_id")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 4000)
        if (
            not isinstance(offset, int)
            or offset < 0
            or not isinstance(limit, int)
            or not 1 <= limit <= 8000
        ):
            raise ValueError("invalid pagination")
        record = self.records[identity]
        path = self.root / identity / "answer.txt"
        answer = path.read_text(encoding="utf-8") if path.exists() else ""
        if arguments.get("section", "answer") == "evidence":
            answer = json.dumps(record.get("evidence", []), ensure_ascii=False, indent=2)
        return ToolResult(
            "success",
            json.dumps(
                {
                    "task": self._summary(record),
                    "text": answer[offset : offset + limit],
                    "next_offset": offset + limit if offset + limit < len(answer) else None,
                },
                ensure_ascii=False,
            ),
        )

    def tools(self) -> list[Tool]:
        item = {
            "type": "object",
            "properties": {
                "instruction": {"type": "string"},
                "role": {"enum": ["explore"]},
                "session_id": {"type": "string"},
                "output_schema": {
                    "type": "object",
                    "description": "Subset: type, properties, required, additionalProperties:boolean, items, enum, description.",
                },
            },
            "required": ["instruction"],
            "additionalProperties": False,
        }
        return [
            Tool(
                "task_list",
                "List saved foreground subtask attempts and statuses; use task_result to expand.",
                self.list_results,
                read_only=True,
                input_schema={
                    "type": "object",
                    "properties": {"offset": {"type": "integer", "minimum": 0}},
                    "additionalProperties": False,
                },
            ),
            Tool(
                "task",
                "Foreground read-only Explore batch. Supply context and tasks[] with instruction. "
                "Tasks must be independent; wait for this batch before dependent work. Optional session_id "
                "is a previous task id for follow-up/retry using saved context. Results include short full "
                "output or a preview marked output_truncated=true; use task_result only to expand long output.",
                self.batch,
                read_only=True,
                input_schema={
                    "type": "object",
                    "properties": {
                        "context": {"type": "string"},
                        "tasks": {"type": "array", "items": item, "minItems": 1, "maxItems": 16},
                    },
                    "required": ["context", "tasks"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                "task_result",
                "Read a saved task result by task_id; section=answer/evidence; paginate using offset and limit.",
                self.result,
                read_only=True,
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8000},
                        "section": {"enum": ["answer", "evidence"]},
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
            ),
        ]

    def install(self, context: Any) -> None:
        self.trace = context.trace_writer

        def before(event: Any) -> None:
            self.parent_run_id, self.step_number = event.run_id, event.step_number

        context.event_bus.on("before_tool_call", before, name="subagents.parent")

        def start(event: Any) -> None:
            event.metadata["subagent_store"] = str(self.root)

        context.event_bus.on("on_run_start", start, name="subagents.store")

        def end(event: Any) -> None:
            event.metadata["subagents"] = [
                self._summary(r)
                for r in self.records.values()
                if r["parent_run_id"] == event.run_id
            ]
            children = event.metadata["subagents"]
            event.metadata["subagent_estimated_tokens"] = sum(
                r.get("estimated_input_tokens", 0) + r.get("estimated_output_tokens", 0)
                for r in children
            )
            event.metadata["subagent_usage_complete"] = all(
                r.get("usage_complete", False) for r in children
            )

        context.event_bus.on("on_run_end", end, name="subagents.accounting")
