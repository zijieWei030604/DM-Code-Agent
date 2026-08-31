"""Verified edit transaction: snapshot, validate, commit, or roll back."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeToolCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.workspace import ImpactReport, SemanticWorkspaceEngine


@dataclass
class _Snapshot:
    path: Path
    existed: bool
    content: bytes
    mode: int | None
    before_hash: str
    after_hash: str = ""


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    detail: str


class VerifiedEditCapability:
    """Treat all writes in one run as a transaction verified at the completion gate."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        engine: SemanticWorkspaceEngine | None = None,
        test_timeout: int = 120,
        run_affected_tests: bool = True,
        run_lint: bool = True,
        run_type_check: bool = True,
        command_runner: Callable[[list[str], int], tuple[int, str]] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.engine = engine
        self.test_timeout = test_timeout
        self.run_affected_tests = run_affected_tests
        self.run_lint = run_lint
        self.run_type_check = run_type_check
        self.command_runner = command_runner
        self._snapshots: dict[Path, _Snapshot] = {}
        self._changed: set[Path] = set()
        self._trace_writer: Any | None = None
        self._verified = False

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        bus = context.event_bus
        bus.on("on_run_start", self._on_run_start, name="verified_edit.run_start")
        bus.on("before_tool_call", self._before_tool_call, name="verified_edit.snapshot")
        bus.on("after_tool_result", self._after_tool_result, name="verified_edit.track")
        bus.on("before_finish", self._before_finish, name="verified_edit.finish_gate")
        bus.on("on_run_end", self._on_run_end, name="verified_edit.run_end")

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._snapshots.clear()
        self._changed.clear()
        self._verified = False
        event.metadata.update(
            {
                "verified_edits_enabled": True,
                "edit_transaction_status": "idle",
                "edit_transaction_files": 0,
                "edit_validation_count": 0,
                "edit_rollback_count": 0,
            }
        )

    def _before_tool_call(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name not in WRITE_ACTIONS:
            return None
        path = self._resolve_path(event.arguments.get("path"))
        if path is None:
            return None
        if path not in self._snapshots:
            existed = path.is_file()
            content = path.read_bytes() if existed else b""
            mode = path.stat().st_mode if existed else None
            self._snapshots[path] = _Snapshot(
                path, existed, content, mode, _hash_bytes(content) if existed else "missing"
            )
            event.metadata["edit_transaction_status"] = "active"
            event.metadata["edit_transaction_files"] = len(self._snapshots)
            self._record(
                "edit_transaction_file_enrolled",
                {"step_number": event.step_number, "path": str(path), "existed": existed},
            )
        return None

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        if event.tool_name not in WRITE_ACTIONS or not event.tool_succeeded:
            return
        path = self._resolve_path(event.arguments.get("path"))
        snapshot = self._snapshots.get(path) if path else None
        if snapshot is None:
            return
        current_hash = _hash_path(snapshot.path)
        if current_hash != snapshot.before_hash:
            snapshot.after_hash = current_hash
            self._changed.add(snapshot.path)
            self._verified = False
            self._record(
                "edit_transaction_changed",
                {"step_number": event.step_number, "path": str(snapshot.path)},
            )

    def _before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        if not self._changed or self._verified:
            return None
        impact = self.engine.analyze_impact(self._changed) if self.engine else None
        if impact is not None:
            event.metadata.update(
                {
                    "edit_impact_risk": impact.risk_level,
                    "edit_impact_score": impact.risk_score,
                    "edit_impact_files": len(impact.affected_files),
                    "edit_impact_tests": len(impact.related_tests),
                }
            )
            self._record("edit_transaction_impact", impact.to_dict())
        results = self._validate(impact)
        event.metadata["edit_validation_count"] = int(
            event.metadata.get("edit_validation_count", 0)
        ) + len(results)
        failures = [result for result in results if not result.passed]
        self._record(
            "edit_transaction_validation",
            {"results": [result.__dict__ for result in results]},
        )
        if failures:
            restored, conflicts = self._rollback()
            event.metadata["edit_transaction_status"] = "rolled_back"
            event.metadata["edit_rollback_count"] = (
                int(event.metadata.get("edit_rollback_count", 0)) + 1
            )
            details = "; ".join(f"{item.name}: {item.detail}" for item in failures)
            conflict_text = f" Rollback conflicts: {', '.join(conflicts)}." if conflicts else ""
            return {
                "block": True,
                "reason": (
                    f"Verified edit transaction rejected completion and restored {restored} file(s). "
                    f"Validation failed: {details}.{conflict_text} Re-read the files and apply a corrected edit."
                ),
            }
        self._verified = True
        event.metadata["edit_transaction_status"] = "committed"
        self._record("edit_transaction_committed", {"files": len(self._changed)})
        return None

    def _on_run_end(self, event: RunEndEvent) -> None:
        if self._changed and not self._verified and event.metadata.get("status") != "success":
            restored, conflicts = self._rollback()
            event.metadata["edit_transaction_status"] = "rolled_back"
            event.metadata["edit_rollback_count"] = (
                int(event.metadata.get("edit_rollback_count", 0)) + 1
            )
            self._record(
                "edit_transaction_rolled_back",
                {"reason": "run_not_successful", "restored": restored, "conflicts": conflicts},
            )

    def _validate(self, impact: ImpactReport | None = None) -> list[ValidationResult]:
        results = [self._validate_python_syntax()]
        changed_python = [
            str(path.relative_to(self.root))
            for path in sorted(self._changed)
            if path.suffix == ".py" and path.is_file()
        ]
        if self.command_runner is not None and changed_python:
            results.append(
                self._run_command("runtime_python_syntax", ["-m", "py_compile", *changed_python])
            )
        if self.run_lint and changed_python:
            results.append(self._run_optional_module("ruff", ["check", *changed_python]))
        if self.run_type_check and changed_python:
            results.append(self._run_optional_module("mypy", changed_python))
        tests = list(impact.related_tests) if impact is not None else []
        if not tests and self.engine:
            tests = self.engine.affected_tests(self._changed)
        if self.run_affected_tests and tests:
            self._record(
                "affected_tests_selected",
                {
                    "tests": tests,
                    "risk": impact.risk_level if impact else "unknown",
                    "reasons": list(impact.reasons) if impact else [],
                },
            )
            results.append(self._run_tests(tests))
        return results

    def _validate_python_syntax(self) -> ValidationResult:
        for path in sorted(self._changed):
            if path.suffix != ".py" or not path.is_file():
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as exc:
                return ValidationResult("python_syntax", False, f"{path}: {exc}")
        return ValidationResult("python_syntax", True, "changed Python files parse successfully")

    def _run_tests(self, tests: list[str]) -> ValidationResult:
        return self._run_command("affected_tests", ["-m", "pytest", "-q", *tests])

    def _run_optional_module(self, module: str, arguments: list[str]) -> ValidationResult:
        if self.command_runner is not None:
            probe = (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
            )
            try:
                returncode, _ = self.command_runner(["-c", probe], self.test_timeout)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return ValidationResult(module, False, str(exc))
            if returncode != 0:
                return ValidationResult(
                    module, True, f"skipped: {module} is not installed in validation runtime"
                )
            return self._run_command(module, ["-m", module, *arguments])
        if importlib.util.find_spec(module) is None:
            return ValidationResult(module, True, f"skipped: {module} is not installed")
        return self._run_command(module, ["-m", module, *arguments])

    def _run_command(self, name: str, arguments: list[str]) -> ValidationResult:
        try:
            if self.command_runner is not None:
                returncode, output = self.command_runner(arguments, self.test_timeout)
            else:
                completed = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout,
                    shell=False,
                )
                returncode = completed.returncode
                output = (completed.stdout + "\n" + completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ValidationResult(name, False, str(exc))
        detail = output[-2000:] if output else f"exit code {returncode}"
        return ValidationResult(name, returncode == 0, detail)

    def _rollback(self) -> tuple[int, list[str]]:
        restored = 0
        conflicts: list[str] = []
        for path in sorted(self._changed):
            snapshot = self._snapshots[path]
            if snapshot.after_hash and _hash_path(path) != snapshot.after_hash:
                conflicts.append(str(path))
                continue
            if snapshot.existed:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.dm-agent-rollback-{os.getpid()}")
                temporary.write_bytes(snapshot.content)
                os.replace(temporary, path)
                if snapshot.mode is not None and os.name != "nt":
                    path.chmod(snapshot.mode)
            elif path.exists():
                path.unlink()
            restored += 1
        if self.engine:
            self.engine.update(self._changed)
        self._changed.clear()
        self._verified = False
        self._record("edit_transaction_rolled_back", {"restored": restored, "conflicts": conflicts})
        return restored, conflicts

    def _resolve_path(self, raw: Any) -> Path | None:
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        return path if path.is_relative_to(self.root) else None

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_path(path: Path) -> str:
    try:
        return _hash_bytes(path.read_bytes()) if path.is_file() else "missing"
    except OSError:
        return "unreadable"
