"""Lifecycle adapter for policy-driven verified edit transactions."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
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
from dm_agent.verification import (
    CommandRunner,
    VerificationPolicy,
    VerificationReport,
    VerificationResult,
    VerificationRunner,
)
from dm_agent.workspace import ImpactReport, SemanticWorkspaceEngine

ValidationResult = VerificationResult


@dataclass
class _Snapshot:
    path: Path
    existed: bool
    content: bytes
    mode: int | None
    before_hash: str
    after_hash: str = ""


class VerifiedEditCapability:
    """Coordinate snapshots, verification, commit, and rollback across one run."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        engine: SemanticWorkspaceEngine | None = None,
        policy: VerificationPolicy | None = None,
        command_runner: CommandRunner | None = None,
        test_timeout: int | None = None,
        run_affected_tests: bool | None = None,
        run_lint: bool | None = None,
        run_type_check: bool | None = None,
        max_affected_tests: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.engine = engine
        self.policy = _with_legacy_overrides(
            policy or VerificationPolicy(),
            test_timeout=test_timeout,
            run_affected_tests=run_affected_tests,
            run_lint=run_lint,
            run_type_check=run_type_check,
            max_affected_tests=max_affected_tests,
        )
        self.runner = VerificationRunner(
            self.root, policy=self.policy, command_runner=command_runner
        )
        self._snapshots: dict[Path, _Snapshot] = {}
        self._changed: set[Path] = set()
        self._trace_writer: Any | None = None
        self._verified = False

    @property
    def command_runner(self) -> CommandRunner | None:
        return self.runner.command_runner

    @property
    def test_timeout(self) -> int:
        return self.policy.timeout

    @property
    def run_affected_tests(self) -> bool:
        return self.policy.run_affected_tests

    @property
    def run_lint(self) -> bool:
        return self.policy.check_lint

    @property
    def run_type_check(self) -> bool:
        return self.policy.check_types

    @property
    def max_affected_tests(self) -> int | None:
        return self.policy.max_affected_tests

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
        self.runner.reset()
        event.metadata.update(
            {
                "verified_edits_enabled": True,
                "edit_transaction_status": "idle",
                "edit_transaction_files": 0,
                "edit_validation_count": 0,
                "edit_validation_cache_hit_count": 0,
                "edit_type_baseline_count": 0,
                "edit_rollback_count": 0,
                "edit_run_end_finalization_count": 0,
                "edit_run_end_salvaged": False,
            }
        )

    def _before_tool_call(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        if event.tool_name not in WRITE_ACTIONS:
            return None
        path = self._resolve_path(event.arguments.get("path"))
        if path is None or path in self._snapshots:
            return None

        baseline = self.runner.capture_type_baseline(path)
        relative = path.relative_to(self.root).as_posix()
        if baseline.captured:
            event.metadata["edit_type_baseline_count"] = (
                int(event.metadata.get("edit_type_baseline_count", 0)) + 1
            )
            self._record(
                "edit_type_baseline",
                {"path": relative, "diagnostics": baseline.diagnostics},
            )
        elif baseline.error:
            self._record("edit_type_baseline_error", {"path": relative, "error": baseline.error})

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
        if event.tool_name not in WRITE_ACTIONS or not event.has_effect:
            return
        path = self._resolve_path(event.arguments.get("path"))
        snapshot = self._snapshots.get(path) if path else None
        if snapshot is None:
            return
        current_hash = _hash_path(snapshot.path)
        if current_hash == snapshot.before_hash:
            snapshot.after_hash = ""
            self._changed.discard(snapshot.path)
            if not self._changed:
                event.metadata["edit_transaction_status"] = "idle"
            return
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
        report = self._verify(event.metadata, trigger="finish")
        if report.passed:
            self._commit(event.metadata, trigger="finish")
            return None

        restored, conflicts = self._rollback()
        self._mark_rollback(event.metadata)
        details = "; ".join(f"{item.name}: {item.detail}" for item in report.failures)
        conflict_text = f" Rollback conflicts: {', '.join(conflicts)}." if conflicts else ""
        return {
            "block": True,
            "reason": (
                f"Verified edit transaction rejected completion and restored {restored} file(s). "
                f"Validation failed: {details}.{conflict_text} "
                "Re-read the files and apply a corrected edit."
            ),
        }

    def _on_run_end(self, event: RunEndEvent) -> None:
        if not self._changed or self._verified:
            return
        if not self.policy.finalize_on_run_end:
            restored, conflicts = self._rollback()
            self._mark_rollback(event.metadata)
            self._record(
                "edit_transaction_run_end_rejected",
                {"reason": "finalization_disabled", "restored": restored, "conflicts": conflicts},
            )
            return

        event.metadata["edit_run_end_finalization_count"] = (
            int(event.metadata.get("edit_run_end_finalization_count", 0)) + 1
        )
        report = self._verify(event.metadata, trigger="run_end")
        if report.passed:
            self._commit(event.metadata, trigger="run_end")
            event.metadata["edit_run_end_salvaged"] = True
            self._record(
                "edit_transaction_run_end_salvaged",
                {"files": len(self._changed), "status": event.metadata.get("status", "")},
            )
            return

        restored, conflicts = self._rollback()
        self._mark_rollback(event.metadata)
        self._record(
            "edit_transaction_run_end_rejected",
            {
                "restored": restored,
                "conflicts": conflicts,
                "failures": [item.__dict__ for item in report.failures],
            },
        )

    def _verify(self, metadata: dict[str, Any], *, trigger: str) -> VerificationReport:
        impact = self._impact(metadata)
        tests = self._selected_tests(impact) if self.policy.run_affected_tests else []
        if tests:
            self._record(
                "affected_tests_selected",
                {
                    "tests": tests,
                    "risk": impact.risk_level if impact else "unknown",
                    "reasons": list(impact.reasons) if impact else [],
                },
            )
        report = self.runner.verify(self._changed, affected_tests=tests)
        metadata["edit_validation_count"] = int(metadata.get("edit_validation_count", 0)) + len(
            report.results
        )
        if report.cache_hit:
            metadata["edit_validation_cache_hit_count"] = (
                int(metadata.get("edit_validation_cache_hit_count", 0)) + 1
            )
            self._record(
                "edit_transaction_validation_cache_hit",
                {"key": report.cache_key, "results": len(report.results), "trigger": trigger},
            )
        self._record(
            "edit_transaction_validation",
            {"trigger": trigger, "results": [item.__dict__ for item in report.results]},
        )
        return report

    def _impact(self, metadata: dict[str, Any]) -> ImpactReport | None:
        if not self.engine:
            return None
        self.engine.update(self._changed)
        impact = self.engine.analyze_impact(self._changed)
        metadata.update(
            {
                "edit_impact_risk": impact.risk_level,
                "edit_impact_score": impact.risk_score,
                "edit_impact_files": len(impact.affected_files),
                "edit_impact_tests": len(impact.related_tests),
            }
        )
        self._record("edit_transaction_impact", impact.to_dict())
        return impact

    def _commit(self, metadata: dict[str, Any], *, trigger: str) -> None:
        self._verified = True
        metadata["edit_transaction_status"] = (
            "committed_on_run_end" if trigger == "run_end" else "committed"
        )
        self._record(
            "edit_transaction_committed", {"files": len(self._changed), "trigger": trigger}
        )

    def _mark_rollback(self, metadata: dict[str, Any]) -> None:
        metadata["edit_transaction_status"] = "rolled_back"
        metadata["edit_rollback_count"] = int(metadata.get("edit_rollback_count", 0)) + 1

    def _selected_tests(self, impact: ImpactReport | None) -> list[str]:
        tests = list(impact.related_tests) if impact is not None else []
        if not tests and self.engine:
            tests = self.engine.affected_tests(self._changed)
        return list(self.runner.select_tests(tests))

    def _rollback(self) -> tuple[int, list[str]]:
        restored = 0
        conflicts: list[str] = []
        changed = tuple(sorted(self._changed))
        for path in changed:
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
            self.engine.update(changed)
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


def _with_legacy_overrides(
    policy: VerificationPolicy,
    *,
    test_timeout: int | None,
    run_affected_tests: bool | None,
    run_lint: bool | None,
    run_type_check: bool | None,
    max_affected_tests: int | None,
) -> VerificationPolicy:
    updates: dict[str, Any] = {}
    if test_timeout is not None:
        updates["timeout"] = test_timeout
    if run_affected_tests is not None:
        updates["run_affected_tests"] = run_affected_tests
    if run_lint is not None:
        updates["check_lint"] = run_lint
    if run_type_check is not None:
        updates["check_types"] = run_type_check
    if max_affected_tests is not None:
        updates["max_affected_tests"] = max_affected_tests
    return replace(policy, **updates) if updates else policy


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_path(path: Path) -> str:
    try:
        return _hash_bytes(path.read_bytes()) if path.is_file() else "missing"
    except OSError:
        return "unreadable"


__all__ = ["ValidationResult", "VerifiedEditCapability"]
