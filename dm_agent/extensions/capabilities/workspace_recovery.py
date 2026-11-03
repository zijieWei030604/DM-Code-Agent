"""Workspace recovery policies and persistence-boundary adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.checkpoint import RunCheckpoint
from dm_agent.core.events import AfterToolResultEvent, BeforeFinishEvent, BeforeToolCallEvent
from dm_agent.recovery.points import RecoveryPoints
from dm_agent.recovery.storage import SnapshotPolicy, WorkspaceStore, digest, encode


class WorkspaceRecovery:
    def __init__(
        self, store: Path, policy: SnapshotPolicy, *, ancestry: dict[str, Any] | None = None
    ) -> None:
        self.points = RecoveryPoints(WorkspaceStore(store, policy))
        self.ancestry = ancestry or {}
        self.ready = False
        self.trace: Any = None
        self.checked_version = ""
        self.reminded = False

    def install(self, context: CapabilityContext) -> None:
        self.trace = context.trace_writer
        context.event_bus.on("before_tool_call", self.before_tool, kind="policy")
        context.event_bus.on("after_tool_result", self.after_tool)
        context.event_bus.on("before_finish", self.before_finish, kind="policy")

    def save(self, path: Path, checkpoint: RunCheckpoint) -> None:
        checkpoint.recovery = dict(self.ancestry)
        if checkpoint.pending_tool_calls or checkpoint.metadata.get("tool_batch_active"):
            checkpoint.recovery["incomplete"] = True
            return
        self.ready = False
        journal = path.resolve().with_name(path.name + ".calls.jsonl")
        point = self.points.commit(checkpoint, journal)
        self.ancestry = dict(checkpoint.recovery)
        self.ready = True
        if self.trace:
            self.trace.record(
                "workspace_checkpoint_committed",
                {
                    "point": point,
                    "manifest": checkpoint.recovery["manifest"],
                    "step_number": checkpoint.step_count,
                },
            )

    def before_tool(self, event: BeforeToolCallEvent) -> dict[str, Any] | None:
        if not self.ready:
            return {
                "block": True,
                "reason": "No complete workspace recovery point; writes are paused.",
            }
        # Do not allow old absolute structural paths to target the previous checkout.
        roots = event.metadata.get("recovery_source_workspaces", [])
        for old in roots:
            if Path(old).resolve() == Path.cwd().resolve():
                continue
            for key in ("path", "file_path", "cwd", "workspace"):
                value = event.arguments.get(key)
                if (
                    isinstance(value, str)
                    and Path(value).is_absolute()
                    and Path(value).is_relative_to(Path(old))
                ):
                    return {
                        "block": True,
                        "reason": "Old workspace path: replan using the current workspace.",
                    }
            # Shell/program text is not safely rewritable. Reject an explicit old-root reference.
            for value in event.arguments.values():
                if (
                    isinstance(value, str)
                    and str(old).replace("\\", "/").casefold()
                    in value.replace("\\", "/").casefold()
                ):
                    return {
                        "block": True,
                        "reason": "Command references the old workspace; do not replay it.",
                    }
        return None

    def after_tool(self, event: AfterToolResultEvent) -> None:
        fact = event.execution_fact
        if fact and fact.changed_paths:
            self.checked_version = ""
            if event.metadata.get("recovery_source_point"):
                event.metadata["recovery_requires_validation"] = True
        elif fact and fact.fact_type == "verification_recorded":
            self.checked_version = ""
            if fact.verification_outcome == "passed" and fact.verification_scope:
                self.checked_version = self.current_version()

    def current_version(self) -> str:
        snapshot = self.points.store.scan(Path.cwd())
        return digest(encode([snapshot["files"], snapshot["directories"]]))

    def before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        if event.metadata.get("recovery_requires_validation") and (
            not self.checked_version or self.checked_version != self.current_version()
        ):
            event.metadata["recovery_verification_limit"] = "No fresh validation after recovery"
            event.metadata["evidence_verification_state"] = "not_run"
            event.metadata["evidence_completion_status"] = "unverified"
            if not self.reminded:
                self.reminded = True
                return {
                    "block": True,
                    "reason": "Recovery invalidated old verification. Run current checks, or explicitly report verification limitations on the next completion.",
                }
        else:
            event.metadata.pop("recovery_verification_limit", None)
        return None
