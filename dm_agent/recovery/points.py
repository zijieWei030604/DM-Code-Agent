"""Commit manifests with Agent state and retain call-journal ancestry."""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dm_agent.core.call_journal import CallJournal
from dm_agent.core.checkpoint import RunCheckpoint

from .storage import (
    RecoveryError,
    SnapshotPolicy,
    WorkspaceStore,
    digest,
    durable_write,
    encode,
    git,
    relative_path,
)


class RecoveryPoints:
    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store

    def commit(self, checkpoint: RunCheckpoint, journal: Path) -> str:
        if checkpoint.pending_tool_calls or checkpoint.metadata.get("tool_batch_active"):
            raise RecoveryError("Only a complete tool-batch boundary is recoverable")
        root = Path(checkpoint.cwd).resolve()
        manifest = self.store.capture(root)
        point = uuid.uuid4().hex
        ancestry = list(checkpoint.recovery.get("journals", []))
        source = {
            "path": str(journal.resolve()),
            "run_id": checkpoint.metadata.get("run_id", ""),
            "step": checkpoint.step_count,
        }
        ancestry = [
            item
            for item in ancestry
            if (item["path"], item["run_id"]) != (source["path"], source["run_id"])
        ]
        ancestry.append(source)
        if not journal.exists():
            durable_write(journal, b"")
        checkpoint.recovery = {
            "store": str(self.store.path),
            "point": point,
            "manifest": manifest,
            "journals": ancestry,
            "baseline": checkpoint.recovery.get("baseline")
            or {
                "store": str(self.store.path),
                "manifest": manifest,
            },
        }
        folder = self.store.path / "points" / point
        data = encode(checkpoint.to_dict())
        durable_write(folder / "checkpoint.json", data)
        durable_write(folder / "COMMITTED.json", encode({"checkpoint_sha256": digest(data)}))
        return point

    def load(self, point: str) -> RunCheckpoint:
        if len(point) != 32 or any(c not in "0123456789abcdef" for c in point):
            raise RecoveryError("Invalid recovery point identifier")
        folder = self.store.path / "points" / point
        try:
            marker = json.loads((folder / "COMMITTED.json").read_bytes())
            raw = (folder / "checkpoint.json").read_bytes()
        except (OSError, ValueError) as exc:
            raise RecoveryError("Recovery point is incomplete or missing") from exc
        if digest(raw) != marker["checkpoint_sha256"]:
            raise RecoveryError("Checkpoint hash mismatch")
        checkpoint = RunCheckpoint.from_dict(json.loads(raw))
        self.store.manifest(checkpoint.recovery["manifest"])
        if checkpoint.compressor_state and checkpoint.compressor_state.get("schema") == "lcm-2":
            database = Path(checkpoint.compressor_state["database"])
            if not database.is_file():
                raise RecoveryError(
                    "Source LCM database is missing; preserve the referenced history"
                )
        return checkpoint

    @staticmethod
    def check_journals(checkpoint: RunCheckpoint) -> None:
        for source in checkpoint.recovery.get("journals", []):
            path = Path(source["path"])
            if not path.is_file():
                raise RecoveryError(f"Missing source call journal: {path}")
            journal = CallJournal(Path("unused"))
            journal.path = path
            journal.check_resume(int(source["step"]), str(source["run_id"]))

    def restore(self, point: str, target: Path) -> Path:
        checkpoint = self.load(point)
        self.check_journals(checkpoint)
        manifest = self.store.manifest(checkpoint.recovery["manifest"])
        target = target.absolute()
        if target.exists() or target.is_symlink():
            raise RecoveryError("Recovery target must not exist")
        source = Path(manifest["root"]).resolve()
        if target.resolve().is_relative_to(source) or target.resolve().is_relative_to(
            self.store.path
        ):
            raise RecoveryError("Restore outside the source workspace and recovery store")
        # The unique target is a staging workspace until READY is committed in the store.
        # --no-checkout avoids deleting or overwriting any pre-existing checkout contents.
        if manifest["git_commit"]:
            git(source, "cat-file", "-e", manifest["git_commit"] + "^{commit}")
            git(
                source,
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(target),
                manifest["git_commit"],
            )
            git(target, "read-tree", manifest["git_commit"])
        else:
            target.mkdir(parents=True)
        for name in manifest["directories"]:
            (target / relative_path(name)).mkdir(parents=True, exist_ok=True)
        for name, item in manifest["files"].items():
            path = target / relative_path(name)
            durable_write(path, self.store.get(item["hash"]))
            os.chmod(path, item["mode"])
        check_store = WorkspaceStore(self.store.path, SnapshotPolicy(**manifest["policy"]))
        actual = check_store.scan(target)
        if actual["files"] != manifest["files"] or actual["directories"] != manifest["directories"]:
            raise RecoveryError("Restored workspace failed verification; do not run the agent here")
        prepared = prepare_state(checkpoint, target)
        receipt = uuid.uuid4().hex
        prepared.recovery["receipt"] = receipt
        data = encode(prepared.to_dict())
        destination = self.store.path / "restores" / receipt / "checkpoint.json"
        durable_write(destination, data)
        durable_write(
            destination.with_name("READY.json"), encode({"checkpoint_sha256": digest(data)})
        )
        return destination


def prepare_state(checkpoint: RunCheckpoint, target: Path) -> RunCheckpoint:
    """Keep historical text unchanged; retire executable/version-sensitive state."""
    result = copy.deepcopy(checkpoint)
    old_root = result.cwd
    source_roots = list(result.metadata.get("recovery_source_workspaces", []))
    if old_root not in source_roots:
        source_roots.append(old_root)
    result.cwd = str(target.resolve())
    result.pending_tool_calls = []
    # A fresh evidence/LSP capability rebuilds current facts. Old facts stay in source checkpoint.
    result.capability_state = {}
    baseline = checkpoint.recovery.get("baseline")
    requires_validation = True
    if baseline:
        original = WorkspaceStore(Path(baseline["store"])).manifest(baseline["manifest"])
        current = WorkspaceStore(Path(checkpoint.recovery["store"])).manifest(
            checkpoint.recovery["manifest"]
        )
        requires_validation = any(original[key] != current[key] for key in ("files", "directories"))
    result.metadata = {
        "run_id": checkpoint.metadata.get("run_id", ""),
        "status": "running",
        "recovery_requires_validation": requires_validation,
        "recovery_source_workspace": old_root,
        "recovery_source_workspaces": source_roots,
        "recovery_source_point": checkpoint.recovery["point"],
    }
    if result.plan_state:
        result.plan_state = remap_paths(result.plan_state, Path(old_root), target)
    notice = (
        f"Workspace recovered from {old_root} to {target}. Historical text and test results "
        "describe the previous run; they are not current verification. Use relative paths in "
        "the current workspace. Do not replay old shell commands or pending calls. Recheck "
        "dependencies, configuration and current tests before completion."
    )
    result.conversation_history.append({"role": "user", "content": notice})
    return result


def remap_paths(value: Any, old: Path, new: Path) -> Any:
    """Only known structural path fields, never prose or arbitrary shell strings."""
    if isinstance(value, list):
        return [remap_paths(item, old, new) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"path", "file_path", "cwd", "workspace"} and isinstance(item, str):
            path = Path(item)
            if path.is_absolute() and path.is_relative_to(old):
                item = str(new / path.relative_to(old))
        result[key] = remap_paths(item, old, new)
    return result


def validate_resume(checkpoint: RunCheckpoint) -> None:
    ref = checkpoint.recovery
    if not ref:
        return
    if ref.get("incomplete"):
        raise RecoveryError("Intermediate audit checkpoint: select a committed workspace point")
    points = RecoveryPoints(WorkspaceStore(Path(ref["store"])))
    original = points.load(ref["point"])
    if receipt := ref.get("receipt"):
        if len(receipt) != 32 or any(c not in "0123456789abcdef" for c in receipt):
            raise RecoveryError("Invalid restore receipt")
        folder = points.store.path / "restores" / receipt
        try:
            marker = json.loads((folder / "READY.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise RecoveryError("Restore did not complete") from exc
        expected = marker["checkpoint_sha256"]
    else:
        expected = digest(encode(original.to_dict()))
    if digest(encode(checkpoint.to_dict())) != expected:
        raise RecoveryError("State does not match committed recovery point")
    points.check_journals(checkpoint)
    manifest = points.store.manifest(ref["manifest"])
    if Path.cwd().resolve() != Path(checkpoint.cwd).resolve():
        raise RecoveryError(f"Resume from the verified workspace: {checkpoint.cwd}")
    store = WorkspaceStore(points.store.path, SnapshotPolicy(**manifest["policy"]))
    actual = store.scan(Path(checkpoint.cwd))
    if actual["files"] != manifest["files"] or actual["directories"] != manifest["directories"]:
        raise RecoveryError("Workspace differs from checkpoint; restore into a new directory")
