"""Offline recovery behavior, including failure at durable commit boundaries."""

import json
import subprocess

import pytest

from dm_agent.core.agent import ReactAgent
from dm_agent.core.call_journal import CallJournal
from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.checkpoint import RunCheckpoint
from dm_agent.core.events import BeforeFinishEvent, BeforeToolCallEvent
from dm_agent.core.persistence import load_resume_state
from dm_agent.extensions.capabilities.workspace_recovery import WorkspaceRecovery
from dm_agent.recovery import points as points_module
from dm_agent.recovery.points import RecoveryPoints
from dm_agent.recovery.storage import RecoveryError, SnapshotPolicy, WorkspaceStore
from dm_agent.tools.base import Tool


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "main.py").write_text("original\n", encoding="utf-8")
    return root


def state(root, step=0):
    return RunCheckpoint("repair", step, cwd=str(root), metadata={"run_id": "run-a"})


def setup_point(root):
    points = RecoveryPoints(WorkspaceStore(root.parent / "store"))
    checkpoint = state(root)
    journal = CallJournal(root.parent / "session.jsonl")
    point = points.commit(checkpoint, journal.path)
    return points, checkpoint, journal, point


def test_restore_preserves_original_and_full_scoped_contents(workspace, monkeypatch):
    (workspace / "空目录").mkdir()
    for folder in ("a", "b"):
        (workspace / folder).mkdir()
        (workspace / folder / "same.bin").write_bytes(folder.encode() + b"\0\xff")
    points, checkpoint, _, point = setup_point(workspace)
    (workspace / "main.py").write_text("broken")
    (workspace / "a" / "same.bin").unlink()
    (workspace / "extra.py").write_text("extra")
    target = workspace.parent / "restored"
    resume = points.restore(point, target)
    assert (workspace / "main.py").read_text() == "broken"
    assert (target / "main.py").read_text() == "original\n"
    assert (target / "a" / "same.bin").read_bytes() == b"a\0\xff"
    assert (target / "b" / "same.bin").read_bytes() == b"b\0\xff"
    assert (target / "空目录").is_dir()
    assert not (target / "extra.py").exists()
    monkeypatch.chdir(target)
    loaded = load_resume_state(resume)
    assert loaded.metadata["recovery_source_workspace"] == str(workspace)
    assert loaded.capability_state == {}
    assert loaded.recovery["point"] == checkpoint.recovery["point"]
    (target / "main.py").write_text("changed again")
    with pytest.raises(RecoveryError, match="differs"):
        load_resume_state(resume)


@pytest.mark.parametrize("file", ["checkpoint.json", "COMMITTED.json"])
def test_interrupted_commit_never_listed_as_recoverable(workspace, monkeypatch, file):
    points = RecoveryPoints(WorkspaceStore(workspace.parent / "store"))
    original = points_module.durable_write

    def fail(path, data):
        if path.name == file:
            raise OSError("simulated disk failure")
        original(path, data)

    monkeypatch.setattr(points_module, "durable_write", fail)
    with pytest.raises(OSError):
        points.commit(state(workspace), workspace.parent / "calls.jsonl")
    assert not list((points.store.path / "points").glob("*/COMMITTED.json"))


def test_corrupt_blob_and_partial_restore_are_refused(workspace, monkeypatch):
    points, checkpoint, _, point = setup_point(workspace)
    manifest = points.store.manifest(checkpoint.recovery["manifest"])
    blob = points.store.path / "objects" / manifest["files"]["main.py"]["hash"]
    blob.write_bytes(b"corrupt")
    with pytest.raises(RecoveryError, match="Corrupt"):
        points.restore(point, workspace.parent / "target")
    assert not (workspace.parent / "target").exists()


def test_store_deduplicates_and_limits_fail_closed(workspace):
    store = WorkspaceStore(workspace.parent / "store")
    one = store.capture(workspace)
    assert store.capture(workspace) == one
    assert len(list((store.path / "objects").iterdir())) == 2
    limited = WorkspaceStore(workspace.parent / "small", SnapshotPolicy(max_file_bytes=1))
    with pytest.raises(RecoveryError, match="limit"):
        limited.capture(workspace)
    with pytest.raises(RecoveryError, match="outside"):
        WorkspaceStore(workspace / "nested").capture(workspace)


def test_unconfirmed_effect_blocks_across_forks_until_reconciled(workspace):
    points, _, journal, point = setup_point(workspace)
    call = journal.begin("run_shell", 1, "run-a")
    journal.finish(call)
    with pytest.raises(ValueError, match="reconciliation"):
        points.restore(point, workspace.parent / "blocked")
    before = journal.path.read_bytes()
    audit = CallJournal(workspace.parent / "unused")
    audit.path = journal.path.with_name(journal.path.name + ".reconciliations.jsonl")
    audit.append(
        {
            "actor": "user",
            "note": "verified local effect only",
            "outcome": "verified",
            "fingerprint": journal.fingerprint(journal.records()[call]),
        }
    )
    assert journal.path.read_bytes() == before
    receipt = points.restore(point, workspace.parent / "fork")
    fork = RunCheckpoint.from_dict(json.loads(receipt.read_bytes()))
    next_point = points.commit(fork, workspace.parent / "fork.calls.jsonl")
    assert len(points.load(next_point).recovery["journals"]) == 2
    journal.begin("run_shell", 2, "run-a")
    with pytest.raises(ValueError, match="reconciliation"):
        points.restore(next_point, workspace.parent / "fork2")


def test_intermediate_batch_is_audit_only(workspace):
    recovery = WorkspaceRecovery(workspace.parent / "store", SnapshotPolicy())
    checkpoint = state(workspace)
    recovery.save(workspace.parent / "session.jsonl", checkpoint)
    first = checkpoint.recovery["point"]
    checkpoint.metadata["tool_batch_active"] = True
    (workspace / "main.py").write_text("half batch")
    recovery.save(workspace.parent / "session.jsonl", checkpoint)
    assert checkpoint.recovery["incomplete"] is True
    assert checkpoint.recovery["point"] == first
    assert len(list((recovery.points.store.path / "points").glob("*/COMMITTED.json"))) == 1
    with pytest.raises(RecoveryError, match="batch"):
        recovery.points.commit(checkpoint, workspace.parent / "calls")
    checkpoint.metadata.pop("tool_batch_active")
    recovery.save(workspace.parent / "session.jsonl", checkpoint)
    assert checkpoint.recovery["point"] != first
    assert "incomplete" not in checkpoint.recovery


def test_git_restore_retains_uncommitted_changes_and_excludes_secrets(workspace):
    def git(*args):
        subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)

    git("init")
    git("add", "main.py")
    git(
        "-c",
        "user.name=Recovery Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    (workspace / "main.py").write_text("uncommitted")
    (workspace / ".env").write_text("secret")
    (workspace / ".gitignore").write_text("ignored/\n")
    (workspace / "ignored").mkdir()
    (workspace / "ignored" / "data").write_text("ignored")
    (workspace / "new.py").write_text("new")
    points, _, _, point = setup_point(workspace)
    target = workspace.parent / "worktree"
    points.restore(point, target)
    assert (target / ".git").is_file()
    assert (target / "main.py").read_text() == "uncommitted"
    assert (target / "new.py").read_text() == "new"
    assert not (target / ".env").exists()
    assert not (target / "ignored").exists()


def test_failed_ready_marker_cannot_resume(workspace, monkeypatch):
    points, _, _, point = setup_point(workspace)
    original = points_module.durable_write

    def fail(path, data):
        if path.name == "READY.json":
            raise OSError("simulated interrupted restore")
        original(path, data)

    monkeypatch.setattr(points_module, "durable_write", fail)
    target = workspace.parent / "target"
    with pytest.raises(OSError):
        points.restore(point, target)
    monkeypatch.chdir(target)
    prepared = next((points.store.path / "restores").glob("*/checkpoint.json"))
    with pytest.raises(RecoveryError, match="did not complete"):
        load_resume_state(prepared)


@pytest.mark.parametrize("interrupt", [False, True])
def test_real_runtime_only_commits_whole_batches(workspace, monkeypatch, interrupt):
    monkeypatch.chdir(workspace)
    original_bytes = (workspace / "main.py").read_bytes()

    class Client:
        supports_tool_calling = True
        last_response_mode = "native_tool_call"
        last_tool_call_count = 2

        def __init__(self):
            self.first = True

        def respond(self, messages, **kwargs):
            if self.first:
                self.first = False
                return json.dumps(
                    {
                        "tool_calls": [
                            {"action": "first", "action_input": {}},
                            {"action": "second", "action_input": {}},
                        ]
                    }
                )
            return json.dumps({"action": "finish", "action_input": "done"})

    def first(args):
        (workspace / "main.py").write_text("first")
        return "written"

    def second(args):
        if interrupt:
            raise KeyboardInterrupt()
        (workspace / "main.py").write_text("second")
        return "written"

    client = Client()
    agent = ReactAgent(
        client,
        [Tool("first", "first", first), Tool("second", "second", second)],
        enable_planning=False,
        enable_compression=False,
    )
    recovery = WorkspaceRecovery(workspace.parent / "store", SnapshotPolicy())
    recovery.install(CapabilityContext(agent.event_bus, lambda phase: client))
    agent._persistence.checkpoint_observer = recovery.save
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            agent.run("change", checkpoint_path=workspace.parent / "session.jsonl", max_steps=3)
    else:
        agent.run("change", checkpoint_path=workspace.parent / "session.jsonl", max_steps=3)
    contents = []
    for marker in (recovery.points.store.path / "points").glob("*/COMMITTED.json"):
        checkpoint = recovery.points.load(marker.parent.name)
        assert not checkpoint.pending_tool_calls
        manifest = recovery.points.store.manifest(checkpoint.recovery["manifest"])
        contents.append(recovery.points.store.get(manifest["files"]["main.py"]["hash"]))
    assert original_bytes in contents
    assert b"first" not in contents
    assert (b"second" in contents) is not interrupt


def test_restored_runtime_rejects_old_paths_and_invalidates_checked_version(workspace, monkeypatch):
    recovery = WorkspaceRecovery(workspace.parent / "store", SnapshotPolicy())
    recovery.ready = True
    target = workspace.parent / "new"
    target.mkdir()
    monkeypatch.chdir(target)
    metadata = {
        "recovery_source_workspaces": [str(workspace)],
        "recovery_requires_validation": True,
    }
    event = BeforeToolCallEvent(
        "read_file", {"path": str(workspace / "main.py")}, 1, "run", metadata
    )
    assert recovery.before_tool(event)["block"]
    event.arguments = {"path": "main.py"}
    assert recovery.before_tool(event) is None
    finish = BeforeFinishEvent("task", "finish", "done", [], 1, "run", metadata)
    assert recovery.before_finish(finish)["block"]
    assert recovery.before_finish(finish) is None
    assert metadata["evidence_completion_status"] == "unverified"
    recovery.checked_version = recovery.current_version()
    assert recovery.before_finish(finish) is None
    assert "recovery_verification_limit" not in metadata
    (target / "new.py").write_text("changed")
    recovery.before_finish(finish)
    assert metadata["recovery_verification_limit"]


def test_capture_failure_stops_runtime_before_any_write(workspace, monkeypatch):
    monkeypatch.chdir(workspace)
    recovery = WorkspaceRecovery(workspace.parent / "store", SnapshotPolicy(max_file_bytes=1))

    class Client:
        def respond(self, *args, **kwargs):
            pytest.fail("No model request should happen without an initial recovery point")

    agent = ReactAgent(
        Client(),
        [Tool("read", "read", lambda args: "unused", read_only=True)],
        enable_planning=False,
        enable_compression=False,
    )
    agent._persistence.checkpoint_observer = recovery.save
    with pytest.raises(RecoveryError, match="limit"):
        agent.run("change", checkpoint_path=workspace.parent / "session.jsonl")
    assert (workspace / "main.py").read_text() == "original\n"


def test_git_include_overrides_ignore_but_keeps_other_contents_out(workspace):
    subprocess.run(["git", "-C", str(workspace), "init"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )
    (workspace / ".gitignore").write_text("assets/\n")
    (workspace / "assets").mkdir()
    (workspace / "assets" / "keep.bin").write_bytes(b"keep")
    (workspace / "assets" / "skip.bin").write_bytes(b"skip")
    store = WorkspaceStore(workspace.parent / "store", SnapshotPolicy(include=("assets/keep.bin",)))
    manifest = store.manifest(store.capture(workspace))
    assert "assets/keep.bin" in manifest["files"]
    assert "assets/skip.bin" not in manifest["files"]


def test_oversize_total_and_missing_memory_refuse_recovery(workspace):
    store = WorkspaceStore(workspace.parent / "limited", SnapshotPolicy(max_snapshot_bytes=1))
    with pytest.raises(RecoveryError, match="size limit"):
        store.capture(workspace)
    points, _, journal, _ = setup_point(workspace)
    checkpoint = state(workspace)
    checkpoint.compressor_state = {
        "schema": "lcm-2",
        "database": str(workspace.parent / "missing.sqlite"),
    }
    point = points.commit(checkpoint, journal.path)
    with pytest.raises(RecoveryError, match="LCM database"):
        points.restore(point, workspace.parent / "target")
    assert not (workspace.parent / "target").exists()


def test_blob_commit_failure_never_publishes_a_point(workspace, monkeypatch):
    from dm_agent.recovery import storage

    store = WorkspaceStore(workspace.parent / "store")

    def fail(path, data):
        raise OSError("disk full during blob write")

    monkeypatch.setattr(storage, "durable_write", fail)
    with pytest.raises(OSError):
        RecoveryPoints(store).commit(state(workspace), workspace.parent / "calls")
    assert not list((store.path / "points").glob("*/COMMITTED.json"))


def test_link_in_scope_refused_and_excluded_link_not_traversed(workspace):
    link = workspace / "link"
    try:
        link.symlink_to(workspace.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Windows user cannot create symlinks")
    store = WorkspaceStore(workspace.parent / "store")
    with pytest.raises(RecoveryError, match="Linked"):
        store.capture(workspace)
    excluded = WorkspaceStore(store.path, SnapshotPolicy(exclude=("link",)))
    manifest = excluded.manifest(excluded.capture(workspace))
    assert "link/" in manifest["omitted"]


def test_readonly_recovery_does_not_require_tests_but_changed_checkpoint_does(workspace):
    points, checkpoint, journal, point = setup_point(workspace)
    readonly = RunCheckpoint.from_dict(
        json.loads(points.restore(point, workspace.parent / "readonly").read_bytes())
    )
    assert readonly.metadata["recovery_requires_validation"] is False
    (workspace / "main.py").write_text("changed")
    changed_point = points.commit(checkpoint, journal.path)
    changed = RunCheckpoint.from_dict(
        json.loads(points.restore(changed_point, workspace.parent / "changed").read_bytes())
    )
    assert changed.metadata["recovery_requires_validation"] is True
