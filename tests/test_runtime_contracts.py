import pytest

from dm_agent.core.call_journal import CallJournal
from dm_agent.core.events import BeforeToolCallEvent, EventBus
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.core.observation import ObservationBounder
from dm_agent.core.persistence import RunPersistence
from dm_agent.core.run_state import RunContext
from dm_agent.core.tool_invoker import ToolInvoker
from dm_agent.core.workspace_version import workspace_version
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tools.write_journal import begin_write, recover_writes


def test_observer_cannot_modify_arguments():
    bus = EventBus()

    def observer(event):
        event.arguments["nested"]["path"] = "elsewhere"
        return {"block": True}

    bus.on("before_tool_call", observer, kind="observer")
    event = BeforeToolCallEvent("edit_file", {"nested": {"path": "original"}}, 1, "run")
    assert bus.emit_before_tool_call(event) is None
    assert event.arguments["nested"]["path"] == "original"


def test_policy_failure_blocks_execution():
    bus = EventBus()

    def broken(event):
        raise RuntimeError("offline")

    bus.on("before_tool_call", broken, kind="policy")
    assert bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", {}, 1, "run"))["block"]


def test_verification_is_versioned_and_retries_supersede():
    graph = EvidenceGraph("Fix users")
    graph.add_change(tool="edit_file", path="users.py", step_number=1)
    graph.workspace_version = "a"
    graph.add_verification(
        tool="run_tests", step_number=2, passed=False, workspace_version="a", check="X"
    )
    graph.add_verification(
        tool="run_tests", step_number=3, passed=True, workspace_version="a", check="X"
    )
    graph.add_conclusion(text="done", step_number=4)
    assert graph.status() == "verified"
    assert graph.audit()["failed_verifications"] == 1
    graph.workspace_version = "b"
    assert graph.status() == "implemented"
    assert EvidenceGraph.from_dict(graph.to_dict()).status() == "implemented"


def test_version_detects_source_and_config_changes(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("x = 1")
    first = workspace_version(tmp_path)
    source.write_text("x = 2")
    assert workspace_version(tmp_path) != first


@pytest.mark.parametrize("applied", [True, False])
def test_interrupted_write_reconciles_without_replay(tmp_path, monkeypatch, applied):
    monkeypatch.setattr("dm_agent.tools.write_journal.tempfile.gettempdir", lambda: str(tmp_path))
    target = tmp_path / "app.py"
    target.write_bytes(b"before")
    journal = begin_write(target, b"after")
    if applied:
        target.write_bytes(b"after")
    recover_writes(target)
    assert "reconciled_" in journal.read_text()
    assert target.read_bytes() == (b"after" if applied else b"before")


def test_interrupted_write_conflict_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr("dm_agent.tools.write_journal.tempfile.gettempdir", lambda: str(tmp_path))
    target = tmp_path / "app.py"
    target.write_bytes(b"before")
    begin_write(target, b"after")
    target.write_bytes(b"user edit")
    with pytest.raises(RuntimeError, match="Uncertain"):
        recover_writes(target)
    assert target.read_bytes() == b"user edit"


def test_resume_blocks_uncertain_shell_and_checkpoint_gap(tmp_path):
    journal = CallJournal(tmp_path / "checkpoint.json")
    call_id = journal.begin("run_shell", 2, "run")
    with pytest.raises(ValueError, match="reconciliation"):
        journal.check_resume(1, "run")
    journal.finish(call_id)
    with pytest.raises(ValueError, match="reconciliation"):
        journal.check_resume(1, "run")
    journal.check_resume(2, "run")


def test_resume_allows_read_retry(tmp_path):
    journal = CallJournal(tmp_path / "checkpoint.json")
    journal.begin("read_file", 2, "run")
    journal.check_resume(1, "run")


def test_structured_status_does_not_depend_on_output_words():
    invoker = ToolInvoker(
        event_bus=EventBus(),
        bounder=ObservationBounder(max_chars=1000),
        persistence=RunPersistence(),
    )
    context = RunContext(run_id="run", step_number=1)
    failed = invoker.invoke(
        Tool("check", "", lambda args: ToolResult("failed", "looks fine")),
        action="check",
        action_input={},
        context=context,
    )
    assert not failed.tool_succeeded
    passed = invoker.invoke(
        Tool("check", "", lambda args: ToolResult("success", "example contains Error")),
        action="check",
        action_input={},
        context=context,
    )
    assert passed.tool_succeeded
