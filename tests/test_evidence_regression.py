from __future__ import annotations

import json
import subprocess

import pytest

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeToolCallEvent,
    EventBus,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.core.evidence_policy import EvidenceCompletionPolicy
from dm_agent.extensions.capabilities.evidence import EvidenceGraphCapability
from dm_agent.extensions.capabilities.evidence_checks import EvidenceChecks
from dm_agent.tools.base import ToolResult


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "service.py").write_text("value = 1\n")
    (tmp_path / "test_service.py").write_text("def test_value(): assert True\n")
    bus = EventBus()
    cap = EvidenceGraphCapability()
    cap.install(CapabilityContext(bus, lambda phase: None))
    monkeypatch.setattr(cap._checks, "environment", lambda root: "env-1")
    metadata = {}
    bus.emit_run_start(RunStartEvent("Fix service", 1, "run", metadata=metadata))
    return tmp_path, bus, cap, metadata


def check(rig, step, outcome, *, arguments=None, mutate=None):
    _, bus, cap, metadata = rig
    args = arguments or {"targets": ["test_service.py::test_value"]}
    bus.emit_before_tool_call(BeforeToolCallEvent("run_tests", args, step, "run", metadata))
    if mutate:
        mutate()
    result = ToolResult(
        "success" if outcome == "passed" else "failed",
        outcome,
        check_scope=tuple(args["targets"]),
        metadata={
            "verification": {
                "outcome": outcome,
                "execution_status": "unavailable" if outcome == "unknown" else "completed",
                "failure_kind": "assertion_failure" if outcome == "failed" else "",
                "scope_level": "direct",
            }
        },
    )
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "run_tests",
            args,
            outcome,
            step,
            "run",
            outcome == "passed",
            metadata,
            result=result,
        )
    )
    return [n for n in cap.graph.nodes.values() if n.kind == "verification"][-1]


def edit(rig, step, *, tool="run_shell", failed=False):
    root, bus, _, metadata = rig
    args = {"command": "rewrite", "path": "service.py"}
    bus.emit_before_tool_call(BeforeToolCallEvent(tool, args, step, "run", metadata))
    (root / "service.py").write_text(f"value = {step}\n")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            tool,
            args,
            "error" if failed else "updated",
            step,
            "run",
            not failed,
            metadata,
            result=ToolResult("failed" if failed else "success", "output"),
        )
    )


def test_regression_requires_baseline_and_preserves_it_across_unavailable_retry(rig):
    _, bus, cap, metadata = rig
    baseline = check(rig, 1, "passed")
    edit(rig, 2)
    failure = check(rig, 3, "failed")
    assert failure.metadata["blocking"]
    assert failure.metadata["baseline_node_id"] == baseline.node_id
    assert any(e.relation == "compared_with" for e in cap.graph.edges)
    check(rig, 4, "unknown")
    assert cap.graph.current_verifications() == [failure]
    finish = BeforeFinishEvent("fix", "finish", "done", [], 5, "run", metadata)
    assert bus.emit_before_finish(finish)["block"]
    assert bus.emit_before_finish(finish)["block"]
    assert metadata["evidence_terminal_completion_rejection"]
    check(rig, 6, "passed")
    assert cap.completion_policy.evaluate(cap.graph).verification_state == "tested"
    assert len([n for n in cap.graph.nodes.values() if n.kind == "verification"]) == 4


def test_new_version_drops_old_contradiction_even_when_retry_unavailable(rig):
    check(rig, 1, "passed")
    edit(rig, 2)
    check(rig, 3, "failed")
    edit(rig, 4)
    check(rig, 5, "unknown")
    assert rig[2].completion_policy.evaluate(rig[2].graph).decision == "warn"


@pytest.mark.parametrize("initial", ["failed", None])
def test_precise_failure_without_passing_baseline_is_not_blocking(rig, initial):
    if initial:
        check(rig, 1, initial)
    edit(rig, 2)
    assert not check(rig, 3, "failed").metadata["blocking"]


@pytest.mark.parametrize("changed", ["test", "configuration", "environment", "parameters"])
def test_changed_verification_context_is_not_a_regression(rig, monkeypatch, changed):
    root, _, cap, _ = rig
    check(rig, 1, "passed")
    edit(rig, 2)
    args = None
    if changed == "test":
        (root / "test_service.py").write_text("def test_value(): assert False\n")
    elif changed == "configuration":
        (root / "pytest.ini").write_text("[pytest]\n")
    elif changed == "environment":
        monkeypatch.setattr(cap._checks, "environment", lambda root: "env-2")
    else:
        args = {"targets": ["test_service.py::test_value"], "framework": "unittest"}
    assert not check(rig, 3, "failed", arguments=args).metadata["blocking"]


def test_environment_changes_during_check_cannot_establish_baseline(rig, monkeypatch):
    cap = rig[2]
    node = check(
        rig,
        1,
        "passed",
        mutate=lambda: monkeypatch.setattr(cap._checks, "environment", lambda root: "changed"),
    )
    assert node.metadata["identity"] == ""
    edit(rig, 2)
    assert not check(rig, 3, "failed").metadata["blocking"]


@pytest.mark.parametrize("tool", ["run_shell", "run_python"])
def test_failed_command_records_actual_modifications_and_deletions(rig, tool):
    root, bus, cap, metadata = rig
    args = {"command": "partial write", "code": "partial write"}
    bus.emit_before_tool_call(BeforeToolCallEvent(tool, args, 1, "run", metadata))
    (root / "service.py").unlink()
    (root / "new.py").write_text("value = 1\n")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            tool,
            args,
            "failed",
            1,
            "run",
            False,
            metadata,
            result=ToolResult("failed", "failed"),
        )
    )
    assert set(cap.graph.changed_paths()) == {"service.py", "new.py"}
    assert cap.graph.change_revision == 1
    assert not cap.graph.current_verifications()


def test_read_only_command_has_no_change_node(rig):
    _, bus, cap, metadata = rig
    bus.emit_before_tool_call(BeforeToolCallEvent("run_shell", {}, 1, "run", metadata))
    bus.emit_after_tool_result(
        AfterToolResultEvent("run_shell", {}, "ok", 1, "run", True, metadata)
    )
    assert cap.graph.changed_paths() == ()


def test_syntax_words_do_not_make_valid_code_a_contradiction(rig):
    edit(rig, 2)
    blocked, _ = rig[2]._blocking_verification(
        {"outcome": "unknown"},
        tool="run_python",
        details="SyntaxError referenced in service.py documentation",
    )
    assert not blocked


def test_syntax_uses_runtime_runner_and_soft_fails(rig):
    root, _, cap, _ = rig
    edit(rig, 2)
    calls = []

    def runner(args, timeout):
        calls.append((args, timeout))
        return 0, 'DM_EVIDENCE_PROBE={"errors":[{"path":"service.py","line":1}]}\nreturncode: 0'

    cap._checks = EvidenceChecks(runner, "docker:test")
    blocked, kind = cap._blocking_verification(
        {"outcome": "error", "failure_kind": "syntax_error"},
        tool="run_python",
        details="SyntaxError",
    )
    assert blocked and kind == "syntax_error"
    assert json.loads(calls[0][0][-1]) == ["service.py"]
    cap._checks = EvidenceChecks(lambda args, timeout: (124, "timeout"), "docker:test")
    assert not cap._blocking_verification(
        {"outcome": "error", "failure_kind": "syntax_error"},
        tool="run_python",
        details="SyntaxError",
    )[0]
    assert (root / "service.py").read_text() == "value = 2\n"


def test_runtime_syntax_probe_does_not_execute_source(tmp_path):
    target = tmp_path / "source.py"
    target.write_text("raise RuntimeError('must not execute')\n")
    checks = EvidenceChecks()
    assert checks.syntax(tmp_path, ["source.py"]) == {"errors": []}
    target.write_text("def invalid(:\n")
    assert checks.syntax(tmp_path, ["source.py"])["errors"][0]["path"] == "source.py"
    assert checks.environment(tmp_path)


def test_probe_launch_failure_is_unavailable(tmp_path):
    def runner(args, timeout):
        raise subprocess.TimeoutExpired("python", timeout)

    assert EvidenceChecks(runner).syntax(tmp_path, ["source.py"]) is None


def test_different_commands_cannot_overwrite_failed_check():
    graph = EvidenceGraph("fix")
    graph.workspace_version = "v1"
    graph.add_verification(
        tool="run_tests",
        step_number=1,
        passed=False,
        workspace_version="v1",
        check="test -x",
        outcome="failed",
    )
    graph.add_verification(
        tool="run_tests",
        step_number=2,
        passed=True,
        workspace_version="v1",
        check="test --ignore=broken",
    )
    assert EvidenceCompletionPolicy().evaluate(graph).decision == "block"
