from __future__ import annotations

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeToolCallEvent,
    EventBus,
    RunStartEvent,
)
from dm_agent.extensions.capabilities import VerifiedEditCapability
from dm_agent.workspace import SemanticWorkspaceEngine


def test_semantic_workspace_persists_symbols_references_and_affected_tests(tmp_path):
    (tmp_path / "service.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_service.py").write_text(
        "from service import calculate_total\n\ndef test_total():\n    assert calculate_total([1]) == 1\n",
        encoding="utf-8",
    )
    database = tmp_path / "index.db"
    engine = SemanticWorkspaceEngine(tmp_path, database_path=database)

    first = engine.update()
    second = engine.update()
    hits = engine.search_symbols("calculate_total")
    impact = engine.find_references("calculate_total")

    assert first.indexed_files == 2
    assert second.cache_hits == 2
    assert hits[0].qualified_name == "calculate_total"
    assert "tests/test_service.py" in impact.tests
    assert engine.affected_tests(["service.py"]) == ["tests/test_service.py"]
    engine.close()

    reopened = SemanticWorkspaceEngine(tmp_path, database_path=database)
    assert reopened.update().cache_hits == 2
    assert reopened.search_symbols("calculate_total")
    reopened.close()


def test_semantic_workspace_propagates_change_impact_through_callers(tmp_path):
    (tmp_path / "service.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    (tmp_path / "api.py").write_text(
        "from service import calculate_total\n\n"
        "def checkout(items):\n    return calculate_total(items)\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_api.py").write_text(
        "from api import checkout\n\n"
        "def test_checkout():\n    assert checkout([1]) == 1\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    engine.update()
    report = engine.analyze_impact(["service.py"], max_depth=2)

    assert "api.py" in report.affected_files
    assert "tests/test_api.py" in report.affected_files
    assert report.related_tests == ("tests/test_api.py",)
    assert any(item.symbol == "checkout" and item.distance == 1 for item in report.affected_symbols)
    assert any(item.symbol == "test_checkout" and item.distance == 2 for item in report.affected_symbols)
    assert report.risk_level in {"medium", "high"}
    assert "api.py:checkout" in report.render()
    engine.close()


def test_semantic_workspace_incrementally_removes_stale_impact_edges(tmp_path):
    service = tmp_path / "service.py"
    caller = tmp_path / "caller.py"
    service.write_text("def value():\n    return 1\n", encoding="utf-8")
    caller.write_text(
        "from service import value\n\ndef use_value():\n    return value()\n", encoding="utf-8"
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()
    assert "caller.py" in engine.analyze_impact(["service.py"]).affected_files

    caller.write_text("def use_value():\n    return 2\n", encoding="utf-8")
    engine.update([caller])

    assert "caller.py" not in engine.analyze_impact(["service.py"]).affected_files
    engine.close()


def test_verified_edit_rolls_back_invalid_python_at_finish(tmp_path):
    target = tmp_path / "module.py"
    original = "def value():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path, run_affected_tests=False, run_lint=False, run_type_check=False
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit module", 1, "run", metadata=metadata))

    arguments = {"path": "module.py"}
    assert (
        bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
        is None
    )
    target.write_text("def value(:\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("edit module", "finish", "done", [], 2, "run", metadata)
    )

    assert block is not None
    assert "rejected completion" in block["reason"]
    assert target.read_text(encoding="utf-8") == original
    assert metadata["edit_transaction_status"] == "rolled_back"


def test_verified_edit_commits_valid_python(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path, run_affected_tests=False, run_lint=False, run_type_check=False
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit module", 1, "run", metadata=metadata))
    arguments = {"path": "module.py"}
    bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
    target.write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("edit module", "finish", "done", [], 2, "run", metadata)
    )

    assert block is None
    assert "return 2" in target.read_text(encoding="utf-8")
    assert metadata["edit_transaction_status"] == "committed"


def test_verified_edit_uses_injected_validation_runner(tmp_path):
    calls: list[tuple[list[str], int]] = []

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        calls.append((arguments, timeout))
        return 0, "1 passed"

    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=runner,
        run_lint=False,
        run_type_check=False,
    )

    result = capability._run_tests(["tests/test_service.py"])

    assert result.passed is True
    assert calls == [(["-m", "pytest", "-q", "tests/test_service.py"], 120)]
