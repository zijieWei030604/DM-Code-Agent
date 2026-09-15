from __future__ import annotations

import json

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeLLMRequestEvent,
    BeforeToolCallEvent,
    EventBus,
    RunEndEvent,
    RunStartEvent,
)
from dm_agent.extensions.capabilities import SemanticWorkspaceCapability, VerifiedEditCapability
from dm_agent.tools.structured_edit_tools import edit_python_symbol, inspect_python_symbol
from dm_agent.verification import VerificationPolicy
from dm_agent.workspace import ImpactReport, SemanticWorkspaceEngine


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


def test_impact_graph_uses_semantic_index_as_single_source_of_truth(tmp_path):
    (tmp_path / "service.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n", encoding="utf-8"
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    engine.update()
    tables = {
        str(row["name"])
        for row in engine._connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        )
    }
    report = engine.analyze_impact(["service.py"])

    assert {"files", "symbols", "refs", "impact_edges"}.issubset(tables)
    assert not {"impact_files", "impact_symbols", "impact_refs"} & tables
    assert "consumer.py" in report.affected_files
    engine.close()


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
        "from api import checkout\n\ndef test_checkout():\n    assert checkout([1]) == 1\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    engine.update()
    report = engine.analyze_impact(["service.py"], max_depth=2)

    assert "api.py" in report.affected_files
    assert "tests/test_api.py" in report.affected_files
    assert report.related_tests == ("tests/test_api.py",)
    assert any(item.symbol == "checkout" and item.distance == 1 for item in report.affected_symbols)
    assert any(
        item.symbol == "test_checkout" and item.distance == 2 for item in report.affected_symbols
    )
    assert report.risk_level in {"medium", "high"}
    assert "api.py:checkout" in report.render()
    engine.close()


def test_ambiguous_edges_are_reported_but_do_not_propagate(tmp_path):
    (tmp_path / "first.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    (tmp_path / "caller.py").write_text(
        "def invoke():\n    return run()\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_caller.py").write_text(
        "from caller import invoke\n\ndef test_invoke():\n    assert invoke()\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    engine.update()
    report = engine.analyze_impact(["first.py"], max_depth=2)

    assert any(item.path == "caller.py" for item in report.ambiguous_symbols)
    assert "tests/test_caller.py" not in report.affected_files
    assert "tests/test_caller.py" not in report.related_tests
    engine.close()


def test_test_companion_matching_uses_name_tokens_not_substrings(tmp_path):
    (tmp_path / "user.py").write_text("def load_user():\n    return 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_user_cache.py").write_text("def test_cache():\n    pass\n", encoding="utf-8")
    (tests / "test_superuser.py").write_text("def test_admin():\n    pass\n", encoding="utf-8")
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    engine.update()
    report = engine.analyze_impact(["user.py"])

    assert report.fallback_tests == ("tests/test_user_cache.py",)
    assert report.graph_tests == ()
    assert report.related_tests == ("tests/test_user_cache.py",)
    assert "related_tests:" not in report.render()
    assert "fallback_tests: tests/test_user_cache.py" in report.render()
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


def _impact_edges(engine):
    return {
        tuple(row)
        for row in engine._connection.execute(
            """SELECT source_path, source_symbol, target_path, target_symbol,
                      relation, line, confidence
               FROM impact_edges ORDER BY source_path, source_symbol, target_path, target_symbol"""
        )
    }


def test_semantic_workspace_locally_rebuilds_changed_callers_only(tmp_path):
    service = tmp_path / "service.py"
    caller = tmp_path / "caller.py"
    unrelated = tmp_path / "unrelated.py"
    service.write_text(
        "def old_value():\n    return 1\n\ndef new_value():\n    return 2\n", encoding="utf-8"
    )
    caller.write_text(
        "from service import old_value\n\ndef use_value():\n    return old_value()\n",
        encoding="utf-8",
    )
    unrelated.write_text(
        "from math import ceil\n\ndef round_up(value):\n    return ceil(value)\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()
    before = _impact_edges(engine)
    assert any(edge[0] == "caller.py" and edge[3] == "old_value" for edge in before)

    caller.write_text(
        "from service import new_value\n\ndef use_value():\n    return new_value()\n",
        encoding="utf-8",
    )
    engine.update([caller])
    after = _impact_edges(engine)

    assert not any(edge[0] == "caller.py" and edge[3] == "old_value" for edge in after)
    assert any(edge[0] == "caller.py" and edge[3] == "new_value" for edge in after)
    assert {edge for edge in before if edge[0] == "unrelated.py"} == {
        edge for edge in after if edge[0] == "unrelated.py"
    }
    engine.close()


def test_semantic_workspace_locally_rebuilds_consumers_for_symbol_addition_and_removal(tmp_path):
    service = tmp_path / "service.py"
    service.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import process\n\ndef consume():\n    return process()\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()

    service.write_text("def process():\n    return 1\n", encoding="utf-8")
    engine.update([service])
    added = _impact_edges(engine)
    assert any(edge[0] == "consumer.py" and edge[3] == "process" for edge in added)

    service.write_text("VALUE = 2\n", encoding="utf-8")
    engine.update([service])
    removed = _impact_edges(engine)
    assert not any(edge[0] == "consumer.py" and edge[3] == "process" for edge in removed)
    engine.close()


def test_semantic_workspace_local_edges_match_full_rebuild(tmp_path):
    service = tmp_path / "service.py"
    service.write_text("def old_value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import old_value\n\ndef consume():\n    return old_value()\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()

    service.write_text("def new_value():\n    return 2\n", encoding="utf-8")
    engine.update([service])
    local_edges = _impact_edges(engine)
    engine._impact_graph.update({}, full_refresh=True)

    assert _impact_edges(engine) == local_edges
    engine.close()


def test_semantic_workspace_deduplicates_property_getter_and_setter(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        "class Example:\n"
        "    @property\n"
        "    def value(self):\n"
        "        return self._value\n\n"
        "    @value.setter\n"
        "    def value(self, new_value):\n"
        "        self._value = new_value\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")

    stats = engine.update()
    report = engine.analyze_impact([module])

    assert stats.parse_errors == 0
    assert report.changed_symbols.count("module.py:Example.value") == 1
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


def test_verified_edit_drops_transaction_when_file_returns_to_baseline(tmp_path):
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

    bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
    target.write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )
    target.write_text(original, encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "restored", 2, "run", True, metadata)
    )

    assert metadata["edit_transaction_status"] == "idle"
    assert (
        bus.emit_before_finish(
            BeforeFinishEvent("edit module", "finish", "done", [], 3, "run", metadata)
        )
        is None
    )
    assert metadata["edit_validation_count"] == 0


def test_verified_edit_salvages_valid_patch_at_run_end(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path,
        policy=VerificationPolicy(
            check_lint=False,
            check_types=False,
            run_affected_tests=False,
            finalize_on_run_end=True,
        ),
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {"status": "max_steps_exceeded"}
    bus.emit_run_start(RunStartEvent("edit module", 1, "run", metadata=metadata))
    arguments = {"path": "module.py"}
    bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
    target.write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )

    bus.emit_run_end(RunEndEvent("edit module", 1, "run", {"metadata": metadata}, metadata))

    assert target.read_text(encoding="utf-8").endswith("return 2\n")
    assert metadata["edit_transaction_status"] == "committed_on_run_end"
    assert metadata["edit_run_end_salvaged"] is True
    assert metadata["status"] == "max_steps_exceeded"


def test_verified_edit_rolls_back_invalid_patch_at_run_end(tmp_path):
    target = tmp_path / "module.py"
    original = "def value():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path,
        run_affected_tests=False,
        run_lint=False,
        run_type_check=False,
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {"status": "max_steps_exceeded"}
    bus.emit_run_start(RunStartEvent("edit module", 1, "run", metadata=metadata))
    arguments = {"path": "module.py"}
    bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
    target.write_text("def value(:\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )

    bus.emit_run_end(RunEndEvent("edit module", 1, "run", {"metadata": metadata}, metadata))

    assert target.read_text(encoding="utf-8") == original
    assert metadata["edit_transaction_status"] == "rolled_back"
    assert metadata["edit_run_end_salvaged"] is False


def test_structured_edit_commits_after_verified_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "module.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    inspected = json.loads(inspect_python_symbol({"path": "module.py", "qualified_name": "value"}))
    arguments = {
        "path": "module.py",
        "qualified_name": "value",
        "expected_hash": inspected["source_hash"],
        "operation": "replace_body",
        "content": "return 2",
    }
    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path, run_affected_tests=False, run_lint=False, run_type_check=False
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit symbol", 1, "run", metadata=metadata))

    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_python_symbol", arguments, 1, "run", metadata)
    )
    observation = edit_python_symbol(arguments)
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_python_symbol", arguments, observation, 1, "run", True, metadata)
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("edit symbol", "finish", "done", [], 2, "run", metadata)
    )

    assert block is None
    assert target.read_text(encoding="utf-8") == "def value():\n    return 2\n"
    assert metadata["edit_transaction_status"] == "committed"


def test_structured_edit_rolls_back_when_verified_validation_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "module.py"
    original = "def value():\n    return 1\n"
    target.write_text(original, encoding="utf-8")
    inspected = json.loads(inspect_python_symbol({"path": "module.py", "qualified_name": "value"}))
    arguments = {
        "path": "module.py",
        "qualified_name": "value",
        "expected_hash": inspected["source_hash"],
        "operation": "replace_body",
        "content": "return 2",
    }

    def reject_validation(command: list[str], timeout: int) -> tuple[int, str]:
        return 1, f"rejected: {' '.join(command)} ({timeout}s)"

    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=reject_validation,
        policy=VerificationPolicy(check_runtime_syntax=True),
        run_affected_tests=False,
        run_lint=False,
        run_type_check=False,
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit symbol", 1, "run", metadata=metadata))

    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_python_symbol", arguments, 1, "run", metadata)
    )
    observation = edit_python_symbol(arguments)
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_python_symbol", arguments, observation, 1, "run", True, metadata)
    )
    block = bus.emit_before_finish(
        BeforeFinishEvent("edit symbol", "finish", "done", [], 2, "run", metadata)
    )

    assert block is not None
    assert "runtime_python_syntax" in block["reason"]
    assert target.read_text(encoding="utf-8") == original
    assert metadata["edit_transaction_status"] == "rolled_back"


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

    result = capability.runner.run_tests(["tests/test_service.py"])

    assert result.passed is True
    assert calls == [(["-m", "pytest", "-q", "tests/test_service.py"], 120)]


def test_verified_edit_limits_affected_tests(tmp_path):
    capability = VerifiedEditCapability(
        tmp_path,
        max_affected_tests=2,
        run_affected_tests=True,
        run_lint=False,
        run_type_check=False,
    )
    impact = ImpactReport(
        changed_files=("service.py",),
        changed_symbols=(),
        affected_symbols=(),
        affected_files=(),
        related_tests=("tests/test_a.py", "tests/test_b.py", "tests/test_c.py"),
        risk_score=0.5,
        risk_level="medium",
        reasons=(),
    )

    assert capability._selected_tests(impact) == ["tests/test_a.py", "tests/test_b.py"]


def test_verified_edit_reuses_validation_for_identical_patch(tmp_path):
    target = tmp_path / "module.py"
    original = "value = 1\n"
    changed = "value = 2\n"
    target.write_text(original, encoding="utf-8")
    calls: list[list[str]] = []

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        calls.append(arguments)
        return (1, "rejected") if arguments[:2] == ["-m", "py_compile"] else (0, "ok")

    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=runner,
        policy=VerificationPolicy(check_runtime_syntax=True),
        run_affected_tests=False,
        run_lint=False,
        run_type_check=False,
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit", 1, "run", metadata=metadata))
    arguments = {"path": "module.py"}

    for step in (1, 3):
        bus.emit_before_tool_call(
            BeforeToolCallEvent("edit_file", arguments, step, "run", metadata)
        )
        target.write_text(changed, encoding="utf-8")
        bus.emit_after_tool_result(
            AfterToolResultEvent("edit_file", arguments, "written", step, "run", True, metadata)
        )
        block = bus.emit_before_finish(
            BeforeFinishEvent("edit", "finish", "done", [], step + 1, "run", metadata)
        )
        assert block is not None
        assert target.read_text(encoding="utf-8") == original

    assert calls == [["-m", "py_compile", "module.py"]]
    assert metadata["edit_validation_cache_hit_count"] == 1


def test_verified_edit_mypy_allows_preexisting_error_after_line_shift(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    mypy_calls = 0

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        nonlocal mypy_calls
        if arguments[0] == "-c":
            return 0, ""
        mypy_calls += 1
        line = 926 if mypy_calls == 1 else 932
        return 1, f'module.py:{line}: error: "Legacy" not callable  [misc]'

    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=runner,
        policy=VerificationPolicy(check_types=True, baseline_aware_types=True),
    )
    capability.runner.capture_type_baseline(target)
    target.write_text("\nvalue = 2\n", encoding="utf-8")

    result = capability.runner.validate_mypy_delta(["module.py"])

    assert result.passed is True
    assert "1 pre-existing diagnostic" in result.detail


def test_verified_edit_mypy_rejects_new_error(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    mypy_calls = 0

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        nonlocal mypy_calls
        if arguments[0] == "-c":
            return 0, ""
        mypy_calls += 1
        old = 'module.py:926: error: "Legacy" not callable  [misc]'
        if mypy_calls == 1:
            return 1, old
        return 1, f"{old}\nmodule.py:10: error: Incompatible return value  [return-value]"

    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=runner,
        policy=VerificationPolicy(check_types=True, baseline_aware_types=True),
    )
    capability.runner.capture_type_baseline(target)
    target.write_text("value = 'wrong'\n", encoding="utf-8")

    result = capability.runner.validate_mypy_delta(["module.py"])

    assert result.passed is False
    assert "Incompatible return value" in result.detail
    assert "Legacy" not in result.detail


def test_verified_edit_uses_posix_paths_for_container_validation(tmp_path):
    target = tmp_path / "package" / "module.py"
    target.parent.mkdir()
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        calls.append(arguments)
        return 0, f"completed in {timeout}s"

    capability = VerifiedEditCapability(
        tmp_path,
        command_runner=runner,
        policy=VerificationPolicy(check_runtime_syntax=True),
        run_affected_tests=False,
        run_lint=False,
        run_type_check=False,
    )
    capability._changed.add(target)

    results = list(capability.runner.verify(capability._changed).results)

    assert all(result.passed for result in results)
    assert ["-m", "py_compile", "package/module.py"] in calls


def test_semantic_capability_injects_bounded_impact_once_after_change(tmp_path):
    (tmp_path / "service.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n", encoding="utf-8"
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    bus = EventBus()
    recorded: list[tuple[str, dict[str, object]]] = []

    class TraceWriter:
        def record(self, event: str, payload: dict[str, object]) -> None:
            recorded.append((event, payload))

    capability = SemanticWorkspaceCapability(engine)
    capability.install(CapabilityContext(bus, lambda phase: None, TraceWriter()))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("change value", 1, "run", metadata=metadata))

    (tmp_path / "service.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file", {"path": "service.py"}, "written", 1, "run", True, metadata
        )
    )
    messages = [{"role": "user", "content": "continue\n\n<repository_map></repository_map>"}]
    bus.emit_before_llm_request(BeforeLLMRequestEvent(messages, 2, "run", "agent", metadata))

    assert messages[0]["content"] == "continue\n\n<repository_map></repository_map>"
    assert len(messages) == 2
    assert "<change_impact>" in messages[1]["content"]
    assert "service.py" in messages[1]["content"]
    assert "consumer.py" in messages[1]["content"]
    assert "likely_affected: consumer.py" in messages[1]["content"]
    assert "ambiguous_candidates: none" in messages[1]["content"]
    assert metadata["semantic_impact_injection_count"] == 1
    injected = next(payload for event, payload in recorded if event == "semantic_impact_injected")
    assert injected["content"] == messages[1]["content"]
    assert injected["chars"] == len(messages[1]["content"])
    assert len(str(injected["sha256"])) == 64

    next_messages = [{"role": "user", "content": "continue again"}]
    bus.emit_before_llm_request(BeforeLLMRequestEvent(next_messages, 3, "run", "agent", metadata))
    assert next_messages == [{"role": "user", "content": "continue again"}]
    assert metadata["semantic_impact_files"] == 1
    engine.close()


def test_verified_edit_selects_graph_related_tests(tmp_path):
    target = tmp_path / "service.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_consumer.py").write_text(
        "from service import value\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()
    calls: list[list[str]] = []

    def runner(arguments: list[str], timeout: int) -> tuple[int, str]:
        calls.append(arguments)
        return 0, "passed"

    bus = EventBus()
    capability = VerifiedEditCapability(
        tmp_path,
        engine=engine,
        command_runner=runner,
        run_affected_tests=True,
        run_lint=False,
        run_type_check=False,
    )
    capability.install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("edit service", 1, "run", metadata=metadata))
    arguments = {"path": "service.py"}
    bus.emit_before_tool_call(BeforeToolCallEvent("edit_file", arguments, 1, "run", metadata))
    target.write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent("edit_file", arguments, "written", 1, "run", True, metadata)
    )

    block = bus.emit_before_finish(
        BeforeFinishEvent("edit service", "finish", "done", [], 2, "run", metadata)
    )

    assert block is None
    assert ["-m", "pytest", "-q", "tests/test_consumer.py"] in calls
    assert metadata["edit_impact_tests"] == 1
    engine.close()


def test_impact_graph_keeps_callers_visible_when_symbol_is_deleted(tmp_path):
    target = tmp_path / "service.py"
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n", encoding="utf-8"
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    engine.update()

    target.write_text("VALUE = 1\n", encoding="utf-8")
    engine.update([target])
    impact = engine.analyze_impact([target])

    assert "consumer.py" in impact.affected_files
    assert any("consumer.py:consume" in reason for reason in impact.reasons)
    engine.close()


def test_semantic_capability_keeps_replanner_context_model_driven(tmp_path):
    (tmp_path / "service.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n", encoding="utf-8"
    )
    engine = SemanticWorkspaceEngine(tmp_path, database_path=tmp_path / "index.db")
    bus = EventBus()
    SemanticWorkspaceCapability(engine).install(CapabilityContext(bus, lambda phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("change value", 1, "run", metadata=metadata))
    (tmp_path / "service.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file", {"path": "service.py"}, "written", 1, "run", True, metadata
        )
    )
    messages = [{"role": "user", "content": "Replan after failed test"}]

    bus.emit_before_llm_request(BeforeLLMRequestEvent(messages, 2, "run", "planner", metadata))

    assert messages[0]["content"] == "Replan after failed test"
    assert metadata["semantic_impact_enabled"] is True
    engine.close()
