from __future__ import annotations

from pathlib import Path
from typing import Any

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
from dm_agent.extensions.capabilities.evidence import EvidenceGraphCapability
from dm_agent.extensions.capabilities.lsp_impact import LspImpactCapability
from dm_agent.lsp_impact.analyzer import ImpactAnalyzer
from dm_agent.lsp_impact.client import PyrightLspClient, _document_key, path_to_uri
from dm_agent.lsp_impact.service import LspImpactService
from dm_agent.tools.base import ToolResult


def test_document_change_sends_top_level_content_and_registers_wait_first(tmp_path: Path) -> None:
    class RecordingClient(PyrightLspClient):
        def _notify(self, method: str, params: dict[str, Any]) -> None:
            document = params["textDocument"]
            assert self._pending_diagnostics[_document_key(document["uri"])] == document["version"]
            messages.append((method, params))

    messages: list[tuple[str, dict[str, Any]]] = []
    client = RecordingClient()
    client.available = True
    path = tmp_path / "source.py"
    client._sync(path, "old")
    client._sync(path, "new")
    method, params = messages[-1]
    assert method == "textDocument/didChange"
    assert params == {
        "textDocument": {"uri": path_to_uri(path), "version": 2},
        "contentChanges": [{"text": "new"}],
    }


def test_missing_diagnostics_is_not_reported_as_clean(tmp_path: Path) -> None:
    client = PyrightLspClient(timeout_seconds=0.01)
    client.available = True
    with pytest.raises(TimeoutError, match="diagnostics timed out"):
        client.diagnostics(tmp_path / "source.py", "broken")


class FakeLspClient:
    available = True
    unavailable_reason = ""

    def __init__(self, *, error_after: bool = False) -> None:
        self.error_after = error_after

    def start(self, workspace_root: Path) -> bool:
        return True

    def close(self) -> None:
        return None

    def document_symbols(self, path: Path, text: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "normalize",
                "kind": 12,
                "range": {"start": {"line": 0}, "end": {"line": 2}},
                "selectionRange": {"start": {"line": 0, "character": 4}},
            }
        ]

    def references(self, path: Path, line: int, character: int, text: str) -> list[dict[str, Any]]:
        return [{"uri": path_to_uri(path.parent / "consumer.py")}]

    def definition(self, path: Path, line: int, character: int, text: str) -> list[dict[str, Any]]:
        return [{"uri": path_to_uri(path)}]

    def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]:
        if self.error_after and "broken" in text:
            return [
                {
                    "severity": 1,
                    "code": "reportUndefinedVariable",
                    "message": "broken is not defined",
                    "range": {"start": {"line": 1}, "end": {"line": 1}},
                }
            ]
        return []


def test_analyzer_reports_cross_file_reference_and_new_error(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text("from source import normalize\n", encoding="utf-8")
    analyzer = ImpactAnalyzer(FakeLspClient(error_after=True), tmp_path)

    analyzer.snapshot(source)
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    report = analyzer.analyze(source)

    assert report.status == "ok"
    assert report.changed_symbols == ["normalize"]
    assert report.candidate_files == ["consumer.py"]
    assert len(report.new_error_diagnostics) == 1


def test_analyzer_ignores_unchanged_diagnostic_after_line_shift(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")

    class MovingDiagnosticClient(FakeLspClient):
        def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]:
            line = 11 if text.startswith("# comment") else 10
            return [
                {
                    "severity": 1,
                    "code": "reportOptionalMemberAccess",
                    "message": "legacy optional access",
                    "source": "Pyright",
                    "range": {"start": {"line": line}, "end": {"line": line}},
                }
            ]

    analyzer = ImpactAnalyzer(MovingDiagnosticClient(), tmp_path)
    analyzer.snapshot(source)
    source.write_text("# comment\ndef normalize(value):\n    return value\n", encoding="utf-8")
    report = analyzer.analyze(source)

    assert report.new_error_diagnostics == []
    assert report.blocking_error_diagnostics == []


def test_analyzer_keeps_dynamic_type_warning_out_of_completion_block(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")

    class DynamicTypeClient(FakeLspClient):
        def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]:
            return (
                [
                    {
                        "severity": 1,
                        "code": "reportCallIssue",
                        "message": "dynamic helper is not callable",
                        "source": "Pyright",
                        "range": {"start": {"line": 1}, "end": {"line": 1}},
                    }
                ]
                if "dynamic_helper" in text
                else []
            )

    analyzer = ImpactAnalyzer(DynamicTypeClient(), tmp_path)
    analyzer.snapshot(source)
    source.write_text("def normalize(value):\n    return dynamic_helper(value)\n", encoding="utf-8")
    report = analyzer.analyze(source)

    assert len(report.new_error_diagnostics) == 1
    assert report.blocking_error_diagnostics == []
    assert report.uncertain_error_diagnostics == report.new_error_diagnostics


def test_analyzer_retries_when_a_cold_index_returns_only_the_declaration(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    consumer = tmp_path / "consumer.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    consumer.write_text("from source import normalize\n", encoding="utf-8")

    class ColdIndexClient(FakeLspClient):
        def __init__(self) -> None:
            super().__init__()
            self.reference_calls = 0

        def references(
            self, path: Path, line: int, character: int, text: str
        ) -> list[dict[str, Any]]:
            self.reference_calls += 1
            if self.reference_calls == 1:
                return [{"uri": path_to_uri(path), "range": {"start": {"line": line}}}]
            return [{"uri": path_to_uri(consumer)}]

    client = ColdIndexClient()
    analyzer = ImpactAnalyzer(client, tmp_path)
    analyzer.snapshot(source)
    source.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")

    report = analyzer.analyze(source)

    assert client.reference_calls == 2
    assert report.candidate_files == ["consumer.py"]


def test_capability_appends_summary_and_blocks_new_error(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient(error_after=True))
    capability = LspImpactCapability(service)
    start = RunStartEvent(task="fix", attempt=1, run_id="run-1")
    capability._on_run_start(start)
    capability._before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": str(source)}, 1, "run-1")
    )
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    event = AfterToolResultEvent(
        "edit_file",
        {"path": str(source)},
        "edited",
        1,
        "run-1",
        True,
        result=ToolResult("success", "edited", changed_files=(str(source),)),
    )
    observation = capability._after_tool_result(event)

    assert "LSP impact" in observation
    assert event.metadata["lsp_impact_reports"][0]["path"] == "source.py"
    decision = capability._before_finish(
        BeforeFinishEvent("fix", "task_complete", "done", [], 2, "run-1")
    )
    assert decision is not None
    assert decision["block"] is True


def test_lsp_failure_is_recorded_by_evidence_before_completion_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient(error_after=True))
    lsp = LspImpactCapability(service)
    evidence = EvidenceGraphCapability()
    bus = EventBus()
    context = CapabilityContext(bus, lambda phase: None)
    lsp.install(context)
    evidence.install(context)
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("fix", 1, "run", metadata=metadata))
    event = BeforeToolCallEvent("edit_file", {"path": "source.py"}, 1, "run", metadata)
    bus.emit_before_tool_call(event)
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "source.py"},
            "edited",
            1,
            "run",
            True,
            metadata,
            result=ToolResult("success", "edited", changed_files=("source.py",)),
        )
    )

    assert bus.emit_before_finish(
        BeforeFinishEvent("fix", "finish", "done", [], 2, "run", metadata)
    )
    assert any(
        node.kind == "verification" and node.metadata.get("tool") == "lsp_impact"
        for node in evidence.graph.nodes.values()
    )
    assert bus.emit_before_finish(
        BeforeFinishEvent("fix", "finish", "done", [], 3, "run", metadata)
    )
    assert metadata["evidence_terminal_completion_rejection"]


def test_evidence_stops_blocking_after_lsp_error_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior LSP failure must not be projected onto a later clean version."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient(error_after=True))
    lsp = LspImpactCapability(service)
    evidence = EvidenceGraphCapability()
    bus = EventBus()
    context = CapabilityContext(bus, lambda phase: None)
    lsp.install(context)
    evidence.install(context)
    metadata: dict[str, Any] = {}
    bus.emit_run_start(RunStartEvent("fix", 1, "run", metadata=metadata))

    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": "source.py"}, 1, "run", metadata)
    )
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "source.py"},
            "edited",
            1,
            "run",
            True,
            metadata,
            result=ToolResult("success", "edited", changed_files=("source.py",)),
        )
    )
    assert bus.emit_before_finish(
        BeforeFinishEvent("fix", "finish", "broken", [], 2, "run", metadata)
    )
    assert metadata["lsp_impact_confirmed_failure"]

    bus.emit_before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": "source.py"}, 3, "run", metadata)
    )
    source.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
    bus.emit_after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": "source.py"},
            "edited",
            3,
            "run",
            True,
            metadata,
            result=ToolResult("success", "edited", changed_files=("source.py",)),
        )
    )

    # Strict evidence pauses once for missing verification, but the repaired
    # LSP error is no longer an explicit contradiction.
    assert bus.emit_before_finish(
        BeforeFinishEvent("fix", "finish", "repaired", [], 4, "run", metadata)
    )
    assert "lsp_impact_confirmed_failure" not in metadata
    assert all(
        item["status"] != "confirmed_code_error" for item in metadata["evidence_policy_issues"]
    )
    assert (
        bus.emit_before_finish(
            BeforeFinishEvent("fix", "finish", "repaired", [], 5, "run", metadata)
        )
        is None
    )


def test_repaired_latest_report_replaces_old_error_for_completion(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient(error_after=True))
    capability = LspImpactCapability(service)
    capability._on_run_start(RunStartEvent(task="fix", attempt=1, run_id="run-1"))

    capability._before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": str(source)}, 1, "run-1")
    )
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    capability._after_tool_result(
        AfterToolResultEvent(
            "edit_file",
            {"path": str(source)},
            "edited",
            1,
            "run-1",
            True,
            result=ToolResult("success", "edited", changed_files=(str(source),)),
        )
    )
    assert service.current_new_errors()

    capability._before_tool_call(
        BeforeToolCallEvent("edit_file", {"path": str(source)}, 2, "run-1")
    )
    source.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
    repaired = AfterToolResultEvent(
        "edit_file",
        {"path": str(source)},
        "edited",
        2,
        "run-1",
        True,
        result=ToolResult("success", "edited", changed_files=(str(source),)),
    )
    capability._after_tool_result(repaired)
    decision = capability._before_finish(
        BeforeFinishEvent("fix", "task_complete", "done", [], 3, "run-1")
    )

    assert decision is None
    assert not service.current_new_errors()
    assert repaired.metadata["lsp_impact_reports"][0]["report_id"] in {
        report.report_id for report in service.completion_state().effective_reports
    }


def test_lsp_uses_the_first_task_snapshot_as_its_error_baseline(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient(error_after=True))
    assert service.start("run")
    service.snapshot("source.py")
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    first = service.analyze("source.py")
    assert first.blocking_error_diagnostics

    # A second edit must not turn the first task-introduced error into baseline.
    service.snapshot("source.py")
    source.write_text("def normalize(value):\n    return broken(value).strip()\n", encoding="utf-8")
    second = service.analyze("source.py")
    assert second.blocking_error_diagnostics


def test_lsp_marks_dependency_resolution_errors_as_uncertain(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")

    class DependencyErrorClient(FakeLspClient):
        def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]:
            return (
                [{"severity": 1, "code": "reportMissingImports", "message": "missing", "range": {}}]
                if "broken" in text
                else []
            )

    analyzer = ImpactAnalyzer(DependencyErrorClient(), tmp_path)
    analyzer.snapshot(source)
    source.write_text("def normalize(value):\n    return broken(value)\n", encoding="utf-8")
    report = analyzer.analyze(source)
    assert report.new_error_diagnostics
    assert not report.blocking_error_diagnostics
    assert report.uncertain_error_diagnostics


def test_report_becomes_stale_after_unanalyzed_later_edit(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    service = LspImpactService(tmp_path, client=FakeLspClient())
    assert service.start("run")
    service.snapshot("source.py")
    source.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
    report = service.analyze("source.py")
    assert service.completion_state().effective_reports == (report,)

    source.write_text("def normalize(value):\n    return value.lower()\n", encoding="utf-8")
    state = service.completion_state()

    assert state.effective_reports == ()
    assert state.stale_reports == (report,)


def test_evidence_graph_links_lsp_observation_to_change_and_conclusion() -> None:
    graph = EvidenceGraph("fix")
    changes, _ = graph.add_change(tool="edit_file", path="source.py", step_number=1)
    observations, edges = graph.add_lsp_impact(
        report={
            "report_id": "report-1",
            "path": "source.py",
            "status": "ok",
            "candidate_files": ["consumer.py"],
            "new_error_diagnostics": [],
        },
        step_number=1,
    )
    later, _ = graph.add_lsp_impact(
        report={
            "report_id": "report-2",
            "path": "source.py",
            "status": "ok",
            "candidate_files": [],
            "new_error_diagnostics": [],
        },
        step_number=2,
    )
    conclusion, conclusion_edges = graph.add_conclusion(
        text="done", step_number=3, lsp_report_ids=("report-2",)
    )

    assert observations[0].metadata["kind"] == "lsp_impact"
    assert (changes[0].node_id, observations[0].node_id, "derived_from") in {
        (edge.source_id, edge.target_id, edge.relation) for edge in edges
    }
    assert (conclusion[0].node_id, later[0].node_id, "checked_at_completion") in {
        (edge.source_id, edge.target_id, edge.relation) for edge in conclusion_edges
    }
    assert (conclusion[0].node_id, observations[0].node_id, "checked_at_completion") not in {
        (edge.source_id, edge.target_id, edge.relation) for edge in conclusion_edges
    }


def test_reference_query_retries_cold_index_and_limits_context_to_first_fifty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    consumer = tmp_path / "consumer.py"
    source.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
    consumer.write_text("\n".join(f"normalize({index})" for index in range(51)), encoding="utf-8")

    class ColdReferencesClient(FakeLspClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def references(
            self, path: Path, line: int, character: int, text: str
        ) -> list[dict[str, Any]]:
            self.calls += 1
            if self.calls == 1:
                return []
            return [
                {
                    "uri": path_to_uri(consumer),
                    "range": {"start": {"line": index, "character": 0}},
                }
                for index in range(51)
            ]

    client = ColdReferencesClient()
    service = LspImpactService(tmp_path, client=client)
    assert service.start("run") is True

    result = service.query("source.py", "references", 0, 4)

    assert client.calls == 2
    assert result["total"] == 51
    assert result["contextual_items"] == 50
    assert result["location_only_items"] == 1
    assert "context" in result["items"][0]
    assert "context" not in result["items"][50]


def _impact_symbol(name, kind, first, last, children=()):
    return {
        "name": name,
        "kind": kind,
        "range": {
            "start": {"line": first, "character": 0},
            "end": {"line": last + 1, "character": 0},
        },
        "selectionRange": {"start": {"line": first, "character": 4}},
        "children": list(children),
    }


@pytest.mark.parametrize(
    ("replacements", "expected_names", "expected_positions", "unmapped"),
    [
        ([("discount = 0.9", "discount = 0.8")], ["total"], [2], []),
        ([("value = 1", "value = 2")], ["helper"], [4], []),
        ([("class Calculator:", "class Calculator(Base):")], ["Calculator"], [0], []),
        ([("limit = 3", "limit = 4")], ["limit"], [1], []),
        ([("MAX_RETRIES = 3", "MAX_RETRIES = 5")], ["MAX_RETRIES"], [10], []),
        ([("local = 1", "local = 2")], ["standalone"], [11], []),
        (
            [("discount = 0.9", "discount = 0.8"), ("return 0", "return 1")],
            ["total", "reset"],
            [2, 8],
            [],
        ),
        (
            [("class Calculator:", "class Calculator(Base):"), ("return 0", "return 1")],
            ["Calculator", "reset"],
            [0, 8],
            [],
        ),
        ([("# trailing comment", "# changed comment")], [], [], [15]),
        ([], [], [], []),
    ],
)
def test_impact_queries_scope_owners_instead_of_local_variables(
    tmp_path, replacements, expected_names, expected_positions, unmapped
):
    source = tmp_path / "source.py"
    before = (
        "class Calculator:\n"
        "    limit = 3\n"
        "    def total(self, price):\n"
        "        discount = 0.9\n"
        "        def helper():\n"
        "            value = 1\n"
        "            return value\n"
        "        return price * discount * helper()\n"
        "    def reset(self):\n"
        "        return 0\n"
        "MAX_RETRIES = 3\n"
        "def standalone():\n"
        "    local = 1\n"
        "    return local\n"
        "# trailing comment\n"
    )

    class HierarchicalClient(FakeLspClient):
        def __init__(self):
            super().__init__()
            self.positions = []

        def document_symbols(self, path, text):
            return [
                _impact_symbol(
                    "Calculator",
                    5,
                    0,
                    9,
                    [
                        _impact_symbol("limit", 8, 1, 1),
                        _impact_symbol(
                            "total",
                            6,
                            2,
                            7,
                            [
                                _impact_symbol("discount", 13, 3, 3),
                                _impact_symbol(
                                    "helper",
                                    12,
                                    4,
                                    6,
                                    [_impact_symbol("value", 13, 5, 5)],
                                ),
                            ],
                        ),
                        _impact_symbol("reset", 6, 8, 9),
                    ],
                ),
                _impact_symbol("MAX_RETRIES", 14, 10, 10),
                _impact_symbol(
                    "standalone",
                    12,
                    11,
                    13,
                    [_impact_symbol("local", 13, 12, 12)],
                ),
            ]

        def references(self, path, line, character, text):
            self.positions.append(line)
            return [{"uri": path_to_uri(tmp_path / f"consumer_{line}.py")}]

    source.write_text(before, encoding="utf-8")
    client = HierarchicalClient()
    analyzer = ImpactAnalyzer(client, tmp_path)
    analyzer.snapshot(source)
    after = before
    for old, new in replacements:
        after = after.replace(old, new)
    source.write_text(after, encoding="utf-8")
    report = analyzer.analyze(source)
    assert report.status == "ok"
    assert report.changed_symbols == expected_names
    assert client.positions == expected_positions
    assert report.unmapped_changed_lines == unmapped
    assert report.candidate_files == sorted(f"consumer_{line}.py" for line in expected_positions)
    if unmapped:
        summary = LspImpactService(tmp_path, client=client).summary(report)
        assert "Impact scope incomplete" in summary
        assert "no cross-file candidates" not in summary


@pytest.mark.parametrize("member_kind", [7, 8, 13, 14])
def test_class_data_inside_a_local_class_is_not_promoted_to_outer_function(member_kind):
    from dm_agent.lsp_impact.analyzer import _changed_symbols

    member = _impact_symbol("member", member_kind, 2, 2)
    symbols = [_impact_symbol("outer", 12, 0, 5, [_impact_symbol("Local", 5, 1, 3, [member])])]
    selected, unmapped = _changed_symbols(symbols, {2})
    assert selected == [member]
    assert unmapped == []


def test_impact_falls_back_to_class_only_when_the_changed_line_is_in_its_range():
    from dm_agent.lsp_impact.analyzer import _changed_symbols

    symbol = _impact_symbol("Calculator", 5, 0, 2)
    selected, unmapped = _changed_symbols([symbol], {1, 3})
    assert selected == [symbol]
    assert unmapped == [4]


def test_unknown_symbol_kind_is_reported_as_unmapped():
    from dm_agent.lsp_impact.analyzer import _changed_symbols

    selected, unmapped = _changed_symbols([_impact_symbol("unknown", 999, 0, 1)], {1})
    assert selected == []
    assert unmapped == [2]
