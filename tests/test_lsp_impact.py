from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeFinishEvent,
    BeforeToolCallEvent,
    RunStartEvent,
)
from dm_agent.core.evidence import EvidenceGraph
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
            return [{"severity": 1, "message": "incompatible type", "range": {}}]
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
