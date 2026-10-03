from __future__ import annotations

from pathlib import Path
from typing import Any

from dm_agent.core.events import AfterToolResultEvent, BeforeFinishEvent, BeforeToolCallEvent, RunStartEvent
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.extensions.capabilities.lsp_impact import LspImpactCapability
from dm_agent.lsp_impact.analyzer import ImpactAnalyzer
from dm_agent.lsp_impact.client import path_to_uri
from dm_agent.lsp_impact.service import LspImpactService
from dm_agent.tools.base import ToolResult


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

    def references(
        self, path: Path, line: int, character: int, text: str
    ) -> list[dict[str, Any]]:
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
    conclusion, conclusion_edges = graph.add_conclusion(text="done", step_number=2)

    assert observations[0].metadata["kind"] == "lsp_impact"
    assert (changes[0].node_id, observations[0].node_id, "derived_from") in {
        (edge.source_id, edge.target_id, edge.relation) for edge in edges
    }
    assert (conclusion[0].node_id, observations[0].node_id, "checked_at_completion") in {
        (edge.source_id, edge.target_id, edge.relation) for edge in conclusion_edges
    }
