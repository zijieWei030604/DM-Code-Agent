"""Run-scoped persistence and presentation for LSP impact reports."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .analyzer import ImpactAnalyzer, ImpactReport
from .client import LspClient, PyrightLspClient, path_to_uri

REFERENCE_CONTEXT_LIMIT = 50
REFERENCE_RESULT_LIMIT = 200
REFERENCE_RETRY_COUNT = 2
REFERENCE_RETRY_DELAY_SECONDS = 0.25


@dataclass(frozen=True)
class LspCompletionState:
    """Reports that do and do not apply to the workspace at completion."""

    effective_reports: tuple[ImpactReport, ...]
    stale_reports: tuple[ImpactReport, ...]
    unavailable_reports: tuple[ImpactReport, ...]


class LspImpactService:
    """Keeps snapshots outside the target repository and saves report artifacts."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        command: str = "pyright-langserver",
        timeout_seconds: float = 5.0,
        client: LspClient | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.client = client or PyrightLspClient(command, timeout_seconds)
        self.analyzer = ImpactAnalyzer(self.client, self.workspace_root)
        self.run_id = ""
        self.reports: list[ImpactReport] = []

    @property
    def cache_root(self) -> Path:
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        digest = hashlib.sha256(str(self.workspace_root).encode("utf-8")).hexdigest()[:16]
        return local / "dm-code-agent" / "lsp-impact" / digest

    def start(self, run_id: str) -> bool:
        self.run_id = run_id
        self.reports.clear()
        self.analyzer.reset()
        return self.client.start(self.workspace_root)

    def close(self) -> None:
        self.client.close()

    def snapshot(self, raw_path: str) -> None:
        try:
            self.analyzer.snapshot(Path(raw_path))
        except (OSError, RuntimeError, TimeoutError):
            return

    def analyze(self, raw_path: str) -> ImpactReport:
        report = self.analyzer.analyze(Path(raw_path))
        self.reports.append(report)
        self._persist(report)
        return report

    def summary(self, report: ImpactReport) -> str:
        if report.status == "ok":
            details = ", ".join(report.candidate_files) or "no cross-file candidates"
            errors = len(report.new_error_diagnostics)
            suffix = " (reference limit reached)" if report.truncated else ""
            return (
                f"LSP impact for {report.path}: changed symbols={report.changed_symbols or ['unknown']}; "
                f"candidates={details}; new Error diagnostics={errors}.{suffix}"
            )
        return f"LSP impact {report.status} for {report.path}: {report.reason or 'no report'}"

    def completion_state(self) -> LspCompletionState:
        """Select only the latest, current-content report for every changed path."""
        latest_by_path: dict[str, ImpactReport] = {}
        for report in self.reports:
            latest_by_path[report.path] = report
        effective: list[ImpactReport] = []
        stale: list[ImpactReport] = []
        unavailable: list[ImpactReport] = []
        for report in latest_by_path.values():
            if report.status != "ok":
                unavailable.append(report)
            elif report.after_hash and report.after_hash == self._current_hash(report.path):
                effective.append(report)
            else:
                stale.append(report)
        return LspCompletionState(tuple(effective), tuple(stale), tuple(unavailable))

    def current_new_errors(self) -> list[ImpactReport]:
        return [
            report
            for report in self.completion_state().effective_reports
            if report.new_error_diagnostics
        ]

    def _current_hash(self, raw_path: str) -> str:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else self.workspace_root / path
        try:
            text = resolved.read_text(encoding="utf-8")
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
        except (OSError, UnicodeError):
            return ""

    def tool_report(self, raw_path: str) -> dict[str, Any]:
        report = self.analyze(raw_path)
        return {"report": report.to_dict(), "summary": self.summary(report)}

    def query(
        self, raw_path: str, action: str, line: int = 0, character: int = 0
    ) -> dict[str, Any]:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else self.workspace_root / path
        if not self.client.available:
            return {"status": "unavailable", "reason": self.client.unavailable_reason, "items": []}
        try:
            text = resolved.read_text(encoding="utf-8")
            if action == "symbols":
                items = self.client.document_symbols(resolved, text)
            elif action == "references":
                items = self._references_with_retry(resolved, line, character, text)
            elif action == "definition":
                items = self.client.definition(resolved, line, character, text)
            elif action == "diagnostics":
                items = self.client.diagnostics(resolved, text)
            else:
                return {
                    "status": "failed",
                    "reason": f"unsupported LSP query: {action}",
                    "items": [],
                }
            if action == "references":
                return self._render_references(items)
            return {
                "status": "ok",
                "action": action,
                "items": items[:REFERENCE_RESULT_LIMIT],
                "truncated": len(items) > REFERENCE_RESULT_LIMIT,
            }
        except (OSError, RuntimeError, TimeoutError, UnicodeError) as error:
            return {"status": "unavailable", "reason": str(error), "items": []}

    def _references_with_retry(
        self, path: Path, line: int, character: int, text: str
    ) -> list[dict[str, Any]]:
        """Retry a cold LSP index when it reports no use sites beyond the declaration."""
        references = self.client.references(path, line, character, text)
        for _ in range(REFERENCE_RETRY_COUNT):
            if not _empty_or_declaration_only(references, path, line):
                break
            time.sleep(REFERENCE_RETRY_DELAY_SECONDS)
            references = self.client.references(path, line, character, text)
        return references

    def _render_references(self, references: list[dict[str, Any]]) -> dict[str, Any]:
        """Give early references context while keeping the remainder cheap for the model."""
        visible = references[:REFERENCE_RESULT_LIMIT]
        contextual_count = min(len(visible), REFERENCE_CONTEXT_LIMIT)
        items = [
            _reference_for_model(
                reference, self.workspace_root, include_context=index < contextual_count
            )
            for index, reference in enumerate(visible)
        ]
        return {
            "status": "ok",
            "action": "references",
            "items": items,
            "total": len(references),
            "contextual_items": contextual_count,
            "location_only_items": len(visible) - contextual_count,
            "truncated": len(references) > REFERENCE_RESULT_LIMIT,
        }

    def _persist(self, report: ImpactReport) -> None:
        run = self.run_id or "manual"
        destination = self.cache_root / "reports" / run / f"{report.report_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)


def _empty_or_declaration_only(references: list[dict[str, Any]], path: Path, line: int) -> bool:
    if not references:
        return True
    target_uri = path_to_uri(path)
    for item in references:
        if str(item.get("uri", "")) != target_uri:
            return False
        range_data = item.get("range")
        start = range_data.get("start") if isinstance(range_data, dict) else None
        if not isinstance(start, dict) or int(start.get("line", -1)) != line:
            return False
    return True


def _reference_for_model(
    reference: dict[str, Any], workspace_root: Path, *, include_context: bool
) -> dict[str, Any]:
    item: dict[str, Any] = {"uri": reference.get("uri", ""), "range": reference.get("range", {})}
    if include_context:
        context = _reference_context(reference, workspace_root)
        if context is not None:
            item["context"] = context
    return item


def _reference_context(reference: dict[str, Any], workspace_root: Path) -> dict[str, Any] | None:
    uri = reference.get("uri")
    if not isinstance(uri, str):
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = Path(unquote(parsed.path.lstrip("/"))).resolve()
    try:
        relative = path.relative_to(workspace_root.resolve())
    except ValueError:
        return None
    range_data = reference.get("range")
    start = range_data.get("start") if isinstance(range_data, dict) else None
    if not isinstance(start, dict) or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    line_number = max(0, int(start.get("line", 0)))
    first = max(0, line_number - 2)
    last = min(len(lines), line_number + 3)
    return {
        "path": relative.as_posix(),
        "start_line": first + 1,
        "end_line": last,
        "source": "\n".join(lines[first:last]),
    }
