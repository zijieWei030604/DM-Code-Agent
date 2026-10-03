"""Run-scoped persistence and presentation for LSP impact reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .analyzer import ImpactAnalyzer, ImpactReport
from .client import LspClient, PyrightLspClient


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

    def current_new_errors(self) -> list[ImpactReport]:
        return [report for report in self.reports if report.new_error_diagnostics]

    def tool_report(self, raw_path: str) -> dict[str, Any]:
        report = self.analyze(raw_path)
        return {"report": report.to_dict(), "summary": self.summary(report)}

    def query(self, raw_path: str, action: str, line: int = 0, character: int = 0) -> dict[str, Any]:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else self.workspace_root / path
        if not self.client.available:
            return {"status": "unavailable", "reason": self.client.unavailable_reason, "items": []}
        try:
            text = resolved.read_text(encoding="utf-8")
            if action == "symbols":
                items = self.client.document_symbols(resolved, text)
            elif action == "references":
                items = self.client.references(resolved, line, character, text)
            elif action == "definition":
                items = self.client.definition(resolved, line, character, text)
            elif action == "diagnostics":
                items = self.client.diagnostics(resolved, text)
            else:
                return {"status": "failed", "reason": f"unsupported LSP query: {action}", "items": []}
            return {"status": "ok", "action": action, "items": items[:200], "truncated": len(items) > 200}
        except (OSError, RuntimeError, TimeoutError, UnicodeError) as error:
            return {"status": "unavailable", "reason": str(error), "items": []}

    def _persist(self, report: ImpactReport) -> None:
        run = self.run_id or "manual"
        destination = self.cache_root / "reports" / run / f"{report.report_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
