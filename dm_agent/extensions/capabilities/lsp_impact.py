"""Lifecycle extension that projects LSP impact facts into the Agent run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, BeforeFinishEvent, BeforeToolCallEvent, RunEndEvent, RunStartEvent
from dm_agent.core.workspace_version import workspace_files
from dm_agent.lsp_impact.service import LspImpactService


class LspImpactCapability:
    """Snapshot before Python writes and analyze their current LSP impact afterwards."""

    checkpoint_key = "lsp_impact"

    def __init__(self, service: LspImpactService) -> None:
        self.service = service
        self._trace_writer: Any | None = None
        self._shell_snapshots: dict[int, dict[str, str]] = {}

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        context.event_bus.on("on_run_start", self._on_run_start, name="lsp-impact.run-start")
        context.event_bus.on("before_tool_call", self._before_tool_call, name="lsp-impact.snapshot", kind="observer")
        context.event_bus.on("after_tool_result", self._after_tool_result, name="lsp-impact.after-write")
        context.event_bus.on("before_finish", self._before_finish, name="lsp-impact.before-finish", kind="policy")
        context.event_bus.on("on_run_end", self._on_run_end, name="lsp-impact.run-end", kind="observer")

    def export_state(self) -> dict[str, Any]:
        return {"report_ids": [report.report_id for report in self.service.reports]}

    def restore_state(self, state: dict[str, Any]) -> None:
        # Snapshots are intentionally ephemeral and external to checkpoints.
        return None

    def _on_run_start(self, event: RunStartEvent) -> str:
        started = self.service.start(event.run_id)
        event.metadata.update(
            {
                "lsp_impact_enabled": True,
                "lsp_impact_available": started,
                "lsp_impact_status": "ready" if started else "unavailable",
                "lsp_impact_reason": self.service.client.unavailable_reason,
            }
        )
        self._shell_snapshots.clear()
        self._record("lsp_impact_started", {"available": started, "reason": self.service.client.unavailable_reason})
        return ""

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        path = _write_path(event.tool_name, event.arguments)
        if path:
            self.service.snapshot(path)
        if event.tool_name in {"run_shell", "run_python"}:
            self._shell_snapshots[event.step_number] = workspace_files(self.service.workspace_root)

    def _after_tool_result(self, event: AfterToolResultEvent) -> str:
        paths = tuple(event.result.changed_files) if event.result else ()
        before = self._shell_snapshots.pop(event.step_number, None)
        if before is not None:
            after = workspace_files(self.service.workspace_root)
            paths = tuple(
                sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
            )
        python_paths = [path for path in paths if Path(path).suffix == ".py"]
        if not python_paths or event.no_change:
            return event.observation
        summaries: list[str] = []
        reports: list[dict[str, Any]] = []
        for path in python_paths:
            report = self.service.analyze(path)
            summaries.append(self.service.summary(report))
            reports.append(report.to_dict())
            self._record("lsp_impact_report", report.to_dict())
        event.metadata["lsp_impact_reports"] = reports
        event.metadata["lsp_impact_latest_status"] = reports[-1]["status"]
        return event.observation + "\n\n" + "\n".join(summaries)

    def _before_finish(self, event: BeforeFinishEvent) -> dict[str, Any] | None:
        failures = self.service.current_new_errors()
        event.metadata["lsp_impact_report_count"] = len(self.service.reports)
        event.metadata["lsp_impact_new_error_count"] = sum(
            len(report.new_error_diagnostics) for report in failures
        )
        if not failures:
            return None
        paths = ", ".join(report.path for report in failures)
        reason = f"Completion blocked: LSP found new Error diagnostics in {paths}. Fix them or inspect the diagnostics before finishing."
        event.metadata["lsp_impact_completion_blocked"] = True
        self._record("lsp_impact_completion_blocked", {"paths": paths, "step_number": event.step_number})
        return {"block": True, "reason": reason}

    def _on_run_end(self, event: RunEndEvent) -> None:
        event.metadata["lsp_impact_report_count"] = len(self.service.reports)
        self._record("lsp_impact_summary", {"reports": [report.to_dict() for report in self.service.reports]})
        self.service.close()

    def _record(self, name: str, payload: dict[str, Any]) -> None:
        if self._trace_writer is not None:
            self._trace_writer.record(name, payload)


def _write_path(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name not in {"edit_file", "create_file", "edit_python_symbol"}:
        return ""
    value = arguments.get("path")
    return value if isinstance(value, str) and value else ""
