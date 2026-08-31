"""Keep the semantic index current during a run without changing model context."""

from __future__ import annotations

from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, RunStartEvent
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.workspace import SemanticWorkspaceEngine


class SemanticWorkspaceCapability:
    """Maintain index and impact telemetry; ``ReactAgent`` owns Repo Map injection."""

    def __init__(
        self,
        engine: SemanticWorkspaceEngine,
    ) -> None:
        self.engine = engine
        self._changed_paths: set[str] = set()
        self._trace_writer: Any | None = None

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        context.event_bus.on("on_run_start", self._on_run_start, name="workspace.index.run_start")
        context.event_bus.on(
            "after_tool_result", self._after_tool_result, name="workspace.index.after_tool"
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._changed_paths.clear()
        stats = self.engine.update()
        event.metadata.update(
            {
                "semantic_workspace_enabled": True,
                "semantic_impact_enabled": True,
                "semantic_index_files": stats.scanned_files,
                "semantic_index_cache_hits": stats.cache_hits,
                "semantic_index_parse_errors": stats.parse_errors,
            }
        )

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        if event.tool_name not in WRITE_ACTIONS or not event.tool_succeeded or event.no_change:
            return
        path = event.arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        stats = self.engine.update([path])
        self._changed_paths.add(path)
        impact = self.engine.analyze_impact(self._changed_paths)
        event.metadata["semantic_index_incremental_updates"] = (
            int(event.metadata.get("semantic_index_incremental_updates", 0)) + 1
        )
        self._record(
            "semantic_index_updated",
            {"step_number": event.step_number, "path": path, "indexed_files": stats.indexed_files},
        )
        event.metadata.update(
            {
                "semantic_impact_analyses": int(event.metadata.get("semantic_impact_analyses", 0))
                + 1,
                "semantic_impact_risk": impact.risk_level,
                "semantic_impact_score": impact.risk_score,
                "semantic_impact_files": len(impact.affected_files),
                "semantic_impact_tests": len(impact.related_tests),
            }
        )
        self._record(
            "semantic_impact_computed",
            {
                "step_number": event.step_number,
                **impact.to_dict(),
            },
        )

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)
