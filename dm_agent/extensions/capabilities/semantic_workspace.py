"""Keep the semantic index current during a run without changing model context."""

from __future__ import annotations

import hashlib
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, BeforeLLMRequestEvent, RunStartEvent
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.workspace import ImpactReport, SemanticWorkspaceEngine


class SemanticWorkspaceCapability:
    """Maintain index and impact telemetry; ``ReactAgent`` owns Repo Map injection."""

    def __init__(
        self,
        engine: SemanticWorkspaceEngine,
        *,
        impact_summary_chars: int = 1200,
    ) -> None:
        if impact_summary_chars < 200:
            raise ValueError("impact_summary_chars must be at least 200")
        self.engine = engine
        self.impact_summary_chars = impact_summary_chars
        self._changed_paths: set[str] = set()
        self._pending_impact: ImpactReport | None = None
        self._impact_revision = 0
        self._injected_revision = 0
        self._trace_writer: Any | None = None

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        context.event_bus.on("on_run_start", self._on_run_start, name="workspace.index.run_start")
        context.event_bus.on(
            "after_tool_result", self._after_tool_result, name="workspace.index.after_tool"
        )
        context.event_bus.on(
            "before_llm_request",
            self._before_llm_request,
            name="workspace.impact.before_llm",
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._changed_paths.clear()
        self._pending_impact = None
        self._impact_revision = 0
        self._injected_revision = 0
        stats = self.engine.update()
        event.metadata.update(
            {
                "semantic_workspace_enabled": True,
                "semantic_impact_enabled": True,
                "semantic_index_files": stats.scanned_files,
                "semantic_index_cache_hits": stats.cache_hits,
                "semantic_index_parse_errors": stats.parse_errors,
                "semantic_impact_injection_count": 0,
            }
        )

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        if event.tool_name not in WRITE_ACTIONS or not event.has_effect or event.no_change:
            return
        path = event.arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        stats = self.engine.update([path])
        self._changed_paths.add(path)
        impact = self.engine.analyze_impact(self._changed_paths)
        self._pending_impact = impact
        self._impact_revision += 1
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

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        if (
            event.phase != "agent"
            or self._pending_impact is None
            or self._injected_revision == self._impact_revision
        ):
            return
        summary = _render_impact_summary(
            self._pending_impact,
            max_chars=self.impact_summary_chars,
        )
        event.messages.append({"role": "system", "content": summary})
        self._injected_revision = self._impact_revision
        event.metadata["semantic_impact_injection_count"] = (
            int(event.metadata.get("semantic_impact_injection_count", 0)) + 1
        )
        self._record(
            "semantic_impact_injected",
            {
                "step_number": event.step_number,
                "revision": self._impact_revision,
                "chars": len(summary),
                # Persist exactly the bounded text sent to the model so offline
                # analysis can audit candidate paths rather than just counts.
                "content": summary,
                "sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            },
        )

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)


def _render_impact_summary(impact: ImpactReport, *, max_chars: int) -> str:
    lines = [
        "<change_impact>",
        "Heuristic candidates only; inspect files before editing and validate with tests.",
        f"risk: {impact.risk_level} ({impact.risk_score:.2f})",
        f"changed: {', '.join(impact.changed_files[:5]) or 'none'}",
        f"possibly_affected: {', '.join(impact.affected_files[:5]) or 'none'}",
        f"related_tests: {', '.join(impact.related_tests[:5]) or 'none'}",
    ]
    lines.extend(f"reason: {reason}" for reason in impact.reasons[:3])
    lines.append("</change_impact>")
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    closing = "\n</change_impact>"
    return summary[: max_chars - len(closing)].rstrip() + closing
