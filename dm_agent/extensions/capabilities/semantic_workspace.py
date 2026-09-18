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
        impact_summary_chars: int = 600,
    ) -> None:
        if impact_summary_chars < 200:
            raise ValueError("impact_summary_chars must be at least 200")
        self.engine = engine
        self.impact_summary_chars = impact_summary_chars
        self._changed_paths: set[str] = set()
        self._pending_impact: ImpactReport | None = None
        self._impact_revision = 0
        self._injected_revision = 0
        self._last_injected_sha256 = ""
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
        self._last_injected_sha256 = ""
        event.metadata.update(
            {
                "semantic_workspace_enabled": True,
                "semantic_impact_enabled": True,
                "semantic_impact_injection_count": 0,
            }
        )
        try:
            stats = self.engine.update()
        except Exception as exc:  # Auxiliary indexing must not abort the Agent run.
            event.metadata["semantic_workspace_errors"] = 1
            self._record(
                "semantic_workspace_error",
                {"step_number": 0, "phase": "run_start", "error": str(exc)},
            )
            return
        event.metadata.update(
            {
                "semantic_index_files": stats.scanned_files,
                "semantic_index_cache_hits": stats.cache_hits,
                "semantic_index_parse_errors": stats.parse_errors,
            }
        )

    def _after_tool_result(self, event: AfterToolResultEvent) -> None:
        if event.tool_name not in WRITE_ACTIONS or not event.has_effect or event.no_change:
            return
        path = event.arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        try:
            stats = self.engine.update([path])
            self._changed_paths.add(path)
            impact = self.engine.analyze_impact(self._changed_paths)
        except Exception as exc:  # Keep ordinary tools available when semantic analysis fails.
            event.metadata["semantic_workspace_errors"] = (
                int(event.metadata.get("semantic_workspace_errors", 0)) + 1
            )
            self._record(
                "semantic_workspace_error",
                {
                    "step_number": event.step_number,
                    "phase": "incremental_update",
                    "path": path,
                    "error": str(exc),
                },
            )
            return
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
        summary_sha256 = hashlib.sha256(summary.encode("utf-8")).hexdigest()
        self._injected_revision = self._impact_revision
        if summary_sha256 == self._last_injected_sha256:
            event.metadata["semantic_impact_duplicate_suppressions"] = (
                int(event.metadata.get("semantic_impact_duplicate_suppressions", 0)) + 1
            )
            self._record(
                "semantic_impact_duplicate_suppressed",
                {
                    "step_number": event.step_number,
                    "revision": self._impact_revision,
                    "sha256": summary_sha256,
                },
            )
            return
        event.messages.append({"role": "system", "content": summary})
        self._last_injected_sha256 = summary_sha256
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
                "sha256": summary_sha256,
            },
        )

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)


def _render_impact_summary(impact: ImpactReport, *, max_chars: int) -> str:
    confirmed_files = tuple(
        sorted(
            {
                item.path
                for item in impact.confirmed_symbols
                if item.path not in impact.changed_files
            }
        )
    )
    lines = [
        "<change_impact>",
        "Candidate impact only; inspect before editing and validate with tests.",
        f"changed: {', '.join(impact.changed_files[:5]) or 'none'}",
        f"confirmed_affected: {', '.join(confirmed_files[:4]) or 'none'}",
        f"suggested_tests: {', '.join(impact.graph_tests[:4]) or 'none'}",
        "Use inspect_change_impact for provenance, ambiguous candidates, and fallbacks.",
    ]
    lines.append("</change_impact>")
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    closing = "\n</change_impact>"
    return summary[: max_chars - len(closing)].rstrip() + closing
