"""Keep the semantic index and repository map current during a run."""

from __future__ import annotations

import re
from typing import Any

from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import (
    AfterToolResultEvent,
    BeforeLLMRequestEvent,
    RunStartEvent,
)
from dm_agent.core.guards import WRITE_ACTIONS
from dm_agent.workspace import ImpactReport, SemanticWorkspaceEngine

_MAP_RE = re.compile(r"<repository_map\b.*?</repository_map>", re.DOTALL)
_IMPACT_RE = re.compile(r"<change_impact\b.*?</change_impact>", re.DOTALL)


class SemanticWorkspaceCapability:
    """Refresh changed files and replace stale maps before each Agent LLM request."""

    def __init__(
        self,
        engine: SemanticWorkspaceEngine,
        *,
        max_files: int = 30,
        max_chars: int = 6000,
    ) -> None:
        self.engine = engine
        self.max_files = max_files
        self.max_chars = max_chars
        self._task = ""
        self._dirty = False
        self._map = ""
        self._changed_paths: set[str] = set()
        self._impact: ImpactReport | None = None
        self._trace_writer: Any | None = None

    def install(self, context: CapabilityContext) -> None:
        self._trace_writer = context.trace_writer
        context.event_bus.on("on_run_start", self._on_run_start, name="workspace.index.run_start")
        context.event_bus.on(
            "after_tool_result", self._after_tool_result, name="workspace.index.after_tool"
        )
        context.event_bus.on(
            "before_llm_request", self._before_llm_request, name="workspace.map.before_llm"
        )

    def _on_run_start(self, event: RunStartEvent) -> None:
        self._task = event.task
        self._changed_paths.clear()
        self._impact = None
        stats = self.engine.update()
        self._map, included, truncated = self.engine.build_repo_map(
            event.task, max_files=self.max_files, max_chars=self.max_chars
        )
        self._dirty = False
        event.metadata.update(
            {
                "semantic_workspace_enabled": True,
                "semantic_impact_enabled": True,
                "semantic_index_files": stats.scanned_files,
                "semantic_index_cache_hits": stats.cache_hits,
                "semantic_index_parse_errors": stats.parse_errors,
                "dynamic_repo_map_files": included,
                "dynamic_repo_map_truncated": truncated,
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
        self._impact = self.engine.analyze_impact(self._changed_paths)
        self._dirty = True
        event.metadata["semantic_index_incremental_updates"] = (
            int(event.metadata.get("semantic_index_incremental_updates", 0)) + 1
        )
        self._record(
            "semantic_index_updated",
            {"step_number": event.step_number, "path": path, "indexed_files": stats.indexed_files},
        )
        event.metadata.update(
            {
                "semantic_impact_analyses": int(
                    event.metadata.get("semantic_impact_analyses", 0)
                )
                + 1,
                "semantic_impact_risk": self._impact.risk_level,
                "semantic_impact_score": self._impact.risk_score,
                "semantic_impact_files": len(self._impact.affected_files),
                "semantic_impact_tests": len(self._impact.related_tests),
            }
        )
        self._record(
            "semantic_impact_computed",
            {
                "step_number": event.step_number,
                **self._impact.to_dict(),
            },
        )

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        if event.phase not in {"agent", "planner"}:
            return
        if self._dirty:
            self._map, included, truncated = self.engine.build_repo_map(
                self._task,
                max_files=self.max_files,
                max_chars=self.max_chars,
                impact=self._impact,
            )
            event.metadata["dynamic_repo_map_files"] = included
            event.metadata["dynamic_repo_map_truncated"] = truncated
            event.metadata["dynamic_repo_map_refresh_count"] = (
                int(event.metadata.get("dynamic_repo_map_refresh_count", 0)) + 1
            )
            self._dirty = False
        map_was_present = False
        for message in event.messages:
            content = message.get("content", "")
            if "<repository_map" in content:
                message["content"] = _MAP_RE.sub(lambda _: self._map, content)
                map_was_present = True
        if event.phase == "planner" and not map_was_present:
            for message in reversed(event.messages):
                if message.get("role") == "user":
                    message["content"] = f"{message.get('content', '')}\n\n{self._map}"
                    break
        if self._impact is not None:
            impact_text = self._impact.render()
            for message in reversed(event.messages):
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if "<change_impact" in content:
                    message["content"] = _IMPACT_RE.sub(lambda _: impact_text, content)
                else:
                    message["content"] = f"{content}\n\n{impact_text}"
                break

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)
