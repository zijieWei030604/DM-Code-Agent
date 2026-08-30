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
from dm_agent.workspace import SemanticWorkspaceEngine

_MAP_RE = re.compile(r"<repository_map\b.*?</repository_map>", re.DOTALL)


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
        stats = self.engine.update()
        self._map, included, truncated = self.engine.build_repo_map(
            event.task, max_files=self.max_files, max_chars=self.max_chars
        )
        self._dirty = False
        event.metadata.update(
            {
                "semantic_workspace_enabled": True,
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
        self._dirty = True
        event.metadata["semantic_index_incremental_updates"] = (
            int(event.metadata.get("semantic_index_incremental_updates", 0)) + 1
        )
        self._record(
            "semantic_index_updated",
            {"step_number": event.step_number, "path": path, "indexed_files": stats.indexed_files},
        )

    def _before_llm_request(self, event: BeforeLLMRequestEvent) -> None:
        if event.phase != "agent":
            return
        if self._dirty:
            self._map, included, truncated = self.engine.build_repo_map(
                self._task, max_files=self.max_files, max_chars=self.max_chars
            )
            event.metadata["dynamic_repo_map_files"] = included
            event.metadata["dynamic_repo_map_truncated"] = truncated
            event.metadata["dynamic_repo_map_refresh_count"] = (
                int(event.metadata.get("dynamic_repo_map_refresh_count", 0)) + 1
            )
            self._dirty = False
        for message in event.messages:
            content = message.get("content", "")
            if "<repository_map" in content:
                message["content"] = _MAP_RE.sub(lambda _: self._map, content)

    def _record(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace_writer:
            self._trace_writer.record(event, payload)
