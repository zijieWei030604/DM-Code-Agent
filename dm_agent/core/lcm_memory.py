"""Runtime adapter for immutable LCM history, artifacts and checkpoint branches."""

from __future__ import annotations

import copy
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dm_agent.clients.base_client import BaseLLMClient
from dm_agent.memory.lcm.compaction import CompactionPolicy, Compactor, SummaryResponse
from dm_agent.memory.lcm.store import LCMStore
from dm_agent.memory.lcm.tools import build_lcm_tools
from dm_agent.tools.base import Tool
from dm_agent.tracing.writer import read_lcm_artifact, redact_text

LCM_RECALL_GUIDANCE = """

## Historical-memory recall

Use the current context when it is sufficient. For exact historical details, search with
`lcm_grep` using distinctive terms, then use the returned `offset` with `lcm_expand`.
Use `record_types` to narrow a search to tool results or summaries when helpful. Historical
summaries and test results are retrieval clues, not proof that the current workspace is valid;
verify the current version before relying on them for completion.
"""


class LCMMemory:
    """One conversation, one branch; resume creates a child at the saved cutoff."""

    def __init__(self, client: BaseLLMClient, *, token_budget: int, trace_writer: Any) -> None:
        self.client = client
        self.token_budget = token_budget
        self.trace_writer = trace_writer
        self._store: LCMStore | None = None
        self.branch = ""
        self.frontier: list[str] = []
        self.history_ids: list[str] = []
        self.pending_call = ""
        self.compactor: Compactor | None = None

    @property
    def store(self) -> LCMStore:
        if self._store is None:
            directory = Path.home() / ".dm_agent" / "lcm"
            directory.mkdir(parents=True, exist_ok=True)
            self._store = LCMStore(directory / f"{uuid.uuid4().hex}.sqlite3")
            self.branch = self._store.create_branch()
            self.compactor = Compactor(self._store, self.branch, self._summarize)
        return self._store

    def _summarize(self, messages: list[dict[str, str]], max_tokens: int) -> SummaryResponse:
        # Use the raw client, not LLMRequestClient: no request hooks, plans or tools.
        data = self.client.complete_summary(messages, max_tokens=max_tokens, timeout=60)
        raw_usage = data.get("usage")
        if raw_usage is None:
            response = data.get("response")
            raw_usage = getattr(response, "usage", None) or getattr(
                response, "usage_metadata", None
            )
        dump_usage = getattr(raw_usage, "model_dump", None)
        if callable(dump_usage):
            raw_usage = dump_usage()
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return SummaryResponse(redact_text(self.client.extract_text(data)), usage)

    def tools(self) -> list[Tool]:
        def get_branch() -> str:
            _ = self.store
            return self.branch

        return build_lcm_tools(lambda: self.store, get_branch, read_lcm_artifact)

    def adopt(self, history: list[dict[str, str]]) -> None:
        if len(history) < len(self.history_ids):
            raise ValueError("LCM history changed without restoring its checkpoint")
        for message in history[len(self.history_ids) :]:
            self.append(message["role"], message["content"], kind="imported")

    def append(
        self,
        role: str,
        content: str,
        *,
        kind: str,
        context_content: str | None = None,
    ) -> None:
        store = self.store
        metadata: dict[str, Any] = {"role": role, "kind": kind}
        # One Runtime decision and its observation form a complete unit, including
        # rejected/unknown calls. Native providers currently normalize to this unit.
        if kind == "model_response":
            if self.pending_call:
                raise ValueError("Previous Runtime decision has no observation")
            self.pending_call = uuid.uuid4().hex
            metadata["call_ids"] = [self.pending_call]
        elif kind in {"tool_result", "observation", "completion"} and self.pending_call:
            metadata["result_ids"] = [self.pending_call]
            self.pending_call = ""
        if context_content is not None:
            metadata["context_content"] = redact_text(context_content)
        record = store.append(
            self.branch, uuid.uuid4().hex, redact_text(content), metadata=metadata
        )
        self.history_ids.append(record)
        self.frontier.append(record)
        store.save_frontier(self.branch, self.frontier)

    def preserve_output(self, full_text: str, preview: str) -> str:
        full_text = redact_text(full_text)
        preview = redact_text(preview)
        reference = self.trace_writer.store_lcm_artifact(full_text)
        metadata = {"trace_reference": reference} if reference else {}
        record = self.store.append(
            self.branch,
            uuid.uuid4().hex,
            preview if reference else full_text,
            kind="artifact",
            metadata=metadata,
        )
        return f"{preview}\n[Full historical output: lcm_expand record_id={record}]"

    @property
    def memory_count(self) -> int:
        if self._store is None:
            return 0
        return sum(self.store.get(self.branch, item)["kind"] == "summary" for item in self.frontier)

    def export_state(self) -> dict[str, Any]:
        store = self.store
        return {
            "schema": "lcm-2",
            "database": str(Path(store.path).resolve()),
            "branch": self.branch,
            "cutoff": store.head(),
            "frontier": list(self.frontier),
            "history_ids": list(self.history_ids),
            "pending_call": self.pending_call,
            "summary_calls": copy.deepcopy(self.compactor.calls if self.compactor else []),
            "policy": asdict(self.compactor.policy) if self.compactor else {},
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "lcm-2":
            raise ValueError("Legacy atomic-memory or LCM checkpoints are not compatible with schema lcm-2")
        path = Path(state["database"])
        if not path.is_file():
            raise ValueError("LCM checkpoint database is missing")
        store = LCMStore(path)
        try:
            parent = str(state["branch"])
            cutoff = int(state["cutoff"])
            frontier = [str(item) for item in state["frontier"]]
            history_ids = [str(item) for item in state["history_ids"]]
            for record_id in {*frontier, *history_ids}:
                if store.get(parent, record_id)["seq"] > cutoff:
                    raise ValueError("Checkpoint references records beyond its cutoff")
            branch = store.create_branch(parent=parent, cutoff=cutoff)
            store.save_frontier(branch, frontier)
        except Exception:
            store.close()
            raise
        if self._store is not None:
            self._store.close()
        self._store = store
        self.branch = branch
        self.frontier = frontier
        self.history_ids = history_ids
        self.pending_call = str(state.get("pending_call", ""))
        self.compactor = Compactor(
            store, branch, self._summarize, policy=CompactionPolicy(**state.get("policy", {}))
        )
        self.compactor.calls = copy.deepcopy(state.get("summary_calls", []))

    def snapshot_runtime_state(self) -> dict[str, Any]:
        return self.export_state()

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        self.restore_state(state)

    def reset(self) -> None:
        if self._store is not None:
            self.branch = self._store.create_branch()
            self.compactor = Compactor(self._store, self.branch, self._summarize)
        self.frontier = []
        self.history_ids = []
        self.pending_call = ""
        if self._store is not None:
            self._store.save_frontier(self.branch, self.frontier)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
