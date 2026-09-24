"""End-to-end offline tests of the active Agent/LCM integration."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dm_agent.core.agent import ReactAgent
from dm_agent.core.checkpoint import load_checkpoint
from dm_agent.core.events import EventBus
from dm_agent.core.lcm_memory import LCMMemory
from dm_agent.memory.lcm.dag import SummaryDAG
from dm_agent.tools.base import Tool
from dm_agent.tracing.writer import SessionWriter, TraceWriter, load_trace_events, read_lcm_artifact


def action(name, args=None):
    return json.dumps({"thought": "next", "action": name, "action_input": args or {}})


class Client:
    model = "offline"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.summary_requests = []

    def respond(self, messages, **kwargs):
        self.requests.append(messages)
        return self.responses.pop(0)

    def complete_summary(self, messages, **kwargs):
        self.summary_requests.append((messages, kwargs))
        return {
            "text": "Historical observations recorded; continue the task.",
            "usage": {"input_tokens": 123, "output_tokens": 12},
        }

    def extract_text(self, data):
        return data["text"]


def make_agent(client, **options):
    agent = ReactAgent(
        client,
        [Tool("echo", "echo", lambda _: "observation " + "x" * 1800, read_only=True)],
        enable_planning=False,
        system_prompt="Perform the task using tools.",
        context_token_budget=1000,
        **options,
    )
    agent._context_window.output_token_reserve = 0
    agent._context_window.safety_margin_tokens = 0
    if agent.compressor:
        _ = agent.compressor.store
        agent.compressor.compactor.policy = replace(
            agent.compressor.compactor.policy, keep_recent=2
        )
    return agent


def test_agent_uses_summaries_without_hooks_or_tools(tmp_path):
    bus = EventBus()
    phases = []
    bus.on("before_llm_request", lambda event: phases.append(event.phase))
    client = Client([action("echo") for _ in range(6)] + [action("finish", "done")])
    writer = TraceWriter(tmp_path / "trace.jsonl", capture_llm_io=True)
    agent = make_agent(client, event_bus=bus, trace_writer=writer)
    try:
        result = agent.run("Read observations and finish", max_steps=8)
        assert result["metadata"]["status"] == "success"
        assert result["metadata"]["context_backend"] == "lcm"
        assert client.summary_requests
        assert phases == ["agent"] * 7
        assert all(
            set(kwargs) == {"max_tokens", "timeout"} for _, kwargs in client.summary_requests
        )
        assert any("<historical_summary" in m["content"] for m in client.requests[-1])
        assert not any("<agent_memory>" in m["content"] for m in client.requests[-1])
        assert len(agent.compressor.history_ids) == len(agent.conversation_history)
        assert {"lcm_grep", "lcm_describe", "lcm_expand"} <= set(agent.tools)
        calls = [
            event
            for event in load_trace_events(writer.path)
            if event["event"] == "lcm_summary_call"
        ]
        assert len(calls) == len(client.summary_requests)
        assert calls[0]["payload"]["usage"]["input_tokens"] == 123
    finally:
        agent.close()
        writer.close()


@pytest.mark.parametrize("traced", [True, False])
def test_full_output_preserved_and_only_sqlite_searched(tmp_path, monkeypatch, traced):
    monkeypatch.setenv("LCM_TEST_SECRET_KEY", "secret-should-not-persist")
    writer = TraceWriter(tmp_path / "trace.jsonl") if traced else None
    memory = LCMMemory(Client([]), token_budget=1000, trace_writer=SessionWriter(writer))
    try:
        result = memory.preserve_output(
            "head secret-should-not-persist middle_unique tail", "head...tail"
        )
        record_id = result.split("record_id=")[1].rstrip("]")
        row = memory.store.get(memory.branch, record_id)
        if traced:
            assert row["body"] == "head...tail"
            assert memory.store.search(memory.branch, "middle_unique") == []
        else:
            assert memory.store.search(memory.branch, "middle_unique")
        dag = SummaryDAG(memory.store, memory.branch, read_lcm_artifact)
        text = dag.expand(record_id)["content"]
        assert "middle_unique" in text
        assert "secret-should-not-persist" not in text
        assert "redacted" in text
        if traced:
            writer.close()
            writer.path.unlink()
            tool = next(t for t in memory.tools() if t.name == "lcm_expand")
            assert tool.execute({"record_id": record_id}).status == "failed"
    finally:
        memory.close()
        if writer:
            writer.close()


def test_agent_truncation_writes_recoverable_artifact(tmp_path):
    client = Client([action("echo"), action("finish", "done")])
    writer = TraceWriter(tmp_path / "trace.jsonl")
    agent = make_agent(client, trace_writer=writer, max_observation_chars=300)
    try:
        result = agent.run("Read long output", max_steps=2)
        assert result["metadata"]["status"] == "success"
        observation = result["steps"][0]["observation"]
        record_id = observation.split("record_id=")[1].rstrip("]")
        response = agent.tools["lcm_expand"].execute(
            {"record_id": record_id, "offset": 1000, "limit": 200}
        )
        assert response.status == "success"
        assert len(json.loads(response.message)["content"]) == 200
    finally:
        agent.close()
        writer.close()


def test_checkpoint_resume_preserves_frontier_and_excludes_later_records(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = make_agent(Client([action("echo") for _ in range(4)]))
    resumed = None
    try:
        first.run("Long task", max_steps=4, checkpoint_path=path)
        saved = load_checkpoint(path)
        memory = first.compressor
        assert memory.memory_count
        frontier = list(memory.frontier)
        memory.append("user", "future_branch_secret", kind="task")
        future_id = memory.history_ids[-1]
        client = Client([action("finish", "done")])
        resumed = make_agent(client)
        # Test reuse, not a fresh compaction triggered by the final tool result.
        resumed.compressor.token_budget = 10000
        resumed.run(saved.task, max_steps=5, resume_state=saved)
        assert resumed.compressor.frontier[: len(frontier)] == frontier
        assert not client.summary_requests
        assert "<historical_summary" in client.requests[0][1]["content"]
        assert (
            resumed.compressor.store.search(resumed.compressor.branch, "future_branch_secret") == []
        )
        denied = resumed.tools["lcm_expand"].execute({"record_id": future_id})
        assert denied.status == "failed"
    finally:
        first.close()
        if resumed:
            resumed.close()


def test_summary_failure_stops_on_overflow_without_discarding_history():
    client = Client([action("echo") for _ in range(6)])

    def fail(*args, **kwargs):
        raise TimeoutError("offline timeout")

    client.complete_summary = fail
    agent = make_agent(client)
    try:
        result = agent.run("Long task", max_steps=6)
        assert result["metadata"]["status"] == "context_overflow"
        assert len(agent.compressor.history_ids) == len(agent.conversation_history)
        assert agent.compressor.frontier == agent.compressor.history_ids
        assert len(agent.compressor.compactor.calls) <= 4
    finally:
        agent.close()


def test_compression_disabled_does_not_register_memory_tools():
    client = Client([action("finish", "done")])
    agent = make_agent(client, enable_compression=False)
    try:
        assert agent.compressor is None
        assert "lcm_grep" not in agent.tools
        assert agent.run("Finish", max_steps=1)["metadata"]["status"] == "success"
        assert not client.summary_requests
    finally:
        agent.close()


@pytest.mark.parametrize("provider", ["openai", "claude", "gemini", "deepseek"])
def test_provider_summary_transport_limits(provider):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    def with_options(**kwargs):
        captured["transport"] = kwargs
        return SimpleNamespace(
            responses=SimpleNamespace(create=create), messages=SimpleNamespace(create=create)
        )

    messages = [{"role": "system", "content": "summarize"}, {"role": "user", "content": "history"}]
    if provider == "openai":
        from dm_agent.clients.openai_client import OpenAIClient as cls
    elif provider == "claude":
        from dm_agent.clients.claude_client import ClaudeClient as cls
    elif provider == "gemini":
        from dm_agent.clients.gemini_client import GeminiClient as cls
    else:
        from dm_agent.clients.deepseek_client import DeepSeekClient as cls
    instance = object.__new__(cls)
    instance.model = "offline"
    instance.client = SimpleNamespace(
        with_options=with_options, models=SimpleNamespace(generate_content=create)
    )
    if provider == "deepseek":

        def post(url, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

        instance.session = SimpleNamespace(post=post)
        instance.base_url = "https://example.invalid"
        instance.endpoint = "/chat/completions"
    instance.complete_summary(messages, max_tokens=77, timeout=12)
    assert "tools" not in captured
    if provider in {"openai", "claude"}:
        assert captured["transport"] == {"timeout": 12, "max_retries": 0}
        assert captured.get("max_output_tokens", captured.get("max_tokens")) == 77
    elif provider == "gemini":
        assert captured["config"]["http_options"]["timeout"] == 12000
        assert captured["config"]["max_output_tokens"] == 77
    else:
        assert captured["timeout"] == 12
        assert captured["json"]["max_tokens"] == 77
