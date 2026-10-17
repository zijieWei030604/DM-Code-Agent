"""Behavioral coverage for ordered calls, policies and interrupted batches."""

import json

import pytest

from dm_agent.core.agent import ReactAgent
from dm_agent.core.events import EventBus
from dm_agent.core.persistence import load_resume_state
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing.writer import TraceWriter, load_trace_events


class Client:
    supports_tool_calling = True
    last_response_mode = "native_tool_call"
    last_tool_call_count = 2

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def respond(self, messages, **kwargs):
        self.requests.append(messages)
        return json.dumps(next(self.responses))


def batch(*names):
    return {
        "tool_calls": [
            {"call_id": f"call-{i}", "action": name, "action_input": {}}
            for i, name in enumerate(names)
        ]
    }


def finish():
    return {"action": "finish", "action_input": "finished"}


def agent(client, tools, **kwargs):
    return ReactAgent(client, tools, enable_planning=False, enable_compression=False, **kwargs)


def test_ordered_calls_use_hooks_and_return_all_observations_before_next_request(tmp_path):
    calls = []
    bus = EventBus()
    bus.on("before_tool_call", lambda e: calls.append(f"before:{e.tool_name}"))
    bus.on("after_tool_result", lambda e: calls.append(f"after:{e.tool_name}"))
    tools = [
        Tool(name, name, lambda a, n=name: calls.append(n) or f"result:{n}") for name in ("a", "b")
    ]
    client = Client([batch("a", "b"), finish()])
    trace = TraceWriter(tmp_path / "trace.jsonl")
    try:
        result = agent(client, tools, event_bus=bus, trace_writer=trace).run("both", max_steps=3)
    finally:
        trace.close()
    assert calls == ["before:a", "a", "after:a", "before:b", "b", "after:b"]
    assert len(client.requests) == 2
    assert "result:a" in str(client.requests[1]) and "result:b" in str(client.requests[1])
    assert [s["action"] for s in result["steps"]] == ["a", "b", "finish"]
    assert result["metadata"]["discarded_tool_call_count"] == 0
    entries = [e for e in load_trace_events(trace.path) if e["event"] == "tool_batch_call"]
    assert [e["payload"]["call_id"] for e in entries] == ["call-0", "call-1"]


@pytest.mark.parametrize(
    "failure", ["failed", "unavailable", "cancelled", "exception", "block", "unknown"]
)
def test_unsuccessful_call_preserves_siblings_and_returns_all_results(failure):
    calls = []
    bus = EventBus()
    if failure == "block":
        bus.on(
            "before_tool_call",
            lambda e: {"block": True, "reason": "denied"} if e.tool_name == "a" else None,
            kind="policy",
        )

    def first(args):
        if failure == "exception":
            raise RuntimeError("failed")
        return ToolResult(
            failure if failure in {"failed", "unavailable", "cancelled"} else "success", "check"
        )

    tools = [Tool("b", "b", lambda a: calls.append("b") or "ok")]
    if failure != "unknown":
        tools.append(Tool("a", "a", first))
    client = Client([batch("a", "b"), finish()])
    result = agent(client, tools, event_bus=bus).run("both", max_steps=3)
    assert calls == ["b"]
    results = [m for m in client.requests[-1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call-0", "call-1"]
    assert results[0]["is_error"] is True and results[1]["status"] == "success"
    assert result["metadata"]["skipped_tool_call_count"] == 0
    assert len(client.requests) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "task_complete", "action_input": {}},
        {"action": "finish", "action_input": {}},
        {"action": "submit", "action_input": {}},
    ],
)
def test_invalid_envelope_executes_no_prefix(bad):
    envelope = batch("a")
    envelope["tool_calls"].append(bad)
    calls = []
    client = Client([envelope, finish()])
    tools = [
        Tool("a", "a", lambda a: calls.append("a") or "ok"),
        Tool("submit", "submit", lambda a: "done", completes_task=True),
    ]
    result = agent(client, tools).run("both", max_steps=2)
    assert calls == []
    assert result["metadata"]["parse_error_count"] == 1


def test_budget_checkpoint_resumes_only_pending_tail_without_new_model_request(tmp_path):
    checkpoint = tmp_path / "state.json"
    calls = []
    tools = [
        Tool(n, n, lambda a, name=n: calls.append(name) or "ok", read_only=True) for n in ("a", "b")
    ]
    first = agent(Client([batch("a", "b")]), tools)
    result = first.run("both", max_steps=1, checkpoint_path=checkpoint)
    assert result["metadata"]["status"] == "max_steps_exceeded"
    assert calls == ["a"]
    client = Client([finish()])
    result = agent(client, tools).run(
        "both", max_steps=3, checkpoint_path=checkpoint, resume_state=load_resume_state(checkpoint)
    )
    assert calls == ["a", "b"]
    assert len(client.requests) == 1
    assert result["metadata"]["status"] == "success"


def test_read_then_edit_in_one_batch_uses_guard(tmp_path, monkeypatch):
    from dm_agent.tools import default_tools

    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    envelope = {
        "tool_calls": [
            {"action": "read_file", "action_input": {"path": "a.py"}},
            {
                "action": "edit_file",
                "action_input": {"path": "a.py", "old_string": "old", "new_string": "new"},
            },
        ]
    }
    result = agent(Client([envelope, finish()]), default_tools()).run("edit", max_steps=3)
    assert (tmp_path / "a.py").read_text() == "new\n"
    assert result["metadata"]["edit_guard_block_count"] == 0


def test_streamed_interleaved_calls_execute_only_after_entire_turn_is_complete():
    from dm_agent.clients.deepseek_client import DeepSeekClient

    order = []

    class Response:
        ok = True

        def iter_lines(self, **kwargs):
            chunks = [
                {
                    "tool_calls": [
                        {"index": 1, "id": "b", "function": {"name": "b", "arguments": "{"}}
                    ]
                },
                {
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "a", "arguments": "{}"}}
                    ]
                },
                {"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]},
            ]
            for delta in chunks:
                assert not any(item in {"a", "b"} for item in order)
                yield "data: " + json.dumps({"choices": [{"delta": delta}]})
            yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
            order.append("done")
            yield "data: [DONE]"

        def close(self):
            pass

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    client = DeepSeekClient("offline", max_retries=0)
    client.session.close()
    client.session = Session()
    tools = [Tool(name, name, lambda a, n=name: order.append(n) or "ok") for name in ("a", "b")]
    result = agent(client, tools, stream_callback=lambda e: order.append(e.type)).run(
        "both", max_steps=2
    )
    assert order.index("done") < order.index("stream_done") < order.index("a") < order.index("b")
    assert result["metadata"]["native_tool_call_count"] == 1


def test_invalid_later_native_arguments_never_execute_valid_prefix():
    from dm_agent.clients.tool_calls import serialize_tool_calls

    raw = serialize_tool_calls(
        [
            {"function": {"name": "a", "arguments": "{}"}},
            {"function": {"name": "b", "arguments": "{"}},
        ]
    )
    parsed = json.loads(raw)["tool_calls"]
    assert "argument_error" not in parsed[0]
    assert "argument_error" in parsed[1]


def test_batch_trace_hides_pending_arguments_but_checkpoint_preserves_them(tmp_path):
    marker = "PRIVATE_BATCH_ARGUMENT_8142"
    envelope = batch("a", "b")
    envelope["tool_calls"][1]["action_input"] = {"value": marker}
    trace = TraceWriter(tmp_path / "trace.jsonl")
    checkpoint = tmp_path / "checkpoint.json"
    try:
        result = agent(
            Client([envelope]), [Tool("a", "a", lambda a: "ok")], trace_writer=trace
        ).run("both", max_steps=1, checkpoint_path=checkpoint)
    finally:
        trace.close()
    assert marker in checkpoint.read_text(encoding="utf-8")
    assert marker not in trace.path.read_text(encoding="utf-8")
    assert marker not in str(result["metadata"])


@pytest.mark.parametrize("calls", [[], [{}] * 33])
def test_batch_size_is_bounded(calls):
    from dm_agent.core.tool_batches import prepare_batch

    with pytest.raises(ValueError, match=r"1\.\.32"):
        prepare_batch({"tool_calls": calls}, set())


def test_corrupt_pending_checkpoint_is_rejected_before_model_or_tool_execution():
    from dm_agent.core.checkpoint import RunCheckpoint

    client = Client([])
    state = RunCheckpoint(task="both", step_count=0, pending_tool_calls=[{}])
    with pytest.raises(ValueError, match="checkpoint tool"):
        agent(client, [Tool("a", "a", lambda a: pytest.fail("must not execute"))]).run(
            "both", resume_state=state
        )
    assert not client.requests


def test_interrupted_side_effect_still_blocks_blind_resume(tmp_path):
    checkpoint = tmp_path / "state.json"

    def write_then_crash(args):
        (tmp_path / "effect.txt").write_text("written", encoding="utf-8")
        raise SystemExit("simulated process exit")

    tools = [Tool("a", "a", write_then_crash), Tool("b", "b", lambda a: pytest.fail("tail"))]
    with pytest.raises(SystemExit):
        agent(Client([batch("a", "b")]), tools).run("both", checkpoint_path=checkpoint)
    assert (tmp_path / "effect.txt").exists()
    with pytest.raises(ValueError, match="side-effect reconciliation"):
        load_resume_state(checkpoint)
