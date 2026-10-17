"""Native protocol, shared barriers, cancellation and memory pairing contracts."""

import json
import threading

from dm_agent.clients.messages import chat_messages, responses_input
from dm_agent.core.agent import ReactAgent
from dm_agent.core.events import EventBus
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing.session import load_session_entries, rebuild_context


class Client:
    supports_tool_calling = True
    last_response_mode = "native_tool_call"

    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    def respond(self, messages, **kwargs):
        self.requests.append(messages)
        if len(self.requests) == 1:
            return json.dumps(
                {
                    "tool_calls": [
                        {"action": name, "action_input": {}, "call_id": f"id-{i}"}
                        for i, name in enumerate(self.calls)
                    ]
                }
            )
        # Every provider request must have a completely paired transcript.
        chat_messages(messages)
        return json.dumps({"action": "finish", "action_input": "done"})


def build(client, tools, **kwargs):
    return ReactAgent(client, tools, enable_compression=False, enable_planning=False, **kwargs)


def test_shared_tools_overlap_exclusive_waits_and_commits_are_on_coordinator():
    barrier = threading.Barrier(2)
    main_thread = threading.get_ident()
    completed = []
    hooks = []
    bus = EventBus()
    bus.on(
        "before_tool_call", lambda e: hooks.append(("before", e.tool_name, threading.get_ident()))
    )
    bus.on(
        "after_tool_result", lambda e: hooks.append(("after", e.tool_name, threading.get_ident()))
    )

    def read(args, name):
        assert threading.get_ident() != main_thread
        barrier.wait(timeout=4)
        completed.append(name)
        return name

    def write(args):
        assert set(completed) == {"a", "b"}
        assert [h[1] for h in hooks if h[0] == "after"] == ["a", "b"]
        completed.append("write")
        return "written"

    tools = [
        Tool(n, n, lambda a, name=n: read(a, name), read_only=True, concurrency="shared")
        for n in ("a", "b")
    ]
    tools += [Tool("write", "write", write), Tool("last", "last", lambda a: "after-write")]
    client = Client(["a", "b", "write", "last"])
    result = build(client, tools, event_bus=bus).run("investigate", max_steps=5)
    assert result["metadata"]["status"] == "success"
    assert all(h[2] == main_thread for h in hooks)
    assert [m["tool_call_id"] for m in client.requests[-1] if m["role"] == "tool"] == [
        "id-0",
        "id-1",
        "id-2",
        "id-3",
    ]


def test_argument_dependent_concurrency_failure_falls_back_to_exclusive():
    calls = []

    def policy(args):
        raise RuntimeError("policy unavailable")

    tools = [
        Tool(
            n,
            n,
            lambda a, name=n: calls.append(name) or "ok",
            read_only=True,
            concurrency=policy if n == "b" else "shared",
        )
        for n in ("a", "b", "c")
    ]
    build(Client(["a", "b", "c"]), tools).run("all", max_steps=4)
    assert calls == ["a", "b", "c"]


def test_error_and_invalid_arguments_are_paired_without_dropping_successes():
    from dm_agent.clients.tool_calls import serialize_tool_calls

    class InvalidClient(Client):
        def respond(self, messages, **kwargs):
            if not self.requests:
                self.requests.append(messages)
                return serialize_tool_calls(
                    [
                        {"id": "bad", "function": {"name": "bad", "arguments": "{"}},
                        {"id": "good", "function": {"name": "good", "arguments": "{}"}},
                    ]
                )
            return super().respond(messages, **kwargs)

    calls = []
    client = InvalidClient([])
    tools = [
        Tool("bad", "bad", lambda a: calls.append("bad")),
        Tool("good", "good", lambda a: calls.append("good") or "ok"),
    ]
    build(client, tools).run("all", max_steps=3)
    assert calls == ["good"]
    results = [m for m in client.requests[-1] if m["role"] == "tool"]
    assert [m["status"] for m in results] == ["failed", "success"]
    wire = responses_input(client.requests[-1])
    assert [m["call_id"] for m in wire if m.get("type") == "function_call_output"] == [
        "bad",
        "good",
    ]
    assert (
        json.loads(
            next(
                m["output"]
                for m in wire
                if m.get("call_id") == "bad" and m.get("type") == "function_call_output"
            )
        )["status"]
        == "failed"
    )


def test_cancel_drains_workers_preserves_completed_result_and_pairs_unstarted_tail(tmp_path):
    fast_done = threading.Event()
    slow_started = threading.Event()
    slow_finished = threading.Event()
    output = []
    writes = []

    def fast(args):
        fast_done.set()
        return "fast result"

    def slow(args, cancelled):
        slow_started.set()
        assert cancelled.wait(5)
        slow_finished.set()
        return ToolResult("cancelled", "stopped cooperatively")

    client = Client(["fast", "slow", "write"])
    runtime = build(
        client,
        [
            Tool("fast", "fast", fast, read_only=True, concurrency="shared"),
            Tool(
                "slow",
                "slow",
                lambda a: "unused",
                read_only=True,
                concurrency="shared",
                cancel_runner=slow,
            ),
            Tool("write", "write", lambda a: writes.append("bad") or "written"),
        ],
    )
    thread = threading.Thread(
        target=lambda: output.append(
            runtime.run("all", max_steps=4, checkpoint_path=tmp_path / "state.jsonl")
        )
    )
    thread.start()
    try:
        assert fast_done.wait(4) and slow_started.wait(4)
        runtime.cancel()
        thread.join(5)
        assert not thread.is_alive() and slow_finished.is_set()
        assert writes == []
        assert output[0]["metadata"]["status"] == "cancelled"
        messages = runtime.conversation_history
        results = [m for m in messages if m["role"] == "tool"]
        assert [m["status"] for m in results] == ["success", "cancelled", "cancelled"]
        assert results[-1]["synthetic"] is True
        chat_messages(messages)
        replayed = rebuild_context(load_session_entries(tmp_path / "state.jsonl"))
        assert [m["tool_call_id"] for m in replayed if m["role"] == "tool"] == [
            "id-0",
            "id-1",
            "id-2",
        ]
    finally:
        runtime.cancel()
        thread.join(5)


def test_native_call_result_ids_survive_lcm_and_safe_boundaries():
    from dm_agent.memory.lcm.fresh_tail import safe_boundaries

    client = Client(["a", "b"])
    runtime = ReactAgent(
        client,
        [Tool(n, n, lambda a: "ok", read_only=True, concurrency="shared") for n in ("a", "b")],
        enable_planning=False,
        enable_compression=True,
        context_token_budget=0,
    )
    try:
        runtime.run("all", max_steps=3)
        memory = runtime.compressor
        rendered = memory.compactor.render(memory.frontier)
        chat_messages(rendered)
        begin = next(i for i, m in enumerate(rendered) if m.get("tool_calls"))
        boundaries = safe_boundaries(rendered)
        assert begin + 1 not in boundaries and begin + 2 not in boundaries
        assert begin + 3 in boundaries
        assert memory.store.search(memory.branch, "ok", record_types=["tool_result"])
    finally:
        runtime.close()


def test_deepseek_followup_request_uses_native_tool_messages():
    from dm_agent.clients.deepseek_client import DeepSeekClient

    class Response:
        ok = True

        def __init__(self, message):
            self.message = message

        def json(self):
            return {"choices": [{"message": self.message}]}

    class Session:
        def __init__(self):
            self.requests = []

        def post(self, url, *, json, **kwargs):
            self.requests.append(json)
            if len(self.requests) == 1:
                return Response(
                    {
                        "tool_calls": [
                            {"id": name, "function": {"name": name, "arguments": "{}"}}
                            for name in ("a", "b")
                        ]
                    }
                )
            return Response({"content": '{"action":"finish","action_input":"done"}'})

    client = DeepSeekClient("offline", max_retries=0)
    client.supports_streaming = False
    client.session.close()
    session = Session()
    client.session = session
    result = build(client, [Tool(name, name, lambda a: "ok") for name in ("a", "b")]).run(
        "all", max_steps=3
    )
    assert result["metadata"]["status"] == "success"
    messages = session.requests[1]["messages"]
    source = next(m for m in messages if m.get("tool_calls"))
    assert [c["id"] for c in source["tool_calls"]] == ["a", "b"]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["a", "b"]
    assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id"} for m in messages)


def test_duplicate_provider_ids_are_repaired_and_remain_paired():
    class DuplicateClient(Client):
        def respond(self, messages, **kwargs):
            if not self.requests:
                self.requests.append(messages)
                return json.dumps(
                    {
                        "tool_calls": [
                            {"call_id": "duplicate", "action": name, "action_input": {}}
                            for name in ("a", "b")
                        ]
                    }
                )
            return super().respond(messages, **kwargs)

    client = DuplicateClient([])
    build(client, [Tool(name, name, lambda a: "ok") for name in ("a", "b")]).run("all", max_steps=3)
    calls = next(m["tool_calls"] for m in client.requests[-1] if m.get("tool_calls"))
    assert len({c["id"] for c in calls}) == 2
    chat_messages(client.requests[-1])


def test_single_native_completion_still_passes_finish_gate_and_returns_paired_error():
    bus = EventBus()
    decisions = []

    def review(event):
        decisions.append(event.action)
        if len(decisions) == 1:
            return {"block": True, "reason": "Run the missing check first."}
        return None

    bus.on("before_finish", review, kind="policy")
    client = Client(["submit"])
    runtime = build(
        client, [Tool("submit", "submit", lambda a: "report", completes_task=True)], event_bus=bus
    )
    result = runtime.run("all", max_steps=2)
    assert result["metadata"]["status"] == "success"
    assert decisions == ["submit", "finish"]
    error = next(m for m in client.requests[-1] if m["role"] == "tool")
    assert error["is_error"] is True and "missing check" in error["content"]
    chat_messages(client.requests[-1])


def test_structured_result_details_survive_checkpoint(tmp_path):
    from dm_agent.core.persistence import load_resume_state

    checkpoint = tmp_path / "state.json"
    client = Client(["a", "b"])
    result = ToolResult(
        "failed",
        "assertion failed",
        error_code="assertion_failure",
        exit_code=1,
        check_scope=("tests/test_a.py",),
        metadata={"verification": {"outcome": "failed"}},
    )
    runtime = build(client, [Tool("a", "a", lambda a: result), Tool("b", "b", lambda a: "ok")])
    runtime.run("all", max_steps=1, checkpoint_path=checkpoint)
    saved = load_resume_state(checkpoint)
    message = next(m for m in saved.conversation_history if m["role"] == "tool")
    assert message["exit_code"] == 1
    assert message["check_scope"] == ["tests/test_a.py"]
    assert message["result_metadata"]["verification"]["outcome"] == "failed"
