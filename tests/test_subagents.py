"""Deterministic delegation tests: in-process workers, no network or keys."""

import json
import threading
import time
from pathlib import Path

import pytest

from dm_agent.evals.scripted_client import ScriptedLLMClient
from dm_agent.subagents.manager import TaskManager
from dm_agent.subagents.worker import execute, parse_report, result_output_preview, scoped_tools
from dm_agent.tools.base import ToolResult


class NativeScriptedLLMClient(ScriptedLLMClient):
    """Simulate the normalized result of a provider-native function call."""

    supports_tool_calling = True

    def __init__(self, responses):
        super().__init__(responses)
        self.options = []
        self.last_response_mode = ""
        self.last_tool_call_count = 0
        self.last_selected_tool = ""

    def respond(self, messages, **extra):
        self.options.append(extra)
        self.last_response_mode = "native_tool_call"
        self.last_tool_call_count = 1
        self.last_selected_tool = "submit_exploration_result"
        return super().respond(messages, **extra)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_worker(request, *, control):
        path = Path(request["attempt"])
        (path / "started").write_text(str(time.monotonic()))
        (path / "previous.txt").write_text(request.get("previous", ""))
        delay = 30 if request["instruction"] == "slow" else 0.1
        end = time.monotonic() + delay
        while time.monotonic() < end:
            control.check()
            control.cancelled.wait(0.005)
        control.check()
        if request["instruction"] == "fail":
            raise RuntimeError("scripted failure")
        (path / "answer.txt").write_text("x" * 9000)
        (path / "ended").write_text(str(time.monotonic()))
        return {"status": "succeeded", "summary": request["instruction"]}

    monkeypatch.setattr("dm_agent.subagents.manager.execute", fake_worker)
    mgr = TaskManager(
        tmp_path / "store",
        {"workspace": str(workspace)},
        workers=2,
        timeout=5,
    )
    yield mgr
    mgr.close()


def batch(*instructions):
    return {"context": "test", "tasks": [{"instruction": i} for i in instructions]}


def test_real_parallelism_and_limit(manager):
    result = manager.batch(batch("a", "b", "c"))
    records = json.loads(result.message)["results"]
    spans = [
        (
            float((manager.root / r["id"] / "started").read_text()),
            float((manager.root / r["id"] / "ended").read_text()),
        )
        for r in records
    ]
    assert max(spans[0][0], spans[1][0]) < min(spans[0][1], spans[1][1])
    assert spans[2][0] >= min(spans[0][1], spans[1][1])
    assert [r["summary"] for r in records] == ["a", "b", "c"]


def test_partial_failure_and_pagination(manager):
    records = json.loads(manager.batch(batch("fail", "ok")).message)["results"]
    assert [r["status"] for r in records] == ["failed", "succeeded"]
    result = json.loads(manager.result({"task_id": records[1]["id"], "limit": 100}).message)
    assert len(result["text"]) == 100
    assert result["next_offset"] == 100


def test_timeout_stops_worker_before_batch_returns(manager):
    manager.timeout = 0.3
    records = json.loads(manager.batch(batch("slow")).message)["results"]
    assert records[0]["status"] == "timed_out"
    assert not (manager.root / records[0]["id"] / "ended").exists()


def test_cancel_inflight_and_queue(manager):
    thread = threading.Thread(target=lambda: manager.batch(batch("slow", "slow", "slow")))
    thread.start()
    deadline = time.monotonic() + 5
    while not any(r["status"] == "running" for r in list(manager.records.values())):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    manager.cancel()
    thread.join(5)
    assert not thread.is_alive()
    assert all(r["status"] == "cancelled" for r in manager.records.values())


def test_recover_interrupted_without_reexecuting(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    record = {"id": "old", "status": "running", "session_id": "old"}
    (store / "tasks.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manager = TaskManager(store, {})
    assert manager.records["old"]["status"] == "interrupted"
    assert len((store / "tasks.jsonl").read_text().splitlines()) == 2
    manager.close()


def test_scope_and_no_execution_tools(tmp_path):
    tools = {t.name: t for t in scoped_tools(tmp_path)}
    assert set(tools) == {
        "read_file",
        "list_directory",
        "find_files",
        "search_code",
        "search_in_file",
    }
    with pytest.raises(ValueError, match="escapes"):
        tools["read_file"].execute({"path": "../secret"})


def test_scoped_read_requires_a_targeted_range_for_large_files(tmp_path):
    large = tmp_path / "large.py"
    large.write_text("\n".join(f"line_{index}" for index in range(200)), encoding="utf-8")
    tool = next(item for item in scoped_tools(tmp_path) if item.name == "read_file")

    whole_file = tool.execute({"path": "large.py"})
    assert isinstance(whole_file, ToolResult)
    assert whole_file.status == "failed"
    assert whole_file.error_code == "read_range_required"

    too_wide = tool.execute({"path": "large.py", "line_start": 1, "line_end": 121})
    assert isinstance(too_wide, ToolResult)
    assert too_wide.status == "failed"
    assert too_wide.error_code == "read_range_too_wide"

    targeted = tool.execute({"path": "large.py", "line_start": 1, "line_end": 120})
    assert isinstance(targeted, str)
    assert "line_119" in targeted


def test_worker_uses_real_react_and_preserves_followup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.py").write_text("VALUE = 42\n")
    answer = json.dumps({"summary": "VALUE is 42", "findings": ["hello.py:1"], "uncertainties": []})
    client = ScriptedLLMClient(
        [
            json.dumps(
                {"thought": "read", "action": "read_file", "action_input": {"path": "hello.py"}}
            ),
            json.dumps({"thought": "done", "action": "finish", "action_input": answer}),
        ]
    )
    request = {
        "settings": {"workspace": str(workspace)},
        "attempt": str(tmp_path / "one"),
        "context": "investigate",
        "instruction": "read hello.py",
    }
    result = execute(request, client)
    assert result["status"] == "succeeded"
    assert result["evidence"][0]["tool"] == "read_file"
    assert result["evidence"][0]["succeeded"]
    next_client = ScriptedLLMClient([json.dumps({"action": "finish", "action_input": answer})])
    execute(
        {**request, "previous": request["attempt"], "attempt": str(tmp_path / "two")}, next_client
    )
    assert "VALUE = 42" in str(next_client.requests[0])
    assert (tmp_path / "one" / "checkpoint.jsonl").exists()
    entries = [
        json.loads(line)
        for line in (tmp_path / "one" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        "messages" not in e["payload"] and "raw_response" not in e["payload"]
        for e in entries
        if e["event"] == "llm_call"
    )


def test_short_child_report_is_inlined_for_the_parent(tmp_path):
    report = {"summary": "located", "findings": ["hello.py:1"], "uncertainties": []}
    client = ScriptedLLMClient(
        [json.dumps({"action": "submit_exploration_result", "action_input": {"report": report}})]
    )

    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "locate hello.py",
        },
        client,
    )

    assert result["summary"] == "located"
    assert result["output"] == json.dumps(report, ensure_ascii=False)
    assert result["output_truncated"] is False
    assert result["output_char_count"] == len(result["output"])


def test_long_child_report_uses_a_marked_preview(tmp_path):
    report = {"summary": "x" * 6_000, "findings": [], "uncertainties": []}
    client = ScriptedLLMClient(
        [json.dumps({"action": "submit_exploration_result", "action_input": {"report": report}})]
    )

    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "produce a long report",
        },
        client,
    )

    assert result["output_truncated"] is True
    assert len(result["output"]) == 5_000
    assert result["output_char_count"] > len(result["output"])


def test_result_output_preview_prefers_a_late_line_boundary():
    text = "a" * 3_000 + "\n" + "b" * 3_000
    preview, truncated = result_output_preview(text)
    assert truncated is True
    assert preview == "a" * 3_000


def test_read_only_worker_skips_workspace_fingerprint(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.py").write_text("VALUE = 42\n", encoding="utf-8")
    report = {"summary": "located", "findings": ["hello.py:1"], "uncertainties": []}
    client = ScriptedLLMClient(
        [
            json.dumps({"action": "read_file", "action_input": {"path": "hello.py"}}),
            json.dumps(
                {
                    "action": "submit_exploration_result",
                    "action_input": {"report": report},
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "dm_agent.core.tool_invoker.workspace_version",
        lambda _root: pytest.fail("read-only workers must not fingerprint the workspace"),
    )

    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "read hello.py",
        },
        client,
    )

    assert result["status"] == "succeeded"
    evidence = result["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["tool"] == "read_file"
    assert evidence[0]["arguments"] == {"path": "hello.py"}
    assert evidence[0]["succeeded"] is True


def test_read_only_worker_bounds_large_source_observations(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.py").write_text("x" * 12_000, encoding="utf-8")
    report = {"summary": "located", "findings": [], "uncertainties": []}
    client = ScriptedLLMClient(
        [
            json.dumps({"action": "read_file", "action_input": {"path": "large.py"}}),
            json.dumps(
                {
                    "action": "submit_exploration_result",
                    "action_input": {"report": report},
                }
            ),
        ]
    )

    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "read large.py",
        },
        client,
    )

    tool_observation = next(
        message["content"]
        for message in client.requests[1]
        if message["role"] == "user" and message["content"].startswith("执行工具 read_file")
    )
    assert len(tool_observation) < 4_300
    assert result["status"] == "succeeded"


def test_read_only_worker_reserves_a_turn_for_structured_delivery(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 42\n", encoding="utf-8")
    report = {"summary": "located", "findings": [], "uncertainties": []}
    client = ScriptedLLMClient(
        [
            *[
                json.dumps({"action": "read_file", "action_input": {"path": "module.py"}})
                for _ in range(6)
            ],
            json.dumps(
                {
                    "action": "submit_exploration_result",
                    "action_input": {"report": report},
                }
            ),
        ]
    )

    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "inspect module.py",
        },
        client,
    )

    assert result["status"] == "succeeded"
    assert len(result["evidence"]) == 6


def test_worker_submits_structured_result_through_terminal_tool(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = {"summary": "located", "findings": ["module.py:7"], "uncertainties": []}
    client = NativeScriptedLLMClient(
        [json.dumps({"action": "submit_exploration_result", "action_input": {"report": report}})]
    )
    monkeypatch.setattr(
        "dm_agent.subagents.worker.parse_report",
        lambda *_args: pytest.fail("structured tool completion must not parse final-answer JSON"),
    )
    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "locate the implementation",
        },
        client,
    )
    assert result["status"] == "succeeded"
    assert result["report"] == report
    assert result["completion_protocol"] == "structured_tool"
    assert len(client.requests) == 1
    definitions = client.options[0]["tool_definitions"]
    submit = next(item for item in definitions if item["name"] == "submit_exploration_result")
    assert submit["parameters"]["required"] == ["report"]
    assert submit["parameters"]["properties"]["report"] == {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "findings", "uncertainties"],
        "additionalProperties": False,
    }
    evidence = result["evidence"]
    assert not evidence


def test_worker_uses_structured_terminal_tool_in_prompt_json_mode(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = {"summary": "located", "findings": [], "uncertainties": ["no tests found"]}
    client = ScriptedLLMClient(
        [json.dumps({"action": "submit_exploration_result", "action_input": {"report": report}})]
    )
    monkeypatch.setattr(
        "dm_agent.subagents.worker.parse_report",
        lambda *_args: pytest.fail(
            "the JSON protocol must submit action_input, not a final JSON string"
        ),
    )
    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "locate tests",
        },
        client,
    )
    assert result["status"] == "succeeded"
    assert result["report"] == report
    assert result["completion_protocol"] == "structured_tool"


def test_structured_submission_accepts_source_text_that_mentions_failure(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = {
        "summary": "manager records worker_failed in its fallback path",
        "findings": ["manager.py:271 reads result.json"],
        "uncertainties": [],
    }
    client = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "action": "submit_exploration_result",
                    "action_input": {"report": report},
                }
            )
        ]
    )

    result = execute(
        {
            "settings": {"workspace": str(workspace)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "report the fallback path",
        },
        client,
    )

    assert result["status"] == "succeeded"
    assert result["report"] == report
    assert result["model_calls"] == 1


def test_worker_starts_its_own_lsp_service_for_read_only_queries(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    class FakeLspService:
        def __init__(self, root, *, command, timeout_seconds):
            calls.append(("init", root, command, timeout_seconds))

        def start(self, run_id):
            calls.append(("start", run_id))
            return True

        def query(self, path, action, line, character):
            calls.append(("query", path, action, line, character))
            return {"status": "ok", "action": action, "items": [], "truncated": False}

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr("dm_agent.lsp_impact.service.LspImpactService", FakeLspService)
    report = {"summary": "no references", "findings": [], "uncertainties": []}
    client = ScriptedLLMClient(
        [
            json.dumps(
                {
                    "action": "lsp_query",
                    "action_input": {
                        "path": "sample.py",
                        "action": "references",
                        "line": 0,
                        "character": 0,
                    },
                }
            ),
            json.dumps({"action": "submit_exploration_result", "action_input": {"report": report}}),
        ]
    )
    result = execute(
        {
            "settings": {"workspace": str(workspace), "lsp": True, "lsp_command": "fake-lsp"},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "find references",
        },
        client,
    )
    assert result["status"] == "succeeded"
    assert calls[1][0] == "start"
    assert calls[2][0] == "query"
    assert calls[-1] == ("close",)
    assert result["evidence"][0]["tool"] == "lsp_query"


def test_invalid_report_is_not_success():
    with pytest.raises(ValueError):
        parse_report('{"summary": "ok"}')


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("channel", ["submit_exploration_result", "finish"])
def test_delivery_can_correct_twice_then_succeed(tmp_path, native, channel):
    report = {"summary": "located", "findings": [], "uncertainties": []}
    invalid = {"summary": "missing fields"}

    def action(value):
        return json.dumps(
            {
                "action": channel,
                "action_input": {"report": value} if channel != "finish" else json.dumps(value),
            }
        )

    client_type = NativeScriptedLLMClient if native else ScriptedLLMClient
    client = client_type([action(invalid), action(invalid), action(report)])
    attempt = tmp_path / "attempt"
    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(attempt),
            "context": "investigate",
            "instruction": "locate implementation",
        },
        client,
    )
    assert result["status"] == "succeeded"
    assert result["schema_valid"] is True
    assert result["report"] == report
    assert result["rejected_delivery_count"] == 2
    assert len(client.requests) == 3
    assert "Correction attempts remaining: 2" in str(client.requests[1])
    entries = [
        json.loads(line)
        for line in (attempt / "rejected_submissions.jsonl").read_text().splitlines()
    ]
    assert len(entries) == 2
    assert entries[0]["candidate"] == (
        {"report": invalid} if channel != "finish" else json.dumps(invalid)
    )


def test_delivery_budget_shared_between_channels_and_stops_requests(tmp_path):
    invalid = {"summary": "missing fields"}
    client = ScriptedLLMClient(
        [
            json.dumps(
                {"action": "submit_exploration_result", "action_input": {"report": invalid}}
            ),
            json.dumps({"action": "finish", "action_input": "I found something"}),
            json.dumps({"action": "submit_exploration_result", "action_input": None}),
            json.dumps(
                {
                    "action": "finish",
                    "action_input": json.dumps(
                        {"summary": "late", "findings": [], "uncertainties": []}
                    ),
                }
            ),
        ]
    )
    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "investigate",
            "instruction": "locate implementation",
        },
        client,
    )
    assert len(client.requests) == 3
    assert result["status"] == "failed"
    assert result["error"] == "report_corrections_exhausted"
    assert result["report"] is None
    assert result["schema_valid"] is False
    assert result["rejected_delivery_count"] == 3


def test_corrections_do_not_extend_step_budget(tmp_path):
    client = ScriptedLLMClient([json.dumps({"action": "finish", "action_input": "not JSON"})])
    result = execute(
        {
            "settings": {"workspace": str(tmp_path), "steps": 1},
            "attempt": str(tmp_path / "attempt"),
            "context": "x",
            "instruction": "x",
        },
        client,
    )
    assert result["status"] == "failed"
    assert result["report"] is None
    assert result["rejected_delivery_count"] == 1
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "report,schema",
    [(None, {"type": "null"}), ([1], {"type": "array", "items": {"type": "integer"}})],
)
def test_legacy_finish_uses_caller_schema_including_null(tmp_path, report, schema):
    client = ScriptedLLMClient(
        [json.dumps({"action": "finish", "action_input": json.dumps(report)})]
    )
    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "x",
            "instruction": "x",
            "output_schema": schema,
        },
        client,
    )
    assert result["status"] == "succeeded"
    assert result["report"] == report
    assert result["completion_protocol"] == "legacy_finish_json"


def test_model_error_after_rejection_remains_failure(tmp_path):
    from dm_agent.clients import LLMError

    class BrokenClient:
        def respond(self, messages, **kwargs):
            raise LLMError("provider unavailable")

    result = execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "x",
            "instruction": "x",
        },
        BrokenClient(),
    )
    assert result["status"] == "failed"
    assert result["error"] == "LLMError"
    assert result["report"] is None
    assert result["completion_protocol"] == "invalid"


def test_invalid_batch_has_no_partial_execution(manager):
    with pytest.raises(ValueError):
        manager.batch({"context": "x", "tasks": [{"instruction": "a"}, {"instruction": ""}]})
    assert not manager.records


def test_followup_uses_latest_attempt(manager):
    first = json.loads(manager.batch(batch("first")).message)["results"][0]
    followup = {"context": "x", "tasks": [{"instruction": "second", "session_id": first["id"]}]}
    second = json.loads(manager.batch(followup).message)["results"][0]
    third = json.loads(manager.batch(followup).message)["results"][0]
    assert third["previous_attempt"] == second["id"]
    assert third["session_id"] == first["session_id"]


def test_store_is_exclusive_and_completed_results_survive(manager):
    result = json.loads(manager.batch(batch("saved")).message)["results"][0]
    with pytest.raises((ValueError, OSError)):
        TaskManager(manager.root, manager.settings)
    manager.close()
    reopened = TaskManager(manager.root, manager.settings)
    try:
        assert reopened.records[result["id"]]["status"] == "succeeded"
        assert json.loads(reopened.result({"task_id": result["id"]}).message)["text"]
    finally:
        reopened.close()


def test_followup_skips_attempt_without_checkpoint(manager):
    first = json.loads(manager.batch(batch("first")).message)["results"][0]
    checkpoint = manager.root / first["id"] / "checkpoint.jsonl"
    checkpoint.write_text("saved history", encoding="utf-8")
    followup = {
        "context": "follow up",
        "tasks": [{"instruction": "fail", "session_id": first["id"]}],
    }
    manager.batch(followup)
    followup["tasks"][0]["instruction"] = "retry"
    latest = json.loads(manager.batch(followup).message)["results"][0]
    assert (manager.root / latest["id"] / "previous.txt").read_text() == str(checkpoint.parent)


def test_torn_journal_preserved_and_recovery_repeatable(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    (root / "tasks.jsonl").write_text('{"unfinished":', encoding="utf-8")
    first = TaskManager(root, {})
    first.close()
    second = TaskManager(root, {})
    second.close()
    assert (root / "tasks.jsonl").read_text().startswith('{"unfinished":')


def test_custom_output_schema_and_unsupported_keywords(manager):
    from dm_agent.subagents.schema import check_schema

    schema = {"type": "array", "items": {"type": "integer"}}
    assert parse_report("[1,2]", schema) == [1, 2]
    with pytest.raises(ValueError):
        parse_report("[true]", schema)
    with pytest.raises(ValueError):
        check_schema({"type": "object", "$ref": "remote"})
    with pytest.raises(ValueError):
        manager.batch(
            {
                "context": "x",
                "tasks": [{"instruction": "a", "output_schema": {"type": "object", "$ref": "x"}}],
            }
        )
    assert not manager.records


def test_parent_react_records_observation_not_verification(manager, monkeypatch):
    from dm_agent.core.agent import ReactAgent
    from dm_agent.extensions.capabilities.evidence import EvidenceGraphCapability

    monkeypatch.chdir(manager.settings["workspace"])
    client = ScriptedLLMClient(
        [
            json.dumps({"action": "task", "action_input": batch("ok")}),
            json.dumps({"action": "finish", "action_input": "investigation complete"}),
        ]
    )
    evidence = EvidenceGraphCapability()
    agent = ReactAgent(
        client,
        manager.tools(),
        enable_planning=False,
        enable_compression=False,
        capabilities=[manager, evidence],
    )
    try:
        result = agent.run("investigate only")
        assert result["metadata"]["subagents"][0]["status"] == "succeeded"
        nodes = list(evidence.graph.nodes.values())
        assert any(
            n.kind == "observation" and n.metadata.get("kind") == "subagent_report" for n in nodes
        )
        assert not any(n.kind == "verification" for n in nodes)
    finally:
        agent.close()


def test_worker_denies_write_even_if_model_requests_it(tmp_path):
    answer = json.dumps({"summary": "denied", "findings": [], "uncertainties": []})
    client = ScriptedLLMClient(
        [
            json.dumps(
                {"action": "create_file", "action_input": {"path": "bad.txt", "content": "bad"}}
            ),
            json.dumps({"action": "finish", "action_input": answer}),
        ]
    )
    execute(
        {
            "settings": {"workspace": str(tmp_path)},
            "attempt": str(tmp_path / "attempt"),
            "context": "test",
            "instruction": "read only",
        },
        client,
    )
    assert not (tmp_path / "bad.txt").exists()


def test_real_sessions_overlap_without_processes_or_context_leaks(tmp_path, monkeypatch):
    import os
    import subprocess

    barrier = threading.Barrier(2)
    clients = []
    pids = []

    class IndependentClient:
        def __init__(self):
            self.messages = None
            self.closed = False
            clients.append(self)

        def respond(self, messages, **kwargs):
            self.messages = messages
            pids.append(os.getpid())
            barrier.wait(timeout=5)
            marker = "alpha" if "alpha" in str(messages) else "beta"
            report = {"summary": marker, "findings": [], "uncertainties": []}
            return json.dumps(
                {"action": "submit_exploration_result", "action_input": {"report": report}}
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no worker processes"))
    monkeypatch.setattr(
        "dm_agent.subagents.worker.create_llm_client", lambda **k: IndependentClient()
    )
    settings = {
        "workspace": str(tmp_path),
        "provider": "fake",
        "api_key": "private",
        "model": "fake",
        "base_url": "",
    }
    manager = TaskManager(tmp_path / "store", settings, workers=2)
    try:
        records = json.loads(manager.batch(batch("alpha", "beta")).message)["results"]
        assert [r["summary"] for r in records] == ["alpha", "beta"]
        assert all(r["execution_backend"] == "in_process" for r in records)
        assert pids == [os.getpid(), os.getpid()]
        assert len(clients) == 2 and all(c.closed for c in clients)
        assert clients[0].messages is not clients[1].messages
        assert all(not ("alpha" in str(c.messages) and "beta" in str(c.messages)) for c in clients)
        assert "private" not in manager.journal.read_text()
        assert all((manager.root / r["id"] / "checkpoint.jsonl").exists() for r in records)
    finally:
        manager.close()


def test_cancel_drains_inflight_request_and_discards_late_tool_call(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    output = []

    class BlockingClient:
        def respond(self, messages, **kwargs):
            entered.set()
            assert release.wait(5)
            return json.dumps({"action": "read_file", "action_input": {"path": "target.py"}})

        def close(self):
            closed.set()

    monkeypatch.setattr("dm_agent.subagents.worker.create_llm_client", lambda **k: BlockingClient())
    monkeypatch.setattr("dm_agent.subagents.worker.scoped_tools", lambda root: [])
    manager = TaskManager(
        tmp_path / "store",
        {
            "workspace": str(tmp_path),
            "provider": "fake",
            "api_key": "x",
            "model": "fake",
            "base_url": "",
        },
        workers=1,
    )
    thread = threading.Thread(target=lambda: output.append(manager.batch(batch("one", "two"))))
    try:
        thread.start()
        assert entered.wait(5)
        manager.cancel()
        assert thread.is_alive()  # Parent cannot advance while synchronous work is still running.
        release.set()
        thread.join(5)
        assert not thread.is_alive() and closed.is_set()
        records = json.loads(output[0].message)["results"]
        assert [r["status"] for r in records] == ["cancelled", "cancelled"]
        assert not (manager.root / records[0]["id"] / "answer.txt").exists()
    finally:
        release.set()
        thread.join(5)
        manager.close()


def test_deadline_expired_response_cannot_finish_or_continue(tmp_path):
    from dm_agent.subagents.control import ExplorationStopped, RunControl

    class LateClient:
        def respond(self, messages, **kwargs):
            control.deadline = time.monotonic() - 1
            return json.dumps({"action": "finish", "action_input": "late"})

    control = RunControl(threading.Event(), 5)
    with pytest.raises(ExplorationStopped) as stopped:
        execute(
            {
                "settings": {"workspace": str(tmp_path)},
                "attempt": str(tmp_path / "attempt"),
                "context": "x",
                "instruction": "x",
            },
            LateClient(),
            control=control,
        )
    assert stopped.value.status == "timed_out"
    assert not (tmp_path / "attempt" / "answer.txt").exists()


@pytest.mark.parametrize("provider", ["openai", "claude", "gemini", "deepseek"])
def test_worker_request_limits_reach_provider_transport(tmp_path, provider):
    from types import SimpleNamespace

    from dm_agent.clients import ClaudeClient, DeepSeekClient, GeminiClient, OpenAIClient
    from dm_agent.subagents.control import RunControl
    from dm_agent.subagents.delivery import DeliveryPolicy
    from dm_agent.subagents.schema import DEFAULT_REPORT
    from dm_agent.subagents.worker import Meter

    seen = {}

    class SDK:
        def with_options(self, **options):
            seen.update(options)
            return self

        def generate_content(self, **options):
            seen.update(options)
            return SimpleNamespace(text="done")

    cls = {
        "openai": OpenAIClient,
        "claude": ClaudeClient,
        "gemini": GeminiClient,
        "deepseek": DeepSeekClient,
    }[provider]
    client = object.__new__(cls)
    sdk = SDK()
    sdk.models = sdk
    client.client = sdk
    client.model = "fake"

    def respond(messages, **options):
        if provider == "gemini":
            client.complete(messages, **options)
        return "done"

    client.respond = respond
    meter = Meter(
        client,
        tmp_path / "usage.jsonl",
        DeliveryPolicy(DEFAULT_REPORT, tmp_path / "rejected.jsonl"),
        RunControl(threading.Event(), 2),
    )
    assert meter.respond([{"role": "user", "content": "test"}]) == "done"
    assert 0 < client.timeout <= 2
    if provider in {"openai", "claude"}:
        assert seen["timeout"] == client.timeout
        assert seen["max_retries"] == 0
    if provider == "gemini":
        http = seen["config"]["http_options"]
        assert 0 < http["timeout"] <= 2000
        assert http["retry_options"] == {"attempts": 1}


def test_close_waits_for_child_cleanup_and_rejects_new_batches(tmp_path, monkeypatch):
    from dm_agent.subagents.control import ExplorationStopped

    entered = threading.Event()
    cleanup = threading.Event()
    release_cleanup = threading.Event()

    def worker(request, *, control):
        entered.set()
        control.cancelled.wait(5)
        try:
            control.check()
        except ExplorationStopped:
            cleanup.set()
            assert release_cleanup.wait(5)
            raise

    monkeypatch.setattr("dm_agent.subagents.manager.execute", worker)
    manager = TaskManager(tmp_path / "store", {"workspace": str(tmp_path)})
    batch_thread = threading.Thread(target=lambda: manager.batch(batch("one")))
    close_thread = threading.Thread(target=manager.close)
    try:
        batch_thread.start()
        assert entered.wait(5)
        close_thread.start()
        assert cleanup.wait(5)
        assert close_thread.is_alive()
        release_cleanup.set()
        close_thread.join(5)
        batch_thread.join(5)
        assert not close_thread.is_alive() and not batch_thread.is_alive()
        assert next(iter(manager.records.values()))["status"] == "cancelled"
        with pytest.raises(RuntimeError, match="closed"):
            manager.batch(batch("later"))
    finally:
        release_cleanup.set()
        batch_thread.join(5)
        if close_thread.ident:
            close_thread.join(5)
        manager.close()
