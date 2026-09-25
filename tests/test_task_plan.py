"""The checklist is a model claim; runtime evidence remains independent."""

import json
from dataclasses import replace

import pytest

from dm_agent.core.agent import ReactAgent
from dm_agent.core.checkpoint import load_checkpoint
from dm_agent.core.events import EventBus
from dm_agent.core.evidence import EvidenceGraph
from dm_agent.core.task_plan import TaskPlan
from dm_agent.extensions.capabilities.evidence import EvidenceGraphCapability
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing import TraceWriter, load_trace_events
from dm_agent.tracing.analysis import analyze_events
from dm_agent.tracing.evidence import rebuild_evidence_graph
from dm_agent.tracing.summary import summarize_events


def action(name, args):
    return json.dumps({"thought": "", "action": name, "action_input": args})


def item(identity="P1", status="in_progress", step="Inspect the implementation"):
    return {"id": identity, "step": step, "status": status}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.options = []

    def respond(self, messages, **kwargs):
        self.requests.append(messages)
        self.options.append(kwargs)
        return next(self.responses)


def agent_for(responses, **kwargs):
    client = Client(responses)
    agent = ReactAgent(
        client,
        [Tool("echo", "Echo", lambda args: "ok", read_only=True)],
        enable_compression=False,
        **kwargs,
    )
    return agent, client


def test_plan_revisions_are_immutable_and_removed_items_are_not_completed():
    plan = TaskPlan()
    original = {"plan": [item(), item("P2", "pending")]}
    assert plan.update(original).status == "success"
    original["plan"][0]["step"] = "Mutated by caller"
    plan.update({"plan": [item(step="New title")]})
    snapshot = plan.snapshot()
    assert snapshot["revisions"][0]["plan"][1]["status"] == "pending"
    assert snapshot["revisions"][0]["plan"][0]["step"] != "Mutated by caller"
    restored = TaskPlan()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    snapshot["revisions"].clear()
    assert restored.revision == 2


@pytest.mark.parametrize(
    "items",
    [
        [item(), item()],
        [item(), item("P2")],
        [item(status="satisfied")],
        [item(identity="../bad")],
        [{"id": "P1", "step": "x"}],
        [item(step=" ")],
    ],
)
def test_invalid_plan_does_not_partially_change_state(items):
    plan = TaskPlan()
    plan.update({"plan": [item()]})
    before = plan.snapshot()
    assert plan.update({"plan": items}).status == "failed"
    assert plan.snapshot() == before


@pytest.mark.parametrize("legacy", [False, True])
def test_errors_do_not_call_another_planner_and_tool_success_does_not_complete_goal(legacy):
    agent, client = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("unknown", {}),
            "invalid json",
            action("echo", {}),
            action("finish", "done"),
        ],
        enable_adaptive_replanning=legacy,
    )
    result = agent.run("inspect", max_steps=5)
    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["replan_count"] == 0
    assert len(client.requests) == 5
    assert agent.task_plan.items == [item()]
    assert "model-reported" in client.requests[-1][0]["content"]


def test_simple_task_can_finish_without_creating_a_plan():
    agent, client = agent_for([action("finish", "done")])
    result = agent.run("answer directly", max_steps=1)
    assert len(client.requests) == 1
    assert result["metadata"]["plan"] == []


def test_native_tool_catalog_includes_update_plan():
    client = Client(
        [
            action("update_plan", {"plan": [item()]}),
            action("finish", "done"),
        ]
    )
    client.supports_tool_calling = True
    agent = ReactAgent(
        client, [Tool("echo", "Echo", lambda _: "ok", read_only=True)], enable_compression=False
    )
    result = agent.run("inspect", max_steps=2)
    assert result["metadata"]["status"] == "success"
    definitions = client.options[0]["tool_definitions"]
    assert next(d for d in definitions if d["name"] == "update_plan")["parameters"]["required"] == [
        "plan"
    ]
    assert agent.task_plan.items == [item()]


def test_trace_summary_uses_latest_checklist_and_resets_for_new_task():
    events = [
        {"event": "run_start", "payload": {"planning_mode": "model_checklist"}},
        {"event": "plan_updated", "payload": {"plan": [item()]}},
        {"event": "plan_updated", "payload": {"plan": [item(status="completed")]}},
    ]
    step = summarize_events(events)["plan_steps"][0]
    assert step["completed"] is True
    assert step["status_source"] == "model_reported"
    events.append(
        {"event": "run_start", "payload": {"planning_mode": "model_checklist", "plan": []}}
    )
    assert summarize_events(events)["plan_steps"] == []


def test_disabled_planning_has_no_update_tool():
    agent, _ = agent_for([action("finish", "done")], enable_planning=False)
    assert "update_plan" not in agent.tools
    assert agent.run("answer", max_steps=1)["metadata"]["status"] == "success"


def test_new_task_resets_plan_scope_and_latest_snapshot_even_with_shared_history():
    agent, client = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("finish", "done"),
            action("finish", "new task done"),
        ]
    )
    agent.run("first task", max_steps=2)
    first_scope = agent.task_plan.scope
    agent.run("second task", max_steps=1)
    assert agent.task_plan.scope != first_scope
    assert agent.task_plan.items == []
    assert '"plan": []' in client.requests[-1][0]["content"]


def test_checkpoint_restores_plan_revisions_and_evidence_together(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "checkpoint.json"
    graph1 = EvidenceGraphCapability()
    first, _ = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("echo", {}),
        ],
        capabilities=[graph1],
    )
    first.run("work", max_steps=2, checkpoint_path=path)
    state = load_checkpoint(path)
    graph2 = EvidenceGraphCapability()
    second, client = agent_for([action("finish", "done")], capabilities=[graph2])
    second.run("work", max_steps=3, resume_state=state)
    assert second.task_plan.snapshot() == first.task_plan.snapshot()
    assert graph2.graph.nodes.keys() >= graph1.graph.nodes.keys()
    assert '"id": "P1"' in client.requests[0][0]["content"]


def test_plan_links_use_call_start_revision_and_removed_history_survives(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = TraceWriter(tmp_path / "trace.jsonl")
    evidence = EvidenceGraphCapability()
    agent, _ = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("echo", {}),
            action("update_plan", {"plan": [item("P2", step="Different route")]}),
            action("echo", {}),
            action("update_plan", {"plan": []}),
            action("echo", {}),
            action("finish", "done"),
        ],
        capabilities=[evidence],
        trace_writer=writer,
    )
    agent.run("work", max_steps=7)
    writer.close()
    graph = evidence.graph
    nodes = {
        node.step_number: node for node in graph.nodes.values() if node.kind == "tool_execution"
    }
    links = {
        edge.source_id: graph.nodes[edge.target_id]
        for edge in graph.edges
        if edge.relation == "occurred_during"
    }
    assert nodes[1].node_id not in links
    assert links[nodes[2].node_id].metadata["plan_id"] == "P1"
    assert links[nodes[3].node_id].metadata["plan_id"] == "P1"
    assert links[nodes[4].node_id].metadata["plan_id"] == "P2"
    assert nodes[6].node_id not in links
    plans = [node for node in graph.nodes.values() if node.kind == "plan_step"]
    assert len(plans) == 2
    assert all(not node.metadata["current"] for node in plans)
    events = load_trace_events(tmp_path / "trace.jsonl")
    restored = rebuild_evidence_graph(events)
    assert restored.nodes == graph.nodes
    assert analyze_events(events)["planning"]["update_count"] == 3


def test_model_completed_plan_cannot_override_direct_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evidence = EvidenceGraphCapability()
    client = Client(
        [
            action("update_plan", {"plan": [item()]}),
            action("run_tests", {}),
            action("update_plan", {"plan": [item(status="completed")]}),
            action("finish", "all done"),
        ]
    )
    failed = ToolResult(
        "failed",
        "AssertionError: wrong result",
        exit_code=1,
        check_scope=("tests/test_x.py::test_target",),
        metadata={
            "verification": {
                "execution_status": "completed",
                "outcome": "failed",
                "failure_kind": "assertion_failure",
                "scope_level": "direct",
            }
        },
    )
    agent = ReactAgent(
        client,
        [Tool("run_tests", "Test", lambda args: failed)],
        capabilities=[evidence],
        enable_compression=False,
    )
    result = agent.run("fix target", max_steps=4)
    assert result["metadata"]["status"] != "success"
    assert result["metadata"]["evidence_completion_block_count"] == 1
    assert agent.task_plan.items[0]["status"] == "completed"


def test_no_progress_hook_does_not_rewrite_model_reported_plan():
    bus = EventBus()

    def no_progress(event):
        event.no_change = True

    bus.on("after_tool_result", no_progress)
    agent, _ = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("echo", {}),
        ],
        event_bus=bus,
    )
    agent.run("work", max_steps=2)
    assert agent.task_plan.items == [item()]


def test_current_plan_survives_lossy_lcm_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class SummaryClient(Client):
        summary_calls = 0

        def complete_summary(self, messages, **kwargs):
            self.summary_calls += 1
            return {"text": "Earlier work summarized without plan details."}

        def extract_text(self, data):
            return data["text"]

    client = SummaryClient(
        [
            action("update_plan", {"plan": [item()]}),
            *[action("echo", {}) for _ in range(6)],
            action("finish", "done"),
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("echo", "Echo", lambda _: "observation " + "x" * 1800, read_only=True)],
        context_token_budget=2000,
        system_prompt="Use tools to work.",
    )
    try:
        agent._context_window.output_token_reserve = 0
        agent._context_window.safety_margin_tokens = 0
        _ = agent.compressor.store
        agent.compressor.compactor.policy = replace(
            agent.compressor.compactor.policy, keep_recent=2
        )
        result = agent.run("Inspect the implementation", max_steps=8)
        assert result["metadata"]["status"] == "success"
        assert client.summary_calls > 0
        assert any("<historical_summary" in m["content"] for m in client.requests[-1])
        system = client.requests[-1][0]["content"]
        assert '"id": "P1"' in system
        assert '"status": "in_progress"' in system
        assert system.count("Current task plan") == 1
        assert not any("Current task plan" in m["content"] for m in agent.conversation_history)
    finally:
        agent.close()


def test_completion_references_only_current_verification_not_recent_reads():
    graph = EvidenceGraph("Fix")
    graph.workspace_version = "v1"
    graph.add_verification(tool="run_tests", step_number=1, passed=False, workspace_version="v1")
    graph.workspace_version = "v2"
    current, _ = graph.add_verification(
        tool="run_tests",
        step_number=2,
        passed=True,
        workspace_version="v2",
    )
    graph.add_observation(tool="read_file", path="x.py", step_number=3, succeeded=True)
    _, edges = graph.add_conclusion(text="done", step_number=4)
    used = [e for e in edges if e.relation == "checked_at_completion"]
    assert [e.target_id for e in used] == [current[0].node_id]
    assert all(e.confidence == "deterministic" for e in used)


def test_shared_trace_rebuild_keeps_only_latest_task_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = TraceWriter(tmp_path / "trace.jsonl")
    evidence = EvidenceGraphCapability()
    agent, _ = agent_for(
        [
            action("update_plan", {"plan": [item()]}),
            action("finish", "first"),
            action("finish", "second"),
        ],
        trace_writer=writer,
        capabilities=[evidence],
    )
    agent.run("first task", max_steps=2)
    agent.run("second task", max_steps=1)
    writer.close()
    rebuilt = rebuild_evidence_graph(load_trace_events(writer.path))
    assert rebuilt.nodes == evidence.graph.nodes
    assert not any(n.kind == "plan_step" for n in rebuilt.nodes.values())


def test_trace_completion_version_does_not_resurrect_old_failure():
    graph = EvidenceGraph("Fix")
    graph.workspace_version = "v1"
    graph.add_verification(tool="run_tests", step_number=1, passed=False, workspace_version="v1")
    graph.workspace_version = "v2"
    graph.add_conclusion(text="unverified", step_number=2)
    events = [
        {"event": "evidence_node", "payload": node.to_dict()} for node in graph.nodes.values()
    ]
    rebuilt = rebuild_evidence_graph(events)
    assert rebuilt.workspace_version == "v2"
    assert rebuilt.current_verifications() == []
