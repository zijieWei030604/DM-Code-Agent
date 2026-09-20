import json

import pytest

from dm_agent.clients.base_client import LLMError
from dm_agent.core.agent import ReactAgent
from dm_agent.core.planner import PHASE_GOALS, AdaptiveReplanPolicy, TaskPlanner
from dm_agent.tools.base import Tool, ToolResult
from dm_agent.tracing import TraceWriter, load_trace_events


class FakeRespondClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def respond(self, messages, **extra):
        self.requests.append((messages, extra))
        if not self.responses:
            raise AssertionError("FakeRespondClient ran out of responses")
        return self.responses.pop(0)


class NativeToolFakeClient(FakeRespondClient):
    supports_tool_calling = True

    def respond(self, messages, **extra):
        response = super().respond(messages, **extra)
        self.last_response_mode = "native_tool_call"
        self.last_tool_call_count = 1
        return response


class StructuredFakeClient(FakeRespondClient):
    supports_json_schema = True


def test_task_planner_uses_strict_schema_when_client_supports_it():
    client = StructuredFakeClient(['{"plan":[{"step":1,"action":"read_file","reason":"inspect"}]}'])
    planner = TaskPlanner(
        client,
        [Tool("read_file", "Read a file", lambda arguments: "content")],
    )

    plan = planner.plan("inspect a file")

    schema = client.requests[0][1]["json_schema"]
    step_schema = schema["properties"]["plan"]["items"]
    assert "goal" not in step_schema["properties"]
    assert step_schema["properties"]["phase"]["enum"] == [
        "locate",
        "inspect",
        "change",
        "validate",
    ]
    assert step_schema["properties"]["preferred_tools"]["items"]["enum"] == [
        "read_file",
    ]
    assert step_schema["additionalProperties"] is False
    assert [step.action for step in plan] == ["read_file"]
    assert plan[0].goal == PHASE_GOALS["inspect"]


def test_task_planner_keeps_prompt_json_fallback_for_other_clients():
    client = FakeRespondClient(['{"plan":[{"step":1,"action":"read_file","reason":"inspect"}]}'])
    planner = TaskPlanner(
        client,
        [Tool("read_file", "Read a file", lambda arguments: "content")],
    )

    planner.plan("inspect a file")

    assert "json_schema" not in client.requests[0][1]


def test_react_agent_closes_owned_resources_once():
    class Closable:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    client = FakeRespondClient([])
    client.close = Closable().close
    resource = Closable()
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
        owned_resources=[resource],
    )

    agent.close()
    agent.close()

    assert resource.close_count == 1
    assert client.close.__self__.close_count == 1


def test_task_planner_parses_json_inside_text():
    client = FakeRespondClient(
        [
            'Plan:\n{"plan": ['
            '{"step": 1, "action": "read_file", "reason": "inspect input"},'
            '{"step": 2, "action": "task_complete", "reason": "finish"}'
            "]}"
        ]
    )
    tools = [
        Tool("read_file", "Read a file", lambda arguments: "content"),
        Tool("task_complete", "Finish", lambda arguments: "done"),
    ]

    planner = TaskPlanner(client, tools)
    plan = planner.plan("inspect a file")

    assert [step.action for step in plan] == ["read_file"]
    assert planner.get_next_step().action == "read_file"
    planner.mark_completed(1, "ok")
    assert planner.get_next_step() is None


def test_react_agent_can_finish_without_tool_call():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "thought": "The answer is ready.",
                    "action": "finish",
                    "action_input": {"answer": "done"},
                }
            )
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish immediately")

    assert result["final_answer"] == "done"
    assert result["steps"][0]["action"] == "finish"
    assert result["metadata"]["status"] == "success"


def test_react_agent_sends_provider_neutral_single_call_tool_definitions():
    client = NativeToolFakeClient(
        [
            json.dumps(
                {
                    "thought": "",
                    "action": "finish",
                    "action_input": "done",
                }
            )
        ]
    )
    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "additionalProperties": False,
    }
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished", input_schema=schema)],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish immediately")

    assert result["final_answer"] == "done"
    assert client.requests[0][1]["tool_choice"] == "auto"
    assert client.requests[0][1]["tool_definitions"] == [
        {"name": "task_complete", "description": "Finish", "parameters": schema}
    ]


def test_react_agent_retries_missing_native_call_without_polluting_history():
    class MissingThenNativeClient(FakeRespondClient):
        supports_tool_calling = True

        def respond(self, messages, **extra):
            response = super().respond(messages, **extra)
            self.last_response_mode = (
                "json_fallback" if len(self.requests) == 1 else "native_tool_call"
            )
            self.last_tool_call_count = 0 if len(self.requests) == 1 else 1
            return response

    response = json.dumps(
        {"thought": "", "action": "finish", "action_input": {"answer": "done"}}
    )
    client = MissingThenNativeClient(["plain text", response])
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish immediately", max_steps=1)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["native_tool_call_missing_count"] == 1
    assert result["metadata"]["native_tool_retry_count"] == 1
    assert len(client.requests) == 2
    assert client.requests[0][1]["tool_choice"] == "auto"
    assert client.requests[1][1]["tool_choice"] == "required"
    assert client.requests[1][0][-1]["role"] == "system"
    assert "native tool call" in client.requests[1][0][-1]["content"]
    assert all(
        "native tool call" not in message.get("content", "")
        for message in agent.conversation_history
    )


def test_react_agent_downgrades_only_explicit_required_tool_choice_rejection():
    class RequiredUnsupportedClient(FakeRespondClient):
        supports_tool_calling = True

        def respond(self, messages, **extra):
            self.requests.append((messages, extra))
            if len(self.requests) == 1:
                self.last_response_mode = "json_fallback"
                self.last_tool_call_count = 0
                return self.responses.pop(0)
            if len(self.requests) == 2:
                raise LLMError("400: tool_choice required is not supported")
            self.last_response_mode = "native_tool_call"
            self.last_tool_call_count = 1
            return self.responses.pop(0)

    invalid = "plain text"
    response = json.dumps(
        {"thought": "", "action": "finish", "action_input": {"answer": "done"}}
    )
    client = RequiredUnsupportedClient([invalid, response])
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish immediately", max_steps=1)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["protocol_downgrade_count"] == 1
    assert [request[1]["tool_choice"] for request in client.requests] == [
        "auto",
        "required",
        "auto",
    ]


def test_react_agent_bounds_native_retries_across_the_run():
    class RepeatedMissingClient(FakeRespondClient):
        supports_tool_calling = True

        def respond(self, messages, **extra):
            response = super().respond(messages, **extra)
            self.last_response_mode = (
                "native_tool_call"
                if extra.get("tool_choice") == "required"
                else "json_fallback"
            )
            self.last_tool_call_count = 1 if self.last_response_mode == "native_tool_call" else 0
            return response

    noop = json.dumps({"thought": "", "action": "noop", "action_input": {}})
    finish = json.dumps(
        {"thought": "", "action": "finish", "action_input": {"answer": "done"}}
    )
    client = RepeatedMissingClient(
        ["plain text", noop, "plain text", noop, "plain text", noop, finish]
    )
    agent = ReactAgent(
        client,
        [
            Tool("noop", "No operation", lambda arguments: "ok"),
            Tool("task_complete", "Finish", lambda arguments: "finished"),
        ],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish after three checks", max_steps=4)

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["native_tool_call_missing_count"] == 4
    assert result["metadata"]["native_tool_retry_count"] == 3
    assert result["metadata"]["native_tool_call_count"] == 3
    assert result["metadata"]["json_fallback_count"] == 1
    assert len(client.requests) == 7


def test_react_agent_stops_on_common_terminal_action_alias():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "thought": "The task is already complete.",
                    "action": "stop",
                    "action_input": {"message": "stopped cleanly"},
                }
            )
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("stop when done", max_steps=3)

    assert result["final_answer"] == "stopped cleanly"
    assert result["steps"][0]["action"] == "finish"
    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["unknown_tool_count"] == 0
    assert result["metadata"]["terminal_action_alias_count"] == 1
    assert result["metadata"]["terminal_action_aliases"][0]["raw"] == "stop"


def test_react_agent_adds_completion_summary_for_terse_finish():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "thought": "The task is complete.",
                    "action": "finish",
                    "action_input": "ok",
                }
            )
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish tersely")

    assert result["final_answer"] == "ok"
    assert result["metadata"]["completion_summary"] == "任务已完成。结果：ok"


def test_task_complete_accepts_answer_shaped_completion():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "thought": "Use the completion tool.",
                    "action": "task_complete",
                    "action_input": {"answer": "answer shaped completion"},
                }
            )
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: f"done: {arguments['answer']}")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("finish with answer dict")

    assert result["final_answer"] == "done: answer shaped completion"
    assert result["metadata"]["status"] == "success"


def test_react_agent_executes_tool_then_finishes():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "thought": "Use the echo tool.",
                    "action": "echo",
                    "action_input": {"text": "hello"},
                }
            ),
            json.dumps(
                {
                    "thought": "The tool result is enough.",
                    "action": "finish",
                    "action_input": "completed",
                }
            ),
        ]
    )
    tool = Tool("echo", "Echo text", lambda arguments: f"echo:{arguments['text']}")
    agent = ReactAgent(
        client,
        [tool, Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("echo hello")

    assert result["final_answer"] == "completed"
    assert [step["action"] for step in result["steps"]] == ["echo", "finish"]
    assert "echo:hello" in result["steps"][0]["observation"]
    assert result["metadata"]["tool_error_count"] == 0


def test_react_agent_reset_conversation_clears_context_memory():
    agent = ReactAgent(
        FakeRespondClient([]),
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=True,
    )
    agent.conversation_history = [
        {"role": "user", "content": "Task: inspect app.py"},
        {"role": "assistant", "content": "Tool read_file app.py succeeded"},
        {"role": "user", "content": "Observation: pytest failed in app.py"},
        {"role": "assistant", "content": "Tool edit_file app.py completed"},
        {"role": "user", "content": "Observation: pytest passed in app.py"},
        {"role": "assistant", "content": "Tool read_file tests/test_app.py succeeded"},
        {"role": "user", "content": "Summarize app.py"},
    ]

    assert agent.compressor is not None
    agent.compressor.keep_recent = 1
    agent.compressor.compress(agent.conversation_history)

    assert agent.get_context_stats()["conversation_messages"] == 7
    assert agent.get_context_stats()["memory_items"] > 0

    agent.reset_conversation()

    assert agent.get_context_stats()["conversation_messages"] == 0
    assert agent.get_context_stats()["memory_items"] == 0


def test_react_agent_throttles_memory_status_output(capsys):
    responses = []
    for index in range(6):
        responses.append(
            json.dumps(
                {
                    "thought": f"Echo step {index}.",
                    "action": "echo",
                    "action_input": {"text": str(index)},
                }
            )
        )
    responses.append(
        json.dumps(
            {
                "thought": "Done.",
                "action": "finish",
                "action_input": "finished after repeated compression",
            }
        )
    )

    agent = ReactAgent(
        FakeRespondClient(responses),
        [
            Tool("echo", "Echo text", lambda arguments: f"echo:{arguments['text']}"),
            Tool("task_complete", "Finish", lambda arguments: "finished"),
        ],
        enable_planning=False,
        enable_compression=True,
    )
    assert agent.compressor is not None
    agent.compressor.compress_every = 1
    agent.compressor.keep_recent = 1

    result = agent.run("exercise memory output", max_steps=10)
    output = capsys.readouterr().out

    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["memory_compression_count"] > result["metadata"]["memory_log_count"]
    assert output.count("[memory]") == result["metadata"]["memory_log_count"]


def test_react_agent_rejects_empty_task():
    agent = ReactAgent(
        FakeRespondClient([]),
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    with pytest.raises(ValueError):
        agent.run(" ")


def test_react_agent_repairs_common_json_drift():
    client = FakeRespondClient(
        [
            "{'thought': 'single quotes', 'action': 'finish', "
            "'action_input': {'answer': 'repaired'},}"
        ]
    )
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda arguments: "finished")],
        enable_planning=False,
        enable_compression=False,
    )

    result = agent.run("repair json")

    assert result["final_answer"] == "repaired"
    assert result["metadata"]["parse_repair_count"] == 1


def test_react_agent_suppresses_first_generic_tool_failure():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "explode", "reason": "trigger failure"},
                        {"step": 2, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Try the failing tool.",
                    "action": "explode",
                    "action_input": {},
                }
            ),
            json.dumps(
                {
                    "thought": "Recover.",
                    "action": "task_complete",
                    "action_input": {"message": "recovered"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool("explode", "Fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom"))),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
    )

    result = agent.run("recover from failure")

    assert result["final_answer"] == "recovered"
    assert result["metadata"]["tool_error_count"] == 1
    assert result["metadata"]["replan_count"] == 0
    assert result["metadata"]["replan_suppressed_count"] == 1


def test_react_agent_keeps_plan_progress_out_of_model_messages():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {
                            "step": 1,
                            "phase": "locate",
                            "goal": "find implementation",
                            "preferred_tools": ["search_code"],
                            "completion_evidence": "candidate_observed",
                        },
                        {
                            "step": 2,
                            "phase": "complete",
                            "goal": "finish",
                            "preferred_tools": ["task_complete"],
                            "completion_evidence": "completion_accepted",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Locate the implementation.",
                    "action": "search_code",
                    "action_input": {"query": "Target"},
                }
            ),
            json.dumps(
                {
                    "thought": "The requested lookup is complete.",
                    "action": "task_complete",
                    "action_input": {"message": "done"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool(
                "search_code",
                "Search",
                lambda _arguments: ToolResult(
                    "success",
                    json.dumps({"match_count": 1, "matches": [{"path": "target.py"}]}),
                ),
            ),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
    )

    result = agent.run("locate the target")

    assert result["final_answer"] == "done"
    second_agent_request = client.requests[2][0]
    contents = [message["content"] for message in second_agent_request]
    tool_index = next(index for index, content in enumerate(contents) if "执行工具 search_code" in content)
    assert tool_index >= 0
    assert not any(content.startswith("[plan]") for content in contents)
    assert result["metadata"]["plan_progress"]["satisfied"] == 1


def test_task_planner_replan_carries_completed_progress():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "read_file", "reason": "inspect"},
                        {"step": 2, "action": "edit_file", "reason": "fix"},
                        {"step": 3, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps(
                {
                    "updates": [
                        {
                            "phase": "validate",
                            "goal": "verify differently",
                            "preferred_tools": ["run_tests"],
                        }
                    ]
                }
            ),
        ]
    )
    tools = [
        Tool("read_file", "Read", lambda arguments: "content"),
        Tool("edit_file", "Edit", lambda arguments: "edited"),
        Tool("run_tests", "Test", lambda arguments: "passed"),
        Tool("task_complete", "Finish", lambda arguments: "done"),
    ]
    planner = TaskPlanner(client, tools)
    plan = planner.plan("fix a bug")
    planner.mark_completed(1, "read ok")

    new_plan = planner.replan("fix a bug", plan, "edit went sideways")

    assert [step.action for step in new_plan] == ["read_file", "edit_file", "run_tests"]
    assert [step.step_number for step in new_plan] == [1, 2, 3]
    assert new_plan[0].completed is True
    assert new_plan[1].completed is False
    assert new_plan[2].goal == PHASE_GOALS["validate"]
    assert planner.get_next_step().action == "edit_file"
    # Progress display keeps completed work visible after the replan.
    assert "1/3" in planner.get_progress()


def test_react_agent_default_mode_does_not_replan_repeated_generic_failures(tmp_path):
    trace_path = tmp_path / "budget.jsonl"
    # Generic tool recovery belongs to the ReAct loop unless adaptive replanning
    # is explicitly enabled.
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "explode", "reason": "trigger failure"},
                        {"step": 2, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps({"thought": "Try once.", "action": "explode", "action_input": {}}),
            json.dumps({"thought": "Try twice.", "action": "explode", "action_input": {}}),
            json.dumps(
                {
                    "thought": "Stop retrying.",
                    "action": "task_complete",
                    "action_input": {"message": "stopped within budget"},
                }
            ),
        ]
    )
    writer = TraceWriter(trace_path)
    agent = ReactAgent(
        client,
        [
            Tool("explode", "Fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom"))),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
        trace_writer=writer,
    )
    result = agent.run("stay within default replan budget")
    writer.close()

    assert result["final_answer"] == "stopped within budget"
    assert result["metadata"]["replan_count"] == 0
    assert result["metadata"]["replan_suppressed_count"] == 2
    triggers = [
        event["payload"]
        for event in load_trace_events(trace_path)
        if event["event"] == "replan_trigger"
    ]
    assert [item["should_replan"] for item in triggers] == [False, False]


def test_adaptive_replan_policy_classifies_failure_signals():
    policy = AdaptiveReplanPolicy()

    tool_signal = policy.classify("Tool execution failed: boom", action="run_shell")
    parse_signal = policy.classify("Agent response parse failed: Response is not valid JSON")
    test_signal = policy.classify("pytest returncode: 1\nAssertionError")

    assert tool_signal.kind == "tool_error"
    assert tool_signal.strategy == "simplify_plan_skip_failed_tool"
    assert parse_signal.kind == "parse_error"
    assert parse_signal.strategy == "repair_response_format"
    assert test_signal.kind == "test_failure"
    assert test_signal.strategy == "inject_test_failure_context"


def test_react_agent_adaptive_replan_records_strategy_metadata():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "explode", "reason": "trigger failure"},
                        {"step": 2, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Try the failing tool.",
                    "action": "explode",
                    "action_input": {},
                }
            ),
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "task_complete", "reason": "recover"},
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Recover.",
                    "action": "task_complete",
                    "action_input": {"message": "recovered"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool("explode", "Fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom"))),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
        enable_adaptive_replanning=True,
        max_replans=2,
    )

    result = agent.run("recover from failure")

    assert result["final_answer"] == "recovered"
    assert result["metadata"]["adaptive_replanning_enabled"] is True
    assert result["metadata"]["replan_count"] == 1
    assert result["metadata"]["replan_decision_count"] == 1
    assert result["metadata"]["replan_signals"][0]["signal"]["kind"] == "tool_error"
    assert result["metadata"]["replan_strategy"] == "simplify_plan_skip_failed_tool"


def test_react_agent_adaptive_replan_respects_budget():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "explode", "reason": "trigger failure"},
                        {"step": 2, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps({"thought": "Try once.", "action": "explode", "action_input": {}}),
            json.dumps({"plan": [{"step": 1, "action": "explode", "reason": "retry once"}]}),
            json.dumps({"thought": "Try twice.", "action": "explode", "action_input": {}}),
            json.dumps(
                {
                    "thought": "Stop retrying.",
                    "action": "task_complete",
                    "action_input": {"message": "done after budget"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool("explode", "Fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom"))),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
        enable_adaptive_replanning=True,
        max_replans=1,
    )

    result = agent.run("recover with one replan")

    assert result["final_answer"] == "done after budget"
    assert result["metadata"]["replan_count"] == 1
    assert result["metadata"]["replan_decision_count"] == 2
    assert result["metadata"]["replan_skipped_count"] == 1
    assert result["metadata"]["replan_maxed_count"] == 1


def test_react_agent_adaptive_replan_handles_parse_error():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "read_file", "reason": "inspect"},
                    ]
                }
            ),
            "not json",
            json.dumps(
                {
                    "updates": [
                        {
                            "phase": "complete",
                            "goal": "return a valid completion response",
                            "preferred_tools": ["task_complete"],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Use strict JSON.",
                    "action": "task_complete",
                    "action_input": {"message": "format repaired"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool("read_file", "Read", lambda _arguments: "content"),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
        enable_adaptive_replanning=True,
        max_replans=1,
    )

    result = agent.run("finish after parse repair")

    assert result["final_answer"] == "format repaired"
    assert result["metadata"]["parse_error_count"] == 1
    assert result["metadata"]["replan_signals"][0]["signal"]["kind"] == "parse_error"
    assert result["metadata"]["replan_strategy"] == "repair_response_format"


def test_react_agent_default_replan_handles_parse_error():
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "read_file", "reason": "inspect"},
                    ]
                }
            ),
            "not json",
            json.dumps(
                {
                    "thought": "Use strict JSON.",
                    "action": "task_complete",
                    "action_input": {"message": "format repaired"},
                }
            ),
        ]
    )
    agent = ReactAgent(
        client,
        [
            Tool("read_file", "Read", lambda _arguments: "content"),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
    )

    result = agent.run("finish after default parse repair")

    assert result["final_answer"] == "format repaired"
    assert result["metadata"]["parse_error_count"] == 1
    assert result["metadata"]["replan_count"] == 0
    assert result["metadata"]["replan_suppressed_count"] == 1
    assert result["metadata"]["replan_decision_count"] == 0


def test_react_agent_adaptive_replan_records_repeated_failures(tmp_path):
    trace_path = tmp_path / "repeat.jsonl"
    client = FakeRespondClient(
        [
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "explode", "reason": "trigger failure"},
                        {"step": 2, "action": "task_complete", "reason": "finish"},
                    ]
                }
            ),
            json.dumps({"thought": "Try once.", "action": "explode", "action_input": {}}),
            json.dumps({"plan": [{"step": 1, "action": "explode", "reason": "retry"}]}),
            json.dumps({"thought": "Try twice.", "action": "explode", "action_input": {}}),
            json.dumps(
                {
                    "plan": [
                        {"step": 1, "action": "task_complete", "reason": "recover"},
                    ]
                }
            ),
            json.dumps(
                {
                    "thought": "Recover.",
                    "action": "task_complete",
                    "action_input": {"message": "done"},
                }
            ),
        ]
    )
    writer = TraceWriter(trace_path)
    agent = ReactAgent(
        client,
        [
            Tool("explode", "Fail", lambda arguments: (_ for _ in ()).throw(RuntimeError("boom"))),
            Tool("task_complete", "Finish", lambda arguments: arguments.get("message", "done")),
        ],
        enable_planning=True,
        enable_compression=False,
        enable_adaptive_replanning=True,
        max_replans=2,
        trace_writer=writer,
    )

    result = agent.run("recover from a repeated failure")
    writer.close()

    assert result["final_answer"] == "done"
    assert result["metadata"]["replan_count"] == 2
    assert result["metadata"]["repeated_failure_count"] == 1
    assert result["metadata"]["repeated_failures"][0]["action"] == "explode"
    assert result["metadata"]["repeated_failures"][0]["kind"] == "tool_error"

    decisions = [
        event["payload"]
        for event in load_trace_events(trace_path)
        if event["event"] == "replan_decision"
    ]
    assert [decision["repeated_failure"] for decision in decisions] == [False, True]
    assert decisions[1]["repeated_failure_details"]["action"] == "explode"
