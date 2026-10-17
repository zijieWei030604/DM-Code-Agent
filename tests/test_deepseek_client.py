import json

import pytest
import requests

from dm_agent.clients.deepseek_client import DeepSeekClient, DeepSeekError, DeepSeekStreamBlocked
from dm_agent.clients.llm_factory import create_llm_client
from dm_agent.clients.streaming import parse_partial_json_object


class FakeResponse:
    def __init__(self, status_code, payload=None, *, reason="OK", text=""):
        self.status_code = status_code
        self._payload = payload
        self.reason = reason
        self.text = text
        self.ok = 200 <= status_code < 400

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.headers = {}
        self.calls = []
        self.closed = False

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


class StreamingResponse(FakeResponse):
    def __init__(self, lines):
        super().__init__(200, {})
        self.lines = lines
        self.closed = False

    def iter_lines(self, *, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


class StreamingSession(FakeSession):
    def post(self, url, *, json, timeout, stream=False):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client_with_session(session, **kwargs):
    client = DeepSeekClient("test-key", retry_backoff=0, **kwargs)
    client.session = session
    return client


def test_deepseek_retries_bad_request_before_success():
    session = FakeSession(
        [
            FakeResponse(
                400, {"error": {"message": "temporary gateway parse error"}}, reason="Bad Request"
            ),
            FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    client = _client_with_session(session, max_retries=1)

    data = client.complete([{"role": "user", "content": "hello"}])

    assert data["choices"][0]["message"]["content"] == "ok"
    assert len(session.calls) == 2


def test_deepseek_stops_after_retry_budget():
    session = FakeSession(
        [
            FakeResponse(400, {"error": {"message": "still bad"}}, reason="Bad Request"),
            FakeResponse(400, {"error": {"message": "still bad"}}, reason="Bad Request"),
            FakeResponse(400, {"error": {"message": "still bad"}}, reason="Bad Request"),
        ]
    )
    client = _client_with_session(session, max_retries=2)

    with pytest.raises(DeepSeekError, match="after 3 attempts"):
        client.complete([{"role": "user", "content": "hello"}])

    assert len(session.calls) == 3


def test_deepseek_does_not_retry_auth_errors():
    session = FakeSession(
        [FakeResponse(401, {"error": {"message": "bad key"}}, reason="Unauthorized")]
    )
    client = _client_with_session(session, max_retries=3)

    with pytest.raises(DeepSeekError, match="401 Unauthorized"):
        client.complete([{"role": "user", "content": "hello"}])

    assert len(session.calls) == 1


def test_deepseek_retries_network_exceptions():
    session = FakeSession(
        [
            requests.Timeout("read timed out"),
            FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    client = _client_with_session(session, max_retries=1)

    data = client.complete([{"role": "user", "content": "hello"}])

    assert data["choices"][0]["message"]["content"] == "ok"
    assert len(session.calls) == 2


def test_deepseek_retry_settings_flow_through_factory():
    client = create_llm_client(
        "deepseek",
        "test-key",
        max_retries=5,
        retry_backoff=0.25,
        retry_status_codes={429},
    )

    assert isinstance(client, DeepSeekClient)
    assert client.max_retries == 5
    assert client.retry_backoff == 0.25
    assert client.retry_status_codes == frozenset({429})


def test_deepseek_reports_multiple_tool_calls_instead_of_discarding_them():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"users.py"}',
                                        }
                                    },
                                    {
                                        "function": {
                                            "name": "run_tests",
                                            "arguments": "{}",
                                        }
                                    },
                                ],
                            }
                        }
                    ]
                },
            )
        ]
    )
    client = _client_with_session(session)
    definition = {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {}},
    }

    text = client.respond(
        [{"role": "user", "content": "read users.py"}],
        tool_definitions=[definition],
        tool_choice="auto",
    )

    function = session.calls[0]["json"]["tools"][0]["function"]
    assert function["name"] == "read_file"
    assert function["strict"] is False
    assert "multiple native tool calls" in text
    assert client.last_response_mode == "native_tool_call_multiple"
    assert client.last_tool_call_count == 2


def test_deepseek_malformed_native_tool_arguments_return_recoverable_text():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"users.py"} trailing',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        ]
    )
    client = _client_with_session(session)

    text = client.respond([{"role": "user", "content": "read users.py"}])

    assert text.startswith("DeepSeek tool arguments are not valid JSON:")


def test_deepseek_uses_responses_api_for_strict_json_schema():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"plan": []}',
                                }
                            ],
                        }
                    ],
                },
            )
        ]
    )
    client = _client_with_session(session)
    schema = {
        "type": "object",
        "properties": {"plan": {"type": "array"}},
        "required": ["plan"],
        "additionalProperties": False,
    }

    text = client.respond(
        [{"role": "user", "content": "create a plan"}],
        json_schema=schema,
        temperature=0.3,
    )

    request = session.calls[0]
    assert client.supports_json_schema is True
    assert request["url"] == "https://api.deepseek.com/responses"
    assert request["json"]["input"] == [{"role": "user", "content": "create a plan"}]
    assert "messages" not in request["json"]
    assert request["json"]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "structured_response",
            "schema": schema,
            "strict": True,
        }
    }
    assert text == '{"plan": []}'
    assert client.last_response_mode == "json_schema"


def test_deepseek_closes_its_http_session():
    session = FakeSession([])
    client = _client_with_session(session)

    client.close()

    assert session.closed is True


def test_partial_json_preview_exposes_only_completed_values():
    assert parse_partial_json_object('{"path":"src/a.py","content":"unterminated') == {
        "path": "src/a.py"
    }
    assert parse_partial_json_object('{"line":12,"nested":{') == {"line": 12}


def test_deepseek_stream_assembles_tool_arguments_and_events():
    lines = [
        'data: {"choices":[{"delta":{"content":"Read ","tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\\"path\\":\\"src/"}}]}}]}',
        'data: {"choices":[{"delta":{"content":"it","tool_calls":[{"index":0,"function":{"arguments":"a.py\\"}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    response = StreamingResponse(lines)
    client = _client_with_session(StreamingSession([response]))
    events = []

    raw = client.respond(
        [{"role": "user", "content": "read"}],
        stream=True,
        on_stream_event=lambda event: events.append(event) or True,
    )

    assert json.loads(raw) == {
        "thought": "",
        "action": "read_file",
        "action_input": {"path": "src/a.py"},
    }
    assert [event.type for event in events] == [
        "stream_start",
        "text_start",
        "text_delta",
        "toolcall_start",
        "toolcall_delta",
        "text_delta",
        "toolcall_delta",
        "text_end",
        "toolcall_end",
        "stream_done",
    ]
    preview = next(event.arguments_preview for event in events if event.type == "toolcall_delta")
    assert preview == {}
    assert response.closed is True


def test_deepseek_stream_stops_when_policy_rejects_completed_path():
    response = StreamingResponse(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\\"path\\":\\"../secret.txt\\"}"}}]}}]}',
            "data: [DONE]",
        ]
    )
    client = _client_with_session(StreamingSession([response]))
    events = []

    def reject_completed_path(event):
        events.append(event)
        return not event.arguments_preview.get("path")

    with pytest.raises(DeepSeekStreamBlocked):
        client.respond(
            [{"role": "user", "content": "read"}],
            stream=True,
            on_stream_event=reject_completed_path,
        )

    assert events[-1].type == "stream_error"
    assert response.closed is True


def test_deepseek_stream_does_not_silently_discard_multiple_calls():
    response = StreamingResponse(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{}"}},{"index":1,"id":"call_b","function":{"name":"run_tests","arguments":"{}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    client = _client_with_session(StreamingSession([response]))

    raw = client.respond([{"role": "user", "content": "do both"}], stream=True)

    assert "multiple native tool calls" in raw
    assert client.last_response_mode == "native_tool_call_multiple"
    assert client.last_tool_call_count == 2
