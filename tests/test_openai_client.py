"""Tests for the OpenAI-compatible client configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import dm_agent.clients.openai_client as openai_client


class FakeOpenAI:
    """Captures SDK construction options without making a network request."""

    options: ClassVar[dict[str, object]] = {}

    def __init__(self, **options: object) -> None:
        type(self).options = options


class FakeResponses:
    request: ClassVar[dict[str, object]] = {}

    @classmethod
    def create(cls, **request: object) -> object:
        cls.request = request
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="read_file",
                    arguments='{"path":"users.py"}',
                ),
                SimpleNamespace(type="function_call", name="run_tests", arguments="{}"),
            ],
            output_text="",
        )


class FakeCallingOpenAI(FakeOpenAI):
    def __init__(self, **options: object) -> None:
        super().__init__(**options)
        self.responses = FakeResponses()


def test_openai_client_passes_configured_base_url(monkeypatch) -> None:
    monkeypatch.setattr(openai_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(openai_client, "OPENAI_AVAILABLE", True)

    openai_client.OpenAIClient(
        "test-key",
        model="gpt-5.5",
        base_url="https://proxy.example/v1",
    )

    assert FakeOpenAI.options["base_url"] == "https://proxy.example/v1"


def test_openai_client_uses_sdk_default_endpoint_when_base_url_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(openai_client, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(openai_client, "OPENAI_AVAILABLE", True)

    openai_client.OpenAIClient("test-key", model="gpt-5")

    assert "base_url" not in FakeOpenAI.options


def test_openai_uses_responses_tools_and_only_first_function_call(monkeypatch) -> None:
    monkeypatch.setattr(openai_client, "OpenAI", FakeCallingOpenAI)
    monkeypatch.setattr(openai_client, "OPENAI_AVAILABLE", True)
    client = openai_client.OpenAIClient("test-key", model="gpt-5")

    text = client.respond(
        [{"role": "user", "content": "read users.py"}],
        tool_definitions=[
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        tool_choice="auto",
        temperature=0,
    )

    assert FakeResponses.request["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    assert "temperature" not in FakeResponses.request
    assert json.loads(text) == {
        "thought": "",
        "action": "read_file",
        "action_input": {"path": "users.py"},
    }
