"""Tests for the OpenAI-compatible client configuration."""

from __future__ import annotations

from typing import ClassVar

import dm_agent.clients.openai_client as openai_client


class FakeOpenAI:
    """Captures SDK construction options without making a network request."""

    options: ClassVar[dict[str, object]] = {}

    def __init__(self, **options: object) -> None:
        type(self).options = options


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
