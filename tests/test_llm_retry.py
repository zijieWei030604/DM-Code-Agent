"""Tests for the unified retryable LLM error handling in the base client."""

from __future__ import annotations

from typing import Any

import pytest

from dm_agent.clients.base_client import (
    BaseLLMClient,
    LLMError,
    classify_retryable_exception,
)
from dm_agent.evals.real_runner import UsageTrackingClient


class FlakyClient(BaseLLMClient):
    """Fails N times with a configurable error before succeeding."""

    def __init__(self, *, failures: int, retryable: bool, respond_retries: int = 2) -> None:
        super().__init__(
            "test-key",
            model="fake-model",
            base_url="https://example.invalid",
            respond_retries=respond_retries,
            respond_retry_backoff=0.0,
        )
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError("transient upstream issue", retryable=self.retryable)
        return {"text": "ok"}

    def extract_text(self, data: dict[str, Any]) -> str:
        return data["text"]


def test_retryable_errors_are_retried_until_success() -> None:
    client = FlakyClient(failures=2, retryable=True, respond_retries=2)

    assert client.respond([{"role": "user", "content": "hi"}]) == "ok"
    assert client.calls == 3
    assert client.total_respond_retries == 2


def test_non_retryable_errors_raise_immediately() -> None:
    client = FlakyClient(failures=1, retryable=False, respond_retries=3)

    with pytest.raises(LLMError):
        client.respond([{"role": "user", "content": "hi"}])
    assert client.calls == 1
    assert client.total_respond_retries == 0


def test_zero_retries_passes_error_through() -> None:
    client = FlakyClient(failures=1, retryable=True, respond_retries=0)

    with pytest.raises(LLMError):
        client.respond([{"role": "user", "content": "hi"}])
    assert client.calls == 1


def test_retry_budget_exhaustion_raises_last_error() -> None:
    client = FlakyClient(failures=10, retryable=True, respond_retries=2)

    with pytest.raises(LLMError) as excinfo:
        client.respond([{"role": "user", "content": "hi"}])
    assert excinfo.value.retryable is True
    assert client.calls == 3  # initial attempt + 2 retries


def test_usage_tracking_client_routes_through_retry_layer() -> None:
    inner = FlakyClient(failures=1, retryable=True, respond_retries=2)
    wrapper = UsageTrackingClient(inner)

    text = wrapper.respond([{"role": "user", "content": "hi"}])

    assert text == "ok"
    assert inner.calls == 2
    assert inner.total_respond_retries == 1
    assert wrapper.usage.request_count == 1


def test_usage_tracking_client_forwards_summary_without_agent_calls() -> None:
    class SummaryClient(FlakyClient):
        def complete_summary(self, messages, *, max_tokens, timeout=60.0):
            self.summary_request = (messages, max_tokens, timeout)
            return {"text": "summary", "usage": {"total_tokens": 23}}

    inner = SummaryClient(failures=0, retryable=False)
    wrapper = UsageTrackingClient(inner)
    messages = [{"role": "user", "content": "Summarize history"}]

    data = wrapper.complete_summary(messages, max_tokens=77, timeout=12)

    assert inner.summary_request == (messages, 77, 12)
    assert wrapper.extract_text(data) == "summary"
    assert data["usage"]["total_tokens"] == 23
    assert inner.calls == 0
    assert wrapper.usage.request_count == 0


def test_usage_tracking_client_propagates_summary_errors() -> None:
    class SummaryClient(FlakyClient):
        def complete_summary(self, messages, *, max_tokens, timeout=60.0):
            raise TimeoutError("summary timed out")

    inner = SummaryClient(failures=0, retryable=False)
    wrapper = UsageTrackingClient(inner)
    with pytest.raises(TimeoutError, match="summary timed out"):
        wrapper.complete_summary([], max_tokens=77)
    assert inner.calls == 0


def test_classify_retryable_exception_by_status_and_text() -> None:
    class WithStatus(Exception):
        status_code = 503

    class Semantic(Exception):
        status_code = 401

    assert classify_retryable_exception(WithStatus("upstream sad")) is True
    assert classify_retryable_exception(Semantic("bad key")) is False
    assert classify_retryable_exception(TimeoutError("request timed out")) is True
    assert classify_retryable_exception(ValueError("invalid schema")) is False


def test_llm_error_defaults_to_non_retryable() -> None:
    assert LLMError("boom").retryable is False
    assert LLMError("boom", retryable=True).retryable is True
