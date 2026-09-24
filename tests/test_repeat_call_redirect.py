from __future__ import annotations

from dm_agent.core import EventBus
from dm_agent.core.capabilities import CapabilityContext
from dm_agent.core.events import AfterToolResultEvent, BeforeLLMRequestEvent, RunStartEvent
from dm_agent.core.execution_facts import ExecutionFact
from dm_agent.extensions.capabilities import RepeatCallRedirectCapability


def _fact(version: str) -> ExecutionFact:
    return ExecutionFact(
        "source_observed",
        "inspect",
        "source_observed",
        "read_file",
        0,
        True,
        "source evidence was inspected",
        workspace_version=version,
    )


def _after(
    step: int,
    metadata: dict[str, object],
    *,
    observation: str = "same result",
    version: str = "v1",
) -> AfterToolResultEvent:
    return AfterToolResultEvent(
        tool_name="read_file",
        arguments={"path": "src/auth.py"},
        observation=observation,
        step_number=step,
        run_id="run-1",
        tool_succeeded=True,
        metadata=metadata,
        execution_fact=_fact(version),
    )


def _request(step: int, metadata: dict[str, object]) -> BeforeLLMRequestEvent:
    return BeforeLLMRequestEvent(
        messages=[{"role": "user", "content": "continue"}],
        step_number=step,
        run_id="run-1",
        phase="agent",
        metadata=metadata,
    )


def test_repeat_call_redirect_steers_after_consecutive_unchanged_results() -> None:
    bus = EventBus()
    capability = RepeatCallRedirectCapability(threshold=4)
    capability.install(CapabilityContext(event_bus=bus, client_for=lambda _phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("task", 1, "run-1", metadata=metadata))

    for step in range(1, 5):
        bus.emit_after_tool_result(_after(step, metadata))

    request = _request(5, metadata)
    outgoing = bus.emit_before_llm_request(request)

    assert len(outgoing) == 2
    assert "4 times consecutively" in outgoing[-1]["content"]
    assert metadata["repeat_call_redirect_detection_count"] == 1
    assert metadata["repeat_call_redirect_injection_count"] == 1


def test_repeat_call_redirect_does_not_duplicate_injection_for_same_result() -> None:
    bus = EventBus()
    capability = RepeatCallRedirectCapability(threshold=2)
    capability.install(CapabilityContext(event_bus=bus, client_for=lambda _phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("task", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(_after(1, metadata))
    bus.emit_after_tool_result(_after(2, metadata))

    first = bus.emit_before_llm_request(_request(3, metadata))
    second = bus.emit_before_llm_request(_request(3, metadata))

    assert len(first) == 2
    assert len(second) == 1
    assert metadata["repeat_call_redirect_injection_count"] == 1


def test_repeat_call_redirect_resets_when_result_or_workspace_changes() -> None:
    bus = EventBus()
    capability = RepeatCallRedirectCapability(threshold=3)
    capability.install(CapabilityContext(event_bus=bus, client_for=lambda _phase: None))
    metadata: dict[str, object] = {}
    bus.emit_run_start(RunStartEvent("task", 1, "run-1", metadata=metadata))
    bus.emit_after_tool_result(_after(1, metadata))
    bus.emit_after_tool_result(_after(2, metadata))
    bus.emit_after_tool_result(_after(3, metadata, observation="new result"))

    outgoing = bus.emit_before_llm_request(_request(4, metadata))
    assert outgoing == [{"role": "user", "content": "continue"}]

    bus.emit_after_tool_result(_after(4, metadata, observation="new result", version="v2"))
    outgoing = bus.emit_before_llm_request(_request(5, metadata))
    assert outgoing == [{"role": "user", "content": "continue"}]
