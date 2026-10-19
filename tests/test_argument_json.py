import json

import pytest

from dm_agent.clients.argument_json import (
    PathFieldScanner,
    parse_final_arguments,
    preview_arguments,
)
from dm_agent.clients.streaming import StreamEvent, StreamToolCall
from dm_agent.clients.tool_calls import serialize_tool_calls
from dm_agent.core.agent import ReactAgent


def test_nested_preview_and_brackets_in_strings():
    assert preview_arguments('{"options":{"path":"a.py"},"text":"} [') == {
        "options": {"path": "a.py"},
        "text": "} [",
    }


def test_numeric_prefix_is_display_only():
    assert preview_arguments('{"limit":1') == {"limit": 1}
    assert preview_arguments('{"limit":100}') == {"limit": 100}
    assert preview_arguments('{"limit":1e') == {}
    assert preview_arguments('{"limit":1e3}') == {"limit": 1000}


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1',
        '{"x":"abc',
        '{"x":[1,2',
        '{"x":1} junk',
        '{"x":1,"x":2}',
        '{"x":NaN}',
        '{"x":1e999}',
        "[]",
        "{'x':1,'x':2}",
        "{'x':1e999}",
    ],
)
def test_final_rejects_truncation_duplicates_and_nonfinite(text):
    with pytest.raises(ValueError):
        parse_final_arguments(text)


def test_final_controlled_repairs():
    assert parse_final_arguments("{'path':'a.py','lines':[1,2,],}") == {
        "path": "a.py",
        "lines": [1, 2],
    }


def test_excessive_nesting_is_recoverable_argument_error():
    with pytest.raises(ValueError, match="nesting"):
        parse_final_arguments('{"x":' + "[" * 200 + "0" + "]" * 200 + "}")


def test_safe_path_independent_of_display_and_nesting():
    call = StreamToolCall(index=0)
    call.append('{"options":{"path":"../bad"},"path":"src/')
    assert call.safety_arguments() == {}
    preview = call.preview()
    call.append('good.py"')
    assert call.preview() == preview  # Below display reparse threshold.
    assert call.safety_arguments() == {"path": "src/good.py"}
    call.append("}")
    assert call.finalize()["path"] == "src/good.py"


def test_path_scanner_every_split_unicode_and_escaped_quotes():
    text = r'{"options":[{"path":"../bad"}],"path":"src/\u0061\"b.py"}'
    for split in range(len(text) + 1):
        scanner = PathFieldScanner()
        scanner.feed(text[:split])
        scanner.feed(text[split:])
        assert scanner.path == 'src/a"b.py'


def test_incomplete_display_path_never_rejects():
    event = StreamEvent("toolcall_delta", "r", arguments_preview={"path": "../bad"})
    assert ReactAgent._stream_safety_reason(event) == ""
    confirmed = StreamEvent("toolcall_delta", "r", safety_arguments={"path": "../bad"})
    assert ReactAgent._stream_safety_reason(confirmed) == "path_outside_workspace"


def test_bad_sibling_keeps_identity_and_original_input():
    raw = [
        {"id": "a", "function": {"name": "read_file", "arguments": '{"path":"a"}'}},
        {"id": "b", "function": {"name": "read_file", "arguments": '{"path":'}},
        {"id": "c", "function": {"name": "read_file", "arguments": "{'path':'c',}"}},
    ]
    calls = json.loads(serialize_tool_calls(raw))["tool_calls"]
    assert "argument_error" not in calls[0]
    assert calls[1]["argument_error"]
    assert calls[1]["raw_arguments"] == '{"path":'
    assert calls[2]["action_input"] == {"path": "c"}


def test_large_payload_throttles_display_but_finalizes(monkeypatch):
    import dm_agent.clients.streaming as streaming

    count = 0
    original = streaming.preview_arguments

    def counted(text):
        nonlocal count
        count += 1
        return original(text)

    monkeypatch.setattr(streaming, "preview_arguments", counted)
    call = StreamToolCall(index=0)
    text = json.dumps({"content": "x" * 20000, "path": "a.py"})
    for start in range(0, len(text), 10):
        call.append(text[start : start + 10])
        call.preview()
    assert count < 100
    assert call.safety_arguments() == {"path": "a.py"}
    assert call.finalize() == json.loads(text)
