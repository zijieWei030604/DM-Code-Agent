from __future__ import annotations

import json

from dm_agent.core.agent import ReactAgent
from dm_agent.memory.repo_map import RepositoryMap
from dm_agent.tools.base import Tool


class _FinishClient:
    model = "scripted"
    total_respond_retries = 0

    def __init__(self) -> None:
        self.requests = []

    def respond(self, messages, **extra):
        self.requests.append((messages, extra))
        return json.dumps({"thought": "done", "action": "finish", "action_input": "mapped"})


def test_repo_map_prioritizes_task_symbols_and_omits_source_bodies(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def authenticate(user):\n    return user is not None\n",
        encoding="utf-8",
    )
    (tmp_path / "payment_service.py").write_text(
        "class PaymentService:\n"
        "    def refund_order(self, order_id: str) -> bool:\n"
        "        secret_body_value = True\n"
        "        return secret_body_value\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_payment_service.py").write_text(
        "def test_refund_order():\n    assert True\n",
        encoding="utf-8",
    )
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("def hidden():\n    pass\n", encoding="utf-8")

    result = RepositoryMap(max_map_files=3).build("fix PaymentService.refund_order", tmp_path)

    assert result.content.index("payment_service.py") < result.content.index("auth.py")
    assert "class PaymentService" in result.content
    assert "def refund_order(self, order_id: str) -> bool" in result.content
    assert "secret_body_value" not in result.content
    assert ".venv" not in result.content
    assert result.scanned_files == 3


def test_repo_map_reuses_unchanged_files_and_refreshes_changed_file(tmp_path):
    first_file = tmp_path / "first.py"
    second_file = tmp_path / "second.py"
    first_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    second_file.write_text("def second():\n    return 2\n", encoding="utf-8")
    repo_map = RepositoryMap(max_map_files=10)

    first = repo_map.build("first", tmp_path)
    second = repo_map.build("first", tmp_path)
    first_file.write_text(
        "def first():\n    return 1\n\ndef added():\n    return 3\n",
        encoding="utf-8",
    )
    refreshed = repo_map.build("added", tmp_path)

    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert refreshed.cache_hits == 1
    assert "def added()" in refreshed.content


def test_repo_map_respects_character_budget(tmp_path):
    for index in range(20):
        (tmp_path / f"module_{index}.py").write_text(
            f"def function_{index}(argument: str) -> str:\n    return argument\n",
            encoding="utf-8",
        )

    result = RepositoryMap(max_map_files=20, max_chars=500).build("module", tmp_path)

    assert len(result.content) <= 500
    assert result.truncated is True
    assert "additional Python files omitted" in result.content


def test_agent_injects_repo_map_and_reports_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "orders.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    client = _FinishClient()
    agent = ReactAgent(
        client,
        [Tool("noop", "No operation", lambda arguments: "ok")],
        enable_planning=False,
        enable_compression=False,
        enable_repo_map=True,
    )

    result = agent.run("检查 calculate_total", max_steps=2)

    task_message = next(
        message["content"] for message in client.requests[0][0] if message["role"] == "user"
    )
    assert "<repository_map" in task_message
    assert "orders.py" in task_message
    assert "def calculate_total(items)" in task_message
    assert result["metadata"]["repo_map_enabled"] is True
    assert result["metadata"]["repo_map_files"] == 1
    assert result["metadata"]["repo_map_chars"] > 0
