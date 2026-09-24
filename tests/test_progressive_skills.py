import json
from pathlib import Path

import pytest

from dm_agent.skills import SkillManager
from dm_agent.skills.packages import MarkdownSkill
from dm_agent.skills.runtime import prepare_skills
from dm_agent.tools.base import Tool


def package(root: Path, body: str, name: str = "python_expert") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Python testing\nkeywords: [python]\n---\n{body}",
        encoding="utf-8",
    )
    return path


def test_discovery_precedence_and_lazy_body(tmp_path):
    user, project = tmp_path / "user", tmp_path / "project"
    package(user, "USER BODY")
    path = package(project, "PROJECT BODY")
    manager = SkillManager(user_directory=user, project_directory=project)
    manager.load_all()
    tools, events, activated = {}, [], []
    prompt = prepare_skills(
        manager, "python", tools, lambda *event: events.append(event), activated
    )
    assert "PROJECT BODY" not in prompt and "USER BODY" not in prompt
    assert "python_expert [recommended]" in prompt
    assert activated == []
    path.write_text(path.read_text().replace("PROJECT BODY", "UPDATED BODY"), encoding="utf-8")
    result = tools["load_skill"].execute({"name": "python_expert"})
    assert result.status == "success"
    assert "UPDATED BODY" in result.message
    assert result.metadata["source"] == str(path.resolve())
    assert activated == ["python_expert"]
    assert "python_best_practices" not in tools  # Markdown overrides cannot attach Python tools.


def test_reload_resources_and_task_reset(tmp_path):
    path = package(tmp_path, "ENTRY")
    resources = path.parent / "resources"
    resources.mkdir()
    (resources / "testing.md").write_text("DETAIL", encoding="utf-8")
    manager = SkillManager()
    manager.load_packages(tmp_path)
    tools, activated = {}, []
    prepare_skills(manager, "python", tools, lambda *args: None, activated)
    for _ in range(2):
        assert "ENTRY" in tools["load_skill"].execute({"name": "python_expert"}).message
        assert (
            "DETAIL"
            in tools["load_skill_resource"]
            .execute({"name": "python_expert", "resource": "testing.md"})
            .message
        )
    assert activated == ["python_expert"]
    prepare_skills(manager, "next task", {}, lambda *args: None, [])
    assert manager.active_skills == []


@pytest.mark.parametrize(
    "resource", ["../SKILL.md", "../../outside", "/etc/passwd", "C:\\secret", "missing"]
)
def test_resource_boundary(tmp_path, resource):
    skill = MarkdownSkill(package(tmp_path, "ENTRY"))
    with pytest.raises(ValueError):
        skill.resource_path(resource)


def test_builtin_tools_load_lazily_and_cannot_replace_existing():
    manager = SkillManager()
    manager.load_builtin_skills()
    tools, activated = {}, []
    prepare_skills(manager, "python", tools, lambda *args: None, activated)
    assert "python_best_practices" not in tools
    assert tools["load_skill"].execute({"name": "python_expert"}).status == "success"
    assert "python_best_practices" in tools
    assert tools["load_skill"].execute({"name": "python_expert"}).status == "success"
    tools = {"python_best_practices": Tool("python_best_practices", "existing", lambda args: "ok")}
    prepare_skills(manager, "python", tools, lambda *args: None, [])
    assert tools["load_skill"].execute({"name": "python_expert"}).status == "failed"
    assert tools["python_best_practices"].execute({}) == "ok"


def test_bad_package_does_not_prevent_valid_discovery(tmp_path):
    package(tmp_path, "good", "good")
    path = package(tmp_path, "bad", "bad")
    path.write_text("---\nname: [\n---\n", encoding="utf-8")
    manager = SkillManager()
    assert manager.load_packages(tmp_path) == 1
    assert "good" in manager.skills


def test_agent_exposes_tools_after_load_and_resets_next_task(tmp_path, monkeypatch):
    from dm_agent.core.agent import ReactAgent

    monkeypatch.chdir(tmp_path)

    class Client:
        model = "fake"
        supports_tool_calling = True

        def __init__(self):
            self.requests = []
            self.actions = iter(
                [
                    ("load_skill", {"name": "python_expert"}),
                    ("python_best_practices", {"topic": "testing"}),
                    ("load_skill", {"name": "python_expert"}),
                    ("finish", {"answer": "done"}),
                    ("finish", {"answer": "next"}),
                ]
            )

        def respond(self, messages, **kwargs):
            self.requests.append((messages, kwargs))
            action, arguments = next(self.actions)
            return json.dumps({"thought": "test", "action": action, "action_input": arguments})

    manager = SkillManager()
    manager.load_builtin_skills()
    client = Client()
    agent = ReactAgent(
        client,
        [Tool("task_complete", "Finish", lambda args: "done")],
        skill_manager=manager,
        enable_planning=False,
        enable_compression=False,
    )
    first = agent.run("python testing", max_steps=5)
    assert first["metadata"]["activated_skills"] == ["python_expert"]
    assert first["metadata"]["tool_error_count"] == 0
    assert "python_best_practices" not in str(client.requests[0][1])
    assert "python_best_practices" in str(client.requests[1][1])
    assert (
        manager.skills["python_expert"].get_prompt_addition()
        not in client.requests[0][0][0]["content"]
    )
    second = agent.run("another task", max_steps=1)
    assert second["metadata"]["activated_skills"] == []
    assert "python_best_practices" not in agent.tools
    assert "python_best_practices" not in str(client.requests[-1][1])


def test_skill_can_reload_after_lcm_compaction(tmp_path, monkeypatch):
    from dataclasses import replace

    from dm_agent.core.agent import ReactAgent
    from dm_agent.tracing.writer import TraceWriter, load_trace_events

    monkeypatch.chdir(tmp_path)
    package(tmp_path / "skills", "UNIQUE_SKILL_INSTRUCTION " + "guidance " * 200)
    manager = SkillManager()
    manager.load_packages(tmp_path / "skills")

    class Client:
        model = "offline"

        def __init__(self):
            self.requests = []
            self.summaries = 0
            self.actions = iter(
                [("load_skill", {"name": "python_expert"})]
                + [("echo", {"step": i}) for i in range(6)]
                + [("load_skill", {"name": "python_expert"}), ("finish", {"answer": "done"})]
            )

        def respond(self, messages, **kwargs):
            self.requests.append(messages)
            action, args = next(self.actions)
            return json.dumps({"thought": "next", "action": action, "action_input": args})

        def complete_summary(self, messages, **kwargs):
            self.summaries += 1
            return {"text": "Earlier guidance and observations summarized."}

        def extract_text(self, response):
            return response["text"]

    client = Client()
    writer = TraceWriter(tmp_path / "trace.jsonl")
    agent = ReactAgent(
        client,
        [Tool("echo", "observation", lambda args: "observation " * 200, read_only=True)],
        system_prompt="Use tools to complete the task.",
        skill_manager=manager,
        enable_planning=False,
        context_token_budget=1600,
        trace_writer=writer,
    )
    agent._context_window.output_token_reserve = 0
    agent._context_window.safety_margin_tokens = 0
    _ = agent.compressor.store
    agent.compressor.compactor.policy = replace(agent.compressor.compactor.policy, keep_recent=2)
    try:
        result = agent.run("python guidance", max_steps=10)
        assert result["metadata"]["status"] == "success"
        assert client.summaries > 0
        assert "UNIQUE_SKILL_INSTRUCTION" not in str(client.requests[-2])
        assert "UNIQUE_SKILL_INSTRUCTION" in str(client.requests[-1])
        loads = [
            event
            for event in load_trace_events(writer.path)
            if event["event"] == "skill_entry_loaded"
        ]
        assert len(loads) == 2
        assert loads[0]["payload"]["content_hash"] == loads[1]["payload"]["content_hash"]
    finally:
        agent.close()
        writer.close()
