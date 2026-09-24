"""Task-scoped progressive skill loading through ordinary tool calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from dm_agent.tools.base import Tool, ToolResult

from .manager import SkillManager
from .packages import MarkdownSkill


def prepare_skills(
    manager: SkillManager,
    task: str,
    tools: dict[str, Tool],
    record: Callable[[str, dict[str, Any]], Any],
    activated: list[str],
) -> str:
    manager.deactivate_all()
    recommended = manager.select_skills_for_task(task)
    record("skill_recommendations", {"recommended": recommended})
    lines = [
        "\nAvailable skills (guidance is loaded only when needed):",
        "Use load_skill(name) for instructions and optional tools; use "
        "load_skill_resource(name, resource) for details. You may reload after context compaction.",
    ]
    for name, skill in manager.skills.items():
        meta = skill.get_metadata()
        marker = " [recommended]" if name in recommended else ""
        lines.append(f"- {name}{marker}: {meta.description}")

    def load(arguments: dict[str, Any], *, resource: bool = False) -> ToolResult:
        name = str(arguments.get("name", ""))
        skill = manager.skills.get(name)
        if skill is None:
            return ToolResult("failed", f"Unknown skill: {name}", error_code="unknown_skill")
        try:
            resource_name = str(arguments.get("resource", "")) if resource else ""
            additions: list[Tool] = []
            if resource:
                if not isinstance(skill, MarkdownSkill):
                    raise ValueError("This skill has no file resources")
                path = skill.resource_path(resource_name)
                text = path.read_text(encoding="utf-8")
                source = str(path)
            else:
                text = skill.get_prompt_addition()
                source = manager.sources.get(name, "registered")
                additions = skill.get_tools()
                if len({tool.name for tool in additions}) != len(additions):
                    raise ValueError("Duplicate skill tool names")
                if name not in manager.active_skills:
                    conflicts = [tool.name for tool in additions if tool.name in tools]
                    if conflicts:
                        raise ValueError(f"Skill tool conflicts: {conflicts}")
                resources = skill.resource_names() if isinstance(skill, MarkdownSkill) else []
            metadata = {
                "skill": name,
                "resource": resource_name,
                "source": source,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            if not resource and name not in manager.active_skills:
                skill.on_activate()
                tools.update({tool.name: tool for tool in additions})
                manager.active_skills.append(name)
                activated.append(name)
            record("skill_resource_loaded" if resource else "skill_entry_loaded", metadata)
            if not resource:
                text += "\n\nResources: " + (", ".join(resources) or "none")
                text += "\nAvailable skill tools: " + json.dumps(
                    [tool.function_definition() for tool in additions], ensure_ascii=False
                )
            return ToolResult("success", text, metadata=metadata)
        except (OSError, ValueError, UnicodeError) as exc:
            return ToolResult("failed", str(exc), error_code="skill_load_error")

    def load_resource(arguments: dict[str, Any]) -> ToolResult:
        return load(arguments, resource=True)

    for tool_name, resource in (("load_skill", False), ("load_skill_resource", True)):
        if tool_name in tools:
            raise ValueError(f"Reserved skill tool name: {tool_name}")
        properties = {"name": {"type": "string", "minLength": 1}}
        if resource:
            properties["resource"] = {"type": "string", "minLength": 1}
        tools[tool_name] = Tool(
            name=tool_name,
            description=(
                "Read a skill resource relative to resources/."
                if resource
                else "Load skill instructions, resource index and optional tools."
            ),
            runner=load_resource if resource else load,
            read_only=True,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        )
    return "\n".join(lines)
