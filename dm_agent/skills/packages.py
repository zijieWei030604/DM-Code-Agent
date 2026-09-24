"""File-backed skills: discover metadata now, read instructions on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import ConfigSkill


def read_header(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        if stream.readline().strip() != "---":
            raise ValueError("SKILL.md requires YAML frontmatter")
        lines = []
        for line in stream:
            if line.strip() == "---":
                break
            lines.append(line)
            if sum(map(len, lines)) > 16384:
                raise ValueError("Skill metadata exceeds 16 KiB")
        else:
            raise ValueError("Unclosed skill frontmatter")
    try:
        metadata = yaml.safe_load("".join(lines))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML frontmatter") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Skill metadata must be a mapping")
    for key in ("name", "description"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Skill requires {key}")
    for key in ("keywords", "patterns"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"{key} must be a list of strings")
    if not isinstance(metadata.get("priority", 10), int):
        raise ValueError("priority must be an integer")
    return metadata


class MarkdownSkill(ConfigSkill):
    def __init__(self, path: Path) -> None:
        self.root = path.parent.resolve()
        self.path = path.resolve()
        if not self.path.is_relative_to(self.root):
            raise ValueError("Skill entry escapes package directory")
        super().__init__(read_header(self.path))

    def get_prompt_addition(self) -> str:
        # Re-check containment when opening: resources may change after discovery.
        path = self.root / "SKILL.md"
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Skill entry escapes package directory")
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            for index, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    return "".join(lines[index + 1 :]).strip()
        raise ValueError("Invalid skill frontmatter")

    def resource_path(self, resource: str) -> Path:
        relative = Path(resource)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("Resource must be relative to resources/ without '..'")
        root = self.root / "resources"
        path = (root / relative).resolve()
        if not root.resolve().is_relative_to(self.root) or not path.is_relative_to(root.resolve()):
            raise ValueError("Resource escapes skill resources directory")
        if not path.is_file():
            raise ValueError("Skill resource does not exist")
        return path

    def resource_names(self) -> list[str]:
        root = self.root / "resources"
        if not root.resolve().is_relative_to(self.root):
            return []
        return sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file() and path.resolve().is_relative_to(root.resolve())
        )
