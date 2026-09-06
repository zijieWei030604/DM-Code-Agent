"""Task-aware, incrementally cached repository maps for Python projects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from dm_agent.workspace import SemanticWorkspaceEngine


@dataclass(frozen=True)
class RepoMapResult:
    content: str
    scanned_files: int
    included_files: int
    parse_errors: int
    cache_hits: int
    truncated: bool
    fingerprint: str


@dataclass
class RepositoryMap:
    """Build a task-aware map backed by the persistent semantic workspace index."""

    max_scan_files: int = 300
    max_map_files: int = 30
    max_chars: int = 6000
    engine: SemanticWorkspaceEngine | None = field(default=None, repr=False)
    _engines: dict[Path, SemanticWorkspaceEngine] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, task: str, root: str | Path = ".") -> RepoMapResult:
        repository_root = Path(root).resolve()
        if not repository_root.is_dir():
            raise ValueError(f"Repo Map 根目录不是有效目录：{repository_root}")

        engine = self.engine
        if engine is not None:
            if engine.root != repository_root:
                raise ValueError(
                    "Repo Map 根目录与共享 Semantic Workspace 根目录不一致："
                    f"{repository_root} != {engine.root}"
                )
        else:
            engine = self._engines.get(repository_root)
            if engine is None:
                engine = SemanticWorkspaceEngine(
                    repository_root,
                    max_scan_files=self.max_scan_files,
                )
                self._engines[repository_root] = engine
        stats = engine.update()
        content, included_files, truncated = engine.build_repo_map(
            task,
            max_files=self.max_map_files,
            max_chars=self.max_chars,
        )
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return RepoMapResult(
            content=content,
            scanned_files=stats.scanned_files,
            included_files=included_files,
            parse_errors=stats.parse_errors,
            cache_hits=stats.cache_hits,
            truncated=truncated,
            fingerprint=fingerprint,
        )


__all__ = ["RepoMapResult", "RepositoryMap"]
