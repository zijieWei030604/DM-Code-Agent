"""Conservative source/config fingerprint for verification evidence."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

_IGNORED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".dm_agent",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "logs",
    "swebench_work",
}
_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".json",
}


def workspace_version(root: Path) -> str:
    digest = hashlib.sha256()
    git_paths = _git_workspace_paths(root)
    if git_paths is not None:
        for relative in git_paths:
            path = root / relative
            if not path.is_file() or path.is_symlink() or path.suffix not in _SUFFIXES:
                continue
            digest.update(relative.as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    for directory, children, files in os.walk(root):
        children[:] = sorted(
            name for name in children if name not in _IGNORED and not name.startswith(".")
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or path.suffix not in _SUFFIXES:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _git_workspace_paths(root: Path) -> list[Path] | None:
    """Return tracked and non-ignored untracked paths when ``root`` is a Git worktree."""
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return sorted(
        (Path(item.decode("utf-8", errors="surrogateescape")) for item in completed.stdout.split(b"\0") if item),
        key=lambda path: path.as_posix(),
    )
