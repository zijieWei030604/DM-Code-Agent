"""Conservative source/config fingerprint for verification evidence."""

from __future__ import annotations

import hashlib
import os
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
