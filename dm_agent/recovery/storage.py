"""Immutable content objects and explicitly scoped workspace manifests."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import stat
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class RecoveryError(ValueError):
    """Fail closed: never claim an incomplete snapshot is recoverable."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def relative_path(raw: str) -> Path:
    parts = PurePosixPath(raw).parts
    if (
        not parts
        or raw != PurePosixPath(raw).as_posix()
        or PurePosixPath(raw).is_absolute()
        or any(p in {".", "..", ".git"} or ":" in p or "\\" in p for p in parts)
    ):
        raise RecoveryError(f"Unsafe snapshot path: {raw}")
    return Path(*parts)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise RecoveryError(result.stderr.decode("utf-8", errors="replace").strip())
    text = result.stdout.decode("utf-8", errors="surrogateescape")
    return text if "-z" in args else text.rstrip("\r\n")


@dataclass(frozen=True)
class SnapshotPolicy:
    # Explicit includes override Git ignore rules, never the hard directory exclusions.
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    max_file_bytes: int = 32 * 1024 * 1024
    max_snapshot_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0 or self.max_snapshot_bytes <= 0:
            raise RecoveryError("Snapshot size limits must be positive")

    def excluded(self, name: str) -> bool:
        parts = PurePosixPath(name).parts
        hard = {
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
        }
        return (
            any(p in hard for p in parts)
            or any(fnmatch.fnmatchcase(name, pattern) for pattern in self.exclude)
            or (
                any(p == ".env" or p.startswith(".env.") for p in parts) and not self.included(name)
            )
        )

    def included(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.include)


class WorkspaceStore:
    def __init__(self, path: Path, policy: SnapshotPolicy | None = None) -> None:
        self.path = path.resolve()
        self.policy = policy or SnapshotPolicy()
        self.path.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        key = digest(data)
        path = self.path / "objects" / key
        if path.exists():
            if digest(path.read_bytes()) != key:
                raise RecoveryError(f"Corrupt content object: {key}")
        else:
            durable_write(path, data)
        return key

    def get(self, key: str) -> bytes:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise RecoveryError("Invalid content object identifier")
        try:
            data = (self.path / "objects" / key).read_bytes()
        except OSError as exc:
            raise RecoveryError(f"Missing content object: {key}") from exc
        if digest(data) != key:
            raise RecoveryError(f"Corrupt content object: {key}")
        return data

    def scan(self, root: Path, *, persist: bool = False) -> dict[str, Any]:
        root = root.resolve()
        if not root.is_dir():
            raise RecoveryError("Workspace directory is missing")
        if self.path.is_relative_to(root):
            raise RecoveryError("Recovery store must be outside the workspace")
        is_git = (root / ".git").exists()
        tracked: set[str] | None = None
        commit = ""
        if is_git:
            if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
                raise RecoveryError("Use the Git worktree root as workspace")
            commit = git(root, "rev-parse", "HEAD")
            for record in git(root, "ls-files", "--stage", "-z").split("\0"):
                if record.startswith("160000 "):
                    name = record.split("\t", 1)[1]
                    if not self.policy.excluded(name):
                        raise RecoveryError(f"Submodule is not supported: {name}")
            tracked = set(
                git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(
                    "\0"
                )
            )
        files: dict[str, Any] = {}
        directories: list[str] = []
        omitted: list[str] = []
        total = 0
        for directory, children, names in os.walk(root, followlinks=False):
            parent = Path(directory)
            keep = []
            for name in sorted(children):
                path = parent / name
                rel = path.relative_to(root).as_posix()
                if self.policy.excluded(rel):
                    omitted.append(rel + "/")
                    continue
                if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                    raise RecoveryError(f"Linked directory is not supported: {rel}")
                # Ignore directories only when Git ignores them and no include might live below.
                if tracked is not None and not self.policy.include:
                    result = subprocess.run(
                        ["git", "-C", str(root), "check-ignore", "-q", "--", rel],
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    if result.returncode == 0 and not any(p.startswith(rel + "/") for p in tracked):
                        omitted.append(rel + "/")
                        continue
                keep.append(name)
                directories.append(rel)
            children[:] = keep
            for name in sorted(names):
                path = parent / name
                rel = path.relative_to(root).as_posix()
                if self.policy.excluded(rel) or (
                    tracked is not None and rel not in tracked and not self.policy.included(rel)
                ):
                    omitted.append(rel)
                    continue
                relative_path(rel)
                info = path.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or path.is_symlink()
                    or getattr(info, "st_file_attributes", 0) & 0x400
                ):
                    raise RecoveryError(f"Non-regular file is not supported: {rel}")
                if info.st_size > self.policy.max_file_bytes:
                    raise RecoveryError(f"File exceeds snapshot limit: {rel}")
                total += info.st_size
                if total > self.policy.max_snapshot_bytes:
                    raise RecoveryError("Workspace exceeds snapshot size limit")
                data = path.read_bytes()
                after = path.stat()
                if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RecoveryError(f"File changed during capture: {rel}")
                key = self.put(data) if persist else digest(data)
                files[rel] = {"hash": key, "mode": stat.S_IMODE(info.st_mode), "size": len(data)}
        return {
            "version": 1,
            "root": str(root),
            "git_commit": commit,
            "files": files,
            "directories": sorted(directories),
            "omitted": sorted(set(omitted)),
            "policy": asdict(self.policy),
        }

    def capture(self, root: Path) -> str:
        manifest = self.scan(root, persist=True)
        check = self.scan(root)
        if (
            manifest["files"] != check["files"]
            or manifest["directories"] != check["directories"]
            or manifest["git_commit"] != check["git_commit"]
        ):
            raise RecoveryError("Workspace changed during capture")
        return self.put(encode(manifest))

    def manifest(self, key: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.get(key))
        if data.get("version") != 1:
            raise RecoveryError("Unsupported manifest")
        for name in data["directories"]:
            relative_path(name)
        for name, item in data["files"].items():
            relative_path(name)
            if len(self.get(item["hash"])) != item["size"]:
                raise RecoveryError(f"Size mismatch: {name}")
        return data
