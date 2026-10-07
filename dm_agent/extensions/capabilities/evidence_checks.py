"""Bounded, read-only probes in the same interpreter as task verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dm_agent.verification import CommandRunner

_MARKER = "DM_EVIDENCE_PROBE="
_ENVIRONMENT = """
import json, os, sys
try:
    from importlib.metadata import distributions
    packages = sorted((d.metadata['Name'], d.version) for d in distributions())
except ImportError:
    import pkg_resources
    packages = sorted((d.project_name, d.version) for d in pkg_resources.working_set)
print('DM_EVIDENCE_PROBE=' + json.dumps(dict(
    python=sys.version, executable=sys.executable, prefix=sys.prefix,
    packages=packages, environment={k: os.environ.get(k, '') for k in
    ('PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PATH')}
), sort_keys=True))
"""
_SYNTAX = """
import json, sys, tokenize
errors = []
for path in json.loads(sys.argv[1]):
    try:
        with tokenize.open(path) as f:
            compile(f.read(), path, 'exec')
    except SyntaxError as e:
        errors.append(dict(path=path, line=e.lineno, message=e.msg))
print('DM_EVIDENCE_PROBE=' + json.dumps(dict(errors=errors)))
"""


class EvidenceChecks:
    def __init__(self, runner: CommandRunner | None = None, environment_id: str = "local"):
        self.runner = runner
        self.environment_id = environment_id

    def _probe(self, root: Path, script: str, *arguments: str) -> dict[str, Any] | None:
        try:
            args = ["-c", script, *arguments]
            if self.runner is not None:
                code, output = self.runner(args, 15)
            else:
                result = subprocess.run(
                    [sys.executable, *args],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                code, output = result.returncode, result.stdout
            if code != 0:
                return None
            for line in reversed(output.splitlines()):
                if line.startswith(_MARKER):
                    value = json.loads(line[len(_MARKER) :])
                    return value if isinstance(value, dict) else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return None

    def environment(self, root: Path) -> str:
        value = self._probe(root, _ENVIRONMENT)
        if value is None:
            return ""
        return fingerprint([self.environment_id, value])

    def syntax(self, root: Path, paths: list[str]) -> dict[str, Any] | None:
        return self._probe(root, _SYNTAX, json.dumps(paths))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def test_identity(
    root: Path, arguments: dict[str, Any], files: dict[str, str], environment: str
) -> str:
    """Missing test sources/environment cannot establish comparable executions."""
    if not environment:
        return ""
    targets = arguments.get("targets") or [arguments.get("test_path", ".")]
    selected: list[str] = []
    for target in targets:
        candidate = Path(str(target).split("::", 1)[0])
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return ""
        matches = [
            p for p in files if relative == "." or p == relative or p.startswith(relative + "/")
        ]
        # A broad pytest target (normally ``.``) includes production files.
        # Those files are intentionally excluded from test identity, otherwise
        # every source edit makes the same regression test look incomparable.
        if relative == "." or candidate.is_dir():
            test_matches = [p for p in matches if _is_test_source(p)]
            matches = test_matches or matches
        if not matches:
            return ""
        selected.extend(matches)
    # Include helpers, conftest and configuration, not just the selected test body.
    sources = {
        p: h
        for p, h in files.items()
        if p in selected
        or _is_test_source(p)
        or Path(p).suffix in {".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"}
        or Path(p).name == "setup.py"
    }
    return fingerprint([arguments, sources, environment])


def _is_test_source(path: str) -> bool:
    candidate = Path(path)
    name = candidate.name.lower()
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in {part.lower() for part in candidate.parts}
        or name == "conftest.py"
    )
