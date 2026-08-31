"""Policy-driven verification primitives for code edit transactions."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CommandRunner = Callable[[list[str], int], tuple[int, str]]


@dataclass(frozen=True)
class VerificationPolicy:
    """Select verification checks without coupling them to Agent control flow."""

    check_syntax: bool = True
    check_runtime_syntax: bool = False
    check_lint: bool = False
    check_types: bool = False
    baseline_aware_types: bool = False
    # Related tests remain available as semantic guidance, but are not safe
    # transaction gates by default: repositories often index fixtures,
    # conftest modules, examples, or tests that already fail in the runtime.
    run_affected_tests: bool = False
    max_affected_tests: int | None = None
    timeout: int = 120
    # A step budget or transient API failure must not discard a syntactically
    # valid patch. Run-end finalization uses the same lightweight policy.
    finalize_on_run_end: bool = True

    def __post_init__(self) -> None:
        if self.max_affected_tests is not None and self.max_affected_tests < 1:
            raise ValueError("max_affected_tests must be >= 1 or None.")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0.")


@dataclass(frozen=True)
class VerificationResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[VerificationResult, ...]
    selected_tests: tuple[str, ...]
    cache_key: str
    cache_hit: bool = False

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[VerificationResult, ...]:
        return tuple(result for result in self.results if not result.passed)


@dataclass(frozen=True)
class BaselineCapture:
    captured: bool
    diagnostics: int = 0
    error: str = ""


class VerificationRunner:
    """Execute one policy against changed files and cache patch-identical reports."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: VerificationPolicy | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or VerificationPolicy()
        self.command_runner = command_runner
        self._cache: dict[str, VerificationReport] = {}
        self._mypy_baselines: dict[str, frozenset[str] | None] = {}
        self._module_availability: dict[str, bool] = {}

    def reset(self) -> None:
        self._cache.clear()
        self._mypy_baselines.clear()
        self._module_availability.clear()

    def select_tests(self, tests: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(str(test) for test in tests))
        limit = self.policy.max_affected_tests
        return selected if limit is None else selected[:limit]

    def capture_type_baseline(self, path: str | Path) -> BaselineCapture:
        target = Path(path).resolve()
        if (
            not self.policy.check_types
            or not self.policy.baseline_aware_types
            or target.suffix != ".py"
            or not target.is_file()
        ):
            return BaselineCapture(False)
        relative = target.relative_to(self.root).as_posix()
        if relative in self._mypy_baselines:
            return BaselineCapture(False)
        try:
            if not self._module_is_available("mypy"):
                self._mypy_baselines[relative] = frozenset()
                return BaselineCapture(False)
            returncode, output = self._execute(["-m", "mypy", relative])
            diagnostics = _mypy_diagnostics(output)
            self._mypy_baselines[relative] = diagnostics if returncode else frozenset()
            return BaselineCapture(True, len(diagnostics))
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._mypy_baselines[relative] = None
            return BaselineCapture(False, error=str(exc))

    def verify(
        self,
        changed_paths: Iterable[str | Path],
        *,
        affected_tests: Iterable[str] = (),
    ) -> VerificationReport:
        changed = tuple(sorted(Path(path).resolve() for path in changed_paths))
        tests = self.select_tests(affected_tests) if self.policy.run_affected_tests else ()
        cache_key = self._cache_key(changed, tests)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return VerificationReport(cached.results, cached.selected_tests, cache_key, True)

        results: list[VerificationResult] = []
        changed_python = tuple(
            path.relative_to(self.root).as_posix()
            for path in changed
            if path.suffix == ".py" and path.is_file()
        )
        if self.policy.check_syntax:
            results.append(self._validate_python_syntax(changed))
        if self.policy.check_runtime_syntax and self.command_runner and changed_python:
            results.append(
                self._run_command("runtime_python_syntax", ["-m", "py_compile", *changed_python])
            )
        if self.policy.check_lint and changed_python:
            results.append(self._run_optional_module("ruff", ["check", *changed_python]))
        if self.policy.check_types and changed_python:
            if self.policy.baseline_aware_types:
                results.append(self.validate_mypy_delta(list(changed_python)))
            else:
                results.append(self._run_optional_module("mypy", list(changed_python)))
        if tests:
            results.append(self.run_tests(list(tests)))

        report = VerificationReport(tuple(results), tests, cache_key)
        self._cache[cache_key] = report
        return report

    def run_tests(self, tests: Sequence[str]) -> VerificationResult:
        return self._run_command("affected_tests", ["-m", "pytest", "-q", *tests])

    def validate_mypy_delta(self, changed_python: list[str]) -> VerificationResult:
        try:
            if not self._module_is_available("mypy"):
                location = " in validation runtime" if self.command_runner else ""
                return VerificationResult("mypy", True, f"skipped: mypy is not installed{location}")
            returncode, output = self._execute(["-m", "mypy", *changed_python])
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult("mypy", False, str(exc))
        if returncode == 0:
            return VerificationResult("mypy", True, "no type errors")
        if not self.policy.baseline_aware_types:
            return VerificationResult("mypy", False, _tail(output))

        baselines = [self._mypy_baselines.get(path) for path in changed_python]
        if any(baseline is None for baseline in baselines):
            return VerificationResult("mypy", False, f"baseline unavailable; {_tail(output)}")
        baseline_diagnostics = frozenset().union(
            *(baseline for baseline in baselines if baseline is not None)
        )
        current_diagnostics = _mypy_diagnostics(output)
        new_diagnostics = current_diagnostics - baseline_diagnostics
        if current_diagnostics and not new_diagnostics:
            return VerificationResult(
                "mypy",
                True,
                f"no new type errors ({len(current_diagnostics)} pre-existing diagnostic(s))",
            )
        detail = "\n".join(sorted(new_diagnostics)) if new_diagnostics else output
        return VerificationResult("mypy", False, _tail(detail))

    def _validate_python_syntax(self, changed: Sequence[Path]) -> VerificationResult:
        for path in changed:
            if path.suffix != ".py" or not path.is_file():
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as exc:
                return VerificationResult("python_syntax", False, f"{path}: {exc}")
        return VerificationResult("python_syntax", True, "changed Python files parse successfully")

    def _run_optional_module(self, module: str, arguments: list[str]) -> VerificationResult:
        try:
            available = self._module_is_available(module)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(module, False, str(exc))
        if not available:
            location = " in validation runtime" if self.command_runner else ""
            return VerificationResult(module, True, f"skipped: {module} is not installed{location}")
        return self._run_command(module, ["-m", module, *arguments])

    def _module_is_available(self, module: str) -> bool:
        if module in self._module_availability:
            return self._module_availability[module]
        if self.command_runner is None:
            available = importlib.util.find_spec(module) is not None
        else:
            probe = (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
            )
            returncode, _ = self._execute(["-c", probe])
            available = returncode == 0
        self._module_availability[module] = available
        return available

    def _run_command(self, name: str, arguments: list[str]) -> VerificationResult:
        try:
            returncode, output = self._execute(arguments)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(name, False, str(exc))
        detail = _tail(output) if output else f"exit code {returncode}"
        return VerificationResult(name, returncode == 0, detail)

    def _execute(self, arguments: list[str]) -> tuple[int, str]:
        if self.command_runner is not None:
            return self.command_runner(arguments, self.policy.timeout)
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.policy.timeout,
            shell=False,
        )
        return completed.returncode, (completed.stdout + "\n" + completed.stderr).strip()

    def _cache_key(self, changed: Sequence[Path], tests: Sequence[str]) -> str:
        payload = {
            "files": [
                (path.relative_to(self.root).as_posix(), _hash_path(path)) for path in changed
            ],
            "tests": list(tests),
            "policy": self.policy,
            "runtime": self.command_runner is not None,
        }
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


_MYPY_DIAGNOSTIC_RE = re.compile(r"^(?P<path>.+?\.py):\d+(?::\d+)?:\s+error:\s+(?P<message>.+)$")


def _mypy_diagnostics(output: str) -> frozenset[str]:
    diagnostics: set[str] = set()
    for line in str(output or "").splitlines():
        match = _MYPY_DIAGNOSTIC_RE.match(line.strip())
        if match:
            path = match.group("path").replace("\\", "/")
            diagnostics.add(f"{path}: error: {match.group('message')}")
    return frozenset(diagnostics)


def _hash_path(path: Path) -> str:
    try:
        content = path.read_bytes() if path.is_file() else b"missing"
    except OSError:
        return "unreadable"
    return hashlib.sha256(content).hexdigest()


def _tail(text: str, limit: int = 2000) -> str:
    return str(text or "")[-limit:]


__all__ = [
    "BaselineCapture",
    "CommandRunner",
    "VerificationPolicy",
    "VerificationReport",
    "VerificationResult",
    "VerificationRunner",
]
