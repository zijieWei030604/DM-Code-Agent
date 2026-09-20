"""SWE-bench task-container backed execution tools.

File discovery and edits stay in the host workspace.  The workspace is bind-mounted at
``/testbed`` so execution observes the same bytes while using the official image's Linux
runtime and dependencies.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from dm_agent.tools.base import Tool, ToolResult, _require_str
from dm_agent.tools.execution_tools import (
    _classify_declared_shell_verification,
    _classify_pytest_result,
    _classify_unittest_result,
    _verification_metadata,
)

_CONTAINER_PREAMBLE = """
set -o pipefail
if [ -f /opt/miniconda3/bin/activate ]; then
  . /opt/miniconda3/bin/activate
  conda activate testbed
fi
cd /testbed
""".strip()

_CONTAINER_WORKSPACE = "/testbed"
_HOST_PATH_KEYS = frozenset({"path", "root"})
_HOST_PATH_LIST_KEYS = frozenset({"paths"})
_CONTAINER_EXECUTION_TOOLS = frozenset({"run_shell", "run_python", "run_tests"})


def _host_workspace_path(value: str) -> str:
    """Translate a Docker ``/testbed`` path into a host-workspace-relative path."""
    normalized = value.replace("\\", "/")
    if normalized == _CONTAINER_WORKSPACE:
        return "."
    prefix = f"{_CONTAINER_WORKSPACE}/"
    if not normalized.startswith(prefix):
        return value

    relative = PurePosixPath(normalized.removeprefix(prefix))
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(f"/testbed 路径不能越出任务工作区: {value}")
    return relative.as_posix()


def _translate_host_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    translated = dict(arguments)
    for key in _HOST_PATH_KEYS:
        value = translated.get(key)
        if isinstance(value, str):
            translated[key] = _host_workspace_path(value)
    for key in _HOST_PATH_LIST_KEYS:
        value = translated.get(key)
        if isinstance(value, list):
            translated[key] = [
                _host_workspace_path(item) if isinstance(item, str) else item
                for item in value
            ]
    return translated


def _wrap_host_runner(
    runner: Callable[[dict[str, Any]], str | ToolResult],
) -> Callable[[dict[str, Any]], str | ToolResult]:
    def wrapped(arguments: dict[str, Any]) -> str | ToolResult:
        return runner(_translate_host_tool_arguments(arguments))

    return wrapped


def bind_host_workspace_paths(tools: list[Tool]) -> list[Tool]:
    """Make container-visible paths usable by host-side repository tools."""
    bound: list[Tool] = []
    for tool in tools:
        if tool.name in _CONTAINER_EXECUTION_TOOLS:
            bound.append(tool)
            continue
        bound.append(
            Tool(
                name=tool.name,
                description=tool.description,
                runner=(
                    tool.runner
                    if tool.result_runner is not None
                    else _wrap_host_runner(tool.runner)
                ),
                result_runner=(
                    _wrap_host_runner(tool.result_runner)
                    if tool.result_runner is not None
                    else None
                ),
                read_only=tool.read_only,
                input_schema=tool.input_schema,
            )
        )
    return bound


@dataclass
class ContainerExecutionStats:
    calls: int = 0
    failures: int = 0


class ContainerExecutionBackend:
    """Run the three execution tools inside one live SWE-bench task container."""

    def __init__(self, container_name: str) -> None:
        self.container_name = container_name
        self.stats = ContainerExecutionStats()

    def _execute(
        self, command: str, *, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                f"{_CONTAINER_PREAMBLE}\n{command}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    @staticmethod
    def _format_result(result: subprocess.CompletedProcess[str]) -> str:
        segments: list[str] = []
        if result.stdout:
            segments.append(result.stdout.strip())
        if result.stderr:
            segments.append(f"stderr:\n{result.stderr.strip()}")
        segments.append(f"returncode: {result.returncode}")
        return "\n".join(segment for segment in segments if segment).strip()

    def _run_completed(self, command: str) -> tuple[subprocess.CompletedProcess[str], str]:
        result = self._execute(command)
        self.stats.calls += 1
        if result.returncode != 0:
            self.stats.failures += 1
        return result, self._format_result(result)

    def _run(self, command: str) -> str:
        return self._run_completed(command)[1]

    def run_validation(self, arguments: list[str], timeout: int) -> tuple[int, str]:
        """Run a verifier with the task image's Python and dependencies."""
        command = " ".join(shlex.quote(part) for part in ["python", *arguments])
        try:
            result = self._execute(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.stats.calls += 1
            self.stats.failures += 1
            return 124, f"validation timed out after {timeout} seconds"
        self.stats.calls += 1
        if result.returncode != 0:
            self.stats.failures += 1
        return result.returncode, self._format_result(result)

    def run_shell(self, arguments: dict[str, Any]) -> str:
        return str(self.run_shell_result(arguments))

    def run_shell_result(self, arguments: dict[str, Any]) -> ToolResult:
        command = _require_str(arguments, "command")
        purpose = arguments.get("purpose", "execution")
        if purpose not in {"execution", "verification"}:
            raise ValueError("purpose 必须是 'execution' 或 'verification'。")
        result, output = self._run_completed(command)
        metadata: dict[str, Any] = {}
        check_scope: tuple[str, ...] = ()
        error_code = ""
        if purpose == "verification":
            verification = _classify_declared_shell_verification(
                result.returncode, output, command
            )
            metadata["verification"] = verification
            check_scope = (command,)
            error_code = str(verification["failure_kind"])
        return ToolResult(
            "success" if result.returncode == 0 else "failed",
            output,
            error_code=error_code,
            exit_code=result.returncode,
            check_scope=check_scope,
            metadata=metadata,
        )

    def run_python(self, arguments: dict[str, Any]) -> str:
        return str(self.run_python_result(arguments))

    def run_python_result(self, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        path_value = arguments.get("path")
        if isinstance(code, str) and code.strip():
            command = f"python -u -c {shlex.quote(code)}"
            scope = ["<inline>"]
        elif isinstance(path_value, str) and path_value.strip():
            command_parts = ["python", "-u", str(PurePosixPath(path_value))]
            scope = [str(PurePosixPath(path_value))]
            extra_args = arguments.get("args")
            if isinstance(extra_args, list):
                command_parts.extend(str(item) for item in extra_args)
            elif isinstance(extra_args, str) and extra_args.strip():
                command_parts.extend(shlex.split(extra_args))
            elif extra_args is not None:
                raise ValueError("工具参数 'args' 必须是字符串或字符串列表。")
            command = " ".join(shlex.quote(part) for part in command_parts)
        else:
            raise ValueError("run_python 工具需要 'code' 或 'path' 参数。")
        result, output = self._run_completed(command)
        passed = result.returncode == 0
        failure_kind = "" if passed else (
            "syntax_error"
            if "SyntaxError" in output or "IndentationError" in output
            else "python_error"
        )
        verification = _verification_metadata(
            execution_status="completed",
            outcome="passed" if passed else "error",
            framework="python",
            scope=scope,
            failure_kind=failure_kind,
        )
        return ToolResult(
            "success" if passed else "failed",
            output,
            error_code=failure_kind,
            exit_code=result.returncode,
            check_scope=tuple(scope),
            metadata={"verification": verification},
        )

    def run_tests(self, arguments: dict[str, Any]) -> str:
        return str(self.run_tests_result(arguments))

    def run_tests_result(self, arguments: dict[str, Any]) -> ToolResult:
        test_path = arguments.get("test_path", ".")
        targets = arguments.get("targets")
        framework = arguments.get("framework", "pytest")
        verbose = arguments.get("verbose", False)
        if targets is not None:
            if not isinstance(targets, list) or not targets or not all(
                isinstance(item, str) and item.strip() for item in targets
            ):
                raise ValueError("targets 必须是非空字符串列表。")
            scopes = list(dict.fromkeys(item.strip() for item in targets))
        elif not isinstance(test_path, str):
            raise ValueError("test_path 必须是字符串。")
        else:
            scopes = [test_path.strip() or "."]
        if framework not in {"pytest", "unittest"}:
            raise ValueError("framework 必须是 'pytest' 或 'unittest'。")

        if framework == "pytest":
            command_parts = ["python", "-m", "pytest"]
            if verbose:
                command_parts.append("-v")
            command_parts.extend(str(PurePosixPath(scope)) for scope in scopes)
        else:
            command_parts = ["python", "-m", "unittest"]
            if verbose:
                command_parts.append("-v")
            if len(scopes) == 1 and not scopes[0].endswith(".py"):
                command_parts.extend(["discover", "-s", str(PurePosixPath(scopes[0]))])
            else:
                command_parts.extend(
                    str(PurePosixPath(scope)).removesuffix(".py").replace("/", ".")
                    for scope in scopes
                )
        result, output = self._run_completed(
            " ".join(shlex.quote(part) for part in command_parts)
        )
        if framework == "pytest":
            counts = _pytest_counts_from_output(output)
            verification = _classify_pytest_result(
                result.returncode, counts=counts, scope=scopes
            )
        else:
            verification = _classify_unittest_result(
                result.returncode, output=output, scope=scopes
            )
        passed = verification["outcome"] == "passed"
        return ToolResult(
            "success" if passed else "failed",
            output,
            error_code=str(verification["failure_kind"]),
            exit_code=result.returncode,
            check_scope=tuple(scopes),
            metadata={"verification": verification},
        )

    def replace_execution_tools(self, tools: list[Tool]) -> list[Tool]:
        runners = {
            "run_shell": (self.run_shell, self.run_shell_result),
            "run_python": (self.run_python, self.run_python_result),
            "run_tests": (self.run_tests, self.run_tests_result),
        }
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                runner=runners[tool.name][0],
                result_runner=runners[tool.name][1],
                read_only=tool.read_only,
                input_schema=tool.input_schema,
            )
            if tool.name in runners
            else tool
            for tool in tools
        ]


def _pytest_counts_from_output(output: str) -> dict[str, int]:
    """Extract enough pytest summary data to distinguish failures from setup errors."""
    import re

    counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    patterns = {
        "passed": r"(?:^|\s)(\d+) passed\b",
        "failed": r"(?:^|\s)(\d+) failed\b",
        "errors": r"(?:^|\s)(\d+) errors?\b",
        "skipped": r"(?:^|\s)(\d+) skipped\b",
    }
    for name, pattern in patterns.items():
        matches = re.findall(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            counts[name] = int(matches[-1])
    counts["collected"] = sum(counts.values())
    return counts
