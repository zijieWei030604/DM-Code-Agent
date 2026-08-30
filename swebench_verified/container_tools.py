"""SWE-bench task-container backed execution tools.

File discovery and edits stay in the host workspace.  The workspace is bind-mounted at
``/testbed`` so execution observes the same bytes while using the official image's Linux
runtime and dependencies.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from dm_agent.tools.base import Tool, _require_str

_CONTAINER_PREAMBLE = """
set -o pipefail
if [ -f /opt/miniconda3/bin/activate ]; then
  . /opt/miniconda3/bin/activate
  conda activate testbed
fi
cd /testbed
""".strip()


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

    def _run(self, command: str) -> str:
        result = self._execute(command)
        self.stats.calls += 1
        if result.returncode != 0:
            self.stats.failures += 1
        return self._format_result(result)

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
        return self._run(_require_str(arguments, "command"))

    def run_python(self, arguments: dict[str, Any]) -> str:
        code = arguments.get("code")
        path_value = arguments.get("path")
        if isinstance(code, str) and code.strip():
            command = f"python -u -c {shlex.quote(code)}"
        elif isinstance(path_value, str) and path_value.strip():
            command_parts = ["python", "-u", str(PurePosixPath(path_value))]
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
        return self._run(command)

    def run_tests(self, arguments: dict[str, Any]) -> str:
        test_path = arguments.get("test_path", ".")
        framework = arguments.get("framework", "pytest")
        verbose = arguments.get("verbose", False)
        if not isinstance(test_path, str):
            raise ValueError("test_path 必须是字符串。")
        if framework not in {"pytest", "unittest"}:
            raise ValueError("framework 必须是 'pytest' 或 'unittest'。")

        if framework == "pytest":
            command_parts = ["python", "-m", "pytest"]
            if verbose:
                command_parts.append("-v")
            command_parts.append(str(PurePosixPath(test_path)))
        else:
            command_parts = ["python", "-m", "unittest"]
            if verbose:
                command_parts.append("-v")
            normalized = str(PurePosixPath(test_path))
            if normalized.endswith(".py"):
                command_parts.append(normalized.removesuffix(".py").replace("/", "."))
            else:
                command_parts.extend(["discover", "-s", normalized])
        return self._run(" ".join(shlex.quote(part) for part in command_parts))

    def replace_execution_tools(self, tools: list[Tool]) -> list[Tool]:
        runners = {
            "run_shell": self.run_shell,
            "run_python": self.run_python,
            "run_tests": self.run_tests,
        }
        return [
            Tool(tool.name, tool.description, runners[tool.name]) if tool.name in runners else tool
            for tool in tools
        ]
