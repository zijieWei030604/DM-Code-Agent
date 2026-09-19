"""代码执行工具"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from .base import ToolResult, _require_str

# run_linter 支持的检查器，按推荐顺序（ruff 最快且覆盖面最广）。
_LINTER_TOOLS = ("ruff", "flake8", "pylint", "mypy", "black")


def available_linters() -> list[str]:
    """探测当前解释器里实际装了哪些检查器，不启动子进程。"""
    available: list[str] = []
    for name in _LINTER_TOOLS:
        try:
            if find_spec(name) is not None:
                available.append(name)
        except (ImportError, ValueError):
            continue
    return available


def run_python(arguments: dict[str, Any]) -> str:
    return str(run_python_result(arguments))


def run_python_result(arguments: dict[str, Any]) -> ToolResult:
    """运行 Python 代码或脚本"""
    code = arguments.get("code")
    path_value = arguments.get("path")

    scope: list[str]
    if isinstance(code, str) and code.strip():
        command = [sys.executable, "-u", "-c", code]
        scope = ["<inline>"]
    elif isinstance(path_value, str) and path_value.strip():
        path = Path(path_value)
        scope = [str(path)]
        if not path.exists():
            verification = _verification_metadata(
                execution_status="invalid",
                outcome="unknown",
                framework="python",
                scope=scope,
                failure_kind="invalid_target",
            )
            return ToolResult(
                "failed",
                f"Python 路径 {path} 不存在。",
                error_code="invalid_target",
                check_scope=tuple(scope),
                metadata={"verification": verification},
            )
        command = [sys.executable, "-u", str(path)]
        extra_args = arguments.get("args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args)
        elif isinstance(extra_args, str) and extra_args.strip():
            command.extend(shlex.split(extra_args))
        elif extra_args is not None:
            raise ValueError("工具参数 'args' 必须是字符串或字符串列表。")
    else:
        raise ValueError("run_python 工具需要 'code' 或 'path' 参数。")

    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    segments: list[str] = []
    if result.stdout:
        segments.append(result.stdout.strip())
    if result.stderr:
        segments.append(f"stderr:\n{result.stderr.strip()}")
    segments.append(f"returncode: {result.returncode}")
    output = "\n".join(segment for segment in segments if segment).strip()
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


def run_shell(arguments: dict[str, Any]) -> str:
    return str(run_shell_result(arguments))


def run_shell_result(arguments: dict[str, Any]) -> ToolResult:
    """运行 Shell 命令"""
    command = _require_str(arguments, "command")
    purpose = arguments.get("purpose", "execution")
    if purpose not in {"execution", "verification"}:
        raise ValueError("purpose 必须是 'execution' 或 'verification'。")
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    segments: list[str] = []
    if result.stdout:
        segments.append(result.stdout.strip())
    if result.stderr:
        segments.append(f"stderr:\n{result.stderr.strip()}")
    segments.append(f"returncode: {result.returncode}")
    output = "\n".join(segment for segment in segments if segment).strip()
    metadata: dict[str, Any] = {}
    check_scope: tuple[str, ...] = ()
    error_code = ""
    if purpose == "verification":
        verification = _classify_declared_shell_verification(result.returncode, output, command)
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


def _classify_declared_shell_verification(
    exit_code: int, output: str, command: str
) -> dict[str, Any]:
    if exit_code == 0:
        execution_status, outcome, failure_kind = "completed", "passed", ""
    elif "SyntaxError" in output or "IndentationError" in output:
        execution_status, outcome, failure_kind = "completed", "error", "syntax_error"
    elif "AssertionError" in output or "FAILED" in output or "failures=" in output:
        execution_status, outcome, failure_kind = "completed", "failed", "assertion_failure"
    else:
        execution_status, outcome, failure_kind = "unavailable", "unknown", "unknown_result"
    return _verification_metadata(
        execution_status=execution_status,
        outcome=outcome,
        framework="shell",
        scope=[command],
        failure_kind=failure_kind,
    )


def run_tests(arguments: dict[str, Any]) -> str:
    return str(run_tests_result(arguments))


def run_tests_result(arguments: dict[str, Any]) -> ToolResult:
    """运行 Python 测试套件（支持 pytest 和 unittest）"""
    framework = arguments.get("framework", "pytest")
    verbose = arguments.get("verbose", False)

    if framework not in ["pytest", "unittest"]:
        raise ValueError("framework 必须是 'pytest' 或 'unittest'。")

    targets = _test_targets(arguments)
    invalid = [target for target in targets if not _test_target_exists(target)]
    if invalid:
        verification = _verification_metadata(
            execution_status="invalid",
            outcome="unknown",
            framework=framework,
            scope=targets,
            failure_kind="invalid_target",
        )
        return ToolResult(
            "failed",
            "测试目标不存在：" + "、".join(invalid),
            error_code="invalid_test_target",
            check_scope=tuple(targets),
            metadata={"verification": verification},
        )

    if framework == "pytest":
        result, counts = _run_pytest(targets, verbose=bool(verbose))
        verification = _classify_pytest_result(
            result.returncode,
            counts=counts,
            scope=targets,
        )
    else:
        result = _run_unittest(targets, verbose=bool(verbose))
        verification = _classify_unittest_result(
            result.returncode,
            output="\n".join((result.stdout or "", result.stderr or "")),
            scope=targets,
        )

    segments: list[str] = []

    if result.stdout:
        segments.append(result.stdout.strip())
    if result.stderr:
        segments.append(f"stderr:\n{result.stderr.strip()}")
    segments.append(f"returncode: {result.returncode}")

    output = "\n".join(segment for segment in segments if segment).strip()
    passed = verification["outcome"] == "passed"
    return ToolResult(
        "success" if passed else "failed",
        output,
        error_code=str(verification["failure_kind"]),
        exit_code=result.returncode,
        check_scope=tuple(targets),
        metadata={"verification": verification},
    )


def _test_targets(arguments: dict[str, Any]) -> list[str]:
    raw_targets = arguments.get("targets")
    if raw_targets is not None:
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("targets 必须是非空字符串列表。")
        if not all(isinstance(item, str) and item.strip() for item in raw_targets):
            raise ValueError("targets 中的每一项都必须是非空字符串。")
        return list(dict.fromkeys(item.strip() for item in raw_targets))

    test_path = arguments.get("test_path", ".")
    if not isinstance(test_path, str):
        raise ValueError("test_path 必须是字符串。")
    if not test_path.strip():
        raise ValueError("test_path 不能为空。")
    return [test_path.strip()]


def _test_target_exists(target: str) -> bool:
    # pytest node ids append ``::Class::test`` to the filesystem path.
    path_part = target.split("::", 1)[0]
    return Path(path_part).exists()


def _run_pytest(
    targets: list[str], *, verbose: bool
) -> tuple[subprocess.CompletedProcess[str], dict[str, int]]:
    with tempfile.TemporaryDirectory(prefix="dm-agent-pytest-") as temp_dir:
        report = Path(temp_dir) / "result.xml"
        command = [sys.executable, "-m", "pytest", f"--junitxml={report}"]
        if verbose:
            command.append("-v")
        command.extend(targets)
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        counts = _read_junit_counts(report)
    return result, counts


def _read_junit_counts(path: Path) -> dict[str, int]:
    counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    if not path.is_file():
        return counts
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return counts

    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    for suite in suites:
        tests = _xml_int(suite.get("tests"))
        failures = _xml_int(suite.get("failures"))
        errors = _xml_int(suite.get("errors"))
        skipped = _xml_int(suite.get("skipped"))
        counts["collected"] += tests
        counts["failed"] += failures
        counts["errors"] += errors
        counts["skipped"] += skipped
    counts["passed"] = max(
        0,
        counts["collected"] - counts["failed"] - counts["errors"] - counts["skipped"],
    )
    return counts


def _xml_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _classify_pytest_result(
    exit_code: int,
    *,
    counts: dict[str, int],
    scope: list[str],
) -> dict[str, Any]:
    if exit_code == 0:
        execution_status, outcome, failure_kind = "completed", "passed", ""
    elif exit_code == 1 and counts["failed"] > 0:
        execution_status, outcome, failure_kind = "completed", "failed", "assertion_failure"
    elif exit_code == 1:
        execution_status, outcome, failure_kind = "completed", "error", "test_error"
    elif exit_code == 2:
        execution_status, outcome, failure_kind = "interrupted", "unknown", "interrupted"
    elif exit_code == 3:
        execution_status, outcome, failure_kind = "unavailable", "error", "pytest_internal_error"
    elif exit_code == 4:
        execution_status, outcome, failure_kind = "invalid", "unknown", "usage_error"
    elif exit_code == 5:
        execution_status, outcome, failure_kind = "invalid", "no_tests", "no_tests_collected"
    elif exit_code == 6:
        execution_status, outcome, failure_kind = "completed", "error", "warnings_error"
    else:
        execution_status, outcome, failure_kind = "unavailable", "unknown", "unknown_exit_code"
    return _verification_metadata(
        execution_status=execution_status,
        outcome=outcome,
        framework="pytest",
        scope=scope,
        failure_kind=failure_kind,
        counts=counts,
    )


def _run_unittest(targets: list[str], *, verbose: bool) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "unittest"]
    if verbose:
        command.append("-v")
    directories = [Path(target) for target in targets if Path(target).is_dir()]
    if directories:
        if len(targets) != 1:
            raise ValueError("unittest 的目录发现模式一次只能接收一个 targets 项。")
        command.extend(["discover", "-s", str(directories[0])])
    else:
        command.extend(_unittest_module(target) for target in targets)
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _unittest_module(target: str) -> str:
    path_part, separator, suffix = target.partition("::")
    module = str(Path(path_part).with_suffix("")).replace("/", ".").replace("\\", ".")
    if separator:
        module += "." + suffix.replace("::", ".")
    return module


def _classify_unittest_result(
    exit_code: int,
    *,
    output: str,
    scope: list[str],
) -> dict[str, Any]:
    if exit_code == 0 and "OK" in output:
        execution_status, outcome, failure_kind = "completed", "passed", ""
    elif "FAILED" in output and "failures=" in output:
        execution_status, outcome, failure_kind = "completed", "failed", "assertion_failure"
    elif "FAILED" in output and "errors=" in output:
        execution_status, outcome, failure_kind = "completed", "error", "test_error"
    else:
        execution_status, outcome, failure_kind = "unavailable", "unknown", "unknown_result"
    return _verification_metadata(
        execution_status=execution_status,
        outcome=outcome,
        framework="unittest",
        scope=scope,
        failure_kind=failure_kind,
    )


def _verification_metadata(
    *,
    execution_status: str,
    outcome: str,
    framework: str,
    scope: list[str],
    failure_kind: str,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    values = counts or {}
    return {
        "execution_status": execution_status,
        "outcome": outcome,
        "framework": framework,
        "scope": list(scope),
        "collected": int(values.get("collected", 0)),
        "passed": int(values.get("passed", 0)),
        "failed": int(values.get("failed", 0)),
        "errors": int(values.get("errors", 0)),
        "skipped": int(values.get("skipped", 0)),
        "failure_kind": failure_kind,
    }


def run_linter(arguments: dict[str, Any]) -> str:
    return str(run_linter_result(arguments))


def run_linter_result(arguments: dict[str, Any]) -> ToolResult:
    """运行代码检查工具（支持 ruff、pylint、flake8、mypy、black）"""
    path_value = _require_str(arguments, "path")
    tool = arguments.get("tool", "ruff")

    if tool not in _LINTER_TOOLS:
        raise ValueError(
            "tool 必须是 " + "、".join(f"'{name}'" for name in _LINTER_TOOLS) + " 之一。"
        )

    path = Path(path_value)
    if not path.exists():
        verification = _verification_metadata(
            execution_status="invalid",
            outcome="unknown",
            framework=str(tool),
            scope=[str(path)],
            failure_kind="invalid_target",
        )
        return ToolResult(
            "failed",
            f"路径 {path} 不存在。",
            error_code="file_not_found",
            check_scope=(str(path),),
            metadata={"verification": verification},
        )

    if tool == "black":
        # black 用于格式化，添加 --check 只检查不修改
        command = [sys.executable, "-m", tool, "--check", str(path)]
    elif tool == "ruff":
        command = [sys.executable, "-m", tool, "check", str(path)]
    else:
        command = [sys.executable, "-m", tool, str(path)]

    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    # 当前解释器没装这个检查器。直接把子进程的 "No module named X" 回给模型，会让它
    # 逐个盲试下一个（实测一道题平均撞两次），因此改为报出本环境实际可用的清单。
    # 文案刻意避开 core/observation.py 的失败标记：换一个检查器重试是局部纠正，
    # 不该白烧一次完整重规划。
    if result.returncode != 0 and f"No module named {tool}" in (result.stderr or ""):
        available = [name for name in available_linters() if name != tool]
        if available:
            verification = _verification_metadata(
                execution_status="unavailable",
                outcome="unknown",
                framework=tool,
                scope=[str(path)],
                failure_kind="checker_unavailable",
            )
            return ToolResult(
                "unavailable",
                (
                    f"当前环境未提供 {tool}。可用的检查工具：{'、'.join(available)}，"
                    f"请改用其中之一重试。"
                ),
                error_code="checker_unavailable",
                check_scope=(str(path),),
                metadata={"verification": verification},
            )
        verification = _verification_metadata(
            execution_status="unavailable",
            outcome="unknown",
            framework=tool,
            scope=[str(path)],
            failure_kind="checker_unavailable",
        )
        return ToolResult(
            "unavailable",
            f"当前环境未提供 {tool}，也没有其他可用的检查工具，本步可跳过。",
            error_code="checker_unavailable",
            check_scope=(str(path),),
            metadata={"verification": verification},
        )

    segments: list[str] = []

    if result.stdout:
        segments.append(result.stdout.strip())
    if result.stderr:
        segments.append(f"stderr:\n{result.stderr.strip()}")
    segments.append(f"returncode: {result.returncode}")

    output = "\n".join(segment for segment in segments if segment).strip()
    passed = result.returncode == 0
    verification = _verification_metadata(
        execution_status="completed",
        outcome="passed" if passed else "failed",
        framework=tool,
        scope=[str(path)],
        failure_kind="" if passed else "lint_failure",
    )
    return ToolResult(
        "success" if passed else "failed",
        output,
        error_code="" if passed else "lint_failure",
        exit_code=result.returncode,
        check_scope=(str(path),),
        metadata={"verification": verification},
    )
