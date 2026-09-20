import ast
import json
import sys

import pytest

import dm_agent.tools.file_tools as file_tools_module
from dm_agent.core.observation import is_failure_observation
from dm_agent.tools import task_complete
from dm_agent.tools.code_analysis_tools import get_code_metrics, get_function_signature, parse_ast
from dm_agent.tools.code_index_tools import dependency_graph, inspect_change_impact, search_symbol
from dm_agent.tools.execution_tools import (
    _classify_declared_shell_verification,
    _classify_pytest_result,
    _validated_shell_scope,
    available_linters,
    run_linter,
    run_python,
    run_python_result,
    run_shell_result,
    run_tests_result,
)
from dm_agent.tools.file_tools import (
    EDIT_ECHO_MAX_LINES,
    _atomic_write_text,
    create_file,
    edit_file,
    find_files,
    read_file,
    search_code,
    search_in_file,
)
from dm_agent.tools.structured_edit_tools import edit_python_symbol, inspect_python_symbol


def test_run_tests_returns_structured_pytest_counts_and_supports_node_ids(tmp_path):
    target = tmp_path / "test_sample.py"
    target.write_text(
        "def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n",
        encoding="utf-8",
    )

    result = run_tests_result({"targets": [f"{target}::test_bad"]})

    verification = result.metadata["verification"]
    assert result.status == "failed"
    assert result.check_scope == (f"{target}::test_bad",)
    assert verification["execution_status"] == "completed"
    assert verification["outcome"] == "failed"
    assert verification["collected"] == 1
    assert verification["failed"] == 1
    assert verification["failure_kind"] == "assertion_failure"

    passed_result = run_tests_result({"targets": [f"{target}::test_ok"]})
    passed_verification = passed_result.metadata["verification"]
    assert passed_result.status == "success"
    assert passed_verification["outcome"] == "passed"
    assert passed_verification["collected"] == 1
    assert passed_verification["passed"] == 1


def test_run_tests_reports_invalid_target_without_calling_it_a_test_failure(tmp_path):
    missing = tmp_path / "missing.py"

    result = run_tests_result({"targets": [str(missing)]})

    verification = result.metadata["verification"]
    assert result.status == "failed"
    assert result.error_code == "invalid_test_target"
    assert verification["execution_status"] == "invalid"
    assert verification["outcome"] == "unknown"


def test_pytest_usage_and_empty_collection_are_non_assertion_outcomes():
    empty = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    usage = _classify_pytest_result(4, counts=empty, scope=["tests"])
    no_tests = _classify_pytest_result(5, counts=empty, scope=["tests"])

    assert usage["execution_status"] == "invalid"
    assert usage["failure_kind"] == "usage_error"
    assert no_tests["outcome"] == "no_tests"
    assert no_tests["failure_kind"] == "no_tests_collected"


def test_run_shell_records_verification_only_when_declared():
    ordinary = run_shell_result({"command": f'"{sys.executable}" -c "print(1)"'})
    verified = run_shell_result(
        {
            "command": f'"{sys.executable}" -c "print(1)"',
            "purpose": "verification",
        }
    )

    assert "verification" not in ordinary.metadata
    assert verified.metadata["verification"]["outcome"] == "passed"
    assert verified.metadata["verification"]["scope_level"] == "related"
    assert verified.check_scope


def test_shell_direct_scope_requires_a_narrow_non_xfail_command():
    assert (
        _validated_shell_scope(
            "pytest tests/test_service.py::test_update", "direct"
        )
        == "direct"
    )
    assert _validated_shell_scope("pytest tests/test_service.py", "direct") == "related"
    assert _validated_shell_scope("pytest --runxfail test_xf.py", "direct") == "related"


def test_declared_shell_verification_requires_real_process_result():
    failed = _classify_declared_shell_verification(
        1,
        "FAILED tests/test_service.py::test_update - AssertionError",
        "pytest tests/test_service.py::test_update",
    )
    unavailable = _classify_declared_shell_verification(
        1,
        "command not found",
        "missing-checker",
    )

    assert failed["outcome"] == "failed"
    assert failed["failure_kind"] == "assertion_failure"
    assert unavailable["execution_status"] == "unavailable"
    assert unavailable["outcome"] == "unknown"


def test_file_tools_create_read_edit_and_search(tmp_path):
    target = tmp_path / "sample.py"

    create_file(
        {
            "path": str(target),
            "content": "def greet():\n    return 'hello'\n",
        }
    )

    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hello'\n"
    assert read_file({"path": str(target), "line_start": 1, "line_end": 1}) == "def greet():"

    edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 2,
            "line_end": 2,
            "content": "    return 'hi'",
        }
    )
    assert "return 'hi'" in target.read_text(encoding="utf-8")

    edit_file(
        {
            "path": str(target),
            "operation": "insert",
            "line_start": 1,
            "content": "# generated by test",
        }
    )

    search_result = search_in_file({"path": str(target), "pattern": "return", "context_lines": 1})
    assert "return 'hi'" in search_result
    assert ">>>" in search_result


def test_repository_search_tools_find_paths_and_content(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "users.py").write_text("def load_user():\n    return 'Ada'\n", encoding="utf-8")
    (package / "binary.bin").write_bytes(b"\x00load_user")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.py").write_text("load_user = 1\n", encoding="utf-8")

    files = json.loads(find_files({"root": str(tmp_path), "pattern": "**/users.py"}))
    assert files["matches"] == ["pkg/users.py"]

    matches = json.loads(search_code({"root": str(tmp_path), "query": "LOAD_USER", "glob": "*.py"}))
    assert matches["match_count"] == 1
    assert matches["matches"][0]["path"] == "pkg/users.py"
    assert matches["matches"][0]["line"] == 1


# --- edit_file 的两处防自伤改动 -------------------------------------------
#
# 背景见 docs/research-log/38-edit-file-precision.md：行号编辑会在区间给宽一行时
# 静默吞掉相邻代码，而旧返回值只有一句「已替换第 N-M 行」，模型平均 2.2 步之后
# 才发现自己改坏了。两处干预：按内容定位（事前防止）与编辑后回显（事后立刻发现）。


def test_old_string_replaces_the_single_exact_match(tmp_path):
    target = tmp_path / "semver.py"
    target.write_text("core = parts[0]\nprerelease = None\n", encoding="utf-8")

    observation = edit_file(
        {
            "path": str(target),
            "old_string": "prerelease = None",
            "new_string": "prerelease = parts[1]",
        }
    )

    assert target.read_text(encoding="utf-8") == "core = parts[0]\nprerelease = parts[1]\n"
    assert "prerelease = parts[1]" in observation


def test_old_string_leaves_the_file_untouched_when_it_does_not_match(tmp_path):
    target = tmp_path / "mod.py"
    original = "value = 1\n"
    target.write_text(original, encoding="utf-8")

    observation = edit_file({"path": str(target), "old_string": "value = 2", "new_string": "x"})

    assert target.read_text(encoding="utf-8") == original
    assert "未命中" in observation
    # 换一段 old_string 重试是局部纠正，不该触发一次完整重规划。
    assert not is_failure_observation(observation)


def test_old_string_refuses_ambiguous_matches_without_writing(tmp_path):
    target = tmp_path / "mod.py"
    original = "a = 1\nb = 2\na = 1\n"
    target.write_text(original, encoding="utf-8")

    observation = edit_file({"path": str(target), "old_string": "a = 1", "new_string": "a = 9"})

    assert target.read_text(encoding="utf-8") == original
    assert "2 处" in observation
    assert not is_failure_observation(observation)


def test_old_string_identity_is_reported_without_writing(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    original = "value = 1\n"
    target.write_text(original, encoding="utf-8")
    writes = []

    monkeypatch.setattr(
        file_tools_module,
        "_atomic_write_text",
        lambda path, content: writes.append((path, content)) or "",
    )

    observation = edit_file({"path": str(target), "old_string": original, "new_string": original})

    assert writes == []
    assert target.read_text(encoding="utf-8") == original
    assert observation.startswith("未改动：")
    assert not is_failure_observation(observation, action="edit_file")


def test_old_string_and_line_numbers_are_mutually_exclusive(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("a = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line_start"):
        edit_file(
            {
                "path": str(target),
                "old_string": "a = 1",
                "new_string": "a = 2",
                "line_start": 1,
            }
        )
    with pytest.raises(ValueError, match="old_string"):
        edit_file({"path": str(target), "new_string": "a = 2"})


def test_edit_echoes_the_resulting_lines_with_new_numbers(tmp_path):
    """这是核心：模型必须当场看见吞掉了什么。复刻 semver_compare 的真实自伤。"""
    target = tmp_path / "semver.py"
    target.write_text(
        "def parse(version):\n"
        "    parts = version.split('-')\n"
        "    core = parts[0]\n"
        "    prerelease = None\n"
        "    return core, prerelease\n",
        encoding="utf-8",
    )

    # 本想只改第 4 行，区间给成 3-4 —— core = parts[0] 被静默吞掉。
    observation = edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 3,
            "line_end": 4,
            "content": "    prerelease = parts[1]",
        }
    )

    assert "core = parts[0]" not in target.read_text(encoding="utf-8")
    # 回显里带新行号，并把改动行标出来，模型据此就能看出 core 那行没了。
    assert "[编辑后]" in observation
    assert ">    3 |     prerelease = parts[1]" in observation
    assert "    4 |     return core, prerelease" in observation


def test_edit_echo_is_bounded(tmp_path):
    target = tmp_path / "big.py"
    target.write_text("".join(f"x{index} = {index}\n" for index in range(200)), encoding="utf-8")

    observation = edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 1,
            "line_end": 100,
            "content": "".join(f"y{index} = {index}\n" for index in range(100)),
        }
    )

    assert "回显截断" in observation
    body = observation.split("[编辑后]", 1)[1]
    assert body.count(" | ") <= EDIT_ECHO_MAX_LINES


def test_python_syntax_is_reported_but_the_write_still_happened(tmp_path):
    target = tmp_path / "broken.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    observation = edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 2,
            "line_end": 2,
            "content": "    return (1",
        }
    )

    # 只报告，不回滚：分步编辑的中间态可能合法地无法解析。
    assert target.read_text(encoding="utf-8") == "def f():\n    return (1\n"
    assert "[语法检查] 未通过" in observation


def test_syntax_check_only_applies_to_python_files(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    observation = edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 1,
            "line_end": 1,
            "content": "(",
        }
    )

    assert "[语法检查]" not in observation


def test_create_file_reports_syntax_without_blocking_the_write(tmp_path):
    target = tmp_path / "new.py"

    observation = create_file({"path": str(target), "content": "def f(:\n"})

    assert target.read_text(encoding="utf-8") == "def f(:\n"
    assert "[语法检查] 未通过" in observation


def test_code_analysis_tools_return_structured_json(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        "import os\n\n"
        "class Greeter:\n"
        "    def greet(self, name: str) -> str:\n"
        "        return f'hello {name}'\n\n"
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n",
        encoding="utf-8",
    )

    ast_data = json.loads(parse_ast({"path": str(module)}))
    assert ast_data["imports"][0]["module"] == "os"
    assert ast_data["classes"][0]["name"] == "Greeter"
    assert any(func["name"] == "add" for func in ast_data["functions"])

    signature = json.loads(get_function_signature({"path": str(module), "function_name": "add"}))
    assert signature["signature"] == "def add(left: int, right: int) -> int"

    metrics = json.loads(get_code_metrics({"path": str(module)}))
    assert metrics["num_functions"] == 2
    assert metrics["num_classes"] == 1


def test_inspect_python_symbol_finds_function_class_and_method(tmp_path):
    module = tmp_path / "service.py"
    module.write_text(
        "def helper(value: int) -> int:\n"
        "    return value + 1\n\n"
        "class UserService:\n"
        "    async def login(self, username: str) -> bool:\n"
        "        return bool(username)\n",
        encoding="utf-8",
    )

    function = json.loads(inspect_python_symbol({"path": str(module), "qualified_name": "helper"}))
    class_info = json.loads(
        inspect_python_symbol({"path": str(module), "qualified_name": "UserService"})
    )
    method = json.loads(
        inspect_python_symbol({"path": str(module), "qualified_name": "UserService.login"})
    )

    assert function["symbol_type"] == "function"
    assert function["signature"] == "def helper(value: int)"
    assert class_info["symbol_type"] == "class"
    assert method["symbol_type"] == "method"
    assert method["signature"] == "async def login(self, username: str)"
    assert len(method["source_hash"]) == 64


def test_edit_python_symbol_replaces_body_with_matching_hash(tmp_path):
    module = tmp_path / "service.py"
    module.write_text(
        "class UserService:\n"
        "    def login(self, username: str) -> bool:\n"
        "        return bool(username)\n\n"
        "def untouched():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    inspected = json.loads(
        inspect_python_symbol({"path": str(module), "qualified_name": "UserService.login"})
    )

    result = json.loads(
        edit_python_symbol(
            {
                "path": str(module),
                "qualified_name": "UserService.login",
                "expected_hash": inspected["source_hash"],
                "operation": "replace_body",
                "content": "if not username:\n    return False\nreturn True",
            }
        )
    )

    updated = module.read_text(encoding="utf-8")
    assert "        if not username:" in updated
    assert "        return True" in updated
    assert "def untouched():\n    return 1" in updated
    assert result["source_hash"] != inspected["source_hash"]
    ast.parse(updated)


def test_edit_python_symbol_rejects_stale_hash_without_writing(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("def value():\n    return 1\n", encoding="utf-8")
    inspected = json.loads(inspect_python_symbol({"path": str(module), "qualified_name": "value"}))
    module.write_text("def value():\n    return 2\n", encoding="utf-8")
    current = module.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="重新调用 inspect_python_symbol"):
        edit_python_symbol(
            {
                "path": str(module),
                "qualified_name": "value",
                "expected_hash": inspected["source_hash"],
                "operation": "replace_body",
                "content": "return 3",
            }
        )

    assert module.read_text(encoding="utf-8") == current


def test_edit_python_symbol_rejects_invalid_python_without_writing(tmp_path):
    module = tmp_path / "module.py"
    original = "def value():\n    return 1\n"
    module.write_text(original, encoding="utf-8")
    inspected = json.loads(inspect_python_symbol({"path": str(module), "qualified_name": "value"}))

    with pytest.raises(ValueError, match="文件未写入"):
        edit_python_symbol(
            {
                "path": str(module),
                "qualified_name": "value",
                "expected_hash": inspected["source_hash"],
                "operation": "replace_body",
                "content": "return (",
            }
        )

    assert module.read_text(encoding="utf-8") == original


def test_code_index_tools_find_symbols_and_dependencies(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "models.py").write_text(
        "class User:\n" "    def display_name(self) -> str:\n" "        return 'Ada'\n",
        encoding="utf-8",
    )
    (package / "service.py").write_text(
        "from . import models\n"
        "from .models import User\n\n\n"
        "def load_user() -> User:\n"
        "    return User()\n",
        encoding="utf-8",
    )

    matches = json.loads(search_symbol({"root": str(tmp_path), "name": "load_user", "exact": True}))
    assert matches["match_count"] == 1
    assert matches["matches"][0]["path"] == "pkg/service.py"
    assert not (tmp_path / ".dm_agent").exists()

    graph = json.loads(dependency_graph({"root": str(tmp_path)}))
    assert {"from": "pkg.service", "to": "pkg.models", "import": "pkg.models"} in graph["edges"]


def test_inspect_change_impact_returns_confidence_tiers_and_provenance(tmp_path):
    (tmp_path / "service.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_service.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")

    result = json.loads(
        inspect_change_impact({"root": str(tmp_path), "paths": ["service.py"]})
    )

    assert any(
        item["path"] == "consumer.py" and item["confidence_level"] == "confirmed"
        for item in result["affected_symbols"]
    )
    assert result["test_candidates"] == [
        {"path": "tests/test_service.py", "confidence_level": "fallback"}
    ]
    assert "candidates" in result["notice"].casefold()


def test_run_python_executes_inline_code():
    result = run_python({"code": "print('agent-ready')"})
    assert "agent-ready" in result
    assert "returncode: 0" in result

    structured = run_python_result({"code": "print('agent-ready')"})
    assert structured.metadata["verification"]["execution_status"] == "completed"
    assert structured.metadata["verification"]["outcome"] == "passed"


def test_task_complete_accepts_message():
    result = task_complete({"message": "ready"})
    assert "ready" in result


def test_task_complete_accepts_common_final_answer_keys():
    result = task_complete({"answer": "ready from answer"})
    assert "ready from answer" in result


def test_run_python_accepts_script_args(tmp_path):
    script = tmp_path / "echo_args.py"
    script.write_text(
        "import sys\nprint('|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = run_python({"path": str(script), "args": ["left", "right"]})
    assert "left|right" in result
    assert "returncode: 0" in result


def test_atomic_write_replaces_content_without_tmp_residue(tmp_path):
    target = tmp_path / "atomic.txt"
    target.write_text("before", encoding="utf-8")

    note = _atomic_write_text(target, "after")

    assert note == ""
    assert target.read_text(encoding="utf-8") == "after"
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_create_and_edit_file_use_atomic_write(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    create_file({"path": str(target), "content": "line1\nline2\n"})
    assert target.read_text(encoding="utf-8") == "line1\nline2\n"

    edit_file(
        {
            "path": str(target),
            "operation": "replace",
            "line_start": 1,
            "line_end": 1,
            "content": "LINE1",
        }
    )
    assert target.read_text(encoding="utf-8") == "LINE1\nline2\n"
    leftovers = [p.name for p in target.parent.iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_missing_linter_reports_the_ones_this_environment_has(tmp_path, monkeypatch):
    """缺检查器时给出可用清单，而不是把 "No module named" 原样丢回给模型。

    实测一轮 30 题里 run_linter 被调用 39 次、其中 29 次撞在没装的 flake8/pylint 上，
    横跨 15 道题——模型拿到裸的 ImportError 只会挨个盲试下一个。
    """
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "dm_agent.tools.execution_tools.available_linters", lambda: ["ruff", "mypy"]
    )
    observation = run_linter({"path": str(target), "tool": "flake8"})

    assert "ruff" in observation and "mypy" in observation
    # 换一个检查器重试是局部纠正，不该触发一次完整重规划。
    assert not is_failure_observation(observation, action="run_linter")


def test_missing_linter_without_any_alternative_says_the_step_can_be_skipped(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr("dm_agent.tools.execution_tools.available_linters", lambda: [])
    observation = run_linter({"path": str(target), "tool": "pylint"})

    assert "可跳过" in observation
    assert not is_failure_observation(observation, action="run_linter")


def test_available_linters_reports_a_subset_of_the_supported_ones():
    supported = {"ruff", "flake8", "pylint", "mypy", "black"}
    assert set(available_linters()) <= supported


def test_editing_an_lf_file_keeps_lf_line_endings(tmp_path):
    """在 Windows 上编辑 LF 文件不得把整份文件转成 CRLF。

    根因是 Path.write_text 默认的 newline=None 会按平台改写行尾。代价实测过：
    在一个真实仓库里改一行，产出的 diff 是 +317/-317 的整文件重写——这样的 patch
    在 SWE-bench 官方 harness 上 git apply 会失败。
    """
    target = tmp_path / "mod.py"
    target.write_bytes(b"a = 1\nb = 2\nc = 3\n")

    edit_file({"path": str(target), "old_string": "b = 2", "new_string": "b = 20"})

    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert raw == b"a = 1\nb = 20\nc = 3\n"


def test_editing_a_crlf_file_keeps_crlf_line_endings(tmp_path):
    """反向同理：原本是 CRLF 的文件，编辑后仍是 CRLF。"""
    target = tmp_path / "mod.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")

    edit_file({"path": str(target), "old_string": "b = 2", "new_string": "b = 20"})

    assert target.read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"


def test_new_files_are_written_with_lf(tmp_path):
    target = tmp_path / "fresh.py"
    create_file({"path": str(target), "content": "x = 1\ny = 2\n"})
    assert target.read_bytes() == b"x = 1\ny = 2\n"
