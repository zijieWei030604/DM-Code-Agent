"""工具模块 - 提供智能体可用的各类工具"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from .base import Tool, ToolResult
from .code_analysis_tools import (
    find_dependencies,
    find_dependencies_result,
    get_code_metrics,
    get_code_metrics_result,
    get_function_signature,
    get_function_signature_result,
    parse_ast,
    parse_ast_result,
)
from .code_index_tools import (
    dependency_graph,
    dependency_graph_result,
    search_symbol,
    search_symbol_result,
)
from .execution_tools import (
    run_linter,
    run_linter_result,
    run_python,
    run_python_result,
    run_shell,
    run_shell_result,
    run_tests,
    run_tests_result,
)
from .file_tools import (
    create_file,
    create_file_result,
    edit_file,
    edit_file_result,
    find_files,
    find_files_result,
    list_directory,
    list_directory_result,
    read_file,
    read_file_result,
    search_code,
    search_code_result,
    search_in_file,
    search_in_file_result,
)
from .structured_edit_tools import (
    edit_python_symbol,
    edit_python_symbol_result,
    inspect_python_symbol,
    inspect_python_symbol_result,
)

if TYPE_CHECKING:
    from dm_agent.extensions import ExtensionAPI, ExtensionRegistry
    from dm_agent.workspace import SemanticWorkspaceEngine


def _object_schema(
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}

BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_directory": _object_schema({"path": _STR, "recursive": _BOOL, "file_type": _STR}),
    "read_file": _object_schema({"path": _STR, "line_start": _INT, "line_end": _INT}, ("path",)),
    "create_file": _object_schema({"path": _STR, "content": _STR}, ("path", "content")),
    "edit_file": _object_schema(
        {
            "path": _STR,
            "old_string": _STR,
            "new_string": _STR,
            "operation": {"type": "string", "enum": ["insert", "replace", "delete"]},
            "line_start": _INT,
            "line_end": _INT,
            "content": _STR,
        },
        ("path",),
    ),
    "search_in_file": _object_schema(
        {"path": _STR, "pattern": _STR, "context_lines": _INT}, ("path", "pattern")
    ),
    "find_files": _object_schema(
        {"pattern": _STR, "root": _STR, "max_results": _INT}, ("pattern",)
    ),
    "search_code": _object_schema(
        {
            "query": _STR,
            "root": _STR,
            "glob": _STR,
            "regex": _BOOL,
            "case_sensitive": _BOOL,
            "max_results": _INT,
        },
        ("query",),
    ),
    "run_python": _object_schema(
        {
            "code": _STR,
            "path": _STR,
            "args": {"anyOf": [_STR, {"type": "array", "items": _STR}]},
        }
    ),
    "run_shell": _object_schema({"command": _STR}, ("command",)),
    "run_tests": _object_schema(
        {
            "test_path": _STR,
            "framework": {"type": "string", "enum": ["pytest", "unittest"]},
            "verbose": _BOOL,
        }
    ),
    "run_linter": _object_schema(
        {
            "path": _STR,
            "tool": {
                "type": "string",
                "enum": ["ruff", "flake8", "pylint", "mypy", "black"],
            },
        },
        ("path",),
    ),
    "parse_ast": _object_schema({"path": _STR}, ("path",)),
    "get_function_signature": _object_schema(
        {"path": _STR, "function_name": _STR}, ("path", "function_name")
    ),
    "find_dependencies": _object_schema({"path": _STR}, ("path",)),
    "get_code_metrics": _object_schema({"path": _STR}, ("path",)),
    "search_symbol": _object_schema(
        {
            "name": _STR,
            "root": _STR,
            "kind": {"type": "string", "enum": ["class", "function", "method"]},
            "exact": _BOOL,
            "max_files": _INT,
            "max_results": _INT,
        },
        ("name",),
    ),
    "dependency_graph": _object_schema(
        {"root": _STR, "max_files": _INT, "include_external": _BOOL}
    ),
    "inspect_python_symbol": _object_schema(
        {"path": _STR, "qualified_name": _STR}, ("path", "qualified_name")
    ),
    "edit_python_symbol": _object_schema(
        {
            "path": _STR,
            "qualified_name": _STR,
            "expected_hash": _STR,
            "operation": {
                "type": "string",
                "enum": ["replace_body", "replace_symbol"],
            },
            "content": _STR,
        },
        ("path", "qualified_name", "expected_hash", "operation", "content"),
    ),
    "task_complete": _object_schema({"message": _STR}),
}


def task_complete(arguments: dict[str, Any]) -> str:
    """
    标记任务完成的工具。调用此工具将自动结束任务。

    当智能体认为任务已经完成时，应调用此工具来终止任务执行流程。
    该工具通常作为任务计划的最后一个步骤被调用。

    Args:
        arguments (Dict[str, Any]): 工具调用参数字典
            - message (str, optional): 任务完成的描述信息，默认为空字符串

    Returns:
        str: 格式化的任务完成消息
            - 如果提供了有效的message字符串，则返回 "任务完成：{message}"
            - 否则返回默认消息 "任务已完成。"

    Examples:
        >>> task_complete({"message": "数据分析已完成"})
        '任务完成：数据分析已完成'

        >>> task_complete({})
        '任务已完成。'

        >>> task_complete({"message": "  "})
        '任务已完成。'
    """
    message = ""
    for key in ("message", "answer", "final_answer", "summary", "result"):
        value = arguments.get(key, "")
        if isinstance(value, str) and value.strip():
            message = value.strip()
            break
    if message:
        return f"任务完成：{message}"
    return "任务已完成。"


def task_complete_result(arguments: dict[str, Any]) -> ToolResult:
    return ToolResult("success", task_complete(arguments))


def _builtin_tools() -> list[Tool]:
    """构造顺序稳定的内置工具实例。"""
    tools = [
        Tool(
            name="list_directory",
            description=(
                "List entries for the given directory path. Arguments: {\"path\": optional string (default '.'), "
                "\"recursive\": optional bool (default false), \"file_type\": optional string filter like '.py' or '.js'}."
            ),
            runner=list_directory,
            result_runner=list_directory_result,
        ),
        Tool(
            name="read_file",
            description=(
                'Read a UTF-8 text file. Arguments: {"path": string, '
                '"line_start": optional int, "line_end": optional int}.'
            ),
            runner=read_file,
            result_runner=read_file_result,
        ),
        Tool(
            name="create_file",
            description='Create or overwrite a text file. Arguments: {"path": string, "content": string}.',
            runner=create_file,
            result_runner=create_file_result,
        ),
        Tool(
            name="edit_file",
            description=(
                "Edit a file. PREFERRED: content-anchored replace with "
                '{"path": string, "old_string": string, "new_string": string} — '
                "old_string must match exactly once (whitespace and indentation "
                "included), and new_string must contain an actual change; if the strings "
                "are identical or old_string matches zero or several times, nothing is written. "
                "Fallback (line-numbered, use only when content anchoring will not do): "
                '{"path": string, "operation": "insert"|"replace"|"delete", '
                '"line_start": int, "line_end": int (for replace/delete), '
                '"content": string (for insert/replace)}. Line numbers shift after every '
                "edit, so re-read the file before using them again. Every edit echoes the "
                "resulting lines back to you — check them before moving on."
            ),
            runner=edit_file,
            result_runner=edit_file_result,
        ),
        Tool(
            name="search_in_file",
            description=(
                'Search for text or regex pattern in a file. Arguments: {"path": string, "pattern": string, '
                '"context_lines": optional int (default 2)}.'
            ),
            runner=search_in_file,
            result_runner=search_in_file_result,
        ),
        Tool(
            name="find_files",
            description=(
                'Recursively find files by path or glob. Arguments: {"pattern": string, '
                '"root": optional string (default "."), "max_results": optional int (default 50)}.'
            ),
            runner=find_files,
            result_runner=find_files_result,
        ),
        Tool(
            name="search_code",
            description=(
                'Search text across repository files. Arguments: {"query": string, '
                '"root": optional string, "glob": optional string, "regex": optional bool, '
                '"case_sensitive": optional bool, "max_results": optional int}. '
                "Search results are candidates only; read a file before editing it."
            ),
            runner=search_code,
            result_runner=search_code_result,
        ),
        Tool(
            name="run_python",
            description=(
                'Execute Python code using the local interpreter. Arguments: either {"code": string} or {"path": string, "args": optional string or list}.'
            ),
            runner=run_python,
            result_runner=run_python_result,
        ),
        Tool(
            name="run_shell",
            description='Execute a shell command. Arguments: {"command": string}.',
            runner=run_shell,
            result_runner=run_shell_result,
        ),
        Tool(
            name="run_tests",
            description=(
                "Run Python test suite. Arguments: {\"test_path\": optional string (default '.'), "
                '"framework": optional "pytest"|"unittest" (default \'pytest\'), "verbose": optional bool (default false)}.'
            ),
            runner=run_tests,
            result_runner=run_tests_result,
        ),
        Tool(
            name="run_linter",
            description=(
                'Run code linter/formatter. Arguments: {"path": string, '
                '"tool": optional "ruff"|"flake8"|"pylint"|"mypy"|"black" (default \'ruff\')}. '
                "If the chosen linter is not installed here, the tool replies with the list "
                "of linters this environment does provide -- switch to one of those."
            ),
            runner=run_linter,
            result_runner=run_linter_result,
        ),
        Tool(
            name="parse_ast",
            description=(
                'Parse Python file AST to extract structure (functions, classes, imports). Arguments: {"path": string}.'
            ),
            runner=parse_ast,
            result_runner=parse_ast_result,
        ),
        Tool(
            name="get_function_signature",
            description=(
                'Get function signature with type hints. Arguments: {"path": string, "function_name": string}.'
            ),
            runner=get_function_signature,
            result_runner=get_function_signature_result,
        ),
        Tool(
            name="find_dependencies",
            description=('Analyze file dependencies (imports). Arguments: {"path": string}.'),
            runner=find_dependencies,
            result_runner=find_dependencies_result,
        ),
        Tool(
            name="get_code_metrics",
            description=(
                'Get code metrics (lines, functions, classes count). Arguments: {"path": string}.'
            ),
            runner=get_code_metrics,
            result_runner=get_code_metrics_result,
        ),
        Tool(
            name="search_symbol",
            description=(
                'Search classes, functions, and methods by name. Arguments: {"name": string, '
                '"root": optional string, "kind": optional "class"|"function"|"method", '
                '"exact": optional bool, "max_files": optional int}.'
            ),
            runner=search_symbol,
            result_runner=search_symbol_result,
        ),
        Tool(
            name="dependency_graph",
            description=(
                'Build a local Python import dependency graph. Arguments: {"root": optional string, '
                '"max_files": optional int, "include_external": optional bool}.'
            ),
            runner=dependency_graph,
            result_runner=dependency_graph_result,
        ),
        Tool(
            name="inspect_python_symbol",
            description=(
                "Inspect one top-level Python function/class or direct class method without knowing line numbers. "
                'Arguments: {"path": string, "qualified_name": string such as "UserService.login"}. '
                "Returns source, location, signature, and source_hash for a guarded follow-up edit."
            ),
            runner=inspect_python_symbol,
            result_runner=inspect_python_symbol_result,
        ),
        Tool(
            name="edit_python_symbol",
            description=(
                "Safely edit a previously inspected Python function, class, or direct method. "
                'Arguments: {"path": string, "qualified_name": string, "expected_hash": string from '
                'inspect_python_symbol, "operation": "replace_body"|"replace_symbol", "content": string}. '
                "The edit is rejected if the symbol changed, and invalid Python is never written."
            ),
            runner=edit_python_symbol,
            result_runner=edit_python_symbol_result,
        ),
        Tool(
            name="task_complete",
            description='Mark the task as complete and finish execution. Arguments: {"message": optional string with completion summary}.',
            runner=task_complete,
            result_runner=task_complete_result,
        ),
    ]
    for tool in tools:
        tool.input_schema = BUILTIN_TOOL_SCHEMAS[tool.name]
    return tools


def bind_semantic_workspace_tools(tools: list[Tool], engine: SemanticWorkspaceEngine) -> None:
    """Bind built-in semantic queries to the run's shared workspace engine."""
    for index, tool in enumerate(tools):
        if tool.name == "search_symbol" and tool.runner is search_symbol:
            tools[index] = Tool(
                name=tool.name,
                description=tool.description,
                runner=partial(search_symbol, engine=engine),
                result_runner=partial(search_symbol_result, engine=engine),
                read_only=tool.read_only,
                input_schema=tool.input_schema,
            )
            return


def register_builtin_tools(api: ExtensionAPI) -> None:
    """通过 ExtensionAPI 注册全部内置工具。"""
    for tool in _builtin_tools():
        api.register_tool(tool)


def default_tools(
    include_mcp: bool = True,
    mcp_tools: list[Tool] | None = None,
    *,
    extension_registry: ExtensionRegistry | None = None,
) -> list[Tool]:
    """返回注册表中的工具，并保持 MCP 工具最后追加的既有行为。"""
    if extension_registry is None:
        from dm_agent.extensions.discovery import create_builtin_registry

        extension_registry = create_builtin_registry()
    tools = extension_registry.get_tools()

    if include_mcp and mcp_tools:
        tools.extend(mcp_tools)

    return tools


__all__ = [
    "Tool",
    "bind_semantic_workspace_tools",
    "default_tools",
    "register_builtin_tools",
    "task_complete",
]
