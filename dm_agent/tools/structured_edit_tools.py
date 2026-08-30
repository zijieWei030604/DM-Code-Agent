"""Python symbol inspection and hash-guarded structural editing tools."""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import _require_str
from .file_tools import _atomic_write_text

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SUPPORTED_OPERATIONS = frozenset({"replace_symbol", "replace_body"})
SymbolNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _LocatedSymbol:
    node: SymbolNode
    qualified_name: str
    symbol_type: str
    start_line: int
    end_line: int


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _node_start_line(node: SymbolNode) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _locate_symbol(tree: ast.Module, qualified_name: str) -> _LocatedSymbol:
    parts = qualified_name.split(".")
    if not all(parts) or len(parts) > 2:
        raise ValueError("qualified_name 只支持顶层函数/类名称，或 ClassName.method_name 形式。")

    top_level = [node for node in tree.body if isinstance(node, (*_FUNCTION_NODES, ast.ClassDef))]
    parent = next((node for node in top_level if node.name == parts[0]), None)
    if parent is None:
        raise ValueError(f"未找到 Python 符号：{qualified_name}")

    node: SymbolNode
    if len(parts) == 1:
        node = parent
        symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
    else:
        if not isinstance(parent, ast.ClassDef):
            raise ValueError(f"{parts[0]} 不是类，无法查找方法 {qualified_name}。")
        method = next(
            (
                item
                for item in parent.body
                if isinstance(item, _FUNCTION_NODES) and item.name == parts[1]
            ),
            None,
        )
        if method is None:
            raise ValueError(f"未找到 Python 方法：{qualified_name}")
        node = method
        symbol_type = "method"

    end_line = getattr(node, "end_lineno", None)
    if end_line is None:
        raise ValueError(f"无法确定符号 {qualified_name} 的结束位置。")
    return _LocatedSymbol(
        node=node,
        qualified_name=qualified_name,
        symbol_type=symbol_type,
        start_line=_node_start_line(node),
        end_line=end_line,
    )


def _read_python(path_value: str) -> tuple[Path, str, ast.Module]:
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"文件 {path} 不存在。")
    if not path.is_file():
        raise ValueError(f"路径 {path} 不是文件。")
    if path.suffix.lower() != ".py":
        raise ValueError("结构化符号工具只支持 .py 文件。")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"无法解析 {path}：第 {exc.lineno} 行 {exc.msg}。") from exc
    return path, source, tree


def _symbol_source(source: str, symbol: _LocatedSymbol) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[symbol.start_line - 1 : symbol.end_line])


def _signature(node: SymbolNode) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({ast.unparse(node.args)})"


def inspect_python_symbol(arguments: dict[str, Any]) -> str:
    """Locate one Python function, class, or direct method and return its source fingerprint."""
    path_value = _require_str(arguments, "path")
    qualified_name = _require_str(arguments, "qualified_name")
    path, source, tree = _read_python(path_value)
    symbol = _locate_symbol(tree, qualified_name)
    selected_source = _symbol_source(source, symbol)
    payload = {
        "path": str(path),
        "qualified_name": symbol.qualified_name,
        "symbol_type": symbol.symbol_type,
        "line_start": symbol.start_line,
        "line_end": symbol.end_line,
        "signature": _signature(symbol.node),
        "source_hash": _source_hash(selected_source),
        "source": selected_source,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _prepare_symbol_replacement(content: str, symbol: _LocatedSymbol) -> str:
    replacement = textwrap.dedent(content).strip("\n")
    if not replacement:
        raise ValueError("replace_symbol 的 content 不能为空。")
    try:
        parsed = ast.parse(replacement)
    except SyntaxError as exc:
        raise ValueError(f"替换符号不是合法 Python 代码：第 {exc.lineno} 行 {exc.msg}。") from exc
    definitions = [
        node for node in parsed.body if isinstance(node, (*_FUNCTION_NODES, ast.ClassDef))
    ]
    if len(parsed.body) != 1 or len(definitions) != 1:
        raise ValueError("replace_symbol 的 content 必须恰好包含一个函数或类定义。")
    replacement_node = definitions[0]
    if replacement_node.name != symbol.node.name:
        raise ValueError(
            f"替换符号名称必须保持为 {symbol.node.name}，实际得到 {replacement_node.name}。"
        )
    if isinstance(symbol.node, ast.ClassDef) != isinstance(replacement_node, ast.ClassDef):
        raise ValueError("替换内容的符号类型与原符号不一致。")
    indentation = " " * symbol.node.col_offset
    return textwrap.indent(replacement, indentation) + "\n"


def _prepare_body_replacement(content: str, symbol: _LocatedSymbol) -> tuple[int, int, str]:
    node = symbol.node
    if not node.body:
        raise ValueError(f"符号 {symbol.qualified_name} 没有可替换的代码体。")
    body_start = node.body[0].lineno
    if body_start <= node.lineno:
        raise ValueError("单行函数/类不支持 replace_body，请改用 replace_symbol。")
    body = textwrap.dedent(content).strip("\n") or "pass"
    indentation = " " * (node.col_offset + 4)
    replacement = textwrap.indent(body, indentation) + "\n"
    return body_start, symbol.end_line, replacement


def edit_python_symbol(arguments: dict[str, Any]) -> str:
    """Edit one Python symbol only when its previously inspected source hash still matches."""
    path_value = _require_str(arguments, "path")
    qualified_name = _require_str(arguments, "qualified_name")
    expected_hash = _require_str(arguments, "expected_hash")
    operation = _require_str(arguments, "operation")
    content = arguments.get("content", "")
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串。")
    if operation not in _SUPPORTED_OPERATIONS:
        raise ValueError("operation 必须是 replace_symbol 或 replace_body。")

    path, source, tree = _read_python(path_value)
    symbol = _locate_symbol(tree, qualified_name)
    current_source = _symbol_source(source, symbol)
    current_hash = _source_hash(current_source)
    if current_hash != expected_hash:
        raise ValueError(
            f"符号 {qualified_name} 自检查后已发生变化；"
            "expected_hash 与当前源码不一致，请重新调用 inspect_python_symbol。"
        )

    if operation == "replace_symbol":
        start_line = symbol.start_line
        end_line = symbol.end_line
        replacement = _prepare_symbol_replacement(content, symbol)
    else:
        start_line, end_line, replacement = _prepare_body_replacement(content, symbol)

    lines = source.splitlines(keepends=True)
    updated = "".join([*lines[: start_line - 1], replacement, *lines[end_line:]])
    if updated == source:
        raise ValueError("结构化编辑没有产生实际变化。")
    try:
        updated_tree = ast.parse(updated)
    except SyntaxError as exc:
        raise ValueError(
            f"结构化编辑会产生无效 Python 代码：第 {exc.lineno} 行 {exc.msg}；文件未写入。"
        ) from exc

    updated_symbol = _locate_symbol(updated_tree, qualified_name)
    updated_source = _symbol_source(updated, updated_symbol)
    note = _atomic_write_text(path, updated)
    payload = {
        "path": str(path),
        "qualified_name": qualified_name,
        "operation": operation,
        "line_start": updated_symbol.start_line,
        "line_end": updated_symbol.end_line,
        "previous_hash": current_hash,
        "source_hash": _source_hash(updated_source),
        "source": updated_source,
        "write_mode": "non-atomic fallback" if note else "atomic",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["edit_python_symbol", "inspect_python_symbol"]
