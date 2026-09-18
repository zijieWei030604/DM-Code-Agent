"""Lightweight code indexing tools for repository-level Python understanding."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dm_agent.workspace import SemanticWorkspaceEngine

from .base import ToolResult, _require_str

DEFAULT_INDEX_EXCLUDES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}


def search_symbol(
    arguments: dict[str, Any],
    *,
    engine: SemanticWorkspaceEngine | None = None,
) -> str:
    """Search the persistent semantic workspace for Python symbols."""
    name = _require_str(arguments, "name")
    root = Path(arguments.get("root", engine.root if engine else ".")).resolve()
    kind = arguments.get("kind")
    exact = bool(arguments.get("exact", False))
    max_files = int(arguments.get("max_files", 200))
    max_results = int(arguments.get("max_results", 50))

    if kind is not None and kind not in {"class", "function", "method"}:
        raise ValueError("kind must be one of: class, function, method")
    if max_results < 1:
        raise ValueError("max_results must be a positive integer")
    if engine is not None and engine.root != root:
        raise ValueError(f"search root {root} does not match semantic workspace {engine.root}")

    owned_engine = engine is None
    current_engine = engine or SemanticWorkspaceEngine(
        root,
        database_path=_standalone_index_path(root),
        max_scan_files=max_files,
    )
    try:
        current_engine.update()
        candidates = current_engine.search_symbols(name, limit=max(max_results * 4, 20))
        matches = []
        needle = name.casefold()
        for symbol in candidates:
            if kind and symbol.kind != kind:
                continue
            leaf = symbol.qualified_name.rsplit(".", 1)[-1]
            if exact:
                matched = name in {leaf, symbol.qualified_name}
            else:
                matched = needle in leaf.casefold() or needle in symbol.qualified_name.casefold()
            if matched:
                matches.append(asdict(symbol))
            if len(matches) >= max_results:
                break
    finally:
        if owned_engine:
            current_engine.close()

    result = {
        "root": str(root),
        "query": name,
        "kind": kind,
        "exact": exact,
        "match_count": len(matches),
        "matches": matches,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def dependency_graph(arguments: dict[str, Any]) -> str:
    """Build a local Python import dependency graph for a repository tree."""
    root = Path(arguments.get("root", ".")).resolve()
    max_files = int(arguments.get("max_files", 200))
    include_external = bool(arguments.get("include_external", False))

    if not root.exists():
        return f"Directory {root} does not exist."
    if not root.is_dir():
        return f"Path {root} is not a directory."

    paths = list(_iter_python_files(root, max_files=max_files, include_tests=True))
    module_to_path = {_module_name(root, path): path.relative_to(root).as_posix() for path in paths}
    nodes = [{"id": module, "path": path} for module, path in sorted(module_to_path.items())]
    edges = []
    external: set[str] = set()

    for path in paths:
        module = _module_name(root, path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for imported in _imported_modules(tree, module):
            target = _match_local_module(imported, module_to_path)
            if target:
                edges.append({"from": module, "to": target, "import": imported})
            elif include_external:
                external.add(imported)

    result = {
        "root": str(root),
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["from"], edge["to"], edge["import"])),
        "external_modules": sorted(external),
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def inspect_change_impact(
    arguments: dict[str, Any],
    *,
    engine: SemanticWorkspaceEngine | None = None,
) -> str:
    """Inspect explainable, confidence-tiered impact candidates for changed files."""
    raw_paths = arguments.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths or not all(
        isinstance(path, str) and path.strip() for path in raw_paths
    ):
        raise ValueError("paths must be a non-empty array of file paths")
    root = Path(arguments.get("root", engine.root if engine else ".")).resolve()
    max_depth = int(arguments.get("max_depth", 2))
    max_nodes = int(arguments.get("max_nodes", 50))
    max_tests = int(arguments.get("max_tests", 8))
    if not 0 <= max_depth <= 5:
        raise ValueError("max_depth must be between 0 and 5")
    if max_nodes < 1 or max_tests < 0:
        raise ValueError("max_nodes must be positive and max_tests must not be negative")
    if engine is not None and engine.root != root:
        raise ValueError(f"inspection root {root} does not match semantic workspace {engine.root}")

    owned_engine = engine is None
    current_engine = engine or SemanticWorkspaceEngine(
        root,
        database_path=_standalone_index_path(root),
    )
    try:
        current_engine.update()
        report = current_engine.analyze_impact(
            raw_paths,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_tests=max_tests,
        )
        result = report.to_dict()
        result["notice"] = (
            "Impact entries are candidates: inspect source before editing and validate with tests."
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    finally:
        if owned_engine:
            current_engine.close()


def _standalone_index_path(root: Path) -> Path:
    """Keep ad-hoc symbol-search indexes outside the inspected repository."""
    root_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "dm_agent_search_indexes" / root_hash / "workspace.db"


def _index_root_failure(arguments: dict[str, Any]) -> ToolResult | None:
    root = Path(arguments.get("root", ".")).resolve()
    if not root.exists():
        return ToolResult(
            "failed",
            f"Directory {root} does not exist.",
            error_code="directory_not_found",
        )
    if not root.is_dir():
        return ToolResult(
            "failed",
            f"Path {root} is not a directory.",
            error_code="not_a_directory",
        )
    return None


def search_symbol_result(
    arguments: dict[str, Any],
    *,
    engine: SemanticWorkspaceEngine | None = None,
) -> ToolResult:
    checked_arguments = dict(arguments)
    if engine is not None and "root" not in checked_arguments:
        checked_arguments["root"] = str(engine.root)
    failure = _index_root_failure(checked_arguments)
    if failure:
        return failure
    try:
        return ToolResult("success", search_symbol(arguments, engine=engine))
    except ValueError as exc:
        return ToolResult("failed", str(exc), error_code="invalid_search")


def dependency_graph_result(arguments: dict[str, Any]) -> ToolResult:
    failure = _index_root_failure(arguments)
    return failure or ToolResult("success", dependency_graph(arguments))


def inspect_change_impact_result(
    arguments: dict[str, Any],
    *,
    engine: SemanticWorkspaceEngine | None = None,
) -> ToolResult:
    checked_arguments = dict(arguments)
    if engine is not None and "root" not in checked_arguments:
        checked_arguments["root"] = str(engine.root)
    failure = _index_root_failure(checked_arguments)
    if failure:
        return failure
    try:
        return ToolResult("success", inspect_change_impact(arguments, engine=engine))
    except (OSError, ValueError, sqlite3.Error) as exc:
        return ToolResult("failed", str(exc), error_code="impact_analysis_failed")


def _iter_python_files(root: Path, *, max_files: int, include_tests: bool) -> Iterable[Path]:
    count = 0
    for path in sorted(root.rglob("*.py")):
        relative_parts = path.relative_to(root).parts
        if any(part in DEFAULT_INDEX_EXCLUDES for part in relative_parts):
            continue
        if not include_tests and any(part in {"test", "tests"} for part in relative_parts):
            continue
        yield path
        count += 1
        if count >= max_files:
            break


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_modules(tree: ast.AST, module: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            modules.update(_resolve_import_from_names(module, node.level, node.module, names))
    return {item for item in modules if item}


def _resolve_import_from(current_module: str, level: int, module: str | None) -> str:
    if level <= 0:
        return module or ""

    parts = current_module.split(".") if current_module else []
    base = parts[: max(len(parts) - level, 0)]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _resolve_import_from_names(
    current_module: str,
    level: int,
    module: str | None,
    names: list[str],
) -> list[str]:
    base = _resolve_import_from(current_module, level, module)
    if level > 0 and not module:
        return [f"{base}.{name}" if base else name for name in names if name != "*"]
    return [base for _ in names]


def _match_local_module(imported: str, module_to_path: dict[str, str]) -> str | None:
    if imported in module_to_path:
        return imported
    parts = imported.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in module_to_path:
            return candidate
    return None
