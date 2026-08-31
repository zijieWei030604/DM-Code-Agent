"""Incremental Python relationship graph and change-impact propagation."""

from __future__ import annotations

import ast
import hashlib
import sqlite3
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImpactNode:
    path: str
    symbol: str
    relation: str
    distance: int
    confidence: float
    via_path: str
    via_symbol: str


@dataclass(frozen=True)
class ImpactReport:
    changed_files: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    affected_symbols: tuple[ImpactNode, ...]
    affected_files: tuple[str, ...]
    related_tests: tuple[str, ...]
    risk_score: float
    risk_level: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["affected_symbols"] = [asdict(item) for item in self.affected_symbols]
        return payload

    def render(self, *, max_nodes: int = 12) -> str:
        lines = [
            "<change_impact>",
            f"risk: {self.risk_level} ({self.risk_score:.2f})",
            f"changed_files: {', '.join(self.changed_files) or 'none'}",
        ]
        if self.changed_symbols:
            lines.append(f"changed_symbols: {', '.join(self.changed_symbols[:12])}")
        if self.affected_symbols:
            lines.append("affected:")
            for item in self.affected_symbols[:max_nodes]:
                lines.append(
                    f"  - {item.path}:{item.symbol} <-{item.relation}- "
                    f"{item.via_path}:{item.via_symbol} "
                    f"(depth={item.distance}, confidence={item.confidence:.2f})"
                )
        if self.related_tests:
            lines.append(f"related_tests: {', '.join(self.related_tests[:12])}")
        lines.extend(f"reason: {reason}" for reason in self.reasons[:8])
        lines.append("</change_impact>")
        return "\n".join(lines)


@dataclass(frozen=True)
class _Symbol:
    path: str
    name: str
    kind: str


@dataclass(frozen=True)
class _Reference:
    path: str
    owner: str
    name: str
    target_hint: str
    relation: str
    line: int


class ImpactGraph:
    """Maintain a scoped call/import/inheritance graph in the workspace database."""

    PARSER_VERSION = "python-impact-v1"

    def __init__(self, root: Path, connection: sqlite3.Connection) -> None:
        self.root = root
        self.connection = connection
        self._create_schema()

    def update(self, paths: Iterable[Path], *, remove_missing: bool = False) -> int:
        candidates = list(paths)
        live = {
            path.relative_to(self.root).as_posix()
            for path in candidates
            if path.is_file() and path.is_relative_to(self.root)
        }
        changed = False
        for path in candidates:
            if not path.is_relative_to(self.root):
                continue
            relative = path.relative_to(self.root).as_posix()
            if not path.is_file():
                if self._tracked(relative):
                    self._delete_file(relative)
                    changed = True
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            row = self.connection.execute(
                "SELECT content_hash, parser_version FROM impact_files WHERE path = ?",
                (relative,),
            ).fetchone()
            if (
                row
                and row["content_hash"] == digest
                and row["parser_version"] == self.PARSER_VERSION
            ):
                continue
            try:
                symbols, references = _parse_python(relative, source)
            except SyntaxError:
                symbols, references = [], []
            self._replace_file(relative, digest, symbols, references)
            changed = True
        if remove_missing:
            stored = {
                str(row["path"]) for row in self.connection.execute("SELECT path FROM impact_files")
            }
            for stale in stored - live:
                self._delete_file(stale)
                changed = True
        if changed:
            self._rebuild_edges()
        return int(self.connection.execute("SELECT COUNT(*) FROM impact_edges").fetchone()[0])

    def analyze(
        self,
        changed_paths: Iterable[str | Path],
        *,
        max_depth: int = 2,
        max_nodes: int = 100,
    ) -> ImpactReport:
        changed_files = tuple(sorted({_relative(self.root, item) for item in changed_paths}))
        symbol_rows = []
        for path in changed_files:
            symbol_rows.extend(
                self.connection.execute(
                    "SELECT path, name, kind FROM impact_symbols WHERE path = ?", (path,)
                ).fetchall()
            )
        changed_symbols = tuple(sorted(f"{row['path']}:{row['name']}" for row in symbol_rows))
        seeds = {(str(row["path"]), str(row["name"])) for row in symbol_rows}
        seeds.update((path, "<module>") for path in changed_files)
        queue = deque((path, symbol, 0) for path, symbol in sorted(seeds))
        visited = set(seeds)
        affected: list[ImpactNode] = []
        reasons: list[str] = []
        while queue and len(affected) < max_nodes:
            target_path, target_symbol, distance = queue.popleft()
            if distance >= max(0, max_depth):
                continue
            if target_symbol == "<module>":
                rows = self.connection.execute(
                    """SELECT source_path, source_symbol, relation, confidence
                       FROM impact_edges
                       WHERE target_path = ?
                       ORDER BY confidence DESC, source_path, source_symbol""",
                    (target_path,),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """SELECT source_path, source_symbol, relation, confidence
                       FROM impact_edges
                       WHERE target_path = ? AND target_symbol IN (?, '<module>')
                       ORDER BY confidence DESC, source_path, source_symbol""",
                    (target_path, target_symbol),
                ).fetchall()
            for row in rows:
                source = (str(row["source_path"]), str(row["source_symbol"]))
                if source in visited:
                    continue
                visited.add(source)
                item = ImpactNode(
                    source[0],
                    source[1],
                    str(row["relation"]),
                    distance + 1,
                    float(row["confidence"]),
                    target_path,
                    target_symbol,
                )
                affected.append(item)
                queue.append((source[0], source[1], distance + 1))
                if len(reasons) < 8:
                    reasons.append(
                        f"{source[0]}:{source[1]} {item.relation} " f"{target_path}:{target_symbol}"
                    )
        affected_files = tuple(
            sorted({item.path for item in affected if item.path not in changed_files})
        )
        tests = {item.path for item in affected if _is_test(item.path)}
        tests.update(self._test_companions(changed_files))
        risk_score = _risk_score(symbol_rows, affected, changed_files)
        risk_level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.35 else "low"
        if tests:
            reasons.append(f"selected {len(tests)} related test file(s)")
        if not affected:
            reasons.append("no reverse dependency edges resolved")
        return ImpactReport(
            changed_files,
            changed_symbols,
            tuple(affected),
            affected_files,
            tuple(sorted(tests)),
            risk_score,
            risk_level,
            tuple(dict.fromkeys(reasons)),
        )

    def _create_schema(self) -> None:
        self.connection.executescript("""CREATE TABLE IF NOT EXISTS impact_files(
                 path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, parser_version TEXT NOT NULL);
               CREATE TABLE IF NOT EXISTS impact_symbols(
                 path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
                 PRIMARY KEY(path, name));
               CREATE TABLE IF NOT EXISTS impact_refs(
                 path TEXT NOT NULL, owner TEXT NOT NULL, name TEXT NOT NULL,
                 target_hint TEXT NOT NULL, relation TEXT NOT NULL, line INTEGER NOT NULL);
               CREATE INDEX IF NOT EXISTS impact_refs_name_idx ON impact_refs(name);
               CREATE TABLE IF NOT EXISTS impact_edges(
                 source_path TEXT NOT NULL, source_symbol TEXT NOT NULL,
                 target_path TEXT NOT NULL, target_symbol TEXT NOT NULL,
                 relation TEXT NOT NULL, line INTEGER NOT NULL, confidence REAL NOT NULL);
               CREATE INDEX IF NOT EXISTS impact_edges_target_idx
                 ON impact_edges(target_path, target_symbol);""")

    def _replace_file(
        self,
        path: str,
        digest: str,
        symbols: list[_Symbol],
        references: list[_Reference],
    ) -> None:
        self._delete_file(path)
        self.connection.execute(
            "INSERT INTO impact_files(path, content_hash, parser_version) VALUES (?, ?, ?)",
            (path, digest, self.PARSER_VERSION),
        )
        self.connection.executemany(
            "INSERT INTO impact_symbols(path, name, kind) VALUES (?, ?, ?)",
            [(item.path, item.name, item.kind) for item in symbols],
        )
        self.connection.executemany(
            """INSERT INTO impact_refs(path, owner, name, target_hint, relation, line)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (item.path, item.owner, item.name, item.target_hint, item.relation, item.line)
                for item in references
            ],
        )

    def _delete_file(self, path: str) -> None:
        self.connection.execute("DELETE FROM impact_edges WHERE source_path = ?", (path,))
        self.connection.execute("DELETE FROM impact_edges WHERE target_path = ?", (path,))
        self.connection.execute("DELETE FROM impact_refs WHERE path = ?", (path,))
        self.connection.execute("DELETE FROM impact_symbols WHERE path = ?", (path,))
        self.connection.execute("DELETE FROM impact_files WHERE path = ?", (path,))

    def _rebuild_edges(self) -> None:
        self.connection.execute("DELETE FROM impact_edges")
        rows = self.connection.execute("SELECT path, name FROM impact_symbols").fetchall()
        by_leaf: dict[str, list[tuple[str, str]]] = {}
        modules: dict[str, str] = {}
        for row in rows:
            path, name = str(row["path"]), str(row["name"])
            by_leaf.setdefault(name.rsplit(".", 1)[-1], []).append((path, name))
            modules[_module_name(Path(path))] = path
        for row in self.connection.execute("SELECT path FROM impact_files"):
            path = str(row["path"])
            modules[_module_name(Path(path))] = path
        edges = set()
        for row in self.connection.execute("SELECT * FROM impact_refs"):
            targets = _resolve_targets(
                str(row["path"]),
                str(row["name"]),
                str(row["target_hint"]),
                by_leaf,
                modules,
            )
            for target_path, target_symbol, confidence in targets:
                edges.add(
                    (
                        str(row["path"]),
                        str(row["owner"]),
                        target_path,
                        target_symbol,
                        str(row["relation"]),
                        int(row["line"]),
                        confidence,
                    )
                )
        self.connection.executemany(
            """INSERT INTO impact_edges(
                 source_path, source_symbol, target_path, target_symbol,
                 relation, line, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            sorted(edges),
        )

    def _tracked(self, path: str) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM impact_files WHERE path = ?", (path,)).fetchone()
            is not None
        )

    def _test_companions(self, changed: Iterable[str]) -> set[str]:
        tests = [
            str(row["path"])
            for row in self.connection.execute("SELECT path FROM impact_files")
            if _is_test(str(row["path"]))
        ]
        result: set[str] = set()
        for path in changed:
            stem = Path(path).stem.removeprefix("test_")
            result.update(test for test in tests if stem and stem in Path(test).stem)
        return result


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, module: str) -> None:
        self.path = path
        self.module = module
        self.symbols: list[_Symbol] = []
        self.references: list[_Reference] = []
        self.aliases: dict[str, str] = {}
        self.classes: list[str] = []
        self.owners: list[str] = []

    @property
    def owner(self) -> str:
        return self.owners[-1] if self.owners else "<module>"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
            self._reference(alias.name.rsplit(".", 1)[-1], alias.name, "imports", node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolve_import(self.module, node.level, node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = f"{base}.{alias.name}" if base else alias.name
            self.aliases[alias.asname or alias.name] = target
            self._reference(alias.name, target, "imports", node.lineno)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        name = ".".join([*self.classes, node.name])
        self.symbols.append(_Symbol(self.path, name, "class"))
        for base in node.bases:
            dotted = _dotted(base)
            if dotted:
                self._reference(
                    dotted.rsplit(".", 1)[-1], self._expand(dotted), "inherits", node.lineno, name
                )
        self.classes.append(node.name)
        self.owners.append(name)
        for child in node.body:
            self.visit(child)
        self.owners.pop()
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = ".".join([*self.classes, node.name])
        self.symbols.append(_Symbol(self.path, name, "method" if self.classes else "function"))
        self.owners.append(name)
        for child in node.body:
            self.visit(child)
        self.owners.pop()

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        if dotted:
            expanded = self._expand(dotted)
            self._reference(expanded.rsplit(".", 1)[-1], expanded, "calls", node.lineno)
        self.generic_visit(node)

    def _expand(self, dotted: str) -> str:
        head, separator, tail = dotted.partition(".")
        target = self.aliases.get(head, head)
        return f"{target}.{tail}" if separator else target

    def _reference(
        self,
        name: str,
        target: str,
        relation: str,
        line: int,
        owner: str | None = None,
    ) -> None:
        self.references.append(
            _Reference(self.path, owner or self.owner, name, target, relation, line)
        )


def _parse_python(path: str, source: str) -> tuple[list[_Symbol], list[_Reference]]:
    visitor = _Visitor(path, _module_name(Path(path)))
    visitor.visit(ast.parse(source, filename=path))
    # A logical symbol may have multiple AST definitions in real projects, for
    # example a property getter/setter pair or overload declarations.  The graph
    # schema identifies symbols by (path, name), so retain the first definition
    # in source order while keeping every reference for impact propagation.
    symbols_by_name: dict[str, _Symbol] = {}
    for symbol in visitor.symbols:
        symbols_by_name.setdefault(symbol.name, symbol)
    return list(symbols_by_name.values()), visitor.references


def _resolve_targets(
    source_path: str,
    leaf: str,
    hint: str,
    by_leaf: dict[str, list[tuple[str, str]]],
    modules: dict[str, str],
) -> list[tuple[str, str, float]]:
    module_hint = hint.rsplit(".", 1)[0] if "." in hint else ""
    module_path = _module_path(module_hint or hint, modules)
    candidates = by_leaf.get(leaf, [])
    if module_path:
        exact = [item for item in candidates if item[0] == module_path]
        if exact:
            return [(path, symbol, 1.0) for path, symbol in exact]
        if not candidates or hint == module_hint:
            return [(module_path, "<module>", 0.95)]
    same_file = [item for item in candidates if item[0] == source_path]
    if same_file:
        return [(path, symbol, 0.95) for path, symbol in same_file[:3]]
    if len(candidates) == 1:
        return [(candidates[0][0], candidates[0][1], 0.9)]
    return [(path, symbol, 0.55) for path, symbol in candidates[:3]]


def _module_path(module: str, modules: dict[str, str]) -> str | None:
    while module:
        if module in modules:
            return modules[module]
        module = module.rpartition(".")[0]
    return None


def _resolve_import(current: str, level: int, module: str | None) -> str:
    if level <= 0:
        return module or ""
    package = current.split(".")[:-1]
    base = package[: max(0, len(package) - level + 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _relative(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _is_test(path: str) -> bool:
    parts = {part.casefold() for part in Path(path).parts}
    return bool(parts & {"test", "tests"}) or Path(path).name.startswith("test_")


def _risk_score(rows: list[Any], nodes: list[ImpactNode], changed: tuple[str, ...]) -> float:
    public = sum(1 for row in rows if not str(row["name"]).rsplit(".", 1)[-1].startswith("_"))
    production = {node.path for node in nodes if not _is_test(node.path)}
    score = 0.08 * min(len(changed), 3)
    score += 0.06 * min(public, 4)
    score += 0.08 * min(len(production), 4)
    score += 0.05 * min(max((node.distance for node in nodes), default=0), 2)
    if any(node.relation in {"inherits", "imports"} for node in nodes):
        score += 0.12
    if len({node.path for node in nodes}) >= 6:
        score += 0.1
    return round(min(1.0, score), 2)


__all__ = ["ImpactGraph", "ImpactNode", "ImpactReport"]
