"""SQLite-backed semantic index for repository-level code understanding."""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_EXCLUDES = frozenset(
    {
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
)
_TASK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{2,}")


@dataclass(frozen=True)
class SymbolRecord:
    path: str
    qualified_name: str
    kind: str
    signature: str
    line_start: int
    line_end: int
    parent: str = ""


@dataclass(frozen=True)
class IndexStats:
    scanned_files: int
    indexed_files: int
    cache_hits: int
    parse_errors: int


@dataclass(frozen=True)
class ImpactResult:
    symbol: str
    references: tuple[tuple[str, int], ...]
    tests: tuple[str, ...]


class SemanticBackend(Protocol):
    """Adapter contract for AST, Tree-sitter, or LSP symbol providers."""

    def parse(
        self, path: Path, source: str
    ) -> tuple[list[SymbolRecord], list[tuple[str, int]]]: ...


class PythonAstBackend:
    """Dependency-free backend used when no Tree-sitter/LSP adapter is installed."""

    def parse(self, path: Path, source: str) -> tuple[list[SymbolRecord], list[tuple[str, int]]]:
        tree = ast.parse(source, filename=str(path))
        relative = path.as_posix()
        symbols: list[SymbolRecord] = []
        references: list[tuple[str, int]] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    SymbolRecord(
                        relative,
                        node.name,
                        "class",
                        _class_signature(node),
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(
                            SymbolRecord(
                                relative,
                                f"{node.name}.{child.name}",
                                "method",
                                _function_signature(child),
                                child.lineno,
                                getattr(child, "end_lineno", child.lineno),
                                node.name,
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    SymbolRecord(
                        relative,
                        node.name,
                        "function",
                        _function_signature(node),
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                    )
                )
        for walked in ast.walk(tree):
            if isinstance(walked, ast.Name) and isinstance(walked.ctx, ast.Load):
                references.append((walked.id, walked.lineno))
            elif isinstance(walked, ast.Attribute) and isinstance(walked.ctx, ast.Load):
                references.append((walked.attr, walked.lineno))
        return symbols, references


class SemanticWorkspaceEngine:
    """Incrementally index symbols/references and build a bounded dynamic repo map."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        database_path: str | Path | None = None,
        max_scan_files: int = 300,
        backend: SemanticBackend | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_scan_files = max_scan_files
        self.backend = backend or PythonAstBackend()
        default_db = self.root / ".dm_agent" / "index" / "workspace.db"
        self.database_path = Path(database_path).resolve() if database_path else default_db
        explicit_database_path = database_path is not None
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            if explicit_database_path:
                raise
            self.database_path = self._fallback_database_path()
        self._connection = self._connect_database(explicit_database_path=explicit_database_path)
        self._connection.row_factory = sqlite3.Row
        try:
            self._fts_enabled = self._create_schema()
        except sqlite3.OperationalError:
            if explicit_database_path:
                raise
            self._connection.close()
            self.database_path = self._fallback_database_path()
            self._connection = self._connect_database(explicit_database_path=False)
            self._connection.row_factory = sqlite3.Row
            self._fts_enabled = self._create_schema()

    def _fallback_database_path(self) -> Path:
        # Read-only workspaces still get a persistent process-independent cache.
        root_key = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:16]
        database_path = Path(tempfile.gettempdir()) / "dm_agent_index" / root_key / "workspace.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        return database_path

    def _connect_database(self, *, explicit_database_path: bool) -> sqlite3.Connection:
        try:
            return sqlite3.connect(self.database_path)
        except sqlite3.OperationalError:
            if explicit_database_path:
                raise
            self.database_path = self._fallback_database_path()
            return sqlite3.connect(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def update(self, paths: Iterable[str | Path] | None = None) -> IndexStats:
        candidates = self._candidate_paths(paths)
        cache_hits = parse_errors = indexed = 0
        live = {path.relative_to(self.root).as_posix() for path in candidates if path.exists()}
        for path in candidates:
            relative = path.relative_to(self.root).as_posix()
            try:
                raw = path.read_bytes()
            except OSError:
                parse_errors += 1
                continue
            digest = hashlib.sha256(raw).hexdigest()
            row = self._connection.execute(
                "SELECT content_hash FROM files WHERE path = ?", (relative,)
            ).fetchone()
            if row is not None and row["content_hash"] == digest:
                cache_hits += 1
                continue
            try:
                symbols, references = self.backend.parse(Path(relative), raw.decode("utf-8"))
            except (UnicodeError, SyntaxError, ValueError):
                parse_errors += 1
                self._replace_file(relative, digest, [], [])
                continue
            self._replace_file(relative, digest, symbols, references)
            indexed += 1
        if paths is None:
            stored = {
                str(row["path"]) for row in self._connection.execute("SELECT path FROM files")
            }
            for stale in stored - live:
                self._delete_file(stale)
        self._connection.commit()
        return IndexStats(len(candidates), indexed, cache_hits, parse_errors)

    def search_symbols(self, query: str, *, limit: int = 20) -> list[SymbolRecord]:
        terms = [token for token in _TASK_TOKEN_RE.findall(query) if token]
        if not terms:
            return []
        if self._fts_enabled:
            expression = " OR ".join(f'"{term}"' for term in terms)
            try:
                rows = self._connection.execute(
                    """SELECT s.* FROM symbol_fts f JOIN symbols s ON s.id = f.rowid
                    WHERE symbol_fts MATCH ? ORDER BY bm25(symbol_fts) LIMIT ?""",
                    (expression, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        else:
            rows = []
        if not rows:
            pattern = f"%{terms[0]}%"
            rows = self._connection.execute(
                "SELECT * FROM symbols WHERE qualified_name LIKE ? OR signature LIKE ? LIMIT ?",
                (pattern, pattern, limit),
            ).fetchall()
        return [_row_to_symbol(row) for row in rows]

    def find_references(self, qualified_name: str, *, limit: int = 100) -> ImpactResult:
        leaf = qualified_name.rsplit(".", 1)[-1]
        rows = self._connection.execute(
            "SELECT path, line FROM refs WHERE name = ? ORDER BY path, line LIMIT ?",
            (leaf, limit),
        ).fetchall()
        refs = tuple((str(row["path"]), int(row["line"])) for row in rows)
        tests = tuple(sorted({path for path, _ in refs if _is_test_path(path)}))
        return ImpactResult(qualified_name, refs, tests)

    def affected_tests(self, changed_paths: Iterable[str | Path]) -> list[str]:
        tests: set[str] = set()
        test_paths = [
            str(row["path"])
            for row in self._connection.execute(
                "SELECT DISTINCT path FROM symbols WHERE path LIKE 'tests/%' OR path LIKE '%/test_%'"
            )
        ]
        for changed in changed_paths:
            relative = _relative_path(self.root, changed)
            stem = Path(relative).stem.removeprefix("test_")
            tests.update(path for path in test_paths if stem and stem in Path(path).stem)
            for row in self._connection.execute(
                "SELECT qualified_name FROM symbols WHERE path = ?", (relative,)
            ):
                tests.update(self.find_references(str(row["qualified_name"])).tests)
        return sorted(tests)

    def build_repo_map(
        self, task: str, *, max_files: int = 30, max_chars: int = 6000
    ) -> tuple[str, int, bool]:
        hits = self.search_symbols(task, limit=max_files * 3)
        selected: list[str] = []
        for hit in hits:
            if hit.path not in selected:
                selected.append(hit.path)
        for row in self._connection.execute(
            "SELECT path FROM files ORDER BY path LIMIT ?", (max_files * 2,)
        ):
            path = str(row["path"])
            if path not in selected:
                selected.append(path)
            if len(selected) >= max_files:
                break
        total = int(self._connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        lines = [
            f'<repository_map root="{self.root}">',
            "# Persistent semantic index; signatures only, source bodies omitted.",
        ]
        included = 0
        for path in selected[:max_files]:
            block = self._render_file(path)
            candidate = "\n".join([*lines, block, "</repository_map>"])
            if max_chars > 0 and len(candidate) > max_chars:
                break
            lines.append(block)
            included += 1
        truncated = included < total
        if truncated:
            lines.append(f"... {total - included} additional Python files omitted")
        lines.append("</repository_map>")
        return "\n".join(lines), included, truncated

    def _create_schema(self) -> bool:
        self._connection.executescript("""PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, content_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS symbols(
              id INTEGER PRIMARY KEY, path TEXT NOT NULL, qualified_name TEXT NOT NULL,
              kind TEXT NOT NULL, signature TEXT NOT NULL, line_start INTEGER NOT NULL,
              line_end INTEGER NOT NULL, parent TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS symbols_path_idx ON symbols(path);
            CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(qualified_name);
            CREATE TABLE IF NOT EXISTS refs(path TEXT NOT NULL, name TEXT NOT NULL, line INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS refs_name_idx ON refs(name);""")
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(qualified_name, signature)"
            )
        except sqlite3.OperationalError:
            return False
        return True

    def _replace_file(
        self,
        relative: str,
        digest: str,
        symbols: list[SymbolRecord],
        references: list[tuple[str, int]],
    ) -> None:
        self._delete_file(relative)
        self._connection.execute(
            "INSERT INTO files(path, content_hash) VALUES (?, ?)", (relative, digest)
        )
        for symbol in symbols:
            cursor = self._connection.execute(
                """INSERT INTO symbols(path, qualified_name, kind, signature, line_start, line_end, parent)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    relative,
                    symbol.qualified_name,
                    symbol.kind,
                    symbol.signature,
                    symbol.line_start,
                    symbol.line_end,
                    symbol.parent,
                ),
            )
            if self._fts_enabled:
                self._connection.execute(
                    "INSERT INTO symbol_fts(rowid, qualified_name, signature) VALUES (?, ?, ?)",
                    (cursor.lastrowid, symbol.qualified_name, symbol.signature),
                )
        self._connection.executemany(
            "INSERT INTO refs(path, name, line) VALUES (?, ?, ?)",
            [(relative, name, line) for name, line in references],
        )

    def _delete_file(self, relative: str) -> None:
        if self._fts_enabled:
            ids = [
                int(row["id"])
                for row in self._connection.execute(
                    "SELECT id FROM symbols WHERE path = ?", (relative,)
                )
            ]
            self._connection.executemany(
                "DELETE FROM symbol_fts WHERE rowid = ?", [(i,) for i in ids]
            )
        self._connection.execute("DELETE FROM refs WHERE path = ?", (relative,))
        self._connection.execute("DELETE FROM symbols WHERE path = ?", (relative,))
        self._connection.execute("DELETE FROM files WHERE path = ?", (relative,))

    def _candidate_paths(self, paths: Iterable[str | Path] | None) -> list[Path]:
        if paths is not None:
            result: list[Path] = []
            for item in paths:
                path = Path(item)
                if not path.is_absolute():
                    path = self.root / path
                path = path.resolve()
                if path.is_file() and path.suffix == ".py" and path.is_relative_to(self.root):
                    result.append(path)
            return sorted(set(result))
        result = []
        for path in sorted(self.root.rglob("*.py")):
            parts = path.relative_to(self.root).parts
            if any(part in DEFAULT_EXCLUDES or part.startswith(".pytest-tmp-") for part in parts):
                continue
            result.append(path)
            if len(result) >= self.max_scan_files:
                break
        return result

    def _render_file(self, path: str) -> str:
        lines = [path]
        rows = self._connection.execute(
            "SELECT signature, parent FROM symbols WHERE path = ? ORDER BY line_start", (path,)
        ).fetchall()
        if not rows:
            lines.append("  (no top-level classes or functions)")
        for row in rows:
            lines.append(f"{'    ' if row['parent'] else '  '}{row['signature']}")
        return "\n".join(lines)


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    return f"class {node.name}({bases})" if bases else f"class {node.name}"


def _row_to_symbol(row: sqlite3.Row) -> SymbolRecord:
    return SymbolRecord(
        str(row["path"]),
        str(row["qualified_name"]),
        str(row["kind"]),
        str(row["signature"]),
        int(row["line_start"]),
        int(row["line_end"]),
        str(row["parent"]),
    )


def _is_test_path(path: str) -> bool:
    return "tests" in Path(path).parts or Path(path).name.startswith("test_")


def _relative_path(root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()
