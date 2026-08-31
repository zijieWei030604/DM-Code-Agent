"""Task-aware, incrementally cached repository maps for Python projects."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from dm_agent.workspace import SemanticWorkspaceEngine

DEFAULT_REPO_MAP_EXCLUDES = frozenset(
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
_STOP_WORDS = frozenset(
    {
        "add",
        "and",
        "bug",
        "code",
        "file",
        "fix",
        "for",
        "from",
        "implement",
        "please",
        "python",
        "the",
        "this",
        "with",
    }
)


@dataclass(frozen=True)
class RepoSymbol:
    kind: str
    name: str
    signature: str
    methods: tuple[RepoSymbol, ...] = ()


@dataclass(frozen=True)
class RepoFileSummary:
    path: str
    symbols: tuple[RepoSymbol, ...]
    imports: tuple[str, ...]


@dataclass(frozen=True)
class RepoMapResult:
    content: str
    scanned_files: int
    included_files: int
    parse_errors: int
    cache_hits: int
    truncated: bool
    fingerprint: str


@dataclass(frozen=True)
class _CacheEntry:
    mtime_ns: int
    size: int
    summary: RepoFileSummary | None
    parse_error: bool = False


@dataclass
class RepositoryMap:
    """Build a task-aware map backed by the persistent semantic workspace index."""

    max_scan_files: int = 300
    max_map_files: int = 30
    max_chars: int = 6000
    _cache: dict[Path, _CacheEntry] = field(default_factory=dict, init=False, repr=False)
    _engines: dict[Path, SemanticWorkspaceEngine] = field(
        default_factory=dict, init=False, repr=False
    )

    def build(self, task: str, root: str | Path = ".") -> RepoMapResult:
        repository_root = Path(root).resolve()
        if not repository_root.is_dir():
            raise ValueError(f"Repo Map 根目录不是有效目录：{repository_root}")

        engine = self._engines.get(repository_root)
        if engine is None:
            engine = SemanticWorkspaceEngine(
                repository_root,
                max_scan_files=self.max_scan_files,
            )
            self._engines[repository_root] = engine
        stats = engine.update()
        content, included_files, truncated = engine.build_repo_map(
            task,
            max_files=self.max_map_files,
            max_chars=self.max_chars,
        )
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return RepoMapResult(
            content=content,
            scanned_files=stats.scanned_files,
            included_files=included_files,
            parse_errors=stats.parse_errors,
            cache_hits=stats.cache_hits,
            truncated=truncated,
            fingerprint=fingerprint,
        )

        # Legacy in-memory implementation is intentionally retained below for one release;
        # it documents the previous ranking behavior and can be removed after index migration.

        paths = list(self._iter_python_files(repository_root))
        summaries: list[RepoFileSummary] = []
        cache_hits = 0
        parse_errors = 0
        live_paths = set(paths)

        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                parse_errors += 1
                continue
            cached = self._cache.get(path)
            if cached and cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size:
                cache_hits += 1
                parse_errors += int(cached.parse_error)
                if cached.summary is not None:
                    summaries.append(cached.summary)
                continue

            summary, parse_error = self._summarize_file(repository_root, path)
            self._cache[path] = _CacheEntry(
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                summary=summary,
                parse_error=parse_error,
            )
            parse_errors += int(parse_error)
            if summary is not None:
                summaries.append(summary)

        for stale_path in set(self._cache) - live_paths:
            del self._cache[stale_path]

        selected = self._select_files(task, summaries)
        content, included_files, truncated = self._render(repository_root, selected, len(summaries))
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return RepoMapResult(
            content=content,
            scanned_files=len(paths),
            included_files=included_files,
            parse_errors=parse_errors,
            cache_hits=cache_hits,
            truncated=truncated,
            fingerprint=fingerprint,
        )

    def _iter_python_files(self, root: Path):
        count = 0
        for path in sorted(root.rglob("*.py")):
            relative_parts = path.relative_to(root).parts
            if any(part in DEFAULT_REPO_MAP_EXCLUDES for part in relative_parts):
                continue
            if any(part.startswith(".pytest-tmp-") for part in relative_parts):
                continue
            yield path
            count += 1
            if count >= self.max_scan_files:
                break

    @staticmethod
    def _summarize_file(root: Path, path: Path) -> tuple[RepoFileSummary | None, bool]:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            return None, True

        symbols: list[RepoSymbol] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = tuple(
                    RepoSymbol(
                        kind="method",
                        name=item.name,
                        signature=_function_signature(item),
                    )
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                symbols.append(
                    RepoSymbol(
                        kind="class",
                        name=node.name,
                        signature=_class_signature(node),
                        methods=methods,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    RepoSymbol(
                        kind="function",
                        name=node.name,
                        signature=_function_signature(node),
                    )
                )

        imports = tuple(sorted(_local_import_hints(tree)))
        return (
            RepoFileSummary(
                path=path.relative_to(root).as_posix(),
                symbols=tuple(symbols),
                imports=imports,
            ),
            False,
        )

    def _select_files(self, task: str, summaries: list[RepoFileSummary]) -> list[RepoFileSummary]:
        terms = {
            token.casefold()
            for token in _TASK_TOKEN_RE.findall(task)
            if token.casefold() not in _STOP_WORDS
        }

        def score(summary: RepoFileSummary) -> tuple[int, int, str]:
            path_text = summary.path.casefold()
            symbol_text = " ".join(_flatten_symbol_text(summary.symbols)).casefold()
            relevance = sum(8 for term in terms if term in path_text)
            relevance += sum(5 for term in terms if term in symbol_text)
            if "test" in task.casefold() and _is_test_path(summary.path):
                relevance += 4
            if Path(summary.path).name in {"main.py", "app.py", "cli.py", "__init__.py"}:
                relevance += 1
            depth = len(Path(summary.path).parts)
            return relevance, -depth, summary.path

        ranked = sorted(summaries, key=score, reverse=True)
        selected = ranked[: self.max_map_files]

        # When a production module is relevant, prefer nearby tests with the same stem.
        selected_paths = {item.path for item in selected}
        stems = {Path(item.path).stem.removeprefix("test_") for item in selected}
        companions = [
            item
            for item in summaries
            if item.path not in selected_paths
            and _is_test_path(item.path)
            and any(stem and stem in Path(item.path).stem for stem in stems)
        ]
        for companion in companions:
            if len(selected) >= self.max_map_files:
                break
            selected.append(companion)
        return selected

    def _render(
        self,
        root: Path,
        summaries: list[RepoFileSummary],
        total_summaries: int,
    ) -> tuple[str, int, bool]:
        header = [
            f'<repository_map root="{root}">',
            "# Generated repository structure; signatures only, source bodies omitted.",
        ]
        footer = ["</repository_map>"]
        blocks: list[str] = []
        included = 0

        for summary in summaries:
            block = _render_file(summary)
            omitted = max(0, total_summaries - included - 1)
            truncation_line = f"... {omitted} additional Python files omitted"
            candidate = "\n".join([*header, *blocks, block, truncation_line, *footer])
            if self.max_chars > 0 and len(candidate) > self.max_chars:
                break
            blocks.append(block)
            included += 1

        truncated = included < total_summaries
        lines = [*header, *blocks]
        if truncated:
            lines.append(f"... {total_summaries - included} additional Python files omitted")
        lines.extend(footer)
        content = "\n".join(lines)
        if self.max_chars > 0 and len(content) > self.max_chars:
            omitted_line = "... repository map omitted by character budget"
            fallback_lines = [*header, omitted_line, *footer]
            content = "\n".join(fallback_lines)
            if len(content) > self.max_chars:
                compact_header = f'<repository_map root="{root}">'
                compact_lines = [compact_header, omitted_line, *footer]
                content = "\n".join(compact_lines)
            if len(content) > self.max_chars:
                content = content[: self.max_chars]
            included = 0
            truncated = bool(total_summaries)
        return content, included, truncated


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    return f"class {node.name}({bases})" if bases else f"class {node.name}"


def _local_import_hints(tree: ast.Module) -> set[str]:
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            imports.add(f"{prefix}{node.module or ''}")
    return {name for name in imports if name}


def _flatten_symbol_text(symbols: tuple[RepoSymbol, ...]):
    for symbol in symbols:
        yield symbol.name
        yield symbol.signature
        for method in symbol.methods:
            yield method.name
            yield method.signature


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or "test" in parts or Path(path).name.startswith("test_")


def _render_file(summary: RepoFileSummary) -> str:
    lines = [summary.path]
    for symbol in summary.symbols:
        lines.append(f"  {symbol.signature}")
        for method in symbol.methods:
            lines.append(f"    {method.signature}")
    if not summary.symbols:
        lines.append("  (no top-level classes or functions)")
    if summary.imports:
        imports = ", ".join(summary.imports[:6])
        suffix = ", ..." if len(summary.imports) > 6 else ""
        lines.append(f"  imports: {imports}{suffix}")
    return "\n".join(lines)


__all__ = ["RepoMapResult", "RepositoryMap"]
