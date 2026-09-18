"""Incremental Python relationship graph and change-impact propagation."""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HIGH_CONFIDENCE = 0.9
AMBIGUOUS_CONFIDENCE = 0.55


@dataclass(frozen=True)
class ImpactNode:
    # path        受影响符号所在文件
    # symbol      受影响符号名称
    # relation    关系：references
    # distance    离修改点几跳
    # confidence  关系推断可信度
    # via_path    它依赖的目标文件
    # via_symbol  它依赖的目标符号
    path: str
    symbol: str
    relation: str
    distance: int
    confidence: float
    via_path: str
    via_symbol: str

    @property
    def confidence_level(self) -> str:
        """Expose a stable category without discarding the internal score."""
        return "confirmed" if self.confidence >= HIGH_CONFIDENCE else "ambiguous"


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
    fallback_tests: tuple[str, ...] = ()

    @property
    def confirmed_symbols(self) -> tuple[ImpactNode, ...]:
        return tuple(item for item in self.affected_symbols if item.confidence >= HIGH_CONFIDENCE)

    @property
    def ambiguous_symbols(self) -> tuple[ImpactNode, ...]:
        return tuple(item for item in self.affected_symbols if item.confidence < HIGH_CONFIDENCE)

    @property
    def graph_tests(self) -> tuple[str, ...]:
        fallback = set(self.fallback_tests)
        return tuple(path for path in self.related_tests if path not in fallback)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["affected_symbols"] = [
            {**asdict(item), "confidence_level": item.confidence_level}
            for item in self.affected_symbols
        ]
        payload["test_candidates"] = [
            {
                "path": path,
                "confidence_level": "fallback" if path in self.fallback_tests else "confirmed",
            }
            for path in self.related_tests
        ]
        return payload

    def render(self, *, max_nodes: int = 12) -> str:
        lines = [
            "<change_impact>",
            f"risk: {self.risk_level} ({self.risk_score:.2f})",
            f"changed_files: {', '.join(self.changed_files) or 'none'}",
        ]
        if self.changed_symbols:
            lines.append(f"changed_symbols: {', '.join(self.changed_symbols[:12])}")
        if self.confirmed_symbols:
            lines.append("affected:")
            for item in self.confirmed_symbols[:max_nodes]:
                lines.append(
                    f"  - {item.path}:{item.symbol} <-{item.relation}- "
                    f"{item.via_path}:{item.via_symbol} "
                    f"(depth={item.distance}, level={item.confidence_level}, "
                    f"confidence={item.confidence:.2f})"
                )
        remaining = max(0, max_nodes - len(self.confirmed_symbols[:max_nodes]))
        if self.ambiguous_symbols and remaining:
            lines.append("ambiguous_candidates:")
            for item in self.ambiguous_symbols[:remaining]:
                lines.append(
                    f"  - {item.path}:{item.symbol} <-{item.relation}- "
                    f"{item.via_path}:{item.via_symbol} "
                    f"(depth={item.distance}, level={item.confidence_level}, "
                    f"confidence={item.confidence:.2f})"
                )
        if self.graph_tests:
            lines.append(f"related_tests: {', '.join(self.graph_tests[:12])}")
        if self.fallback_tests:
            lines.append(f"fallback_tests: {', '.join(self.fallback_tests[:12])}")
        lines.extend(f"reason: {reason}" for reason in self.reasons[:8])
        lines.append("</change_impact>")
        return "\n".join(lines)


class ImpactGraph:
    """Maintain a scoped call/import/inheritance graph in the workspace database."""

    def __init__(self, root: Path, connection: sqlite3.Connection) -> None:
        self.root = root
        self.connection = connection
        self._create_schema()

    def update(
        self,
        changes: dict[str, set[str]],
        *,
        full_refresh: bool = False,
    ) -> int:
        """Refresh derived impact edges from the shared semantic index.

        ``files`` / ``symbols`` / ``refs`` are the single source of truth. The
        impact graph only stores dependency edges derived from those tables.
        A full index scan rebuilds the complete projection. Targeted source
        updates rebuild only sources that could have been affected by the
        changed files' old or new symbols.
        """
        if full_refresh:
            self._rebuild_edges()
        elif changes:
            self._rebuild_changed_edges(changes)
        return int(self.connection.execute("SELECT COUNT(*) FROM impact_edges").fetchone()[0])

    def analyze(
        self,
        changed_paths: Iterable[str | Path],
        *,
        max_depth: int = 2,
        max_nodes: int = 100,
        max_tests: int = 8,
    ) -> ImpactReport:
        changed_files = tuple(sorted({_relative(self.root, item) for item in changed_paths}))
        symbol_rows = []
        for path in changed_files:
            symbol_rows.extend(
                self.connection.execute(
                    "SELECT DISTINCT path, qualified_name AS name, kind FROM symbols WHERE path = ?",
                    (path,),
                ).fetchall()
            )
        changed_symbols = tuple(sorted({f"{row['path']}:{row['name']}" for row in symbol_rows}))
        seeds = {(str(row["path"]), str(row["name"])) for row in symbol_rows}
        seeds.update((path, "<module>") for path in changed_files)
        queue = deque((path, symbol, 0) for path, symbol in sorted(seeds))
        propagated = set(seeds)
        affected_by_source: dict[tuple[str, str], ImpactNode] = {}
        while queue and len(affected_by_source) < max_nodes:
            target_path, target_symbol, distance = queue.popleft()
            test_only = distance >= max(0, max_depth)
            if distance > max(0, max_depth):
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
                confidence = float(row["confidence"])
                if test_only and (not _is_test(source[0]) or confidence < HIGH_CONFIDENCE):
                    continue
                if source in propagated:
                    continue
                item = ImpactNode(
                    source[0],
                    source[1],
                    str(row["relation"]),
                    distance + 1,
                    confidence,
                    target_path,
                    target_symbol,
                )
                previous = affected_by_source.get(source)
                if previous is not None and previous.confidence >= item.confidence:
                    continue
                affected_by_source[source] = item
                # Ambiguous leaf-name matches are useful direct candidates, but
                # propagating through them compounds one guess into many.
                if item.confidence >= HIGH_CONFIDENCE and not test_only:
                    propagated.add(source)
                    queue.append((source[0], source[1], distance + 1))
        affected = list(affected_by_source.values())
        reasons = [
            f"{item.path}:{item.symbol} {item.relation} {item.via_path}:{item.via_symbol}"
            for item in affected[:8]
        ]
        affected_files = tuple(
            sorted({item.path for item in affected if item.path not in changed_files})
        )
        graph_test_nodes: dict[str, ImpactNode] = {}
        for item in affected:
            if not _is_test(item.path) or item.confidence < HIGH_CONFIDENCE:
                continue
            previous = graph_test_nodes.get(item.path)
            if previous is None or _test_rank(item) < _test_rank(previous):
                graph_test_nodes[item.path] = item
        graph_tests = tuple(
            sorted(graph_test_nodes, key=lambda path: (*_test_rank(graph_test_nodes[path]), path))
        )
        fallback_candidates = self._test_companions(changed_files) - set(graph_tests)
        direct_graph_tests = tuple(
            path for path in graph_tests if graph_test_nodes[path].distance == 1
        )
        indirect_graph_tests = tuple(
            path for path in graph_tests if graph_test_nodes[path].distance != 1
        )
        exact_fallback_tests = tuple(
            sorted(
                path
                for path in fallback_candidates
                if _is_exact_test_companion(path, changed_files)
            )
        )
        remaining_fallback_tests = tuple(
            sorted(set(fallback_candidates) - set(exact_fallback_tests))
        )
        ranked_tests = (
            *exact_fallback_tests,
            *direct_graph_tests,
            *indirect_graph_tests,
            *remaining_fallback_tests,
        )
        tests = ranked_tests[: max(0, max_tests)]
        fallback_tests = tuple(path for path in tests if path in fallback_candidates)
        risk_score = _risk_score(symbol_rows, affected, changed_files)
        risk_level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.35 else "low"
        if graph_tests:
            reasons.append(f"selected {len(graph_tests)} graph-related test file(s)")
        if fallback_tests:
            reasons.append(f"selected {len(fallback_tests)} filename fallback test file(s)")
        if not affected:
            reasons.append("no reverse dependency edges resolved")
        return ImpactReport(
            changed_files,
            changed_symbols,
            tuple(affected),
            affected_files,
            tests,
            risk_score,
            risk_level,
            tuple(dict.fromkeys(reasons)),
            fallback_tests,
        )

    def _create_schema(self) -> None:
        self.connection.executescript("""CREATE TABLE IF NOT EXISTS impact_edges(
                 source_path TEXT NOT NULL, source_symbol TEXT NOT NULL,
                 target_path TEXT NOT NULL, target_symbol TEXT NOT NULL,
                 relation TEXT NOT NULL, line INTEGER NOT NULL, confidence REAL NOT NULL);
               CREATE INDEX IF NOT EXISTS impact_edges_target_idx
                 ON impact_edges(target_path, target_symbol);
               CREATE INDEX IF NOT EXISTS impact_edges_source_idx
                 ON impact_edges(source_path);""")

    def _delete_file(self, path: str) -> None:
        self.connection.execute("DELETE FROM impact_edges WHERE source_path = ?", (path,))
        self.connection.execute("DELETE FROM impact_edges WHERE target_path = ?", (path,))

    def _rebuild_edges(self) -> None:
        self.connection.execute("DELETE FROM impact_edges")
        self._insert_edges_for_sources(None)

    def _rebuild_changed_edges(self, changes: dict[str, set[str]]) -> None:
        changed_paths = set(changes)
        sources = self._affected_sources(changed_paths, changes)
        if not sources:
            return
        placeholders = ", ".join("?" for _ in sources)
        self.connection.execute(
            f"DELETE FROM impact_edges WHERE source_path IN ({placeholders})",
            tuple(sorted(sources)),
        )
        self._insert_edges_for_sources(sources)

    def _affected_sources(
        self,
        changed_paths: set[str],
        changes: dict[str, set[str]],
    ) -> set[str]:
        sources = set(changed_paths)
        path_placeholders = ", ".join("?" for _ in changed_paths)
        for row in self.connection.execute(
            f"SELECT DISTINCT source_path FROM impact_edges "
            f"WHERE target_path IN ({path_placeholders})",
            tuple(sorted(changed_paths)),
        ):
            sources.add(str(row["source_path"]))

        leaf_names = {name.rsplit(".", 1)[-1] for names in changes.values() for name in names}
        module_names = {_module_name(Path(path)) for path in changed_paths}
        for row in self.connection.execute("SELECT path, name, target_hint FROM refs"):
            target_hint = str(row["target_hint"] or "")
            if str(row["name"]) in leaf_names or any(
                target_hint == module or target_hint.startswith(f"{module}.")
                for module in module_names
                if module
            ):
                sources.add(str(row["path"]))
        return sources

    def _insert_edges_for_sources(self, sources: set[str] | None) -> None:
        rows = self.connection.execute(
            "SELECT path, qualified_name AS name FROM symbols"
        ).fetchall()
        by_leaf: dict[str, list[tuple[str, str]]] = {}
        modules: dict[str, str] = {}
        for row in rows:
            path, name = str(row["path"]), str(row["name"])
            by_leaf.setdefault(name.rsplit(".", 1)[-1], []).append((path, name))
            modules[_module_name(Path(path))] = path
        for row in self.connection.execute("SELECT path FROM files"):
            path = str(row["path"])
            modules[_module_name(Path(path))] = path
        edges = set()
        query = "SELECT path, name, line, owner, target_hint, relation FROM refs"
        parameters: tuple[str, ...] = ()
        if sources:
            placeholders = ", ".join("?" for _ in sources)
            query += f" WHERE path IN ({placeholders})"
            parameters = tuple(sorted(sources))
        for row in self.connection.execute(query, parameters):
            relation = str(row["relation"] or "references")
            targets = _resolve_targets(
                str(row["path"]),
                str(row["name"]),
                str(row["target_hint"] or row["name"]),
                relation,
                by_leaf,
                modules,
            )
            for target_path, target_symbol, confidence in targets:
                edges.add(
                    (
                        str(row["path"]),
                        str(
                            row["owner"]
                            or _owner_for_reference(
                                self.connection, str(row["path"]), int(row["line"])
                            )
                        ),
                        target_path,
                        target_symbol,
                        relation,
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

    def _test_companions(self, changed: Iterable[str]) -> set[str]:
        tests = [
            str(row["path"])
            for row in self.connection.execute("SELECT path FROM files")
            if _is_test(str(row["path"]))
        ]
        result: set[str] = set()
        for path in changed:
            changed_tokens = _name_tokens(path)
            changed_stem = _normalized_stem(path)
            for test in tests:
                test_tokens = _name_tokens(test)
                test_stem = _normalized_stem(test)
                single_token_prefix = len(changed_tokens) == 1 and test_stem.startswith(
                    f"{changed_stem}_"
                )
                if (
                    test_stem == changed_stem
                    or single_token_prefix
                    or (len(changed_tokens) >= 2 and changed_tokens <= test_tokens)
                ):
                    result.add(test)
        return result


def _owner_for_reference(connection: sqlite3.Connection, path: str, line: int) -> str:
    row = connection.execute(
        """SELECT qualified_name FROM symbols
           WHERE path = ? AND line_start <= ? AND line_end >= ?
           ORDER BY line_start DESC, line_end ASC LIMIT 1""",
        (path, line, line),
    ).fetchone()
    return str(row["qualified_name"]) if row is not None else "<module>"


def _resolve_targets(
    source_path: str,
    leaf: str,
    hint: str,
    relation: str,
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
    if relation == "imports":
        return []
    same_file = [item for item in candidates if item[0] == source_path]
    if same_file:
        return [(path, symbol, 0.95) for path, symbol in same_file[:3]]
    if len(candidates) == 1:
        return [(candidates[0][0], candidates[0][1], AMBIGUOUS_CONFIDENCE)]
    return [(path, symbol, AMBIGUOUS_CONFIDENCE) for path, symbol in candidates[:3]]


def _module_path(module: str, modules: dict[str, str]) -> str | None:
    while module:
        if module in modules:
            return modules[module]
        module = module.rpartition(".")[0]
    return None


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


def _test_rank(item: ImpactNode) -> tuple[int, float]:
    return item.distance, -item.confidence


def _is_exact_test_companion(path: str, changed_files: Iterable[str]) -> bool:
    test_stem = _normalized_stem(path)
    return any(test_stem == _normalized_stem(changed) for changed in changed_files)


def _name_tokens(path: str) -> set[str]:
    stem = _normalized_stem(path)
    return {part for part in stem.split("_") if len(part) >= 2}


def _normalized_stem(path: str) -> str:
    return Path(path).stem.casefold().removeprefix("test_")


def _risk_score(rows: list[Any], nodes: list[ImpactNode], changed: tuple[str, ...]) -> float:
    public = sum(1 for row in rows if not str(row["name"]).rsplit(".", 1)[-1].startswith("_"))
    confirmed = [node for node in nodes if node.confidence >= HIGH_CONFIDENCE]
    production = {node.path for node in confirmed if not _is_test(node.path)}
    score = 0.08 * min(len(changed), 3)
    score += 0.06 * min(public, 4)
    score += 0.08 * min(len(production), 4)
    score += 0.05 * min(max((node.distance for node in confirmed), default=0), 2)
    if any(node.relation in {"inherits", "imports"} for node in confirmed):
        score += 0.12
    if len({node.path for node in confirmed}) >= 6:
        score += 0.1
    return round(min(1.0, score), 2)


__all__ = [
    "AMBIGUOUS_CONFIDENCE",
    "HIGH_CONFIDENCE",
    "ImpactGraph",
    "ImpactNode",
    "ImpactReport",
]
