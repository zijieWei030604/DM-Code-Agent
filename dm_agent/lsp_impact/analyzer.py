"""Deterministic orchestration around LSP facts for changed Python files."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .client import LspClient

REFERENCE_RETRY_COUNT = 4
REFERENCE_RETRY_DELAY_SECONDS = 0.5
CALLABLE_SYMBOL_KINDS = frozenset({6, 9, 12})  # Method, Constructor, Function
DATA_SYMBOL_KINDS = frozenset({7, 8, 13, 14})  # Property, Field, Variable, Constant
CLASS_SYMBOL_KIND = 5

# These diagnostics identify a definite defect in the edited expression itself.
# Broad type-analysis warnings vary across legacy repositories and remain observations.
BLOCKING_DIAGNOSTIC_CODES = frozenset(
    {"reportUndefinedVariable", "reportUnboundVariable", "reportInvalidSyntax"}
)


@dataclass(frozen=True)
class ImpactSnapshot:
    path: str
    content: str
    content_hash: str
    diagnostics: list[dict[str, Any]]


@dataclass
class ImpactReport:
    report_id: str
    path: str
    status: str
    elapsed_ms: int
    changed_symbols: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)
    references_seen: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    new_error_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    blocking_error_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    uncertain_error_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    reason: str = ""
    before_hash: str = ""
    after_hash: str = ""
    unmapped_changed_lines: list[int] = field(default_factory=list)
    unqueryable_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImpactAnalyzer:
    """Captures per-file pre-state and builds bounded LSP impact reports."""

    def __init__(
        self, client: LspClient, workspace_root: Path, *, max_references: int = 200
    ) -> None:
        self.client = client
        self.workspace_root = workspace_root.resolve()
        self.max_references = max_references
        self._snapshots: dict[Path, ImpactSnapshot] = {}
        # A task has one diagnostic baseline per file.  Replacing this with the
        # most recent snapshot would make an error introduced by an earlier edit
        # look "old" after a second edit, which is unsafe at completion time.
        self._initial_diagnostics: dict[Path, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        """Discard run-scoped snapshots before a new Agent task starts."""
        self._snapshots.clear()
        self._initial_diagnostics.clear()

    def snapshot(self, path: Path) -> ImpactSnapshot | None:
        resolved = self._resolve(path)
        if resolved.suffix != ".py" or not resolved.is_file():
            return None
        text = resolved.read_text(encoding="utf-8")
        diagnostics = self.client.diagnostics(resolved, text) if self.client.available else []
        snapshot = ImpactSnapshot(
            path=self._relative(resolved),
            content=text,
            content_hash=_hash(text),
            diagnostics=diagnostics,
        )
        self._snapshots[resolved] = snapshot
        self._initial_diagnostics.setdefault(resolved, diagnostics)
        return snapshot

    def analyze(self, path: Path) -> ImpactReport:
        started = time.monotonic()
        resolved = self._resolve(path)
        before = self._snapshots.get(resolved)
        if not self.client.available:
            return self._report(
                resolved, "unavailable", started, reason=self.client.unavailable_reason
            )
        if resolved.suffix != ".py":
            return self._report(
                resolved, "skipped", started, reason="only Python files are analyzed"
            )
        if not resolved.is_file():
            return self._report(resolved, "deleted", started, before=before)
        try:
            text = resolved.read_text(encoding="utf-8")
            symbols = self.client.document_symbols(resolved, text)
            diagnostics = self.client.diagnostics(resolved, text)
            before_text = before.content if before else ""
            changed_lines = _changed_line_numbers(before_text, text)
            changed, unmapped_lines = _changed_symbols(symbols, changed_lines)
            unqueryable: list[str] = []
            candidates: set[str] = set()
            seen = 0
            truncated = False
            for symbol in changed:
                position = _symbol_position(symbol)
                if position is None:
                    unqueryable.append(_symbol_name(symbol))
                    continue
                references = self._references_with_retry(resolved, position, text)
                for reference in references:
                    seen += 1
                    if seen > self.max_references:
                        truncated = True
                        break
                    candidate = _location_path(reference, self.workspace_root)
                    if candidate and candidate != self._relative(resolved):
                        candidates.add(candidate)
                if truncated:
                    break
            baseline = self._initial_diagnostics.get(resolved)
            # A newly-created Python file has no pre-write document. Its empty
            # diagnostic baseline is still meaningful: every current Error is
            # attributable to the task's new file.
            if baseline is None:
                baseline = []
                self._initial_diagnostics[resolved] = baseline
            old_errors = {_diagnostic_key(item) for item in baseline if _is_error(item)}
            new_errors = [
                item
                for item in diagnostics
                if _is_error(item) and _diagnostic_key(item) not in old_errors
            ]
            blocking_errors = [
                item
                for item in new_errors
                if _is_code_error(item) and _diagnostic_touches_changed_lines(item, changed_lines)
            ]
            report = self._report(
                resolved,
                "ok",
                started,
                before=before,
                changed_symbols=[_symbol_name(item) for item in changed],
                candidate_files=sorted(candidates)[:10],
                references_seen=seen,
                diagnostics=diagnostics,
                new_error_diagnostics=new_errors,
                blocking_error_diagnostics=blocking_errors,
                uncertain_error_diagnostics=[
                    item for item in new_errors if item not in blocking_errors
                ],
                truncated=truncated or len(candidates) > 10,
                unmapped_changed_lines=unmapped_lines,
                unqueryable_symbols=unqueryable,
            )
            self._snapshots[resolved] = ImpactSnapshot(
                self._relative(resolved), text, _hash(text), diagnostics
            )
            return report
        except (OSError, RuntimeError, TimeoutError, UnicodeError) as error:
            return self._report(resolved, "unavailable", started, before=before, reason=str(error))

    def _references_with_retry(
        self, path: Path, position: tuple[int, int], text: str
    ) -> list[dict[str, Any]]:
        """Wait briefly for a cold workspace index to expose non-declaration references."""
        references = self.client.references(path, position[0], position[1], text)
        for _ in range(REFERENCE_RETRY_COUNT):
            if not _empty_or_declaration_only(references, path, position[0]):
                break
            time.sleep(REFERENCE_RETRY_DELAY_SECONDS)
            references = self.client.references(path, position[0], position[1], text)
        return references

    def _report(self, path: Path, status: str, started: float, **kwargs: Any) -> ImpactReport:
        before = kwargs.pop("before", self._snapshots.get(path))
        after_hash = ""
        if path.is_file():
            with suppress(OSError):
                after_hash = _hash(path.read_text(encoding="utf-8"))
        report_id = hashlib.sha256(
            f"{self._relative(path)}:{before.content_hash if before else ''}:{after_hash}".encode()
        ).hexdigest()[:16]
        return ImpactReport(
            report_id=report_id,
            path=self._relative(path),
            status=status,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            before_hash=before.content_hash if before else "",
            after_hash=after_hash,
            **kwargs,
        )

    def _resolve(self, path: Path) -> Path:
        return (path if path.is_absolute() else self.workspace_root / path).resolve()

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _symbol_name(symbol: dict[str, Any]) -> str:
    return str(symbol.get("name", "<anonymous>"))


def _symbol_position(symbol: dict[str, Any]) -> tuple[int, int] | None:
    selection = symbol.get("selectionRange") or symbol.get("range")
    if not isinstance(selection, dict):
        return None
    start = selection.get("start")
    if not isinstance(start, dict):
        return None
    return int(start.get("line", 0)), int(start.get("character", 0))


def _changed_symbols(
    after_symbols: list[dict[str, Any]],
    changed_lines: set[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select reference targets and expose unmapped lines (one-based).

    Local data changes belong to the nearest callable. Module data and class
    members remain independently queryable. No arbitrary symbol is used when
    LSP cannot map a changed line, including a deletion boundary, to a target.
    """
    flattened = _nested_symbols(after_symbols)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    unmapped: list[int] = []
    for line in sorted(changed_lines):
        matching = [lineage for symbol, lineage in flattened if _symbol_touches(symbol, {line})]
        if not matching:
            unmapped.append(line + 1)
            continue
        deepest = max(len(lineage) for lineage in matching)
        mapped = False
        for lineage in matching:
            if len(lineage) != deepest:
                continue
            target = _reference_target(lineage)
            if target is None:
                continue
            mapped = True
            identity = id(target)
            if identity not in chosen_ids:
                chosen.append(target)
                chosen_ids.add(identity)
        if not mapped:
            unmapped.append(line + 1)
    return chosen, unmapped


def _nested_symbols(
    symbols: list[dict[str, Any]], ancestors: tuple[dict[str, Any], ...] = ()
) -> list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]]:
    """Flatten DocumentSymbol trees with their enclosing scopes."""
    result: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = []
    for symbol in symbols:
        lineage = (*ancestors, symbol)
        result.append((symbol, lineage))
        children = symbol.get("children")
        if isinstance(children, list):
            nested = [item for item in children if isinstance(item, dict)]
            result.extend(_nested_symbols(nested, lineage))
    return result


def _reference_target(lineage: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    for index in range(len(lineage) - 1, -1, -1):
        symbol = lineage[index]
        kind = symbol.get("kind")
        if kind == CLASS_SYMBOL_KIND or kind in CALLABLE_SYMBOL_KINDS:
            return symbol
        if kind in DATA_SYMBOL_KINDS:
            for ancestor in reversed(lineage[:index]):
                if ancestor.get("kind") in CALLABLE_SYMBOL_KINDS:
                    return ancestor
                if ancestor.get("kind") == CLASS_SYMBOL_KIND:
                    break
            return symbol
    return None


def _changed_line_numbers(before: str, after: str) -> set[int]:
    before_lines, after_lines = before.splitlines(), after.splitlines()
    import difflib

    result: set[int] = set()
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=before_lines, b=after_lines).get_opcodes():
        if tag != "equal":
            result.update(range(j1, max(j1 + 1, j2)))
    return result


def _symbol_touches(symbol: dict[str, Any], lines: set[int]) -> bool:
    value = symbol.get("range")
    if not isinstance(value, dict):
        return False
    start, end = value.get("start"), value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return False
    first, last = int(start.get("line", 0)), int(end.get("line", 0))
    # LSP ends are exclusive: an end at column zero does not cover that line.
    if end.get("character") == 0:
        last -= 1
    return any(first <= line <= last for line in lines)


def _location_path(location: dict[str, Any], root: Path) -> str:
    uri = location.get("uri")
    if not isinstance(uri, str) or not uri.startswith("file:"):
        return ""
    candidate = Path(unquote(urlparse(uri).path.lstrip("/")))
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _empty_or_declaration_only(references: list[dict[str, Any]], path: Path, line: int) -> bool:
    if not references:
        return True
    for reference in references:
        uri = reference.get("uri")
        if not isinstance(uri, str) or uri != path.resolve().as_uri():
            return False
        value = reference.get("range")
        start = value.get("start") if isinstance(value, dict) else None
        if not isinstance(start, dict) or int(start.get("line", -1)) != line:
            return False
    return True


def _is_error(diagnostic: dict[str, Any]) -> bool:
    return int(diagnostic.get("severity", 1)) == 1


def _is_code_error(diagnostic: dict[str, Any]) -> bool:
    """Return whether Pyright identifies a definite local code defect."""
    code = str(diagnostic.get("code", ""))
    # Test doubles and parser diagnostics may omit a code. They remain blocking
    # only when their range overlaps the edited code.
    return not code or code in BLOCKING_DIAGNOSTIC_CODES


def _diagnostic_touches_changed_lines(diagnostic: dict[str, Any], changed_lines: set[int]) -> bool:
    value = diagnostic.get("range")
    if not isinstance(value, dict):
        return True
    start, end = value.get("start"), value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return True
    first, last = int(start.get("line", 0)), int(end.get("line", 0))
    return any(first <= line <= last for line in changed_lines)


def _diagnostic_key(diagnostic: dict[str, Any]) -> str:
    """Identify a diagnostic without volatile source coordinates.

    Inserting lines above an unchanged legacy issue shifts its LSP range. The
    range must not turn that old issue into a newly introduced one.
    """
    stable = {
        key: diagnostic.get(key)
        for key in ("severity", "code", "message", "source")
        if key in diagnostic
    }
    return json.dumps(stable, sort_keys=True, ensure_ascii=False)
