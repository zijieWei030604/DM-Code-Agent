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


@dataclass(frozen=True)
class ImpactSnapshot:
    path: str
    content: str
    content_hash: str
    symbols: list[dict[str, Any]]
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
    truncated: bool = False
    reason: str = ""
    before_hash: str = ""
    after_hash: str = ""

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

    def snapshot(self, path: Path) -> ImpactSnapshot | None:
        resolved = self._resolve(path)
        if resolved.suffix != ".py" or not resolved.is_file():
            return None
        text = resolved.read_text(encoding="utf-8")
        symbols = self.client.document_symbols(resolved, text) if self.client.available else []
        diagnostics = self.client.diagnostics(resolved, text) if self.client.available else []
        snapshot = ImpactSnapshot(
            path=self._relative(resolved),
            content=text,
            content_hash=_hash(text),
            symbols=symbols,
            diagnostics=diagnostics,
        )
        self._snapshots[resolved] = snapshot
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
            changed = _changed_symbols(
                before.symbols if before else [], symbols, before.content if before else "", text
            )
            candidates: set[str] = set()
            seen = 0
            truncated = False
            for symbol in changed:
                position = _symbol_position(symbol)
                if position is None:
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
            old_errors = {
                _diagnostic_key(item)
                for item in (before.diagnostics if before else [])
                if _is_error(item)
            }
            new_errors = [
                item
                for item in diagnostics
                if _is_error(item) and _diagnostic_key(item) not in old_errors
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
                truncated=truncated or len(candidates) > 10,
            )
            self._snapshots[resolved] = ImpactSnapshot(
                self._relative(resolved), text, _hash(text), symbols, diagnostics
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
    before_symbols: list[dict[str, Any]],
    after_symbols: list[dict[str, Any]],
    before: str,
    after: str,
) -> list[dict[str, Any]]:
    old_names = {_symbol_name(item) for item in before_symbols}
    changed_lines = _changed_line_numbers(before, after)
    chosen = [
        item
        for item in after_symbols
        if _symbol_name(item) not in old_names or _symbol_touches(item, changed_lines)
    ]
    return chosen or after_symbols[:1]


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
    return any(int(start.get("line", 0)) <= line <= int(end.get("line", 0)) for line in lines)


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


def _diagnostic_key(diagnostic: dict[str, Any]) -> str:
    return json.dumps(diagnostic, sort_keys=True, ensure_ascii=False)
