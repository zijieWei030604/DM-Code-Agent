"""Deterministic evaluation for semantic change-impact predictions."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .engine import SemanticWorkspaceEngine


@dataclass(frozen=True)
class ImpactEvaluationCase:
    case_id: str
    description: str
    files: dict[str, str]
    changed_files: tuple[str, ...]
    expected_affected_files: tuple[str, ...]
    expected_related_tests: tuple[str, ...]
    evaluate_affected_files: bool = True
    evaluate_related_tests: bool = True


@dataclass(frozen=True)
class SetMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    exact_matches: int
    cases: int


@dataclass(frozen=True)
class ImpactCaseResult:
    case_id: str
    description: str
    changed_files: tuple[str, ...]
    expected_affected_files: tuple[str, ...]
    predicted_affected_files: tuple[str, ...]
    expected_related_tests: tuple[str, ...]
    predicted_related_tests: tuple[str, ...]
    affected_false_positives: tuple[str, ...]
    affected_false_negatives: tuple[str, ...]
    test_false_positives: tuple[str, ...]
    test_false_negatives: tuple[str, ...]
    evaluate_affected_files: bool
    evaluate_related_tests: bool


@dataclass(frozen=True)
class ImpactEvaluationReport:
    manifest: str
    case_count: int
    affected_files: SetMetrics
    related_tests: SetMetrics
    test_recall_at_1: float
    test_recall_at_3: float
    results: tuple[ImpactCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_impact_cases(path: str | Path) -> list[ImpactEvaluationCase]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("影响分析评测清单必须包含 version=1 和 cases 列表")
    repository_mode = bool(payload.get("repository_root"))
    cases: list[ImpactEvaluationCase] = []
    seen: set[str] = set()
    for raw in payload["cases"]:
        if not isinstance(raw, dict):
            raise ValueError("每个影响分析评测案例必须是 JSON 对象")
        case_id = str(raw.get("id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError(f"案例 id 为空或重复：{case_id!r}")
        seen.add(case_id)
        files = raw.get("files")
        if not repository_mode and (not isinstance(files, dict) or not files):
            raise ValueError(f"案例 {case_id} 必须声明非空 files")
        normalized_files = {
            _safe_relative_path(name): str(content)
            for name, content in (files.items() if isinstance(files, dict) else ())
        }
        changed = tuple(_safe_relative_path(item) for item in raw.get("changed_files", []))
        if not changed or (
            not repository_mode and any(item not in normalized_files for item in changed)
        ):
            raise ValueError(f"案例 {case_id} 的 changed_files 必须引用 files 中的文件")
        expected_files = tuple(
            sorted(_safe_relative_path(item) for item in raw.get("affected_files", []))
        )
        expected_tests = tuple(
            sorted(_safe_relative_path(item) for item in raw.get("related_tests", []))
        )
        missing_labels = [
            item for item in (*expected_files, *expected_tests) if item not in normalized_files
        ]
        if missing_labels and not repository_mode:
            raise ValueError(f"案例 {case_id} 的人工标注必须引用 files 中的文件：{missing_labels}")
        if any(_is_test_path(item) for item in expected_files):
            raise ValueError(f"案例 {case_id} 的 affected_files 不应包含测试文件")
        if any(not _is_test_path(item) for item in expected_tests):
            raise ValueError(f"案例 {case_id} 的 related_tests 只能包含测试文件")
        cases.append(
            ImpactEvaluationCase(
                case_id=case_id,
                description=str(raw.get("description") or ""),
                files=normalized_files,
                changed_files=changed,
                expected_affected_files=expected_files,
                expected_related_tests=expected_tests,
                evaluate_affected_files=bool(raw.get("evaluate_affected_files", True)),
                evaluate_related_tests=bool(raw.get("evaluate_related_tests", True)),
            )
        )
    return cases


def evaluate_impact_manifest(path: str | Path) -> ImpactEvaluationReport:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = load_impact_cases(manifest_path)
    results: list[ImpactCaseResult] = []
    with tempfile.TemporaryDirectory(prefix="dm-agent-impact-eval-") as temp:
        temp_root = Path(temp)
        repository_root_value = str(payload.get("repository_root") or "").strip()
        if repository_root_value:
            repository_root = (manifest_path.parent / repository_root_value).resolve()
            if not repository_root.is_dir():
                raise ValueError(f"真实仓库评测根目录不存在：{repository_root}")
            _validate_repository_cases(repository_root, cases)
            engine = SemanticWorkspaceEngine(
                repository_root,
                database_path=temp_root / "repository.db",
                max_scan_files=int(payload.get("max_scan_files", 2000)),
            )
            try:
                engine.update()
                results.extend(_evaluate_case(engine, case) for case in cases)
            finally:
                engine.close()
        else:
            for index, case in enumerate(cases):
                case_root = temp_root / f"case-{index:03d}"
                _materialize_case(case_root, case)
                engine = SemanticWorkspaceEngine(case_root, database_path=case_root / "index.db")
                try:
                    engine.update()
                    results.append(_evaluate_case(engine, case))
                finally:
                    engine.close()
    return ImpactEvaluationReport(
        manifest=str(manifest_path),
        case_count=len(results),
        affected_files=_aggregate_metrics(
            [
                (item.expected_affected_files, item.predicted_affected_files)
                for item in results
                if item.evaluate_affected_files
            ]
        ),
        related_tests=_aggregate_metrics(
            [
                (item.expected_related_tests, item.predicted_related_tests)
                for item in results
                if item.evaluate_related_tests
            ]
        ),
        test_recall_at_1=_recall_at_k(results, 1),
        test_recall_at_3=_recall_at_k(results, 3),
        results=tuple(results),
    )


def write_impact_report(report: ImpactEvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_impact_markdown(report: ImpactEvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    affected = report.affected_files
    tests = report.related_tests
    lines = [
        "# Semantic Workspace Impact Evaluation",
        "",
        f"- Cases: {report.case_count}",
        (
            "- Affected files: "
            f"precision {affected.precision:.2%}, recall {affected.recall:.2%}, "
            f"F1 {affected.f1:.2%}, exact {affected.exact_matches}/{affected.cases}"
        ),
        (
            "- Related tests: "
            f"precision {tests.precision:.2%}, recall {tests.recall:.2%}, "
            f"F1 {tests.f1:.2%}, exact {tests.exact_matches}/{tests.cases}"
        ),
        f"- Test recall@1: {report.test_recall_at_1:.2%}",
        f"- Test recall@3: {report.test_recall_at_3:.2%}",
        "",
        "## Case Details",
        "",
    ]
    for item in report.results:
        lines.extend(
            [
                f"### {item.case_id}",
                "",
                item.description,
                "",
                f"- Affected false positives: {', '.join(item.affected_false_positives) or 'none'}",
                f"- Affected false negatives: {', '.join(item.affected_false_negatives) or 'none'}",
                f"- Test false positives: {', '.join(item.test_false_positives) or 'none'}",
                f"- Test false negatives: {', '.join(item.test_false_negatives) or 'none'}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def _materialize_case(root: Path, case: ImpactEvaluationCase) -> None:
    root.mkdir(parents=True)
    for relative, content in case.files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _evaluate_case(engine: SemanticWorkspaceEngine, case: ImpactEvaluationCase) -> ImpactCaseResult:
    impact = engine.analyze_impact(case.changed_files)
    predicted_tests = tuple(impact.related_tests)
    predicted_files = tuple(
        sorted(
            {
                item.path
                for item in impact.confirmed_symbols
                if item.path not in impact.changed_files and not _is_test_path(item.path)
            }
        )
    )
    expected_files = set(case.expected_affected_files)
    expected_tests = set(case.expected_related_tests)
    actual_files = set(predicted_files)
    actual_tests = set(predicted_tests)
    return ImpactCaseResult(
        case_id=case.case_id,
        description=case.description,
        changed_files=case.changed_files,
        expected_affected_files=case.expected_affected_files,
        predicted_affected_files=predicted_files,
        expected_related_tests=case.expected_related_tests,
        predicted_related_tests=predicted_tests,
        affected_false_positives=(
            tuple(sorted(actual_files - expected_files)) if case.evaluate_affected_files else ()
        ),
        affected_false_negatives=(
            tuple(sorted(expected_files - actual_files)) if case.evaluate_affected_files else ()
        ),
        test_false_positives=(
            tuple(sorted(actual_tests - expected_tests)) if case.evaluate_related_tests else ()
        ),
        test_false_negatives=(
            tuple(sorted(expected_tests - actual_tests)) if case.evaluate_related_tests else ()
        ),
        evaluate_affected_files=case.evaluate_affected_files,
        evaluate_related_tests=case.evaluate_related_tests,
    )


def _validate_repository_cases(root: Path, cases: list[ImpactEvaluationCase]) -> None:
    for case in cases:
        for item in (
            *case.changed_files,
            *case.expected_affected_files,
            *case.expected_related_tests,
        ):
            if not (root / item).is_file():
                raise ValueError(f"案例 {case.case_id} 引用的仓库文件不存在：{item}")


def _aggregate_metrics(pairs: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> SetMetrics:
    true_positives = false_positives = false_negatives = exact_matches = 0
    for expected_items, predicted_items in pairs:
        expected = set(expected_items)
        predicted = set(predicted_items)
        true_positives += len(expected & predicted)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        exact_matches += expected == predicted
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return SetMetrics(
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
        exact_matches,
        len(pairs),
    )


def _recall_at_k(results: list[ImpactCaseResult], k: int) -> float:
    hits = total = 0
    for result in results:
        if not result.evaluate_related_tests:
            continue
        expected = set(result.expected_related_tests)
        hits += len(expected & set(result.predicted_related_tests[:k]))
        total += len(expected)
    return _ratio(hits, total)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def _safe_relative_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"评测文件路径必须位于案例工作区内：{value!r}")
    return path.as_posix()


def _is_test_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return "tests" in {part.casefold() for part in candidate.parts} or candidate.name.startswith(
        "test_"
    )


__all__ = [
    "ImpactCaseResult",
    "ImpactEvaluationCase",
    "ImpactEvaluationReport",
    "SetMetrics",
    "evaluate_impact_manifest",
    "load_impact_cases",
    "write_impact_markdown",
    "write_impact_report",
]
