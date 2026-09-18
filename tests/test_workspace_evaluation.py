from __future__ import annotations

import json
from pathlib import Path

import pytest

from dm_agent.workspace.evaluation import (
    evaluate_impact_manifest,
    load_impact_cases,
    write_impact_markdown,
    write_impact_report,
)


def _write_manifest(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"version": 1, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_impact_evaluation_reports_false_positives_and_false_negatives(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "cases.json",
        [
            {
                "id": "direct",
                "description": "direct dependency",
                "files": {
                    "service.py": "def value():\n    return 1\n",
                    "consumer.py": (
                        "from service import value\n\ndef consume():\n    return value()\n"
                    ),
                    "missing.py": "VALUE = 1\n",
                    "tests/test_consumer.py": (
                        "from consumer import consume\n\ndef test_consume():\n"
                        "    assert consume() == 1\n"
                    ),
                    "tests/test_missing.py": "def test_missing():\n    assert True\n",
                },
                "changed_files": ["service.py"],
                "affected_files": ["consumer.py", "missing.py"],
                "related_tests": ["tests/test_consumer.py", "tests/test_missing.py"],
            }
        ],
    )

    report = evaluate_impact_manifest(manifest)

    assert report.case_count == 1
    assert report.affected_files.true_positives == 1
    assert report.affected_files.false_negatives == 1
    assert report.related_tests.true_positives == 1
    assert report.related_tests.false_negatives == 1
    assert report.results[0].affected_false_negatives == ("missing.py",)
    assert report.results[0].test_false_negatives == ("tests/test_missing.py",)


def test_impact_evaluation_writes_json_and_markdown(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "cases.json",
        [
            {
                "id": "isolated",
                "description": "isolated module",
                "files": {"module.py": "VALUE = 1\n"},
                "changed_files": ["module.py"],
                "affected_files": [],
                "related_tests": [],
            }
        ],
    )
    report = evaluate_impact_manifest(manifest)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_impact_report(report, json_path)
    write_impact_markdown(report, markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["case_count"] == 1
    assert "Affected files" in markdown_path.read_text(encoding="utf-8")


def test_impact_manifest_rejects_escape_paths(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "cases.json",
        [
            {
                "id": "escape",
                "files": {"../outside.py": "VALUE = 1\n"},
                "changed_files": ["../outside.py"],
            }
        ],
    )

    with pytest.raises(ValueError, match="工作区内"):
        load_impact_cases(manifest)


def test_repository_manifest_indexes_existing_workspace_and_supports_scoped_axes(
    tmp_path: Path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "service.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (repository / "consumer.py").write_text(
        "from service import value\n\ndef consume():\n    return value()\n", encoding="utf-8"
    )
    tests = repository / "tests"
    tests.mkdir()
    (tests / "test_consumer.py").write_text(
        "from consumer import consume\n\ndef test_consume():\n    assert consume() == 1\n",
        encoding="utf-8",
    )
    manifest = repository / "cases.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "repository_root": ".",
                "cases": [
                    {
                        "id": "repository-case",
                        "changed_files": ["service.py"],
                        "affected_files": [],
                        "related_tests": ["tests/test_consumer.py"],
                        "evaluate_affected_files": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_impact_manifest(manifest)

    assert report.affected_files.cases == 0
    assert report.related_tests.exact_matches == 1
    assert report.results[0].predicted_related_tests == ("tests/test_consumer.py",)
    assert report.results[0].evaluate_affected_files is False


def test_checked_in_impact_manifest_is_well_formed():
    root = Path(__file__).resolve().parents[1]
    cases = load_impact_cases(root / "benchmarks/semantic_workspace/impact_cases.json")

    assert len(cases) == 15
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.case_id == "dynamic_getattr_call" for case in cases)
