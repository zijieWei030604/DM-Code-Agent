"""CLI for deterministic Semantic Workspace impact evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .evaluation import evaluate_impact_manifest, write_impact_markdown, write_impact_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Semantic Workspace impact predictions.")
    parser.add_argument("--manifest", type=Path, required=True, help="Labeled case manifest.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_impact_manifest(args.manifest)
    if args.output:
        write_impact_report(report, args.output)
    if args.markdown:
        write_impact_markdown(report, args.markdown)
    affected = report.affected_files
    tests = report.related_tests
    print(f"cases={report.case_count}")
    print(
        "affected_files "
        f"precision={affected.precision:.3f} recall={affected.recall:.3f} f1={affected.f1:.3f} "
        f"exact={affected.exact_matches}/{affected.cases}"
    )
    print(
        "related_tests "
        f"precision={tests.precision:.3f} recall={tests.recall:.3f} f1={tests.f1:.3f} "
        f"exact={tests.exact_matches}/{tests.cases}"
    )
    print(f"test_recall_at_1={report.test_recall_at_1:.3f}")
    print(f"test_recall_at_3={report.test_recall_at_3:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
