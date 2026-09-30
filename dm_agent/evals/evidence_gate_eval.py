"""Deterministic evaluation for the low-intervention evidence completion gate.

This module does not call a model or modify a workspace.  It turns the policy
contract into a small confusion matrix: confirmed current failures must block;
missing or unavailable local verification must warn; a passing check must allow.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dm_agent.core.evidence import EvidenceGraph
from dm_agent.core.evidence_policy import EvidenceCompletionPolicy


@dataclass(frozen=True)
class GateScenario:
    name: str
    expected: str
    build: Callable[[], EvidenceGraph]


def _changed_graph() -> EvidenceGraph:
    graph = EvidenceGraph("Update service")
    graph.workspace_version = "v1"
    graph.add_observation(
        tool="read_file", path="service.py", step_number=1, succeeded=True, workspace_version="v1"
    )
    graph.add_change(
        tool="edit_file", path="service.py", step_number=2, before_version="v1", after_version="v2"
    )
    return graph


def _passing_check() -> EvidenceGraph:
    graph = _changed_graph()
    graph.add_verification(tool="run_tests", step_number=3, passed=True, workspace_version="v2")
    return graph


def _failed_check(*, syntax: bool = False) -> EvidenceGraph:
    graph = _changed_graph()
    graph.add_verification(
        tool="run_python" if syntax else "run_tests",
        step_number=3,
        passed=False,
        workspace_version="v2",
        failure_kind="syntax_error" if syntax else "test_failure",
        blocking=True,
    )
    return graph


def _unavailable_check() -> EvidenceGraph:
    graph = _changed_graph()
    graph.add_verification(
        tool="run_tests",
        step_number=3,
        passed=False,
        workspace_version="v2",
        blocking=False,
        execution_status="invalid",
        outcome="unavailable",
    )
    return graph


def builtin_scenarios() -> tuple[GateScenario, ...]:
    return (
        GateScenario("passing_check", "allow", _passing_check),
        GateScenario("confirmed_test_failure", "block", _failed_check),
        GateScenario("current_syntax_error", "block", lambda: _failed_check(syntax=True)),
        GateScenario("no_local_verification", "warn", _changed_graph),
        GateScenario("unavailable_verification", "warn", _unavailable_check),
    )


def evaluate_gate() -> dict[str, object]:
    policy = EvidenceCompletionPolicy()
    results = []
    for scenario in builtin_scenarios():
        outcome = policy.evaluate(scenario.build())
        results.append(
            {
                "scenario": scenario.name,
                "expected": scenario.expected,
                "actual": outcome.decision,
                "passed": scenario.expected == outcome.decision,
                "verification_state": outcome.verification_state,
            }
        )
    expected_block = [item for item in results if item["expected"] == "block"]
    expected_nonblock = [item for item in results if item["expected"] != "block"]

    def actual_block(item: dict[str, object]) -> bool:
        return item["actual"] == "block"

    summary = {
        "scenarios": len(results),
        "contract_passes": sum(bool(item["passed"]) for item in results),
        "expected_blocks": len(expected_block),
        "actual_blocks": sum(actual_block(item) for item in results),
        "true_positives": sum(actual_block(item) for item in expected_block),
        "false_negatives": sum(not actual_block(item) for item in expected_block),
        "false_positives": sum(actual_block(item) for item in expected_nonblock),
        "true_negatives": sum(not actual_block(item) for item in expected_nonblock),
    }
    return {"mode": "evidence_gate_eval", "summary": summary, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic evidence-gate evaluation.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args(argv)
    report = evaluate_gate()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["summary"]["contract_passes"] == report["summary"]["scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
