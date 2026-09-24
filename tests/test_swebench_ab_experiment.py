from __future__ import annotations

import json

from swebench_verified.ab_experiment import (
    _analysis_summary,
    _prediction_summary,
    _render_markdown,
)


def test_ab_prediction_summary_aggregates_agent_metrics(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "dm_status": "success",
            "model_patch": "diff",
            "dm_steps": 10,
            "dm_duration_seconds": 20,
        },
        {
            "dm_status": "max_steps_exceeded",
            "model_patch": "",
            "dm_steps": 20,
            "dm_duration_seconds": 40,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = _prediction_summary(path)

    assert summary == {
        "runs": 2,
        "success": 1,
        "max_steps": 1,
        "empty_patch": 1,
        "mean_steps": 15.0,
        "mean_duration_seconds": 30.0,
    }


def test_ab_analysis_summary_extracts_audit_metrics(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "all": {
                        "trace": {
                            "verification_gap": {"true_count": 2, "measured_count": 3},
                            "plan_tool_deviations": {"sum": 7},
                        }
                    }
                },
                "instances": [
                    {"evidence_completion": {"status": "tested"}},
                    {"evidence_completion": {"status": "not_run"}},
                    {"evidence_completion": {"status": "tested"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _analysis_summary(path) == {
        "verification_gap_count": 2,
        "verification_gap_measured": 3,
        "plan_tool_deviations": 7,
        "evidence_completion_statuses": {"not_run": 1, "tested": 2},
    }


def test_ab_markdown_renders_acceptance_delta():
    group = {
        "official": {"resolved": 1, "submitted": 2, "resolve_rate": 0.5, "empty_patch": 0},
        "prediction": {"mean_steps": 8.0, "mean_duration_seconds": 12.0},
        "analysis": {"verification_gap_count": 1},
        "analysis_markdown": "detail.md",
    }
    result = {
        "selection_manifest": "selection.json",
        "limit": 2,
        "groups": {"pure_react": group, "planner_evidence": group},
        "delta": {
            "resolved": 0,
            "resolve_rate_percentage_points": 0.0,
            "empty_patch": 0,
            "verification_gap": 0,
            "mean_steps": 0.0,
        },
    }

    rendered = _render_markdown(result)

    assert "Planner + Evidence A/B" in rendered
    assert "Verification gaps" in rendered
    assert "Resolved: +0" in rendered
