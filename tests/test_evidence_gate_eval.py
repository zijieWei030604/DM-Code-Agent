from dm_agent.evals.evidence_gate_eval import evaluate_gate


def test_evidence_gate_eval_reports_the_completion_policy_confusion_matrix() -> None:
    report = evaluate_gate()
    assert report["summary"] == {
        "scenarios": 5,
        "contract_passes": 5,
        "expected_blocks": 2,
        "actual_blocks": 2,
        "true_positives": 2,
        "false_negatives": 0,
        "false_positives": 0,
        "true_negatives": 3,
    }
