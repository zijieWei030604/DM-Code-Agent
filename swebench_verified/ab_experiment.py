"""Run and score the pure-ReAct vs Planner+Evidence SWE-bench experiment."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluate import summarize


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _prediction_summary(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    steps = [
        int(row["dm_steps"])
        for row in rows
        if isinstance(row.get("dm_steps"), int) and not isinstance(row.get("dm_steps"), bool)
    ]
    durations = [
        float(row["dm_duration_seconds"])
        for row in rows
        if isinstance(row.get("dm_duration_seconds"), (int, float))
        and not isinstance(row.get("dm_duration_seconds"), bool)
    ]
    return {
        "runs": len(rows),
        "success": sum(row.get("dm_status") == "success" for row in rows),
        "max_steps": sum(row.get("dm_status") == "max_steps_exceeded" for row in rows),
        "empty_patch": sum(not str(row.get("model_patch") or "").strip() for row in rows),
        "mean_steps": round(statistics.fmean(steps), 2) if steps else None,
        "mean_duration_seconds": round(statistics.fmean(durations), 2) if durations else None,
    }


def _analysis_summary(path: Path) -> dict[str, Any]:
    analysis = json.loads(path.read_text(encoding="utf-8"))
    all_summary = analysis.get("summary", {}).get("all", {})
    trace = all_summary.get("trace", {})
    verification_gap = trace.get("verification_gap", {})
    plan_deviations = trace.get("plan_tool_deviations", {})
    evidence_statuses = Counter(
        str(instance.get("evidence_completion", {}).get("status") or "unmeasured")
        for instance in analysis.get("instances", [])
    )
    return {
        "verification_gap_count": verification_gap.get("true_count"),
        "verification_gap_measured": verification_gap.get("measured_count"),
        "plan_tool_deviations": plan_deviations.get("sum"),
        "evidence_completion_statuses": dict(sorted(evidence_statuses.items())),
    }


def _find_harness_report(root: Path, run_id: str) -> Path:
    matches = sorted(root.glob(f"*.{run_id}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one harness report for run_id={run_id}, found {len(matches)}")
    return matches[0]


def _render_markdown(result: dict[str, Any]) -> str:
    groups = result["groups"]
    pure = groups["pure_react"]
    full = groups["planner_evidence"]
    lines = [
        "# Planner + Evidence A/B",
        "",
        f"- Selection: `{result['selection_manifest']}`",
        f"- Instances: {result['limit']}",
        "",
        "| Group | Resolved | Rate | Empty patches | Verification gaps | Mean steps | Mean duration |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, group in (("Pure ReAct", pure), ("Planner + Evidence", full)):
        official = group["official"]
        prediction = group["prediction"]
        analysis = group["analysis"]
        lines.append(
            "| {label} | {resolved}/{submitted} | {rate:.1%} | {empty} | {gaps} | {steps} | {duration}s |".format(
                label=label,
                resolved=official["resolved"],
                submitted=official["submitted"],
                rate=official["resolve_rate"],
                empty=official["empty_patch"],
                gaps=analysis["verification_gap_count"],
                steps=prediction["mean_steps"],
                duration=prediction["mean_duration_seconds"],
            )
        )
    delta = result["delta"]
    lines.extend(
        [
            "",
            "## Delta (Planner + Evidence minus Pure ReAct)",
            "",
            f"- Resolved: {delta['resolved']:+d}",
            f"- Resolve rate: {delta['resolve_rate_percentage_points']:+.1f} percentage points",
            f"- Empty patches: {delta['empty_patch']:+d}",
            f"- Verification gaps: {delta['verification_gap']:+d}",
            f"- Mean steps: {delta['mean_steps']:+.2f}",
            "",
            "## Detailed reports",
            "",
            f"- Pure ReAct: `{pure['analysis_markdown']}`",
            f"- Planner + Evidence: `{full['analysis_markdown']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pure ReAct and Planner+Evidence on one fixed SWE-bench selection."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--name", default="planner-evidence-ab")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=900, help="LLM request timeout")
    parser.add_argument("--harness-timeout", type=int, default=1800)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--harness-python",
        type=Path,
        default=Path(".swebench-venv/Scripts/python.exe"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("swebench_work"))
    parser.add_argument("--report-dir", type=Path, default=Path("bench_reports"))
    parser.add_argument("--temp-root", type=Path, default=Path(r"E:\DevCache\Temp"))
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(r"E:\DevCache\dm-agent-workspaces")
    )
    parser.add_argument("--trace-root", type=Path, default=Path(r"E:\DevCache\dm-agent-traces"))
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    root = Path.cwd()
    harness_python = (root / args.harness_python).resolve()
    if not harness_python.is_file():
        parser.error(f"SWE-bench Python not found: {harness_python}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.name}-{args.limit}.selection.json"
    groups = {
        "pure_react": ["--disable-planning"],
        "planner_evidence": ["--enable-evidence-graph"],
    }
    artifacts: dict[str, dict[str, Any]] = {}
    common_env = os.environ.copy()
    common_env["TEMP"] = str(args.temp_root.resolve())
    common_env["TMP"] = str(args.temp_root.resolve())
    harness_env = common_env.copy()
    shim = str((root / "swebench_verified" / "_winshim").resolve())
    harness_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (shim, harness_env.get("PYTHONPATH", "")) if part
    )
    harness_env.setdefault("PYTHONUTF8", "1")

    for group, switches in groups.items():
        stem = f"{args.name}-{args.limit}-{group.replace('_', '-')}"
        predictions = args.output_dir / f"{stem}.jsonl"
        trace_dir = args.trace_root / stem
        workspace = args.workspace_root / stem
        run_command = [
            sys.executable,
            "-m",
            "swebench_verified.run",
            "--limit",
            str(args.limit),
            "--selection-manifest",
            str(manifest),
            "--output",
            str(predictions),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--max-steps",
            str(args.max_steps),
            "--timeout",
            str(args.timeout),
            "--workspace-root",
            str(workspace),
            "--trace-dir",
            str(trace_dir),
            *switches,
        ]
        if args.resume:
            run_command.append("--resume")
        _run(run_command, cwd=root, env=common_env)

        run_id = stem
        _run(
            [
                str(harness_python),
                "-m",
                "swebench_verified.evaluate",
                "--predictions",
                str(predictions),
                "--run-id",
                run_id,
                "--max-workers",
                str(args.max_workers),
                "--timeout",
                str(args.harness_timeout),
            ],
            cwd=root,
            env=harness_env,
        )
        harness_report = _find_harness_report(root, run_id)
        analysis_json = args.report_dir / f"{stem}-analysis.json"
        analysis_markdown = args.report_dir / f"{stem}-analysis.md"
        _run(
            [
                sys.executable,
                "-m",
                "swebench_verified.analyze",
                "--predictions",
                str(predictions),
                "--report",
                str(harness_report),
                "--manifest",
                str(manifest),
                "--trace-dir",
                str(trace_dir),
                "--json",
                str(analysis_json),
                "--markdown",
                str(analysis_markdown),
            ],
            cwd=root,
            env=common_env,
        )
        artifacts[group] = {
            "prediction": _prediction_summary(predictions),
            "official": summarize(harness_report),
            "analysis": _analysis_summary(analysis_json),
            "predictions": str(predictions),
            "trace_dir": str(trace_dir),
            "harness_report": str(harness_report),
            "analysis_json": str(analysis_json),
            "analysis_markdown": str(analysis_markdown),
        }

    pure = artifacts["pure_react"]
    full = artifacts["planner_evidence"]
    pure_steps = pure["prediction"]["mean_steps"] or 0.0
    full_steps = full["prediction"]["mean_steps"] or 0.0
    pure_gaps = pure["analysis"]["verification_gap_count"] or 0
    full_gaps = full["analysis"]["verification_gap_count"] or 0
    result = {
        "schema_version": 1,
        "name": args.name,
        "limit": args.limit,
        "selection_manifest": str(manifest),
        "groups": artifacts,
        "delta": {
            "resolved": full["official"]["resolved"] - pure["official"]["resolved"],
            "resolve_rate_percentage_points": round(
                100 * (full["official"]["resolve_rate"] - pure["official"]["resolve_rate"]),
                2,
            ),
            "empty_patch": full["official"]["empty_patch"] - pure["official"]["empty_patch"],
            "verification_gap": full_gaps - pure_gaps,
            "mean_steps": round(full_steps - pure_steps, 2),
        },
    }
    summary_json = args.report_dir / f"{args.name}-{args.limit}-summary.json"
    summary_markdown = args.report_dir / f"{args.name}-{args.limit}-summary.md"
    summary_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_markdown.write_text(_render_markdown(result), encoding="utf-8")
    print(f"\nA/B JSON -> {summary_json}")
    print(f"A/B Markdown -> {summary_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
