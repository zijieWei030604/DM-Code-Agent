"""Explicit offline inspection, workspace restore, and human reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dm_agent.core.call_journal import CallJournal

from .points import RecoveryPoints
from .storage import WorkspaceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--store", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--store", type=Path, required=True)
    restore.add_argument("--point", required=True)
    restore.add_argument("--target", type=Path, required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--journal", type=Path, required=True)
    reconcile.add_argument("--call", required=True)
    reconcile.add_argument(
        "--outcome", choices=("verified", "compensated", "not_executed"), required=True
    )
    reconcile.add_argument("--note", required=True)
    args = parser.parse_args()
    try:
        if args.command == "reconcile":
            if not args.note.strip():
                raise ValueError("A human reconciliation note is required")
            journal = CallJournal(Path("unused"))
            journal.path = args.journal.resolve()
            record = journal.records().get(args.call)
            if record is None:
                raise ValueError("Unknown call ID")
            # Never expose this operation as an Agent tool. It is an operator attestation,
            # not a security boundary against unrestricted shell access.
            response = input(
                f"Confirm YOU inspected {record['action']} ({args.call}). Type CONFIRM: "
            )
            if response != "CONFIRM":
                raise ValueError("Reconciliation cancelled")
            audit = CallJournal(Path("unused"))
            audit.path = journal.path.with_name(journal.path.name + ".reconciliations.jsonl")
            audit.append(
                {
                    "actor": "user",
                    "call_id": args.call,
                    "fingerprint": journal.fingerprint(record),
                    "outcome": args.outcome,
                    "note": args.note,
                }
            )
            print("Reconciliation appended; the original call journal is unchanged.")
            return 0
        points = RecoveryPoints(WorkspaceStore(args.store))
        if args.command == "list":
            for marker in sorted((args.store / "points").glob("*/COMMITTED.json")):
                try:
                    state = points.load(marker.parent.name)
                    print(
                        json.dumps(
                            {
                                "point": marker.parent.name,
                                "step": state.step_count,
                                "cwd": state.cwd,
                                "task": state.task,
                            },
                            ensure_ascii=False,
                        )
                    )
                except (OSError, ValueError) as exc:
                    print(json.dumps({"point": marker.parent.name, "error": str(exc)}))
        else:
            checkpoint = points.restore(args.point, args.target)
            print(f"Verified workspace: {args.target.absolute()}")
            print(f"Resume checkpoint: {checkpoint}")
            print(
                "Configure dependencies/credentials explicitly; use the installed dm-agent in this workspace."
            )
        return 0
    except (OSError, ValueError) as exc:
        print(f"Recovery refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
