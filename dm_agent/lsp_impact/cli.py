"""Standalone inspection entry point for the LSP impact adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import LspImpactService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Python change impact through a local LSP server.")
    parser.add_argument("path", type=Path, help="Changed Python file, relative to --workspace when needed.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--command", default="pyright-langserver")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    service = LspImpactService(
        args.workspace, command=args.command, timeout_seconds=args.timeout_seconds
    )
    try:
        if not service.start("manual"):
            print(json.dumps({"status": "unavailable", "reason": service.client.unavailable_reason}))
            return 2
        service.snapshot(str(args.path))
        report = service.analyze(str(args.path))
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.status == "ok" else 2
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
