"""Top-level CLI for the Audio-Health-Benchmark harness.

Subcommands wired so far:
    status   forward to run_all.cmd_status (Phase 1 placeholder)

Subcommands added in later phases:
    prep, warm, train, train-cv, run, run-cross, run-cross-category,
    run-data-eff, summary
"""

from __future__ import annotations

import argparse
import sys


def _cmd_status(args: argparse.Namespace) -> None:
    """Phase 1: forward to the existing run_all.cmd_status implementation.

    Will be replaced by an in-package implementation in Phase 5 once the
    legacy ``run_all.py`` is retired.
    """
    import run_all

    run_all.cmd_status(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ahb",
        description="Audio-Health-Benchmark harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show completion status of all runs")
    status_parser.add_argument(
        "--test-only",
        action="store_true",
        help="Report CI-presence status instead of completion status",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        _cmd_status(args)
    else:
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
