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


def _cmd_prep(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest

    for task in args.task:
        ensure_manifest(task)


def _cmd_warm(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm import run_warm

    ensure_manifest(args.task)
    run_warm(args.task, args.encoder, probe=args.probe, device=args.device)


def _cmd_train(args: argparse.Namespace) -> None:
    from ahb.train import cmd_train

    cmd_train(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
    )


def _cmd_train_cv(args: argparse.Namespace) -> None:
    from ahb.train_cv import cmd_train_cv

    cmd_train_cv(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
    )


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

    prep_parser = sub.add_parser(
        "prep",
        help="Build manifests for one or more tasks (no-op if already present)",
    )
    prep_parser.add_argument(
        "task",
        nargs="+",
        help="Task stem(s) — looked up in training/config/{tasks,cross_tasks}/",
    )

    warm_parser = sub.add_parser(
        "warm",
        help="Warm the HDF5 cache for one (task, encoder) pair (idempotent)",
    )
    warm_parser.add_argument("task", help="Task stem")
    warm_parser.add_argument("encoder", help="Encoder name (model_name in registry.yaml)")
    warm_parser.add_argument("--probe", default="AvgTProbe", help="Probe name (default: AvgTProbe)")
    warm_parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda:0)")

    train_parser = sub.add_parser(
        "train",
        help="Train probe with Ray Tune HP search; final eval on test set",
    )
    train_parser.add_argument("task", help="Task stem")
    train_parser.add_argument("encoder", help="Encoder name (model_name in registry.yaml)")
    train_parser.add_argument("--probe", default="AvgTProbe", help="Probe name (default: AvgTProbe)")
    train_parser.add_argument("--probe-yaml", default="Probe.yaml",
                              help="Probe yaml filename under training/config/probes/")
    train_parser.add_argument("--tag", default="run1", help="Experiment tag (default: run1)")
    train_parser.add_argument("--overrides", default="",
                              help="Extra YAML overrides forwarded to load_hyperpyyaml")

    train_cv_parser = sub.add_parser(
        "train-cv",
        help="Per-fold CV training (mvdr_*); aggregates fold metrics into test_results.yaml",
    )
    train_cv_parser.add_argument("task", help="Task stem (CV tasks have num_fold set)")
    train_cv_parser.add_argument("encoder", help="Encoder name")
    train_cv_parser.add_argument("--probe", default="AvgTProbe")
    train_cv_parser.add_argument("--probe-yaml", default="Probe.yaml")
    train_cv_parser.add_argument("--tag", default="run1")
    train_cv_parser.add_argument("--overrides", default="")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        _cmd_status(args)
    elif args.command == "prep":
        _cmd_prep(args)
    elif args.command == "warm":
        _cmd_warm(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "train-cv":
        _cmd_train_cv(args)
    else:
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
