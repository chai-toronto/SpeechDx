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
    from ahb.status import cmd_status

    cmd_status(args)


def _cmd_summary(args: argparse.Namespace) -> None:
    from ahb.summary import cmd_summary

    cmd_summary(args)


def _cmd_run(args: argparse.Namespace) -> None:
    from ahb.run import cmd_run

    cmd_run(args)


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
        level_dir=getattr(args, "level_dir", None),
    )


def _cmd_train_cv(args: argparse.Namespace) -> None:
    from ahb.train_cv import cmd_train_cv

    cmd_train_cv(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
        level_dir=getattr(args, "level_dir", None),
    )


def _cmd_warm_cross(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm_cross import run_warm_cross

    ensure_manifest(args.task)
    run_warm_cross(args.task, args.encoder,
                   probe=args.probe, probe_yaml=args.probe_yaml,
                   device=args.device)


def _cmd_train_cross(args: argparse.Namespace) -> None:
    from ahb.train_cross import cmd_train_cross

    cmd_train_cross(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
    )


def _cmd_run_cross(args: argparse.Namespace) -> None:
    from ahb.run_cross import cmd_run_cross

    cmd_run_cross(args)


def _cmd_run_cross_category(args: argparse.Namespace) -> None:
    from ahb.run_cross_category import cmd_run_cross_category

    cmd_run_cross_category(args)


def _cmd_run_data_eff(args: argparse.Namespace) -> None:
    from ahb.run_data_eff import cmd_run_data_eff

    cmd_run_data_eff(args)


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
    train_parser.add_argument("--level-dir", default=None,
                              help="Reroute output_folder + manifest paths from "
                                   "./exps/single_task/ to ./exps/data_eff/<level_dir>/ "
                                   "(used by ahb run-data-eff)")

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
    train_cv_parser.add_argument("--level-dir", default=None,
                                 help="See ahb train --level-dir")

    summary_parser = sub.add_parser(
        "summary", help="Collect results into per-metric CSVs",
    )
    summary_parser.add_argument(
        "--out-dir", type=str, default="exps/single_task/_summary",
        help="Directory to write metric CSVs (default: exps/single_task/_summary)",
    )
    summary_parser.add_argument(
        "--tag", type=str, default="run1",
        help="Experiment tag to scan (default: run1)",
    )

    run_parser = sub.add_parser("run", help="Execute all incomplete runs")
    run_parser.add_argument("--device", type=str, default=None,
                            help="Device override (e.g. cuda:0)")
    run_parser.add_argument("--max-workers", "-j", type=int, default=3,
                            help="Max concurrent tasks (default: 3); writers serialize per (dataset, encoder)")
    run_parser.add_argument("--encoder", type=str, default=None, action="append",
                            help="Run only this encoder (repeatable)")
    run_parser.add_argument("--dataset", type=str, default=None, action="append",
                            help="Run only tasks whose dataset matches (repeatable)")
    run_parser.add_argument("--task", type=str, default=None, action="append",
                            help="Run only these task stems (repeatable)")
    run_parser.add_argument("--test-only", action="store_true",
                            help="Run inference only (no training); requires prior trained model")
    run_parser.add_argument("--cache-only", action="store_true",
                            help="Warm caches and exit before training")
    run_parser.add_argument("--no-writer", action="store_true",
                            help="Run every job as a reader (cache must already be warm)")

    warm_cross_parser = sub.add_parser(
        "warm-cross",
        help="Warm cross-task HDF5 caches (3 caches per pair) — idempotent",
    )
    warm_cross_parser.add_argument("task", help="Cross task stem")
    warm_cross_parser.add_argument("encoder", help="Encoder name")
    warm_cross_parser.add_argument("--probe", default="Probe")
    warm_cross_parser.add_argument("--probe-yaml", default="Probe.yaml")
    warm_cross_parser.add_argument("--device", default=None)

    train_cross_parser = sub.add_parser(
        "train-cross",
        help="Train probe on a cross task — Ray Tune HP search + final eval",
    )
    train_cross_parser.add_argument("task", help="Cross task stem")
    train_cross_parser.add_argument("encoder", help="Encoder name")
    train_cross_parser.add_argument("--probe", default="Probe")
    train_cross_parser.add_argument("--probe-yaml", default="Probe.yaml")
    train_cross_parser.add_argument("--tag", default="run1")
    train_cross_parser.add_argument("--overrides", default="")

    for name, helptext in [
        ("run-cross", "Execute all incomplete cross-task runs"),
        ("run-cross-category", "Execute all incomplete cross-category runs"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--device", type=str, default=None)
        p.add_argument("--max-workers", "-j", type=int, default=3)
        p.add_argument("--encoder", type=str, default=None, action="append")
        p.add_argument("--dataset", type=str, default=None, action="append")
        p.add_argument("--task", type=str, default=None, action="append")
        p.add_argument("--test-only", action="store_true")
        p.add_argument("--cache-only", action="store_true")
        p.add_argument("--no-writer", action="store_true")

    de_parser = sub.add_parser("run-data-eff",
                               help="Execute all incomplete data-efficiency runs (across levels)")
    de_parser.add_argument("--device", type=str, default=None)
    de_parser.add_argument("--max-workers", "-j", type=int, default=3)
    de_parser.add_argument("--encoder", type=str, default=None, action="append")
    de_parser.add_argument("--dataset", type=str, default=None, action="append")
    de_parser.add_argument("--task", type=str, default=None, action="append")
    de_parser.add_argument("--level", type=str, default=None, action="append",
                           help="Restrict to these level dirs (repeatable; "
                                "default: all levels in registry.yaml:data_eff_levels)")
    de_parser.add_argument("--test-only", action="store_true")
    de_parser.add_argument("--cache-only", action="store_true")
    de_parser.add_argument("--no-writer", action="store_true")

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
    elif args.command == "summary":
        _cmd_summary(args)
    elif args.command == "run":
        if args.cache_only and args.test_only:
            parser.error("--cache-only and --test-only are mutually exclusive")
        if args.cache_only and args.no_writer:
            parser.error("--cache-only and --no-writer are mutually exclusive")
        _cmd_run(args)
    elif args.command == "warm-cross":
        _cmd_warm_cross(args)
    elif args.command == "train-cross":
        _cmd_train_cross(args)
    elif args.command in ("run-cross", "run-cross-category"):
        if args.cache_only and args.test_only:
            parser.error("--cache-only and --test-only are mutually exclusive")
        if args.cache_only and args.no_writer:
            parser.error("--cache-only and --no-writer are mutually exclusive")
        if args.command == "run-cross":
            _cmd_run_cross(args)
        else:
            _cmd_run_cross_category(args)
    elif args.command == "run-data-eff":
        if args.cache_only and args.test_only:
            parser.error("--cache-only and --test-only are mutually exclusive")
        if args.cache_only and args.no_writer:
            parser.error("--cache-only and --no-writer are mutually exclusive")
        _cmd_run_data_eff(args)
    else:
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
