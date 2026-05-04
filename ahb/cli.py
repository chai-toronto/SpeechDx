"""Top-level CLI for the Audio-Health-Benchmark harness.

Surface:
    python -m ahb <mode> <command> [flags]

Modes:    single, cross, cross-cat, data-eff, all
Commands: prep, warm, train, run, status, summary

If the first argument is a command rather than a mode, mode defaults to
``single`` — so ``python -m ahb run`` is shorthand for
``python -m ahb single run``. The old flat names (``ahb run-cross``,
``ahb summary-data-eff``, ``ahb run-all``, ...) are gone as of commit 1 of
the rewrite.

Stub cells (``cross status`` / ``cross-cat status`` / ``data-eff status`` /
``all status`` / ``all prep`` / ``all warm`` / ``all train`` /
``all summary``) raise ``NotImplementedError`` with a pointer to the commit
that will fill them in. Everything else routes to today's handlers
unchanged.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

MODES = ("single", "cross", "cross-cat", "data-eff", "all")
COMMANDS = ("prep", "warm", "train", "run", "status", "summary")


# ---------------------------------------------------------------------------
# Handlers — one per (mode, command) cell. Imports are inline so unrelated
# subcommands don't pull heavy dependencies into argparse startup.
# ---------------------------------------------------------------------------

def _h_prep(args: argparse.Namespace) -> None:
    """Mode-agnostic prep — works for single, cross, and cross-cat stems."""
    from ahb.prep.dispatch import ensure_manifest
    for task in args.task:
        ensure_manifest(task)


def _h_single_warm(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm import run_warm
    ensure_manifest(args.task)
    run_warm(args.task, args.encoder, probe=args.probe, device=args.device)


def _h_single_train(args: argparse.Namespace) -> None:
    """Auto-routes to per-fold CV when the task yaml sets ``num_fold``."""
    from ahb.orchestrator import is_cv
    from ahb.prep.dispatch import ensure_manifest
    ensure_manifest(args.task)
    if is_cv(args.task):
        from ahb.train_cv import cmd_train_cv as _train_fn
    else:
        from ahb.train import cmd_train as _train_fn
    _train_fn(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
        level_dir=getattr(args, "level_dir", None),
    )


def _h_single_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run import cmd_run
    cmd_run(args)


def _h_single_status(args: argparse.Namespace) -> None:
    from ahb.status import cmd_status
    cmd_status(args)


def _h_single_summary(args: argparse.Namespace) -> None:
    from ahb.summary import cmd_summary
    cmd_summary(args)


def _h_cross_warm(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm_cross import run_warm_cross
    ensure_manifest(args.task)
    run_warm_cross(args.task, args.encoder,
                   probe=args.probe, probe_yaml=args.probe_yaml,
                   device=args.device)


def _h_cross_train(args: argparse.Namespace) -> None:
    """``cmd_train_cross`` auto-detects category vs non-category via the task
    yaml, so the per-pair entry point is shared with ``cross-cat train``."""
    from ahb.train_cross import cmd_train_cross
    cmd_train_cross(
        args.task, args.encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
    )


def _h_cross_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run_cross import cmd_run_cross
    cmd_run_cross(args)


def _h_cross_status(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb cross status` is not yet implemented (scheduled for commit 7).")


def _h_cross_summary(args: argparse.Namespace) -> None:
    from ahb.summary_cross import cmd_summary_cross
    cmd_summary_cross(args)


def _h_crosscat_warm(args: argparse.Namespace) -> None:
    """Today routes to ``warm_cross`` (which short-circuits for category
    tasks). Commit 5 swaps this to delegate to single warm across the listed
    train/test datasets."""
    _h_cross_warm(args)


def _h_crosscat_train(args: argparse.Namespace) -> None:
    _h_cross_train(args)


def _h_crosscat_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run_cross_category import cmd_run_cross_category
    cmd_run_cross_category(args)


def _h_crosscat_status(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb cross-cat status` is not yet implemented (scheduled for commit 7).")


def _h_crosscat_summary(args: argparse.Namespace) -> None:
    from ahb.summary_cross import cmd_summary_cross_category
    cmd_summary_cross_category(args)


def _h_dataeff_prep(args: argparse.Namespace) -> None:
    """data-eff manifests are identical to single — caches/manifests are
    level-agnostic."""
    _h_prep(args)


def _h_dataeff_warm(args: argparse.Namespace) -> None:
    """data-eff warm == single warm. Caches are shared across levels."""
    _h_single_warm(args)


def _h_dataeff_train(args: argparse.Namespace) -> None:
    """Train one (task, encoder) at every data-eff level by default;
    ``--level`` filters to specific levels."""
    from ahb.registry import data_eff_levels
    levels = args.level or [name for name, _ in data_eff_levels()]
    for level in levels:
        ns = argparse.Namespace(**vars(args))
        ns.level_dir = level
        ns.level = None
        _h_single_train(ns)


def _h_dataeff_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run_data_eff import cmd_run_data_eff
    cmd_run_data_eff(args)


def _h_dataeff_status(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb data-eff status` is not yet implemented (scheduled for commit 7).")


def _h_dataeff_summary(args: argparse.Namespace) -> None:
    from ahb.summary_data_eff import cmd_summary_data_eff
    cmd_summary_data_eff(args)


def _h_all_prep(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb all prep` is a starting-point stub — fleshed out in a later commit. "
        "Use mode-specific prep (e.g. `ahb single prep <task>`) for now.")


def _h_all_warm(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb all warm` is a starting-point stub — fleshed out in a later commit. "
        "Use mode-specific warm or `ahb all run --cache-only` for now.")


def _h_all_train(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb all train` is a starting-point stub — fleshed out in a later commit. "
        "Use mode-specific train or `ahb all run --no-writer` for now.")


def _h_all_run(args: argparse.Namespace) -> None:
    """Sequentially invoke single → cross → cross-cat → data-eff. Each
    underlying orchestrator skips already-complete jobs, so the chain is
    idempotent. Mode filters apply uniformly; a filter that matches no tasks
    in a given mode just records 0 pending."""
    _validate_run_flags(args)
    modes = [
        ("single",    _h_single_run),
        ("cross",     _h_cross_run),
        ("cross-cat", _h_crosscat_run),
        ("data-eff",  _h_dataeff_run),
    ]
    skip = set(args.skip_mode or [])
    failures: list[tuple[str, BaseException]] = []
    for name, fn in modes:
        if name in skip:
            print(f"=== all run: skipping mode {name!r} (--skip-mode) ===")
            continue
        print(f"\n=== all run: starting mode {name!r} ===")
        try:
            fn(args)
        except BaseException as e:
            failures.append((name, e))
            print(f"=== all run: mode {name!r} FAILED: {e!r} ===")
            if not args.continue_on_failure:
                raise
    if failures:
        names = ", ".join(n for n, _ in failures)
        raise SystemExit(f"all run: {len(failures)} mode(s) failed: {names}")


def _h_all_status(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb all status` is not yet implemented (scheduled for commit 7).")


def _h_all_summary(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "`ahb all summary` is a starting-point stub — fleshed out in a later commit. "
        "Run mode-specific summaries (`ahb single summary`, etc.) for now.")


def _validate_run_flags(args: argparse.Namespace) -> None:
    if getattr(args, "cache_only", False) and getattr(args, "test_only", False):
        raise SystemExit("--cache-only and --test-only are mutually exclusive")
    if getattr(args, "cache_only", False) and getattr(args, "no_writer", False):
        raise SystemExit("--cache-only and --no-writer are mutually exclusive")


DISPATCH: dict[tuple[str, str], Callable[[argparse.Namespace], None]] = {
    ("single", "prep"):    _h_prep,
    ("single", "warm"):    _h_single_warm,
    ("single", "train"):   _h_single_train,
    ("single", "run"):     _h_single_run,
    ("single", "status"):  _h_single_status,
    ("single", "summary"): _h_single_summary,
    ("cross", "prep"):     _h_prep,
    ("cross", "warm"):     _h_cross_warm,
    ("cross", "train"):    _h_cross_train,
    ("cross", "run"):      _h_cross_run,
    ("cross", "status"):   _h_cross_status,
    ("cross", "summary"):  _h_cross_summary,
    ("cross-cat", "prep"):    _h_prep,
    ("cross-cat", "warm"):    _h_crosscat_warm,
    ("cross-cat", "train"):   _h_crosscat_train,
    ("cross-cat", "run"):     _h_crosscat_run,
    ("cross-cat", "status"):  _h_crosscat_status,
    ("cross-cat", "summary"): _h_crosscat_summary,
    ("data-eff", "prep"):    _h_dataeff_prep,
    ("data-eff", "warm"):    _h_dataeff_warm,
    ("data-eff", "train"):   _h_dataeff_train,
    ("data-eff", "run"):     _h_dataeff_run,
    ("data-eff", "status"):  _h_dataeff_status,
    ("data-eff", "summary"): _h_dataeff_summary,
    ("all", "prep"):    _h_all_prep,
    ("all", "warm"):    _h_all_warm,
    ("all", "train"):   _h_all_train,
    ("all", "run"):     _h_all_run,
    ("all", "status"):  _h_all_status,
    ("all", "summary"): _h_all_summary,
}


# ---------------------------------------------------------------------------
# Parser builders — argument groupings reused across modes.
# ---------------------------------------------------------------------------

def _add_prep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", nargs="+",
                   help="Task stem(s) — looked up in ahb/configs/{tasks,cross_tasks}/")


def _add_warm_args(p: argparse.ArgumentParser, *, default_probe: str = "AvgTProbe") -> None:
    p.add_argument("task", help="Task stem")
    p.add_argument("encoder", help="Encoder name (model_name in registry.yaml)")
    p.add_argument("--probe", default=default_probe,
                   help=f"Probe name (default: {default_probe})")
    p.add_argument("--probe-yaml", default="Probe.yaml",
                   help="Probe yaml filename under ahb/configs/probes/")
    p.add_argument("--device", default=None,
                   help="Torch device override (e.g. cuda:0)")


def _add_train_args(p: argparse.ArgumentParser, *,
                    default_probe: str = "AvgTProbe",
                    include_level_dir: bool = False,
                    include_level: bool = False) -> None:
    p.add_argument("task", help="Task stem")
    p.add_argument("encoder", help="Encoder name")
    p.add_argument("--probe", default=default_probe,
                   help=f"Probe name (default: {default_probe})")
    p.add_argument("--probe-yaml", default="Probe.yaml")
    p.add_argument("--tag", default="run1",
                   help="Experiment tag (default: run1)")
    p.add_argument("--overrides", default="",
                   help="Extra YAML overrides forwarded to load_hyperpyyaml")
    if include_level_dir:
        p.add_argument("--level-dir", default=None,
                       help="Reroute output_folder + manifest paths from "
                            "./exps/single_task/ to ./exps/data_eff/<level_dir>/")
    if include_level:
        p.add_argument("--level", action="append", default=None,
                       help="Restrict to these data-eff level dirs (repeatable; "
                            "default: all levels in registry.yaml)")


def _add_run_args(p: argparse.ArgumentParser, *,
                  include_level: bool = False) -> None:
    p.add_argument("--device", type=str, default=None,
                   help="Device override (e.g. cuda:0)")
    p.add_argument("--max-workers", "-j", type=int, default=3,
                   help="Max concurrent tasks (default: 3); writers serialize "
                        "per (dataset, encoder)")
    p.add_argument("--encoder", type=str, default=None, action="append",
                   help="Restrict to this encoder (repeatable)")
    p.add_argument("--dataset", type=str, default=None, action="append",
                   help="Restrict to tasks whose dataset matches (repeatable)")
    p.add_argument("--task", type=str, default=None, action="append",
                   help="Restrict to this task stem (repeatable)")
    if include_level:
        p.add_argument("--level", type=str, default=None, action="append",
                       help="Restrict to these level dirs (repeatable; "
                            "default: all levels in registry.yaml)")
    p.add_argument("--test-only", action="store_true",
                   help="Run inference only (requires prior trained model)")
    p.add_argument("--cache-only", action="store_true",
                   help="Warm caches and exit before training")
    p.add_argument("--no-writer", action="store_true",
                   help="Run every job as a reader (cache must already be warm)")
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag forwarded to train (default: run1)")


def _add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag to scan (default: run1)")


def _add_summary_args(p: argparse.ArgumentParser, *,
                      default_root_hint: str,
                      include_level: bool = False) -> None:
    p.add_argument("--out-dir", type=str, default=None,
                   help=f"Output dir (default: {default_root_hint}/_summary_<tag>)")
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag to scan (default: run1)")
    p.add_argument("--encoder", type=str, default=None, action="append",
                   help="Restrict to these encoders (repeatable)")
    p.add_argument("--dataset", type=str, default=None, action="append",
                   help="Restrict to tasks whose dataset matches (repeatable)")
    p.add_argument("--task", type=str, default=None, action="append",
                   help="Restrict to these task stems (repeatable)")
    if include_level:
        p.add_argument("--level", type=str, default=None, action="append",
                       help="Restrict to these level dirs (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ahb",
        description=("Audio-Health-Benchmark harness. Surface: "
                     "`ahb <mode> <command>`. <mode> defaults to 'single' "
                     "if the first arg is a command."),
    )
    mode_sub = parser.add_subparsers(dest="mode", required=True)

    # ---- single ----
    sp = mode_sub.add_parser("single", help="Single-dataset benchmark mode (default).")
    sub = sp.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more tasks."))
    _add_warm_args(sub.add_parser("warm", help="Warm the HDF5 cache for one (task, encoder)."))
    _add_train_args(sub.add_parser(
        "train", help="Train probe (auto-routes to per-fold CV via task yaml)."),
        include_level_dir=True)
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete (task, encoder) pairs."))
    _add_status_args(sub.add_parser("status", help="Per-task × per-encoder completion grid."))
    _add_summary_args(sub.add_parser("summary", help="Aggregate test results into per-metric CSVs."),
                      default_root_hint="exps/single_task")

    # ---- cross ----
    cp = mode_sub.add_parser("cross", help="Zero-shot cross-task mode.")
    sub = cp.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more cross tasks."))
    _add_warm_args(sub.add_parser("warm", help="Warm cross-task caches."),
                   default_probe="Probe")
    _add_train_args(sub.add_parser("train", help="Train probe on a cross task."),
                    default_probe="Probe")
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete cross runs."))
    _add_status_args(sub.add_parser("status", help="(stub — commit 7)"))
    _add_summary_args(sub.add_parser("summary", help="Aggregate cross results."),
                      default_root_hint="exps/cross")

    # ---- cross-cat ----
    cap = mode_sub.add_parser("cross-cat", help="Multi-source cross-category mode.")
    sub = cap.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more category tasks."))
    _add_warm_args(sub.add_parser("warm", help="Warm category-task caches."),
                   default_probe="Probe")
    _add_train_args(sub.add_parser("train", help="Train probe on a category task."),
                    default_probe="Probe")
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete cross-category runs."))
    _add_status_args(sub.add_parser("status", help="(stub — commit 7)"))
    _add_summary_args(sub.add_parser("summary", help="Aggregate cross-category results."),
                      default_root_hint="exps/cross_cat")

    # ---- data-eff ----
    dep = mode_sub.add_parser("data-eff", help="Data-efficiency mode (4 reduced training-set sizes).")
    sub = dep.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Same as single prep."))
    _add_warm_args(sub.add_parser("warm", help="Same as single warm — caches are level-agnostic."))
    _add_train_args(sub.add_parser(
        "train", help="Train at every data-eff level for one (task, encoder); use --level to restrict."),
        include_level=True)
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete data-eff runs (across levels)."),
                  include_level=True)
    _add_status_args(sub.add_parser("status", help="(stub — commit 7)"))
    _add_summary_args(sub.add_parser("summary", help="Aggregate data-eff results."),
                      default_root_hint="exps/data_eff", include_level=True)

    # ---- all ----
    ap = mode_sub.add_parser("all", help="Apply command to every mode in turn.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("prep", help="(stub — fleshed out in a later commit)")
    sub.add_parser("warm", help="(stub — fleshed out in a later commit)")
    sub.add_parser("train", help="(stub — fleshed out in a later commit)")
    all_run = sub.add_parser("run", help="Sequentially run every mode (single → cross → cross-cat → data-eff).")
    _add_run_args(all_run, include_level=True)
    all_run.add_argument("--skip-mode", action="append", default=None,
                         choices=list(MODES[:-1]),
                         help="Mode(s) to skip (repeatable)")
    all_run.add_argument("--continue-on-failure", action="store_true",
                         help="Run later modes even if an earlier one raises")
    sub.add_parser("status", help="(stub — commit 7)")
    sub.add_parser("summary", help="(stub — fleshed out in a later commit)")

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Inject ``single`` if the first arg is a command rather than a mode.

    Allows ``python -m ahb run`` as shorthand for ``python -m ahb single run``.
    """
    if argv and argv[0] in COMMANDS and argv[0] not in MODES:
        return ["single", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    raw = _normalize_argv(raw)
    parser = build_parser()
    args = parser.parse_args(raw)
    handler = DISPATCH.get((args.mode, args.command))
    if handler is None:
        parser.error(f"unknown command: {args.mode} {args.command}")
    handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
