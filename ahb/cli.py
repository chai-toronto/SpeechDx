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
# Filter resolution — every command takes -e/-d/-t (encoder/dataset/task)
# allowlists. Empty filter = "everything found".
# ---------------------------------------------------------------------------

def _resolve_tasks(args: argparse.Namespace, mode: str) -> list[str]:
    """Sorted task stems matching the --task / --dataset filters for ``mode``."""
    datasets = getattr(args, "dataset", None)
    tasks = getattr(args, "task", None)
    if mode in ("single", "data-eff"):
        from ahb.orchestrator import discover_tasks
        return discover_tasks(datasets=datasets, tasks=tasks)
    if mode == "cross":
        from ahb.orchestrator_cross import discover_tasks as cross_disc
        return cross_disc(datasets=datasets, tasks=tasks, include_categories=False)
    if mode == "cross-cat":
        from ahb.orchestrator_cross import discover_tasks as cross_disc
        return cross_disc(datasets=datasets, tasks=tasks, include_categories=True)
    raise ValueError(f"unknown mode for task selection: {mode}")


def _resolve_pairs(args: argparse.Namespace, mode: str) -> list[tuple[str, str]]:
    """Cartesian product of resolved tasks and encoders matching the filters."""
    from ahb.registry import encoders as registry_encoders
    tasks = _resolve_tasks(args, mode)
    enc_filter = set(getattr(args, "encoder", None) or [])
    encs = list(registry_encoders().keys())
    if enc_filter:
        encs = [e for e in encs if e in enc_filter]
    return [(t, e) for t in tasks for e in encs]


def _load_cross_yaml(cross_task: str) -> dict:
    """Read the (cross or cross-cat) task yaml as a plain dict for the
    bookkeeping fields needed by warm delegation. Uses ``TolerantLoader``
    so HyperPyYAML tags like ``!new:torch.nn.BCEWithLogitsLoss`` parse
    without trying to construct the underlying object."""
    import yaml
    from ahb.orchestrator_cross import TASKS_DIR
    from ahb.yaml_io import TolerantLoader
    path = TASKS_DIR / f"{cross_task}.yaml"
    return yaml.load(path.read_text(), Loader=TolerantLoader) or {}


def _proxy_single_tasks_for_cross(cross_yaml: dict) -> list[str]:
    """List of single-mode task stems on the datasets named in the cross
    yaml. Cross uses ``train_dataset`` / ``test_dataset`` (singular);
    cross-cat uses ``train_datasets`` / ``test_datasets`` (plural). Cache
    is keyed by ``(dataset, encoder)``, so listing every task on each
    dataset is fine — duplicates no-op (Q3 b)."""
    from ahb.orchestrator import TASKS_DIR as SINGLE_TASKS_DIR
    datasets: list[str] = []
    for key in ("train_dataset", "test_dataset"):
        v = cross_yaml.get(key)
        if isinstance(v, str):
            datasets.append(v)
    for key in ("train_datasets", "test_datasets"):
        v = cross_yaml.get(key)
        if isinstance(v, list):
            datasets.extend(v)
    seen: set[str] = set()
    out: list[str] = []
    for ds in datasets:
        for p in sorted(SINGLE_TASKS_DIR.glob(f"{ds}_*.yaml")):
            if p.stem in seen:
                continue
            seen.add(p.stem)
            out.append(p.stem)
    return out


def _wipe_warm_cache(task: str, encoder: str, *, probe: str = "AvgTProbe") -> None:
    """Remove the per-(dataset, encoder) cache.hdf5 files so the next
    ``run_warm`` rebuilds from scratch. Used by ``warm --overwrite``."""
    from ahb.warm import _cache_dirs as warm_cache_dirs
    from ahb.config import compose_config
    try:
        hparams = compose_config(task, encoder, probe=probe, mode="read")
    except Exception as e:
        print(f"  ⚠ couldn't resolve cache for {task} × {encoder}: {e}")
        return
    train_dir, val_dir = warm_cache_dirs(hparams)
    for d in (train_dir, val_dir):
        f = d / "cache.hdf5"
        if f.exists():
            f.unlink()
            print(f"  removed {f}")


def _train_one(args: argparse.Namespace, task: str, encoder: str, *,
               level_dir: str | None = None) -> None:
    """Per-pair single-mode train; auto-routes to per-fold CV via task yaml."""
    from ahb.orchestrator import is_cv
    from ahb.prep.dispatch import ensure_manifest
    ensure_manifest(task)
    if is_cv(task):
        from ahb.train_cv import cmd_train_cv as _fn
    else:
        from ahb.train import cmd_train as _fn
    _fn(
        task, encoder,
        probe=args.probe, probe_yaml=args.probe_yaml,
        tag=args.tag, overrides=args.overrides or "",
        level_dir=level_dir,
    )


# ---------------------------------------------------------------------------
# Handlers — one per (mode, command) cell. Imports are inline so unrelated
# subcommands don't pull heavy dependencies into argparse startup.
# ---------------------------------------------------------------------------

def _h_prep(args: argparse.Namespace) -> None:
    """Mode-agnostic prep — works for single, cross, and cross-cat stems."""
    from ahb.orchestrator import manifest_paths
    from ahb.prep.dispatch import ensure_manifest
    tasks = _resolve_tasks(args, args.mode)
    if not tasks:
        print("prep: no matching tasks for filters.")
        return
    print(f"prep: {len(tasks)} task(s)")
    overwrite = getattr(args, "overwrite", False)
    for task in tasks:
        if overwrite:
            for p in manifest_paths(task):
                p.unlink(missing_ok=True)
        ensure_manifest(task)


def _h_single_warm(args: argparse.Namespace) -> None:
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm import run_warm
    pairs = _resolve_pairs(args, "single")
    if not pairs:
        print("warm: no matching (task, encoder) pairs for filters.")
        return
    print(f"warm: {len(pairs)} (task, encoder) pair(s)")
    for task, encoder in pairs:
        ensure_manifest(task)
        if getattr(args, "overwrite", False):
            _wipe_warm_cache(task, encoder, probe=args.probe)
        run_warm(task, encoder, probe=args.probe, device=args.device)


def _h_single_train(args: argparse.Namespace) -> None:
    """Sweep all matching (task, encoder) pairs; per-pair train auto-routes
    to per-fold CV via the task yaml. Skip-if-complete by default; pass
    ``--overwrite`` to wipe and retrain."""
    from ahb.orchestrator import (
        get_output_folder, get_task_info, is_complete,
    )
    pairs = _resolve_pairs(args, "single")
    if not pairs:
        print("train: no matching pairs for filters.")
        return
    overwrite = getattr(args, "overwrite", False)
    level_dir = getattr(args, "level_dir", None)
    print(f"train: {len(pairs)} pair(s)")
    skipped = 0
    for task, encoder in pairs:
        ds, t = get_task_info(task)
        folder = get_output_folder(ds, t, encoder, args.tag)
        if folder.exists() and is_complete(folder, task) and not overwrite:
            print(f"  SKIP (results exist): {task} × {encoder}")
            skipped += 1
            continue
        if folder.exists() and overwrite:
            import shutil
            shutil.rmtree(folder)
        _train_one(args, task, encoder, level_dir=level_dir)
    if skipped:
        print(f"train: skipped {skipped} complete pair(s); pass --overwrite to redo.")


def _h_single_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run import cmd_run
    _strict_phase_run(args, cmd_run, mode_label="single run")


def _h_single_status(args: argparse.Namespace) -> None:
    from ahb.status import cmd_status
    cmd_status(args)


def _h_single_summary(args: argparse.Namespace) -> None:
    from ahb.summary import cmd_summary
    cmd_summary(args)


def _h_cross_warm(args: argparse.Namespace) -> None:
    """Cross / cross-cat warm delegate to single warm.

    The cross trainer reads from the same per-(dataset, encoder) caches
    that single warm populates (paths in main_cross.yaml are
    ``<tmp>/<dataset>/<encoder>/{train,val}`` — identical to single's
    layout, just resolved against train_dataset and test_dataset). So
    rather than maintain a separate warm_cross writer, we iterate the
    underlying single tasks on each listed dataset and call run_warm
    with the cross task's ``num_aug_ver`` so the cache gets extended to
    the aug count the cross trainer needs.

    Cross-cat: every task on every listed dataset (Q3 b — cache is keyed
    by (dataset, encoder), so multiple tasks per dataset no-op after the
    first).
    """
    from ahb.prep.dispatch import ensure_manifest
    from ahb.warm import run_warm
    pairs = _resolve_pairs(args, args.mode)
    if not pairs:
        print(f"{args.mode} warm: no matching pairs for filters.")
        return
    print(f"{args.mode} warm: {len(pairs)} cross pair(s) "
          f"→ delegating to single warm")
    for cross_task, encoder in pairs:
        ensure_manifest(cross_task)
        cross_yaml = _load_cross_yaml(cross_task)
        num_aug = int(cross_yaml.get("num_aug_ver", 1) or 1)
        proxy_tasks = _proxy_single_tasks_for_cross(cross_yaml)
        if not proxy_tasks:
            print(f"  ⚠ {cross_task}: no underlying single tasks found")
            continue
        print(f"  {cross_task} × {encoder}  (num_aug_ver={num_aug}, "
              f"{len(proxy_tasks)} proxy single task(s))")
        for proxy in proxy_tasks:
            ensure_manifest(proxy)
            if getattr(args, "overwrite", False):
                _wipe_warm_cache(proxy, encoder, probe=args.probe)
            run_warm(proxy, encoder, probe=args.probe,
                     device=args.device, num_aug_ver=num_aug)


def _h_cross_train(args: argparse.Namespace) -> None:
    """``cmd_train_cross`` auto-detects category vs non-category via the task
    yaml, so the per-pair entry point is shared with ``cross-cat train``.
    Skip-if-complete by default; ``--overwrite`` wipes and retrains."""
    import shutil
    from ahb.orchestrator import is_complete
    from ahb.orchestrator_cross import (
        get_output_folder as cross_output_folder,
        get_task_info as cross_task_info,
    )
    from ahb.run_cross_category import CATEGORY_EXPS_ROOT
    from ahb.train_cross import cmd_train_cross
    pairs = _resolve_pairs(args, args.mode)
    if not pairs:
        print(f"{args.mode} train: no matching pairs for filters.")
        return
    overwrite = getattr(args, "overwrite", False)
    exps_root = CATEGORY_EXPS_ROOT if args.mode == "cross-cat" else None
    print(f"{args.mode} train: {len(pairs)} pair(s)")
    skipped = 0
    for task, encoder in pairs:
        ds, t = cross_task_info(task)
        folder = cross_output_folder(ds, t, encoder, args.tag, exps_root=exps_root)
        if folder.exists() and is_complete(folder, task) and not overwrite:
            print(f"  SKIP (results exist): {task} × {encoder}")
            skipped += 1
            continue
        if folder.exists() and overwrite:
            shutil.rmtree(folder)
        cmd_train_cross(
            task, encoder,
            probe=args.probe, probe_yaml=args.probe_yaml,
            tag=args.tag, overrides=args.overrides or "",
        )
    if skipped:
        print(f"{args.mode} train: skipped {skipped} complete pair(s); pass --overwrite to redo.")


def _h_cross_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run_cross import cmd_run_cross
    _strict_phase_run(args, cmd_run_cross, mode_label="cross run")


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
    _strict_phase_run(args, cmd_run_cross_category, mode_label="cross-cat run")


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
    """Sweep all matching (task, encoder) pairs at every data-eff level by
    default; ``--level`` filters to specific levels."""
    from ahb.registry import data_eff_levels
    pairs = _resolve_pairs(args, "data-eff")
    if not pairs:
        print("data-eff train: no matching pairs for filters.")
        return
    levels = args.level or [name for name, _ in data_eff_levels()]
    print(f"data-eff train: {len(pairs)} pair(s) × {len(levels)} level(s)")
    for level in levels:
        for task, encoder in pairs:
            _train_one(args, task, encoder, level_dir=level)


def _h_dataeff_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from ahb.run_data_eff import cmd_run_data_eff
    _strict_phase_run(args, cmd_run_data_eff, mode_label="data-eff run")


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


def _confirm_overwrite(args: argparse.Namespace, mode_label: str) -> bool:
    """Interactive ``[y/N]`` prompt before destructive `run --overwrite`.

    Bypassed by ``--yes``. Also bypassed when ``--dry-run`` is set (nothing
    actually gets destroyed). Returns True if the run should proceed.
    """
    if not getattr(args, "overwrite", False):
        return True
    if getattr(args, "yes", False) or getattr(args, "dry_run", False):
        return True
    print(f"\n!! {mode_label} --overwrite will redo every matching pair, "
          f"including ones whose results already exist.")
    print("   Existing results will be wiped. This is not reversible.")
    try:
        resp = input("   Proceed? [y/N] ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted.")
        return False
    return True


def _strict_phase_run(args: argparse.Namespace,
                      run_fn: Callable[[argparse.Namespace], None],
                      *, mode_label: str) -> None:
    """Two-phase run: warm everything, then train everything.

    Underlying ``cmd_run*`` functions are pipelined per-pair; we drive them
    twice with ``--cache-only`` then ``--no-writer`` (so both phases re-use
    today's writer/reader plumbing without changes). Worker pools are sized
    independently — warm dominates GPU time, train dominates CPU; running
    them strictly serial lets each phase fully utilize its bottleneck.

    Skipped:
    - ``--test-only``: no warm needed (just re-eval), single phase only.
    - ``--cache-only``: no train phase (warm-only by user request).
    - ``--no-writer``: no warm phase (caches assumed warm by user request).
    """
    test_only = getattr(args, "test_only", False)
    cache_only = getattr(args, "cache_only", False)
    no_writer = getattr(args, "no_writer", False)
    warm_workers = getattr(args, "warm_workers", 1)
    train_workers = getattr(args, "train_workers", 3)

    if not _confirm_overwrite(args, mode_label):
        return

    if test_only:
        ns = argparse.Namespace(**vars(args))
        ns.max_workers = train_workers
        print(f"=== {mode_label}: test-only (workers={train_workers}) ===")
        run_fn(ns)
        return

    do_warm = not no_writer
    do_train = not cache_only

    if do_warm:
        warm_ns = argparse.Namespace(**vars(args))
        warm_ns.cache_only = True
        warm_ns.no_writer = False
        warm_ns.test_only = False
        warm_ns.max_workers = warm_workers
        print(f"=== {mode_label}: phase 1 — warm caches "
              f"(workers={warm_workers}) ===")
        run_fn(warm_ns)

    if do_train:
        train_ns = argparse.Namespace(**vars(args))
        train_ns.cache_only = False
        # If we just warmed, every pair reads from warm cache.
        # If user invoked --no-writer, respect that (cache assumed warm).
        train_ns.no_writer = True
        train_ns.test_only = False
        train_ns.max_workers = train_workers
        phase_label = ("phase 2 — train" if do_warm else "train (no-writer)")
        print(f"\n=== {mode_label}: {phase_label} "
              f"(workers={train_workers}) ===")
        run_fn(train_ns)


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

def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """The canonical ``-e``/``-d``/``-t`` filters used by every command.

    Each is repeatable; an empty filter means "every match in scope".
    """
    p.add_argument("-e", "--encoder", action="append", default=None,
                   help="Restrict to this encoder (repeatable; default: all)")
    p.add_argument("-d", "--dataset", action="append", default=None,
                   help="Restrict to this dataset (repeatable; default: all)")
    p.add_argument("-t", "--task", action="append", default=None,
                   help="Restrict to this task stem (repeatable; default: all)")


def _add_prep_args(p: argparse.ArgumentParser) -> None:
    _add_filter_args(p)
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild manifests even if they already exist.")


def _add_warm_args(p: argparse.ArgumentParser, *, default_probe: str = "AvgTProbe") -> None:
    _add_filter_args(p)
    p.add_argument("--probe", default=default_probe,
                   help=f"Probe name (default: {default_probe})")
    p.add_argument("--probe-yaml", default="Probe.yaml",
                   help="Probe yaml filename under ahb/configs/probes/")
    p.add_argument("--device", default=None,
                   help="Torch device override (e.g. cuda:0)")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe the per-(dataset, encoder) cache.hdf5 files "
                        "before re-extracting.")


def _add_train_args(p: argparse.ArgumentParser, *,
                    default_probe: str = "AvgTProbe",
                    include_level_dir: bool = False,
                    include_level: bool = False) -> None:
    _add_filter_args(p)
    p.add_argument("--probe", default=default_probe,
                   help=f"Probe name (default: {default_probe})")
    p.add_argument("--probe-yaml", default="Probe.yaml")
    p.add_argument("--tag", default="run1",
                   help="Experiment tag (default: run1)")
    p.add_argument("--overrides", default="",
                   help="Extra YAML overrides forwarded to load_hyperpyyaml")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe the experiment folder and retrain. By default, "
                        "pairs whose results already exist are skipped.")
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
    p.add_argument("--warm-workers", type=int, default=1,
                   help="Concurrent warm workers (default: 1). Each warm "
                        "saturates GPU; ↑ only if you have many distinct "
                        "(dataset, encoder) pairs and headroom.")
    p.add_argument("--train-workers", "-j", type=int, default=3,
                   help="Concurrent train workers (default: 3). `-j` alias.")
    _add_filter_args(p)
    if include_level:
        p.add_argument("--level", type=str, default=None, action="append",
                       help="Restrict to these level dirs (repeatable; "
                            "default: all levels in registry.yaml)")
    p.add_argument("--test-only", action="store_true",
                   help="Re-evaluate saved best trial; no warm or train "
                        "(single phase, uses --train-workers).")
    p.add_argument("--cache-only", action="store_true",
                   help="Warm phase only (skips train phase).")
    p.add_argument("--no-writer", action="store_true",
                   help="Train phase only — cache must already be warm.")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo every matching pair, including ones whose "
                        "results already exist. Prompts for confirmation "
                        "unless --yes is also passed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (which pairs would be queued / "
                        "skipped) and exit without doing any work.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the --overwrite confirmation prompt.")
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag forwarded to train (default: run1)")


def _add_status_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag to scan (default: run1)")
    _add_filter_args(p)


def _add_summary_args(p: argparse.ArgumentParser, *,
                      default_root_hint: str,
                      include_level: bool = False) -> None:
    p.add_argument("--out-dir", type=str, default=None,
                   help=f"Output dir (default: {default_root_hint}/_summary_<tag>)")
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag to scan (default: run1)")
    _add_filter_args(p)
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
