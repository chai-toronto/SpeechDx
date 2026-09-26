"""Top-level CLI for the SpeechDx harness.

Surface:
    python -m sdx <mode> <command> [flags]

Modes:    single, cross, cross-cat, data-eff, all
Commands: prep, warm, train, run, status, summary

If the first argument is a command rather than a mode, mode defaults to
``single`` — so ``python -m sdx run`` is shorthand for
``python -m sdx single run``. The old flat names (``sdx run-cross``,
``sdx summary-data-eff``, ``sdx run-all``, ...) are gone as of commit 1 of
the rewrite.

Every (mode, command) cell is wired today. ``all <cmd>`` cells iterate
single → cross → cross-cat → data-eff, propagating filters and the
command's flags; ``--skip-mode`` and ``--stop-on-failure`` control the
chain (continue past failures by default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from sdx.log import _JobDashboard, _now, _run_logged_job, _slug

MODES = ("single", "cross", "cross-cat", "data-eff", "all")
COMMANDS = ("prep", "warm", "train", "run", "status", "summary")
TRAIN_LOGS_ROOT = Path("logs/train")
TRAIN_CROSS_LOGS_ROOT = Path("logs/train_cross")
TRAIN_CROSS_CAT_LOGS_ROOT = Path("logs/train_cross_cat")
TRAIN_DATA_EFF_LOGS_ROOT = Path("logs/train_data_eff")


# ---------------------------------------------------------------------------
# Filter resolution — every command takes -e/-d/-t (encoder/dataset/task)
# allowlists. Empty filter = "everything found".
# ---------------------------------------------------------------------------

def _resolve_tasks(args: argparse.Namespace, mode: str) -> list[str]:
    """Sorted task stems matching the --task / --dataset filters for ``mode``."""
    datasets = getattr(args, "dataset", None)
    tasks = getattr(args, "task", None)
    if mode in ("single", "data-eff"):
        from sdx.orchestrator import discover_tasks
        return discover_tasks(datasets=datasets, tasks=tasks)
    if mode == "cross":
        from sdx.orchestrator_cross import discover_tasks as cross_disc
        return cross_disc(datasets=datasets, tasks=tasks, include_categories=False)
    if mode == "cross-cat":
        from sdx.orchestrator_cross import discover_tasks as cross_disc
        return cross_disc(datasets=datasets, tasks=tasks, include_categories=True)
    raise ValueError(f"unknown mode for task selection: {mode}")


def _resolve_pairs(args: argparse.Namespace, mode: str) -> list[tuple[str, str]]:
    """Cartesian product of resolved tasks and encoders matching the filters."""
    from sdx.registry import encoders as registry_encoders
    tasks = _resolve_tasks(args, mode)
    enc_filter = set(getattr(args, "encoder", None) or [])
    encs = list(registry_encoders().keys())
    if enc_filter:
        encs = [e for e in encs if e in enc_filter]
    return [(t, e) for t in tasks for e in encs]


def _merged_overrides(args: argparse.Namespace) -> str:
    """``--test-only`` injects ``test_only: true`` into the override string
    forwarded to load_hyperpyyaml; merges with any user-supplied overrides
    via newline join."""
    base = getattr(args, "overrides", "") or ""
    if getattr(args, "test_only", False):
        return (base + "\n" if base else "") + "test_only: true"
    return base


def _train_one(args: argparse.Namespace, task: str, encoder: str, *,
               level_dir: str | None = None) -> None:
    """Per-pair single-mode train; auto-routes to per-fold CV via task yaml.
    Soft-skips when the underlying dataset isn't staged."""
    from sdx.orchestrator import is_cv
    from sdx.prep.dispatch import ensure_manifest
    from sdx.prep.raw import MissingDatasetError
    try:
        ensure_manifest(task)
    except MissingDatasetError as e:
        print(f"⚠ {task} × {encoder}: skipped — {e}", flush=True)
        return
    if is_cv(task):
        from sdx.train_cv import cmd_train_cv as _fn
    else:
        from sdx.train import cmd_train as _fn
    from sdx.config import resolve_probe
    probe_name, probe_yaml = resolve_probe(getattr(args, "probe", None))
    _fn(
        task, encoder,
        probe=probe_name, probe_yaml=probe_yaml,
        tag=args.tag, overrides=_merged_overrides(args),
        level_dir=level_dir,
    )


def _run_single_train_leaf(args: argparse.Namespace) -> None:
    """Hidden leaf path used by the run orchestrators for one-pair train jobs."""
    from sdx.orchestrator import (
        get_output_folder, is_complete,
    )
    from sdx.run import _has_trained_model

    pairs = _resolve_pairs(args, "single")
    if not pairs:
        print("train: no matching pairs for filters.")
        return

    overwrite = getattr(args, "overwrite", False)
    test_only = getattr(args, "test_only", False)
    level_dir = getattr(args, "level_dir", None)
    if level_dir is not None:
        from sdx.orchestrator_data_eff import (
            get_output_folder as _level_output_folder,
        )

    for task, encoder in pairs:
        folder = (_level_output_folder(task, encoder, level_dir, args.tag)
                  if level_dir is not None
                  else get_output_folder(task, encoder, args.tag,
                                         probe=getattr(args, "probe", None)))
        if test_only:
            if not _has_trained_model(folder, task):
                print(f"  SKIP (no trained model): {task} × {encoder}")
                continue
        elif folder.exists() and is_complete(folder, task) and not overwrite:
            print(f"  SKIP (results exist): {task} × {encoder}")
            continue
        if folder.exists() and overwrite:
            import shutil
            shutil.rmtree(folder)
        _train_one(args, task, encoder, level_dir=level_dir)


def _run_cross_train_leaf(args: argparse.Namespace) -> None:
    """Hidden leaf path used by the run orchestrators for one-pair cross jobs."""
    import shutil

    from sdx.orchestrator_cross import (
        get_output_folder as cross_output_folder,
    )
    from sdx.prep.raw import MissingDatasetError
    from sdx.run_cross_category import CATEGORY_EXPS_ROOT
    from sdx.status import _is_complete_any
    from sdx.train_cross import cmd_train_cross

    pairs = _resolve_pairs(args, args.mode)
    if not pairs:
        print(f"{args.mode} train: no matching pairs for filters.")
        return

    overwrite = getattr(args, "overwrite", False)
    test_only = getattr(args, "test_only", False)
    exps_root = CATEGORY_EXPS_ROOT if args.mode == "cross-cat" else None
    skipped_tasks: set[str] = set()
    for task, encoder in pairs:
        if task in skipped_tasks:
            continue
        folder = cross_output_folder(task, encoder, args.tag, exps_root=exps_root)
        if test_only:
            if not (folder / "best_hparams.yaml").exists():
                print(f"  SKIP (no trained model): {task} × {encoder}")
                continue
        elif folder.exists() and _is_complete_any(folder) and not overwrite:
            print(f"  SKIP (results exist): {task} × {encoder}")
            continue
        if folder.exists() and overwrite:
            shutil.rmtree(folder)
        try:
            cmd_train_cross(
                task, encoder,
                tag=args.tag, overrides=_merged_overrides(args),
            )
        except MissingDatasetError as e:
            print(f"⚠ {task} × {encoder}: skipped — {e}", flush=True)
            skipped_tasks.add(task)


def _run_dataeff_train_leaf(args: argparse.Namespace) -> None:
    """Hidden leaf path used by the run orchestrators for one-pair data-eff jobs."""
    import shutil

    from sdx.orchestrator import is_complete
    from sdx.orchestrator_data_eff import get_output_folder as de_output_folder
    from sdx.registry import data_eff_levels
    from sdx.run import _has_trained_model

    pairs = _resolve_pairs(args, "data-eff")
    if not pairs:
        print("data-eff train: no matching pairs for filters.")
        return

    levels = args.level or [name for name, _ in data_eff_levels()]
    overwrite = getattr(args, "overwrite", False)
    test_only = getattr(args, "test_only", False)
    for level in levels:
        for task, encoder in pairs:
            folder = de_output_folder(task, encoder, level, args.tag)
            if test_only:
                if not _has_trained_model(folder, task):
                    print(f"  SKIP (no trained model): {task} × {encoder} @ {level}")
                    continue
            elif folder.exists() and is_complete(folder, task) and not overwrite:
                print(f"  SKIP (results exist): {task} × {encoder} @ {level}")
                continue
            if folder.exists() and overwrite:
                shutil.rmtree(folder)
            _train_one(args, task, encoder, level_dir=level)


def _run_dashboard_jobs(
    jobs: list[tuple[str, Path, str, Callable[[], None], tuple[str, ...]]], *,
    logs_root: Path,
    header_lines: tuple[str, ...],
    done_label: str = "completed",
    skipped: int = 0,
) -> None:
    """Run local jobs under the shared terminal/log dashboard."""
    dashboard = _JobDashboard(
        total_jobs=len(jobs),
        logs_root=logs_root,
        header_lines=header_lines,
    )
    for idx, (label, rel_log_path, role, fn, detail_lines) in enumerate(jobs):
        log_path = dashboard.run_log_dir / rel_log_path
        dashboard.start_job(idx, role, label, log_path)
        ok, elapsed = _run_logged_job(
            label, log_path, fn, detail_lines=detail_lines,
        )
        dashboard.finish_job(idx, label, ok=ok, elapsed=elapsed, log_path=log_path)
    dashboard.print_summary(done_label=done_label, skipped=skipped)


def _delegate_train_to_run(
    args: argparse.Namespace,
    run_fn: Callable[[argparse.Namespace], None],
) -> None:
    """Run the train subcommand via the existing no-writer orchestrator."""
    ns = argparse.Namespace(**vars(args))
    ns.no_writer = True
    ns.cache_only = False
    ns.device = getattr(args, "device", None)
    ns.max_workers = max(1, getattr(args, "workers", 3))
    run_fn(ns)


# ---------------------------------------------------------------------------
# Handlers — one per (mode, command) cell. Imports are inline so unrelated
# subcommands don't pull heavy dependencies into argparse startup.
# ---------------------------------------------------------------------------

def _h_prep(args: argparse.Namespace) -> None:
    """Mode-agnostic prep — works for single, cross, and cross-cat stems.
    Tasks whose datasets aren't staged (and have no auto-download script)
    log a warning and are skipped; the rest of the prep run continues."""
    from sdx.prep.dispatch import ensure_manifest, manifest_paths
    from sdx.prep.raw import MissingDatasetError
    tasks = _resolve_tasks(args, args.mode)
    if not tasks:
        print("prep: no matching tasks for filters.")
        return
    print(f"prep: {len(tasks)} task(s)")
    overwrite = getattr(args, "overwrite", False)
    skipped: list[tuple[str, str]] = []
    for task in tasks:
        if overwrite:
            for p in manifest_paths(task):
                p.unlink(missing_ok=True)
        try:
            ensure_manifest(task)
        except MissingDatasetError as e:
            print(f"⚠ {task}: skipped — {e}", flush=True)
            skipped.append((task, str(e)))
    if skipped:
        print(f"\nprep: completed; skipped {len(skipped)} task(s) with unavailable data.")


def _h_single_warm(args: argparse.Namespace) -> None:
    from sdx.prep.dispatch import ensure_manifest
    from sdx.prep.raw import MissingDatasetError
    from sdx.warm import cmd_warm_jobs
    pairs = _resolve_pairs(args, "single")
    if not pairs:
        print("warm: no matching (task, encoder) pairs for filters.")
        return
    if not _confirm_overwrite(
            args, "warm",
            what=f"wipe cache.hdf5 for {len(pairs)} (task, encoder) "
                 "pair(s) and re-extract from scratch."):
        return
    print(f"warm: {len(pairs)} (task, encoder) pair(s)")
    jobs: list[tuple[str, str, int | None]] = []
    skipped_tasks: set[str] = set()
    for task, encoder in pairs:
        if task in skipped_tasks:
            continue
        try:
            ensure_manifest(task)
        except MissingDatasetError as e:
            print(f"⚠ {task} × {encoder}: skipped — {e}", flush=True)
            skipped_tasks.add(task)
            continue
        jobs.append((task, encoder, None))
    cmd_warm_jobs(
        jobs,
        device=args.device,
        overwrite=getattr(args, "overwrite", False),
        max_workers=getattr(args, "workers", 1),
        probe=getattr(args, "probe", None),
    )


def _h_single_train(args: argparse.Namespace) -> None:
    """Sweep all matching (task, encoder) pairs; per-pair train auto-routes
    to per-fold CV via the task yaml. Skip-if-complete by default; pass
    ``--overwrite`` to wipe and retrain. ``--test-only`` re-evaluates the
    saved best trial without retraining (skips pairs without one)."""
    _validate_train_flags(args)
    if getattr(args, "leaf", False):
        _run_single_train_leaf(args)
        return
    level_dir = getattr(args, "level_dir", None)
    if level_dir is None:
        from sdx.run import cmd_run
        _delegate_train_to_run(
            args,
            lambda ns: cmd_run(ns, logs_root=TRAIN_LOGS_ROOT),
        )
        return
    from sdx.orchestrator import (
        get_output_folder, is_complete,
    )
    from sdx.run import _has_trained_model
    pairs = _resolve_pairs(args, "single")
    if not pairs:
        print("train: no matching pairs for filters.")
        return
    overwrite = getattr(args, "overwrite", False)
    test_only = getattr(args, "test_only", False)
    if level_dir is not None:
        from sdx.orchestrator_data_eff import (
            get_output_folder as _level_output_folder,
        )
    skipped = 0
    jobs: list[tuple[str, Path, str, Callable[[], None], tuple[str, ...]]] = []
    for task, encoder in pairs:
        folder = (_level_output_folder(task, encoder, level_dir, args.tag)
                  if level_dir is not None
                  else get_output_folder(task, encoder, args.tag,
                                         probe=getattr(args, "probe", None)))
        if test_only:
            if not _has_trained_model(folder, task):
                print(f"  SKIP (no trained model): {task} × {encoder}")
                skipped += 1
                continue
        elif folder.exists() and is_complete(folder, task) and not overwrite:
            print(f"  SKIP (results exist): {task} × {encoder}")
            skipped += 1
            continue
        label = f"{task} × {encoder}"
        rel_log_path = Path(f"{_slug(task)}__{_slug(encoder)}.log")

        def _fn(task=task, encoder=encoder, folder=folder, level_dir=level_dir):
            if folder.exists() and overwrite:
                import shutil
                shutil.rmtree(folder)
            _train_one(args, task, encoder, level_dir=level_dir)

        detail_lines = (
            f"tag={args.tag}",
            f"test_only={test_only}",
            f"overwrite={overwrite}",
        )
        jobs.append((label, rel_log_path, "train", _fn, detail_lines))

    _run_dashboard_jobs(
        jobs,
        logs_root=TRAIN_LOGS_ROOT,
        header_lines=(
            f"[{_now()}] Train start: {len(jobs)} pending, {skipped} skipped",
            f"           mode    : train",
            f"           tag     : {args.tag}",
        ),
        skipped=skipped,
    )


def _h_single_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from sdx.run import cmd_run
    from sdx.summary import cmd_summary
    _strict_phase_run(args, cmd_run, mode_label="single run",
                      summary_fn=cmd_summary)


def _h_single_status(args: argparse.Namespace) -> None:
    from sdx.status import cmd_status
    cmd_status(args)


def _h_single_summary(args: argparse.Namespace) -> None:
    from sdx.summary import cmd_summary
    cmd_summary(args)


def _h_cross_warm(args: argparse.Namespace) -> None:
    """Cross warm — self-contained warmer driven by main_cross.yaml +
    cross_tasks/<stem>.yaml only. Writes 3 caches per pair:
    ``<train_dataset>/{train,val}`` and ``<test_dataset>/val``."""
    from sdx.prep.dispatch import ensure_manifest
    from sdx.prep.raw import MissingDatasetError
    from sdx.warm_cross import cmd_warm_cross_jobs
    pairs = _resolve_pairs(args, args.mode)
    if not pairs:
        print(f"{args.mode} warm: no matching pairs for filters.")
        return
    if not _confirm_overwrite(
            args, f"{args.mode} warm",
            what=f"wipe cache.hdf5 for {len(pairs)} cross pair(s) "
                 "(train+test datasets) and re-extract from scratch."):
        return
    print(f"{args.mode} warm: {len(pairs)} cross pair(s)")
    skipped_tasks: set[str] = set()
    available_pairs: list[tuple[str, str]] = []
    for cross_task, encoder in pairs:
        if cross_task in skipped_tasks:
            continue
        try:
            ensure_manifest(cross_task)
        except MissingDatasetError as e:
            print(f"⚠ {cross_task} × {encoder}: skipped — {e}", flush=True)
            skipped_tasks.add(cross_task)
            continue
        available_pairs.append((cross_task, encoder))
    cmd_warm_cross_jobs(
        available_pairs,
        device=args.device,
        overwrite=getattr(args, "overwrite", False),
        max_workers=getattr(args, "workers", 1),
    )


def _h_cross_train(args: argparse.Namespace) -> None:
    """``cmd_train_cross`` auto-detects category vs non-category via the task
    yaml, so the per-pair entry point is shared with ``cross-cat train``.
    Skip-if-complete by default; ``--overwrite`` wipes and retrains.
    ``--test-only`` re-evaluates the saved best trial (skips pairs without
    one).

    Uses a mode-agnostic completeness check (``test_results.{txt,yaml}``);
    cross task stems aren't in single's TASKS_DIR, so the orchestrator's
    ``is_complete`` (which goes through ``is_cv``) would crash."""
    _validate_train_flags(args)
    if getattr(args, "leaf", False):
        _run_cross_train_leaf(args)
        return
    from sdx.run_cross import cmd_run_cross
    from sdx.run_cross_category import CATEGORY_EXPS_ROOT
    if args.mode == "cross-cat":
        _delegate_train_to_run(
            args,
            lambda ns: cmd_run_cross(
                ns,
                include_categories=True,
                exps_root=CATEGORY_EXPS_ROOT,
                logs_root=TRAIN_CROSS_CAT_LOGS_ROOT,
            ),
        )
        return
    _delegate_train_to_run(
        args,
        lambda ns: cmd_run_cross(ns, logs_root=TRAIN_CROSS_LOGS_ROOT),
    )


def _h_cross_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from sdx.run_cross import cmd_run_cross
    from sdx.summary_cross import cmd_summary_cross
    _strict_phase_run(args, cmd_run_cross, mode_label="cross run",
                      summary_fn=cmd_summary_cross)


def _h_cross_status(args: argparse.Namespace) -> None:
    from sdx.status import cmd_status_cross
    cmd_status_cross(args)


def _h_cross_summary(args: argparse.Namespace) -> None:
    from sdx.summary_cross import cmd_summary_cross
    cmd_summary_cross(args)


def _h_crosscat_warm(args: argparse.Namespace) -> None:
    """Cross-cat warm — self-contained warmer driven by
    main_cross_category.yaml + cross_tasks/<stem>.yaml only. Per train
    dataset writes ``<dataset>/{train,val}``; per test dataset writes
    ``<dataset>/val`` only."""
    from sdx.prep.dispatch import ensure_manifest
    from sdx.prep.raw import MissingDatasetError
    from sdx.warm_cross_cat import cmd_warm_crosscat_jobs
    pairs = _resolve_pairs(args, args.mode)
    if not pairs:
        print(f"{args.mode} warm: no matching pairs for filters.")
        return
    if not _confirm_overwrite(
            args, f"{args.mode} warm",
            what=f"wipe cache.hdf5 for {len(pairs)} cross-cat pair(s) "
                 "(every contributing dataset) and re-extract from scratch."):
        return
    print(f"{args.mode} warm: {len(pairs)} cross-cat pair(s)")
    skipped_tasks: set[str] = set()
    available_pairs: list[tuple[str, str]] = []
    for cat_task, encoder in pairs:
        if cat_task in skipped_tasks:
            continue
        try:
            ensure_manifest(cat_task)
        except MissingDatasetError as e:
            print(f"⚠ {cat_task} × {encoder}: skipped — {e}", flush=True)
            skipped_tasks.add(cat_task)
            continue
        available_pairs.append((cat_task, encoder))
    cmd_warm_crosscat_jobs(
        available_pairs,
        device=args.device,
        overwrite=getattr(args, "overwrite", False),
        max_workers=getattr(args, "workers", 1),
    )


def _h_crosscat_train(args: argparse.Namespace) -> None:
    _h_cross_train(args)


def _h_crosscat_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from sdx.run_cross_category import cmd_run_cross_category
    from sdx.summary_cross import cmd_summary_cross_category
    _strict_phase_run(args, cmd_run_cross_category, mode_label="cross-cat run",
                      summary_fn=cmd_summary_cross_category)


def _h_crosscat_status(args: argparse.Namespace) -> None:
    from sdx.status import cmd_status_crosscat
    cmd_status_crosscat(args)


def _h_crosscat_summary(args: argparse.Namespace) -> None:
    from sdx.summary_cross import cmd_summary_cross_category
    cmd_summary_cross_category(args)


def _h_dataeff_prep(args: argparse.Namespace) -> None:
    """Build per-level subsampled manifests under exps/data_eff/<level>/<task>/.

    Ensures the upstream single-task source is up to date first (delegating to
    ``_h_prep``), then subsamples it into each level's manifest dir. Honors
    ``--overwrite`` so changes to the source manifest propagate downstream;
    without it, ``ensure_manifest_data_eff`` short-circuits when the level
    files already exist.
    """
    _h_prep(args)
    from sdx.orchestrator_data_eff import (
        LEVELS, discover_tasks, ensure_manifest_data_eff,
    )
    from sdx.prep.raw import MissingDatasetError
    tasks = discover_tasks(args.dataset, args.task)
    overwrite = bool(getattr(args, "overwrite", False))
    for ts in tasks:
        for level_dir, _ in LEVELS:
            try:
                ensure_manifest_data_eff(ts, level_dir, overwrite=overwrite)
            except MissingDatasetError as e:
                print(f"⚠ {ts}@{level_dir}: skipped — {e}", flush=True)
                break  # skip remaining levels for this task


def _h_dataeff_warm(args: argparse.Namespace) -> None:
    """data-eff warm == single warm. Caches are shared across levels."""
    _h_single_warm(args)


def _h_dataeff_train(args: argparse.Namespace) -> None:
    """Sweep all matching (task, encoder) pairs at every data-eff level by
    default; ``--level`` filters to specific levels. Skip-if-complete by
    default; pass ``--overwrite`` to wipe and retrain. ``--test-only``
    re-evaluates the saved best trial (skips pairs without one)."""
    _validate_train_flags(args)
    if getattr(args, "leaf", False):
        _run_dataeff_train_leaf(args)
        return
    from sdx.run_data_eff import cmd_run_data_eff
    _delegate_train_to_run(
        args,
        lambda ns: cmd_run_data_eff(ns, logs_root=TRAIN_DATA_EFF_LOGS_ROOT),
    )


def _h_dataeff_run(args: argparse.Namespace) -> None:
    _validate_run_flags(args)
    from sdx.run_data_eff import cmd_run_data_eff
    from sdx.summary_data_eff import cmd_summary_data_eff
    _strict_phase_run(args, cmd_run_data_eff, mode_label="data-eff run",
                      summary_fn=cmd_summary_data_eff)


def _h_dataeff_status(args: argparse.Namespace) -> None:
    from sdx.status import cmd_status_dataeff
    cmd_status_dataeff(args)


def _h_dataeff_summary(args: argparse.Namespace) -> None:
    from sdx.summary_data_eff import cmd_summary_data_eff
    cmd_summary_data_eff(args)


def _all_forward(args: argparse.Namespace,
                 modes: list[tuple[str, Callable[[argparse.Namespace], None]]],
                 *, label: str,
                 mutate: Callable[[argparse.Namespace, str], None] | None = None) -> None:
    """Iterate ``modes``, calling each handler with a copy of ``args`` whose
    ``mode`` is set to that mode's name. ``mutate`` is an optional per-mode
    namespace patch (e.g. resolve sentinel defaults). Continues past mode
    failures by default; ``--stop-on-failure`` aborts the chain instead."""
    skip = set(getattr(args, "skip_mode", None) or [])
    failures: list[tuple[str, BaseException]] = []
    for name, fn in modes:
        if name in skip:
            print(f"=== {label}: skipping mode {name!r} (--skip-mode) ===")
            continue
        print(f"\n=== {label}: starting mode {name!r} ===")
        ns = argparse.Namespace(**vars(args))
        ns.mode = name
        if mutate is not None:
            mutate(ns, name)
        try:
            fn(ns)
        except BaseException as e:
            failures.append((name, e))
            print(f"=== {label}: mode {name!r} FAILED: {e!r} ===")
            if getattr(args, "stop_on_failure", False):
                raise
    if failures:
        names = ", ".join(n for n, _ in failures)
        raise SystemExit(f"{label}: {len(failures)} mode(s) failed: {names}")


def _h_all_prep(args: argparse.Namespace) -> None:
    """Build manifests across single → cross → cross-cat → data-eff."""
    modes = [
        ("single",    _h_prep),
        ("cross",     _h_prep),
        ("cross-cat", _h_prep),
        ("data-eff",  _h_dataeff_prep),
    ]
    _all_forward(args, modes, label="all prep")


def _h_all_warm(args: argparse.Namespace) -> None:
    """Warm caches across every mode."""
    modes = [
        ("single",    _h_single_warm),
        ("cross",     _h_cross_warm),
        ("cross-cat", _h_crosscat_warm),
        ("data-eff",  _h_dataeff_warm),
    ]
    _all_forward(args, modes, label="all warm")


def _h_all_train(args: argparse.Namespace) -> None:
    """Sequentially train every mode (single → cross → cross-cat → data-eff).

    Equivalent to ``all run --no-writer``: skips the warm phase in each mode,
    assuming caches are already warm. Use ``all run`` to warm-then-train.
    Filters and ``-j`` (train workers) propagate uniformly to every mode.
    """
    modes = [
        ("single",    _h_single_train),
        ("cross",     _h_cross_train),
        ("cross-cat", _h_crosscat_train),
        ("data-eff",  _h_dataeff_train),
    ]
    _all_forward(args, modes, label="all train")


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
    _all_forward(args, modes, label="all run")


def _h_all_status(args: argparse.Namespace) -> None:
    from sdx.status import cmd_status_all
    cmd_status_all(args)


def _h_all_summary(args: argparse.Namespace) -> None:
    """Aggregate results across every mode. Each mode writes under its own
    ``<mode-root>/_summary_<tag>`` — ``all summary`` does not accept
    ``--out-dir`` because a single dir would collide across modes; run
    per-mode summary if you need that. ``--level`` is consumed by data-eff
    only."""
    modes = [
        ("single",    _h_single_summary),
        ("cross",     _h_cross_summary),
        ("cross-cat", _h_crosscat_summary),
        ("data-eff",  _h_dataeff_summary),
    ]

    def _seed_out_dir(ns: argparse.Namespace, _name: str) -> None:
        ns.out_dir = None

    _all_forward(args, modes, label="all summary", mutate=_seed_out_dir)


def _validate_run_flags(args: argparse.Namespace) -> None:
    if getattr(args, "cache_only", False) and getattr(args, "test_only", False):
        raise SystemExit("--cache-only and --test-only are mutually exclusive")
    if getattr(args, "cache_only", False) and getattr(args, "no_writer", False):
        raise SystemExit("--cache-only and --no-writer are mutually exclusive")


def _validate_train_flags(args: argparse.Namespace) -> None:
    if getattr(args, "test_only", False) and getattr(args, "overwrite", False):
        raise SystemExit("--test-only and --overwrite are mutually exclusive")


def _confirm_overwrite(args: argparse.Namespace, mode_label: str,
                       *, what: str | None = None) -> bool:
    """Interactive ``[y/N]`` prompt before a destructive ``--overwrite``.

    Bypassed by ``--yes``. Also bypassed when ``--dry-run`` is set (nothing
    actually gets destroyed). Returns True if the action should proceed.
    ``what`` overrides the default ``run``-flavored description for callers
    like ``warm`` whose destruction target is different.
    """
    if not getattr(args, "overwrite", False):
        return True
    if getattr(args, "yes", False) or getattr(args, "dry_run", False):
        return True
    detail = what or (
        "redo every matching pair, including ones whose results already "
        "exist.\n   Existing results will be wiped."
    )
    print(f"\n!! {mode_label} --overwrite will {detail}")
    print("   This is not reversible.")
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
                      *, mode_label: str,
                      summary_fn: Callable[[argparse.Namespace], None] | None = None) -> None:
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

    # Phase 3: summary. Skipped under --dry-run (nothing happened) and
    # --cache-only (no new train results to summarize). Summary's parser
    # adds attributes (out_dir, level) that the run parser doesn't, so
    # inject defaults before calling.
    dry = getattr(args, "dry_run", False)
    if summary_fn is not None and do_train and not dry:
        print(f"\n=== {mode_label}: phase 3 — summary ===")
        sum_ns = argparse.Namespace(**vars(args))
        if not hasattr(sum_ns, "out_dir"):
            sum_ns.out_dir = None
        if not hasattr(sum_ns, "level"):
            sum_ns.level = None
        try:
            summary_fn(sum_ns)
        except Exception as e:
            print(f"  ⚠ summary skipped: {e!r}")


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


def _add_probe_arg(p: argparse.ArgumentParser) -> None:
    """``--probe`` for the single-mode commands (warm/train/run/status/summary)."""
    p.add_argument("--probe", default=None,
                   help="Downstream readout: probes/<PROBE>.yaml (e.g. ASP). "
                        "Default: the mean-pool linear probe (AvgTProbe). A "
                        "probe that declares `cache_pool: none` warms/reads "
                        "the T x D single/ cache; results land in "
                        "<encoder>-<PROBE>-<tag>/. Pass the same value to "
                        "warm, train, run, status and summary.")


def _add_prep_args(p: argparse.ArgumentParser) -> None:
    _add_filter_args(p)
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild manifests even if they already exist.")


def _add_warm_args(p: argparse.ArgumentParser) -> None:
    _add_filter_args(p)
    p.add_argument("--device", default=None,
                   help="Torch device override (e.g. cuda:0)")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent warm workers (default: 1). Warm jobs are "
                        "now lock-safe across processes, but each worker still "
                        "loads a full encoder, so increase only with headroom.")
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe the per-(dataset, encoder) cache.hdf5 files "
                        "before re-extracting. Prompts for confirmation "
                        "unless --yes is also passed.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the --overwrite confirmation prompt.")


def _add_train_args(p: argparse.ArgumentParser, *,
                    include_level_dir: bool = False,
                    include_level: bool = False) -> None:
    _add_filter_args(p)
    p.add_argument("--tag", default="run1",
                   help="Experiment tag (default: run1)")
    p.add_argument("--overrides", default="",
                   help="Extra YAML overrides forwarded to load_hyperpyyaml")
    p.add_argument("--workers", "-j", type=int, default=3,
                   help="Concurrent train workers (default: 3).")
    p.add_argument("--leaf", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true",
                   help="Wipe the experiment folder and retrain. By default, "
                        "pairs whose results already exist are skipped.")
    p.add_argument("--test-only", action="store_true",
                   help="Re-evaluate the saved best trial without retraining. "
                        "Pairs without a trained model are skipped. Mutually "
                        "exclusive with --overwrite.")
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
    """``run`` is the union of prep + warm + train + summary, so its flag
    surface forwards each phase's own flags."""
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
    p.add_argument("--overrides", default="",
                   help="Extra YAML overrides forwarded to train's "
                        "load_hyperpyyaml.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Summary output dir (default: "
                        "<mode-root>/_summary_<tag>).")
    p.add_argument("--cache-only", action="store_true",
                   help="Warm phase only (skips train phase). For train-only "
                        "without re-warming, use the `train` subcommand.")
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


def _add_status_args(p: argparse.ArgumentParser, *,
                     include_level: bool = False) -> None:
    p.add_argument("--tag", type=str, default="run1",
                   help="Experiment tag to scan (default: run1)")
    _add_filter_args(p)
    if include_level:
        p.add_argument("--level", type=str, default=None, action="append",
                       help="Restrict to these level dirs (repeatable)")


def _add_all_chain_args(p: argparse.ArgumentParser) -> None:
    """Shared ``all <cmd>`` chain controls: which modes to skip and whether
    to abort on the first per-mode failure."""
    p.add_argument("--skip-mode", action="append", default=None,
                   choices=list(MODES[:-1]),
                   help="Mode(s) to skip (repeatable)")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="Abort the chain on the first mode failure. "
                        "Default: continue and report failures at the end.")


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
        prog="sdx",
        description=("SpeechDx harness. Surface: "
                     "`sdx <mode> <command>`. <mode> defaults to 'single' "
                     "if the first arg is a command."),
    )
    mode_sub = parser.add_subparsers(dest="mode", required=True)

    # ---- single ----
    sp = mode_sub.add_parser("single", help="Single-dataset benchmark mode (default).")
    sub = sp.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more tasks."))
    single_warm = sub.add_parser("warm", help="Warm the HDF5 cache for one (task, encoder).")
    _add_warm_args(single_warm)
    single_train = sub.add_parser(
        "train", help="Train probe (auto-routes to per-fold CV via task yaml).")
    _add_train_args(single_train, include_level_dir=True)
    single_run = sub.add_parser("run", help="Sweep all incomplete (task, encoder) pairs.")
    _add_run_args(single_run)
    single_status = sub.add_parser("status", help="Per-task × per-encoder completion grid.")
    _add_status_args(single_status)
    single_summary = sub.add_parser("summary", help="Aggregate test results into per-metric CSVs.")
    _add_summary_args(single_summary, default_root_hint="exps/single_task")
    for _p in (single_warm, single_train, single_run, single_status, single_summary):
        _add_probe_arg(_p)

    # ---- cross ----
    cp = mode_sub.add_parser("cross", help="Zero-shot cross-task mode.")
    sub = cp.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more cross tasks."))
    _add_warm_args(sub.add_parser("warm", help="Warm cross-task caches."))
    _add_train_args(sub.add_parser("train", help="Train probe on a cross task."))
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete cross runs."))
    _add_status_args(sub.add_parser("status", help="Per-task × per-encoder completion grid for cross."))
    _add_summary_args(sub.add_parser("summary", help="Aggregate cross results."),
                      default_root_hint="exps/cross")

    # ---- cross-cat ----
    cap = mode_sub.add_parser("cross-cat", help="Multi-source cross-category mode.")
    sub = cap.add_subparsers(dest="command", required=True)
    _add_prep_args(sub.add_parser("prep", help="Build manifests for one or more category tasks."))
    _add_warm_args(sub.add_parser("warm", help="Warm category-task caches."))
    _add_train_args(sub.add_parser("train", help="Train probe on a category task."))
    _add_run_args(sub.add_parser("run", help="Sweep all incomplete cross-category runs."))
    _add_status_args(sub.add_parser("status", help="Per-task × per-encoder completion grid for cross-cat."))
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
    _add_status_args(sub.add_parser("status", help="Per-(level, task, encoder) completion grid for data-eff."),
                     include_level=True)
    _add_summary_args(sub.add_parser("summary", help="Aggregate data-eff results."),
                      default_root_hint="exps/data_eff", include_level=True)

    # ---- all ----
    ap = mode_sub.add_parser("all", help="Apply command to every mode in turn.")
    sub = ap.add_subparsers(dest="command", required=True)
    all_prep = sub.add_parser("prep", help="Build manifests across every mode.")
    _add_prep_args(all_prep)
    _add_all_chain_args(all_prep)
    all_warm = sub.add_parser("warm", help="Warm caches across every mode.")
    _add_warm_args(all_warm)
    _add_all_chain_args(all_warm)
    all_train = sub.add_parser(
        "train",
        help="Train every mode assuming caches are warm "
             "(equivalent to `all run --no-writer`).")
    _add_train_args(all_train, include_level=True)
    _add_all_chain_args(all_train)
    all_run = sub.add_parser("run", help="Sequentially run every mode (single → cross → cross-cat → data-eff).")
    _add_run_args(all_run, include_level=True)
    _add_all_chain_args(all_run)
    _add_status_args(sub.add_parser("status", help="Stacked completion grids for every mode."),
                     include_level=True)
    all_summary = sub.add_parser("summary", help="Summarize results across every mode.")
    _add_filter_args(all_summary)
    all_summary.add_argument("--tag", type=str, default="run1",
                             help="Experiment tag to scan (default: run1)")
    all_summary.add_argument("--level", type=str, default=None, action="append",
                             help="Restrict to these data-eff level dirs "
                                  "(repeatable; ignored by single/cross/cross-cat)")
    _add_all_chain_args(all_summary)

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Inject ``single`` if the first arg is a command rather than a mode.

    Allows ``python -m sdx run`` as shorthand for ``python -m sdx single run``.
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
