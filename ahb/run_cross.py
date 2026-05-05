"""``ahb run-cross`` — orchestrator for cross-task training runs.

Salvaged from ``run_all_cross.py:cmd_run`` (lines 558-748). Same
shared-cache reader scheduling as ``ahb/run.py`` but operates on cross
tasks (3 caches per pair instead of 2) and dispatches to the mode-aware
``ahb cross warm`` + ``ahb cross train`` subcommands.

The ``exps_root`` parameter lets ``ahb/run_cross_category.py`` reuse
this entry point with the category-specific output root
(``./exps/cross_cat``) and tasks subset (``category_*``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ahb.log import _emit, _now, _slug, _JobDashboard
from ahb.orchestrator_cross import (
    EXPS_ROOT,
    LOGS_ROOT,
    discover_tasks,
    forget_task_ids,
    get_output_folder,
    get_task_info,
    is_complete,
    needed_keys,
    task_weight,
)
from ahb.prep.dispatch import ensure_manifest
from ahb.registry import encoders as registry_encoders


def _run_subprocess(cmd: list[str], log_path: Path, label: str) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [
            os.getcwd(), os.environ.get("PYTHONPATH", "")
        ])),
    }
    with log_path.open("a", buffering=1) as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  Command: {' '.join(cmd)}\n")
        f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")
        f.flush()
        try:
            subprocess.run(cmd, check=True, stdout=f,
                           stderr=subprocess.STDOUT, env=env)
            f.write(f"\n--- {label} subprocess OK ---\n")
            return True
        except subprocess.CalledProcessError as e:
            f.write(f"\n--- {label} subprocess FAILED (exit {e.returncode}) ---\n")
            return False


def _execute_job_cross(task_stem: str, model_name: str, role: str, *,
                       device: str | None, test_only: bool, cache_only: bool,
                       tag: str, log_path: Path,
                       mode: str = "cross",
                       overwrite: bool = False,
                       overrides: str = "") -> tuple[str, bool, float]:
    label = f"{task_stem} × {model_name}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write(f"{'='*60}\n  Running: {label}\n")
        f.write(f"  Role: {role}, test_only={test_only}, cache_only={cache_only}, tag={tag}\n")
        f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")

    train_extras = ["--leaf", "--tag", tag]
    if overwrite:
        train_extras.append("--overwrite")
    user_ov = ["--overrides", overrides] if overrides else []
    base = ["python", "-m", "ahb", mode]
    target = ["-t", task_stem, "-e", model_name]
    start = time.time()
    success = True

    if cache_only:
        cmd = [*base, "warm", *target]
        if device:
            cmd.append(f"--device={device}")
        success = _run_subprocess(cmd, log_path, f"{mode}-warm")
    elif test_only:
        merged = (overrides + "\n" if overrides else "") + "test_only: true"
        cmd = [*base, "train", *target, *train_extras,
               "--overrides", merged]
        success = _run_subprocess(cmd, log_path, f"{mode}-test")
    elif role == "writer":
        warm_cmd = [*base, "warm", *target]
        if device:
            warm_cmd.append(f"--device={device}")
        if not _run_subprocess(warm_cmd, log_path, f"{mode}-warm"):
            success = False
        if success:
            train_cmd = [*base, "train", *target, *train_extras, *user_ov]
            success = _run_subprocess(train_cmd, log_path, f"{mode}-train")
    else:  # reader
        cmd = [*base, "train", *target, *train_extras, *user_ov]
        success = _run_subprocess(cmd, log_path, f"{mode}-train")

    elapsed = time.time() - start
    with log_path.open("a") as f:
        marker = "✓ OK" if success else "✗ FAILED"
        f.write(f"\n{'='*60}\n  {marker}: {label} in {elapsed/60:.1f} min\n")
        f.write(f"  Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")
    return label, success, elapsed


def cmd_run_cross(args: argparse.Namespace, *,
                  include_categories: bool = False,
                  exps_root: Path | None = None,
                  logs_root: Path | None = None) -> None:
    """Run all cross-task × encoder combos with shared-cache reader scheduling."""
    exps_root = exps_root or EXPS_ROOT
    logs_root = logs_root or LOGS_ROOT

    tasks = discover_tasks(args.dataset, args.task,
                           include_categories=include_categories)
    skipped = 0
    test_only = getattr(args, "test_only", False)
    cache_only = getattr(args, "cache_only", False)
    no_writer = getattr(args, "no_writer", False)
    overwrite = getattr(args, "overwrite", False)
    dry_run = getattr(args, "dry_run", False)
    tag = getattr(args, "tag", "run1")
    overrides = getattr(args, "overrides", "") or ""

    all_encoders = registry_encoders()
    if args.encoder:
        encoders = {e: all_encoders[e] for e in args.encoder}
    else:
        encoders = all_encoders

    pending: list[tuple[str, str]] = []
    completed_by_ds_enc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_stem in tasks:
        dataset, _ = get_task_info(task_stem)
        for model_name in encoders:
            folder = get_output_folder(task_stem, model_name, tag,
                                       exps_root=exps_root)
            if cache_only:
                pass
            elif test_only:
                if not (folder / "best_hparams.yaml").exists():
                    _emit(f"[{_now()}] SKIP (no trained model): {task_stem} × {model_name}")
                    skipped += 1
                    continue
            elif not overwrite and is_complete(folder, task_stem):
                _emit(f"[{_now()}] SKIP (done)            : {task_stem} × {model_name}")
                skipped += 1
                completed_by_ds_enc[(dataset, model_name)].append(task_stem)
                continue
            pending.append((task_stem, model_name))

    if dry_run:
        print(f"\n[dry-run] would queue {len(pending)} pair(s); skipped {skipped}.")
        for ts, mn in pending:
            print(f"  {ts} × {mn}")
        return

    needed_tasks = {ts for ts, _ in pending}
    needed_tasks |= {ts for lst in completed_by_ds_enc.values() for ts in lst}
    for ts in sorted(needed_tasks):
        try:
            ensure_manifest(ts)
            forget_task_ids(ts, exps_root=exps_root)
        except Exception as e:
            print(f"⚠ prepare_data for {ts} failed: {e}", flush=True)

    written: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for (ds, enc), tlist in completed_by_ds_enc.items():
        for ts in tlist:
            try:
                written[(ds, enc)] |= needed_keys(ts, exps_root=exps_root)
            except Exception:
                pass

    total_jobs = len(pending)
    pending.sort(key=lambda j: -task_weight(j[0], exps_root=exps_root))

    max_workers = max(1, args.max_workers)
    header_title = ("Train start" if getattr(args, "command", None) == "train"
                    else "Run start  ")
    mode_name = "cross-cat" if include_categories else "cross"
    mode = (
        "cache-only" if cache_only
        else "test-only" if test_only
        else ("train-only" if no_writer and getattr(args, "command", None) == "train"
              else ("train+test (no-writer)" if no_writer else "train+test"))
    )
    dashboard = _JobDashboard(
        total_jobs=total_jobs,
        logs_root=logs_root,
        header_lines=(
            f"[{_now()}] {header_title}: {total_jobs} pending, {skipped} skipped",
            f"           workers : up to {max_workers} concurrent",
            f"           mode    : {mode}  ({mode_name})",
            f"           tag     : {tag}",
        ),
    )

    # Warm owns the correctness lock at the cache boundary. These per-key
    # locks are just a same-process scheduler hint so one run invocation
    # doesn't launch duplicate warmers for the same shared cache.
    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()

    def _execute(job: tuple[str, str], idx: int, role: str) -> tuple[str, bool, float]:
        task_stem, model_name = job
        label = f"{task_stem} × {model_name}"
        log_path = dashboard.run_log_dir / f"{_slug(task_stem)}__{_slug(model_name)}.log"

        dashboard.start_job(idx, role, label, log_path)
        result = _execute_job_cross(
            task_stem, model_name, role,
            device=args.device, test_only=test_only, cache_only=cache_only,
            tag=tag, log_path=log_path,
            mode=("cross-cat" if include_categories else "cross"),
            overwrite=overwrite,
            overrides=overrides,
        )
        _, ok, elapsed = result
        dashboard.finish_job(idx, label, ok=ok, elapsed=elapsed, log_path=log_path)
        return result

    def dispatch(job: tuple[str, str], idx: int) -> tuple[str, bool, float]:
        task_stem, model_name = job
        dataset, _ = get_task_info(task_stem)
        key = (dataset, model_name)
        label = f"{task_stem} × {model_name}"

        if test_only:
            return _execute(job, idx, "test")
        if no_writer:
            return _execute(job, idx, "reader")

        try:
            needed = needed_keys(task_stem, exps_root=exps_root)
        except Exception:
            needed = None

        is_reader = False
        if needed is not None:
            with written_lock:
                is_reader = needed.issubset(written[key])

        if is_reader:
            if cache_only:
                _emit(f"[{_now()}] SKIP (cache covered)   : {label}")
                return label, True, 0.0
            return _execute(job, idx, "reader")

        ds_lock = ds_enc_locks[key]
        ds_lock.acquire()
        released = False
        try:
            if needed is not None:
                with written_lock:
                    satisfied = needed.issubset(written[key])
            else:
                satisfied = False
            if satisfied:
                ds_lock.release()
                released = True
                if cache_only:
                    _emit(f"[{_now()}] SKIP (cache covered)   : {label}")
                    return label, True, 0.0
                return _execute(job, idx, "reader")
            result = _execute(job, idx, "writer")
            _, success, _ = result
            if success and needed is not None:
                with written_lock:
                    written[key] |= needed
            return result
        finally:
            if not released:
                ds_lock.release()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(dispatch, job, i): job for i, job in enumerate(pending)}
        for future in as_completed(futures):
            future.result()

    done_label = "cache warmed" if cache_only else "completed"
    dashboard.print_summary(done_label=done_label, skipped=skipped)
