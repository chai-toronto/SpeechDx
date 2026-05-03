"""``ahb run-cross`` — orchestrator for cross-task training runs.

Salvaged from ``run_all_cross.py:cmd_run`` (lines 558-748). Same
writer/reader lock semantics as ``ahb/run.py`` but operates on cross
tasks (3 caches per pair instead of 2) and dispatches to
``ahb warm-cross`` + ``ahb train-cross`` subcommands.

The ``exps_root`` parameter lets ``ahb/run_cross_category.py`` reuse
this entry point with the category-specific output root
(``./cross_cat_exps``) and tasks subset (``category_*``).
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

from ahb.log import ROLE_W, _Progress, _emit, _now, _slug, _tail
from ahb.orchestrator_cross import (
    EXPS_ROOT,
    LOGS_ROOT,
    discover_tasks,
    forget_task_ids,
    get_output_folder,
    get_task_info,
    has_ci_results,
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
                       log_path: Path) -> tuple[str, bool, float]:
    label = f"{task_stem} × {model_name}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write(f"{'='*60}\n  Running: {label}\n")
        f.write(f"  Role: {role}, test_only={test_only}, cache_only={cache_only}\n")
        f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")

    start = time.time()
    success = True

    if cache_only:
        cmd = ["python", "-m", "ahb", "warm-cross", task_stem, model_name]
        if device:
            cmd.append(f"--device={device}")
        success = _run_subprocess(cmd, log_path, "warm-cross")
    elif test_only:
        cmd = ["python", "-m", "ahb", "train-cross", task_stem, model_name,
               "--overrides", "test_only: true"]
        success = _run_subprocess(cmd, log_path, "test-cross")
    elif role == "writer":
        warm_cmd = ["python", "-m", "ahb", "warm-cross", task_stem, model_name]
        if device:
            warm_cmd.append(f"--device={device}")
        if not _run_subprocess(warm_cmd, log_path, "warm-cross"):
            success = False
        if success:
            train_cmd = ["python", "-m", "ahb", "train-cross", task_stem, model_name]
            success = _run_subprocess(train_cmd, log_path, "train-cross")
    else:  # reader
        cmd = ["python", "-m", "ahb", "train-cross", task_stem, model_name]
        success = _run_subprocess(cmd, log_path, "train-cross")

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
    """Run all cross-task × encoder combos with writer/reader serialization."""
    exps_root = exps_root or EXPS_ROOT
    logs_root = logs_root or LOGS_ROOT

    tasks = discover_tasks(args.dataset, args.task,
                           include_categories=include_categories)
    skipped = completed = 0
    failed: list[str] = []
    failed_log_paths: dict[str, Path] = {}
    test_only = getattr(args, "test_only", False)
    cache_only = getattr(args, "cache_only", False)
    no_writer = getattr(args, "no_writer", False)

    all_encoders = registry_encoders()
    if args.encoder:
        encoders = {e: all_encoders[e] for e in args.encoder}
    else:
        encoders = all_encoders

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = logs_root / run_stamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str]] = []
    completed_by_ds_enc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name,
                                       exps_root=exps_root)
            if cache_only:
                pass
            elif test_only:
                if has_ci_results(folder, task_stem):
                    _emit(f"[{_now()}] SKIP (CI present)     : {task_stem} × {model_name}")
                    skipped += 1
                    continue
                if not (folder / "best_hparams.yaml").exists():
                    _emit(f"[{_now()}] SKIP (no trained model): {task_stem} × {model_name}")
                    skipped += 1
                    continue
            elif is_complete(folder, task_stem):
                _emit(f"[{_now()}] SKIP (done)            : {task_stem} × {model_name}")
                skipped += 1
                completed_by_ds_enc[(dataset, model_name)].append(task_stem)
                continue
            pending.append((task_stem, model_name))

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
    progress = _Progress(total_jobs)
    pending.sort(key=lambda j: -task_weight(j[0], exps_root=exps_root))

    max_workers = max(1, args.max_workers)
    mode = (
        "cache-only" if cache_only
        else "test-only" if test_only
        else ("train+test (no-writer)" if no_writer else "train+test")
    )
    _emit(
        "",
        f"[{_now()}] Run start  : {total_jobs} pending, {skipped} skipped",
        f"           workers : up to {max_workers} concurrent",
        f"           mode    : {mode}  (cross)",
        f"           logs    : {run_log_dir}/  (one file per job)",
        "",
    )

    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()

    def _execute(job: tuple[str, str], idx: int, role: str) -> tuple[str, bool, float]:
        task_stem, model_name = job
        label = f"{task_stem} × {model_name}"
        log_path = run_log_dir / f"{_slug(task_stem)}__{_slug(model_name)}.log"

        progress.start()
        _emit(
            f"[{_now()}] START [{idx+1:>3}/{total_jobs}] {role:<{ROLE_W}} {label}",
            f"           log : {log_path}",
            f"           prog: {progress.snap()}",
        )
        result = _execute_job_cross(
            task_stem, model_name, role,
            device=args.device, test_only=test_only, cache_only=cache_only,
            log_path=log_path,
        )
        _, ok, elapsed = result
        progress.finish(ok)
        status = "✓ OK  " if ok else "✗ FAIL"
        head = (f"[{_now()}] END   [{idx+1:>3}/{total_jobs}] {status:<{ROLE_W}} {label}    "
                f"elapsed={elapsed/60:.1f} min   prog: {progress.snap()}")
        if not ok:
            extra = [f"           tail of {log_path}:"]
            extra += [f"             | {ln}" for ln in _tail(log_path, 20)]
            _emit(head, *extra)
        else:
            _emit(head)
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
            job = futures[future]
            label, success, _ = future.result()
            if success:
                completed += 1
            else:
                failed.append(label)
                ts, mn = job
                failed_log_paths[label] = run_log_dir / f"{_slug(ts)}__{_slug(mn)}.log"

    done_label = "cache warmed" if cache_only else "completed"
    print(f"\n{'='*60}")
    print(f"  Done — {completed} {done_label}, {skipped} skipped, {len(failed)} failed")
    print(f"  Logs : {run_log_dir}/")
    if failed:
        print("  Failed runs:")
        for name in failed:
            lp = failed_log_paths.get(name)
            print(f"    - {name}" + (f"   →  {lp}" if lp else ""))
    print(f"{'='*60}")
