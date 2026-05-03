"""``ahb run-data-eff`` — data-efficiency orchestrator.

Salvaged from ``run_all_data_eff.py:cmd_run`` (lines 374-571). Same
writer/reader lock semantics as ``ahb/run.py`` but operates on
(task, encoder, level) triples: each level gets its own subsampled
manifest under ``./data_eff_exps/<level>/`` and shares the encoder
cache with the full benchmark.

Subprocess chain per job:
- writer: ``ahb warm`` (the regular cache, shared across levels) →
          ``ahb train --level-dir <level>`` (writes to data_eff_exps/...)
- reader: just ``ahb train --level-dir <level>``
- cache_only: just ``ahb warm``
- test_only: ``ahb train --level-dir <level> --overrides "test_only: true"``

CV tasks (mvdr_*) use ``ahb train-cv`` instead of ``ahb train``.
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
from ahb.orchestrator import task_ids, task_num_ver, task_weight
from ahb.orchestrator_data_eff import (
    LEVELS,
    LOGS_ROOT,
    discover_tasks,
    encoders_default,
    ensure_manifest_data_eff,
    get_output_folder,
    get_task_info,
    has_ci_results,
    is_complete,
    is_cv,
)


def _run_subprocess(cmd: list[str], log_path: Path, label: str) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
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


def _execute_job_de(task_stem: str, model_name: str, level_dir: str,
                    role: str, *, device: str | None,
                    test_only: bool, cache_only: bool,
                    log_path: Path) -> tuple[str, bool, float]:
    label = f"{task_stem} × {model_name} @ {level_dir}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write(f"{'='*60}\n  Running: {label}\n")
        f.write(f"  Role: {role}, test_only={test_only}, cache_only={cache_only}\n")
        f.write(f"  level_dir: {level_dir}\n")
        f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")

    train_subcommand = "train-cv" if is_cv(task_stem) else "train"
    start = time.time()
    success = True

    if cache_only:
        cmd = ["python", "-m", "ahb", "warm", task_stem, model_name]
        if device:
            cmd.append(f"--device={device}")
        success = _run_subprocess(cmd, log_path, "warm")
    elif test_only:
        cmd = ["python", "-m", "ahb", train_subcommand, task_stem, model_name,
               "--level-dir", level_dir,
               "--overrides", "test_only: true"]
        success = _run_subprocess(cmd, log_path, "test")
    elif role == "writer":
        warm_cmd = ["python", "-m", "ahb", "warm", task_stem, model_name]
        if device:
            warm_cmd.append(f"--device={device}")
        if not _run_subprocess(warm_cmd, log_path, "warm"):
            success = False
        if success:
            train_cmd = ["python", "-m", "ahb", train_subcommand, task_stem, model_name,
                         "--level-dir", level_dir]
            success = _run_subprocess(train_cmd, log_path, "train")
    else:  # reader
        cmd = ["python", "-m", "ahb", train_subcommand, task_stem, model_name,
               "--level-dir", level_dir]
        success = _run_subprocess(cmd, log_path, "train")

    elapsed = time.time() - start
    with log_path.open("a") as f:
        marker = "✓ OK" if success else "✗ FAILED"
        f.write(f"\n{'='*60}\n  {marker}: {label} in {elapsed/60:.1f} min\n")
        f.write(f"  Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")
    return label, success, elapsed


def cmd_run_data_eff(args: argparse.Namespace) -> None:
    """Run all (task × encoder × level) combos with shared-cache writer/reader logic."""
    tasks = discover_tasks(args.dataset, args.task)
    skipped = completed = 0
    failed: list[str] = []
    failed_log_paths: dict[str, Path] = {}
    test_only = getattr(args, "test_only", False)
    cache_only = getattr(args, "cache_only", False)
    no_writer = getattr(args, "no_writer", False)

    all_encoders = encoders_default()
    if args.encoder:
        encoders = {e: all_encoders[e] for e in args.encoder if e in all_encoders}
    else:
        encoders = all_encoders

    # --level filters down to a subset of LEVELS.
    if getattr(args, "level", None):
        active_levels = [(d, v) for d, v in LEVELS if d in set(args.level)]
    else:
        active_levels = list(LEVELS)

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = LOGS_ROOT / run_stamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Pending = (task_stem, model_name, level_dir).
    pending: list[tuple[str, str, str]] = []
    completed_by_ds_enc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for level_dir, _ in active_levels:
        for task_stem in tasks:
            dataset, task = get_task_info(task_stem)
            for model_name in encoders:
                folder = get_output_folder(dataset, task, model_name, level_dir)
                if cache_only:
                    pass
                elif test_only:
                    if has_ci_results(folder, task_stem):
                        _emit(f"[{_now()}] SKIP (CI present)     : {task_stem} × {model_name} @ {level_dir}")
                        skipped += 1
                        continue
                    if not (folder / "best_hparams.yaml").exists():
                        _emit(f"[{_now()}] SKIP (no trained model): {task_stem} × {model_name} @ {level_dir}")
                        skipped += 1
                        continue
                elif is_complete(folder, task_stem):
                    _emit(f"[{_now()}] SKIP (done)            : {task_stem} × {model_name} @ {level_dir}")
                    skipped += 1
                    completed_by_ds_enc[(dataset, model_name)].append(task_stem)
                    continue
                pending.append((task_stem, model_name, level_dir))

    # Subsample manifests up-front for every (task, level) we'll touch — keeps
    # dispatch hot-path free of subsampling latency.
    needed_subsamples = {(ts, lv) for ts, _, lv in pending}
    for (ts, lv) in sorted(needed_subsamples):
        try:
            ensure_manifest_data_eff(ts, lv)
        except Exception as e:
            print(f"⚠ subsample for {ts}@{lv} failed: {e}", flush=True)

    # The encoder cache is SHARED across levels — same train_cache_dir for every
    # (task, encoder) regardless of level. So writer/reader is keyed on the
    # source task's id set, not the per-level subsample.
    written: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for (ds, enc), tlist in completed_by_ds_enc.items():
        for ts in tlist:
            try:
                ids = task_ids(ts)
                nv = task_num_ver(ts)
                written[(ds, enc)] |= {(uid, v) for uid in ids for v in range(nv)}
            except Exception:
                pass

    total_jobs = len(pending)
    progress = _Progress(total_jobs)
    pending.sort(key=lambda j: -task_weight(j[0]))

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
        f"           mode    : {mode}  (data-eff)",
        f"           levels  : {[d for d, _ in active_levels]}",
        f"           logs    : {run_log_dir}/  (one file per job)",
        "",
    )

    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn, _ in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()

    def _execute(job, idx, role):
        task_stem, model_name, level_dir = job
        label = f"{task_stem} × {model_name} @ {level_dir}"
        log_path = (run_log_dir / level_dir
                    / f"{_slug(task_stem)}__{_slug(model_name)}.log")

        progress.start()
        _emit(
            f"[{_now()}] START [{idx+1:>3}/{total_jobs}] {role:<{ROLE_W}} {label}",
            f"           log : {log_path}",
            f"           prog: {progress.snap()}",
        )
        result = _execute_job_de(
            task_stem, model_name, level_dir, role,
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

    def dispatch(job, idx):
        task_stem, model_name, level_dir = job
        dataset, _ = get_task_info(task_stem)
        key = (dataset, model_name)
        label = f"{task_stem} × {model_name} @ {level_dir}"

        if test_only:
            return _execute(job, idx, "test")
        if no_writer:
            return _execute(job, idx, "reader")

        try:
            ids = task_ids(task_stem)
            nv = task_num_ver(task_stem)
            needed = {(uid, v) for uid in ids for v in range(nv)}
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
                ts, mn, lv = job
                failed_log_paths[label] = (
                    run_log_dir / lv / f"{_slug(ts)}__{_slug(mn)}.log"
                )

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
