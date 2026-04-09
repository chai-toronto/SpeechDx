#!/usr/bin/env python3
"""Run training across all task × encoder combinations with skip-checking and timing."""

import argparse
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Global configuration ──────────────────────────────────────────────
# Encoders: model_name -> encoder YAML filename
# Edit this dict to change which encoders are run.
ENCODERS = {
    "qwen3voice": "qwen3_voice.yaml",
    # "wavlm": "wavlm.yaml",
    # "whisper": "whisper.yaml",
    # "hubert": "hubert.yaml",
    # "w2v2": "w2v2.yaml",
}

TASK =[]

PROBE_NAME = "AvgTProbe"
PROBE_YAML = "AvgTProbe.yaml"
EXPERIMENT_TAG = "run1"

BASE_CONFIG = Path("training/config/main.yaml")
TMP_CONFIG = Path("training/config/_tmp_run.yaml")
TASKS_DIR = Path("training/config/tasks")
# ────────────────────────────────────────────────────────────────────────


def discover_tasks() -> list[str]:
    """Auto-scan training/config/tasks/*.yaml and return sorted list of stems.
    Override with TASK env var (comma-separated) if set."""
    if TASK:
        return TASK
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def get_task_info(task_stem: str) -> tuple[str, str]:
    """Extract dataset and task fields from a task YAML via regex."""
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    dataset_m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    task_m = re.search(r"^task:\s*(\S+)", text, re.MULTILINE)
    if not dataset_m or not task_m:
        raise ValueError(f"Could not parse dataset/task from {task_stem}.yaml")
    return dataset_m.group(1), task_m.group(1)


def get_output_folder(dataset: str, task: str, model_name: str) -> Path:
    return Path(f"./exps/{dataset}_{task}/{model_name}-{PROBE_NAME}-{EXPERIMENT_TAG}")


def get_train_command(task_stem: str) -> str:
    """mvdr tasks use trainPerFoldCV; everything else uses train."""
    if task_stem.startswith("mvdr"):
        return "python -m training.trainPerFoldCV"
    return "python -m training.train"


def get_results_file(task_stem: str) -> str:
    """trainPerFoldCV writes test_results.yaml; train writes test_results.txt."""
    if task_stem.startswith("mvdr"):
        return "test_results.yaml"
    return "test_results.txt"


def is_complete(output_folder: Path, task_stem: str) -> bool:
    return (output_folder / get_results_file(task_stem)).exists()


def make_config(model_name: str, encoder_yaml: str, task_yaml: str, config_id: str = "") -> Path:
    text = BASE_CONFIG.read_text()
    subs = [
        (r"^model_name:.*$", f"model_name: {model_name}"),
        (r"^probe_name:.*$", f"probe_name: {PROBE_NAME}"),
        (r"^encoder_params: !include:.*$", f"encoder_params: !include:encoders/{encoder_yaml}"),
        (r"^probe_params: !include:.*$", f"probe_params: !include:probes/{PROBE_YAML}"),
        (r"^data_params: !include:.*$", f"data_params: !include:tasks/{task_yaml}"),
    ]
    for pattern, replacement in subs:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    # Remove chunk_at line entirely
    text = re.sub(r"^chunk_at:.*\n?", "", text, flags=re.MULTILINE)
    config_path = Path(f"training/config/_tmp_run{config_id}.yaml")
    config_path.write_text(text)
    return config_path


def run_one(task_stem: str, model_name: str, encoder_yaml: str, device: str | None,
            config_id: str = "", dataset_lock: threading.Lock = None) -> tuple[str, bool, float]:
    """Run a single training job. Returns (label, success, elapsed_seconds).
    Acquires dataset_lock to prevent concurrent jobs on the same dataset."""
    label = f"{task_stem} × {model_name}"
    lock = dataset_lock or threading.Lock()

    with lock:
        task_yaml = f"{task_stem}.yaml"
        config_path = make_config(model_name, encoder_yaml, task_yaml, config_id)

        cmd = get_train_command(task_stem).split()
        cmd.append(str(config_path))
        if device:
            cmd.append(f"--device={device}")

        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start
            print(f"  ✓ {label} completed in {elapsed/60:.1f} min")
            return label, True, elapsed
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"  ✗ {label} FAILED (exit code {e.returncode}) after {elapsed/60:.1f} min")
            return label, False, elapsed
        finally:
            config_path.unlink(missing_ok=True)


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    complete, incomplete = 0, 0

    print(f"{'Task':<30} {'Encoder':<25} {'Status'}")
    print("-" * 70)

    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name in ENCODERS:
            folder = get_output_folder(dataset, task, model_name)
            if is_complete(folder, task_stem):
                status = "COMPLETE"
                complete += 1
            else:
                status = "INCOMPLETE"
                incomplete += 1
            print(f"{task_stem:<30} {model_name:<25} {status}")

    total = complete + incomplete
    print(f"\nSummary: {complete}/{total} complete, {incomplete} remaining")


def cmd_run(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    skipped, completed, failed = 0, 0, []
    max_parallel = args.parallel

    # Collect jobs to run, with per-dataset locks to prevent concurrent same-dataset jobs
    jobs = []
    dataset_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name, encoder_yaml in ENCODERS.items():
            folder = get_output_folder(dataset, task, model_name)
            if is_complete(folder, task_stem):
                print(f"SKIP (done): {task_stem} × {model_name}")
                skipped += 1
                continue
            jobs.append((task_stem, model_name, encoder_yaml, dataset))

    print(f"\n{len(jobs)} jobs to run ({max_parallel} parallel workers)")
    print(f"  (same-dataset tasks run sequentially to avoid conflicts)\n")

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_one, task_stem, model_name, encoder_yaml, args.device, f"_{i}", dataset_locks[dataset]): f"{task_stem} × {model_name}"
            for i, (task_stem, model_name, encoder_yaml, dataset) in enumerate(jobs)
        }
        for future in as_completed(futures):
            label, success, elapsed = future.result()
            if success:
                completed += 1
            else:
                failed.append(label)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done — {completed} completed, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("  Failed runs:")
        for name in failed:
            print(f"    - {name}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training across task × encoder combos")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show completion status of all runs")

    run_parser = sub.add_parser("run", help="Execute all incomplete runs")
    run_parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0)")
    run_parser.add_argument("--parallel", "-j", type=int, default=4, help="Max parallel jobs (default: 4)")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
