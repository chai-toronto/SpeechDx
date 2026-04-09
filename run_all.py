#!/usr/bin/env python3
"""Run training across all task × encoder combinations with skip-checking and timing."""

import argparse
import re
import subprocess
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

TASK = []

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


def run_one(task_stem: str, model_name: str, encoder_yaml: str, device=None,
            config_id: str = "") -> tuple[str, bool, float]:
    """Run a single training job. Returns (label, success, elapsed_seconds)."""
    label = f"{task_stem} × {model_name}"
    task_yaml = f"{task_stem}.yaml"
    config_path = make_config(model_name, encoder_yaml, task_yaml, config_id)

    cmd = get_train_command(task_stem).split()
    cmd.append(str(config_path))
    if device:
        cmd.append(f"--device={device}")

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"  ✓ {label} completed in {elapsed/60:.1f} min", flush=True)
        return label, True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"  ✗ {label} FAILED (exit code {e.returncode}) after {elapsed/60:.1f} min", flush=True)
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


def run_dataset_queue(dataset, queue, device, start_idx):
    """Run all tasks for one dataset sequentially. Called from its own thread."""
    results = []
    for i, (task_stem, model_name, encoder_yaml) in enumerate(queue):
        label, success, elapsed = run_one(
            task_stem, model_name, encoder_yaml, device, f"_{start_idx + i}"
        )
        results.append((label, success, elapsed))
    return results


def cmd_run(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    skipped, completed, failed = 0, 0, []

    # Group jobs by dataset — each group runs sequentially, groups run in parallel
    dataset_queues = defaultdict(list)
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name, encoder_yaml in ENCODERS.items():
            folder = get_output_folder(dataset, task, model_name)
            if is_complete(folder, task_stem):
                print(f"SKIP (done): {task_stem} × {model_name}")
                skipped += 1
                continue
            dataset_queues[dataset].append((task_stem, model_name, encoder_yaml))

    total_jobs = sum(len(q) for q in dataset_queues.values())
    print(f"\n{total_jobs} jobs across {len(dataset_queues)} datasets")
    for ds, q in dataset_queues.items():
        print(f"  {ds}: {len(q)} tasks")
    print()

    # One thread per dataset, each runs its tasks sequentially
    # Cap workers to avoid GPU OOM when many datasets run concurrently
    max_workers = min(len(dataset_queues), args.max_workers)
    print(f"Running with {max_workers} concurrent dataset workers\n")
    idx = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for dataset, queue in dataset_queues.items():
            futures[pool.submit(run_dataset_queue, dataset, queue, args.device, idx)] = dataset
            idx += len(queue)

        for future in as_completed(futures):
            for label, success, elapsed in future.result():
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
    run_parser.add_argument("--max-workers", "-j", type=int, default=4, help="Max concurrent dataset workers (default: 4)")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
