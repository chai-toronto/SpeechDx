#!/usr/bin/env python3
"""Run training across all task × encoder combinations with skip-checking and timing."""

import argparse
import csv
import importlib
import inspect
import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# ─── Global configuration ──────────────────────────────────────────────
# Encoders: model_name -> encoder YAML filename
# Edit this dict to change which encoders are run.
ENCODERS = {
    "qwen3voice": "qwen3_voice.yaml",
    "wavlm": "wavlm.yaml",
    "ast": "ast.yaml",
    "audiomae": "audiomae.yaml",
    "clap": "clap.yaml",  # reserved, run later (needs -j 1 due to 48kHz memory)
    "emotion2vec": "emotion2vec.yaml",
    "hubert": "hubert.yaml",
    "mms": "mms.yaml",
    "opera_gt": "opera_gt.yaml",
    "w2v2": "w2v2.yaml",
    "wavjepa": "wavjepa.yaml",
    "whisper": "whisper.yaml",
}

TASK = []
EXCLUDE_DATASETS = {"daic_woz"}  # permanent dead task; edaic is live

PROBE_NAME = "AvgTProbe"
PROBE_YAML = "AvgTProbe.yaml"
EXPERIMENT_TAG = "run1"

BASE_CONFIG = Path("training/config/main.yaml")
TMP_CONFIG = Path("training/config/_tmp_run.yaml")
TASKS_DIR = Path("training/config/tasks")
ENCODERS_DIR = Path("training/config/encoders")
# ────────────────────────────────────────────────────────────────────────


def discover_all_tasks() -> list[str]:
    """Every task yaml stem under TASKS_DIR, no EXCLUDE filter, no TASK override."""
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def discover_all_encoders() -> dict[str, str]:
    """Every encoder yaml directly under ENCODERS_DIR (skipping _*.yaml partials),
    mapped stem -> output-folder name. Uses ENCODERS for folder overrides
    (e.g. qwen3_voice -> qwen3voice); otherwise stem is the folder name."""
    stem_to_yaml = {p.stem: p.name for p in ENCODERS_DIR.glob("*.yaml")
                    if not p.stem.startswith("_")}
    yaml_to_folder = {v: k for k, v in ENCODERS.items()}
    return {stem: yaml_to_folder.get(yaml_name, stem)
            for stem, yaml_name in sorted(stem_to_yaml.items())}


def discover_tasks(datasets: list[str] | None = None,
                   tasks: list[str] | None = None) -> list[str]:
    """Auto-scan training/config/tasks/*.yaml and return sorted list of stems.
    Override with TASK env var (comma-separated) if set. Datasets listed in
    EXCLUDE_DATASETS are filtered out. If `datasets` is given, keep only those.
    If `tasks` is given, keep only stems matching those names."""
    if TASK:
        stems = TASK
    else:
        stems = sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))
    allowed_ds = set(datasets) if datasets else None
    allowed_tasks = set(tasks) if tasks else None
    out = []
    for s in stems:
        ds = get_task_info(s)[0]
        if ds in EXCLUDE_DATASETS:
            continue
        if allowed_ds is not None and ds not in allowed_ds:
            continue
        if allowed_tasks is not None and s not in allowed_tasks:
            continue
        out.append(s)
    return out


def get_task_info(task_stem: str) -> tuple[str, str]:
    """Extract dataset and task fields from a task YAML via regex."""
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    dataset_m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    task_m = re.search(r"^task:\s*(\S+)", text, re.MULTILINE)
    if not dataset_m or not task_m:
        raise ValueError(f"Could not parse dataset/task from {task_stem}.yaml")
    return dataset_m.group(1), task_m.group(1)


def get_output_folder(dataset: str, task: str, model_name: str, tag: str = EXPERIMENT_TAG) -> Path:
    return Path(f"./exps/{dataset}_{task}/{model_name}-{PROBE_NAME}-{tag}")


# ─── Task metadata (manifests, num_aug_ver) ─────────────────────────────
_task_yaml_cache: dict[str, dict] = {}
_task_ids_cache: dict[str, frozenset[str]] = {}
_main_yaml_cache: dict | None = None


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores hyperpyyaml tags (!new:, !ref, !include:, !apply:, !name:)."""


def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return None


_TolerantLoader.add_multi_constructor("!", _ignore_unknown)
_TolerantLoader.add_multi_constructor("tag:", _ignore_unknown)


def _load_main_yaml() -> dict:
    global _main_yaml_cache
    if _main_yaml_cache is None:
        _main_yaml_cache = yaml.load(BASE_CONFIG.read_text(), Loader=_TolerantLoader) or {}
    return _main_yaml_cache


def _load_task_yaml(task_stem: str) -> dict:
    if task_stem not in _task_yaml_cache:
        _task_yaml_cache[task_stem] = yaml.load(
            (TASKS_DIR / f"{task_stem}.yaml").read_text(), Loader=_TolerantLoader
        ) or {}
    return _task_yaml_cache[task_stem]


def manifest_paths(task_stem: str) -> tuple[Path, Path, Path]:
    dataset, task = get_task_info(task_stem)
    base = Path(f"./exps/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


def task_num_ver(task_stem: str) -> int:
    return int(_load_task_yaml(task_stem).get("num_aug_ver", 1))


def task_ids(task_stem: str) -> frozenset[str]:
    if task_stem in _task_ids_cache:
        return _task_ids_cache[task_stem]
    ids: set[str] = set()
    for p in manifest_paths(task_stem):
        if p.exists():
            with p.open() as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ids |= set(data.keys())
                elif isinstance(data, list):
                    for fold in data:
                        if isinstance(fold, dict):
                            ids |= set(fold.keys())
    frozen = frozenset(ids)
    _task_ids_cache[task_stem] = frozen
    return frozen


def task_weight(task_stem: str) -> int:
    """Sort key: run biggest cache-writers first so readers unlock sooner."""
    return task_num_ver(task_stem) * len(task_ids(task_stem))


def ensure_manifest(task_stem: str) -> None:
    """Call the task's prepare_data_fn if manifest files are missing.

    Different prep_*.py modules take slightly different kwargs (e.g. mvdr
    takes `num_fold` and `raw_label_key` instead of `ratio`/`manifest_test_path`),
    so we build a superset kwargs dict and filter to the function's actual
    signature.
    """
    tr, va, te = manifest_paths(task_stem)
    # Heuristic: most tasks produce all 3; CV tasks produce only train+valid.
    # Call prep only if train or valid is missing.
    if tr.exists() and va.exists():
        return
    tcfg = _load_task_yaml(task_stem)
    mcfg = _load_main_yaml()
    dataset, task = tcfg["dataset"], tcfg["task"]
    data_folder = mcfg.get("data_folder", "./data/")

    module = importlib.import_module(tcfg["data_io_script"])
    fn = getattr(module, tcfg["prepare_data_fn"])
    print(f"Preparing manifests for {task_stem} …", flush=True)
    tr.parent.mkdir(parents=True, exist_ok=True)

    all_kwargs = {
        "wav_folder": f"{data_folder}/{dataset}/processed/audio",
        "metadata_path": f"{data_folder}/{dataset}/processed/{dataset}.csv",
        "manifest_train_path": str(tr),
        "manifest_val_path": str(va),
        "manifest_test_path": str(te),
        "ratio": mcfg.get("ratio"),
        "random_seed": mcfg.get("random_seed"),
        "dataset": dataset,
        "task": task,
        "raw_label_key": tcfg.get("raw_label_key"),
        "num_fold": tcfg.get("num_fold"),
    }
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    fn(**kwargs)
    _task_ids_cache.pop(task_stem, None)  # force re-read from fresh manifests


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


def is_trained(output_folder: Path, task_stem: str) -> bool:
    """True iff HP-opt finished, so test_only can reuse the best config(s).
    mvdr tasks use trainPerFoldCV and need every fold's best_hparams file."""
    if task_stem.startswith("mvdr"):
        num_fold = int(_load_task_yaml(task_stem).get("num_fold", 0))
        if num_fold < 2:
            return False
        return all((output_folder / f"best_hparams_fold_{i}.yaml").exists()
                   for i in range(num_fold))
    return (output_folder / "best_hparams.yaml").exists()


def make_config(model_name: str, encoder_yaml: str, task_yaml: str, config_id: str = "",
                warm_cache_override: bool | None = None, test_only: bool = False) -> Path:
    text = BASE_CONFIG.read_text()
    subs = [
        (r"^model_name:.*$", f"model_name: {model_name}"),
        (r"^probe_name:.*$", f"probe_name: {PROBE_NAME}"),
        (r"^encoder_params: !include:.*$", f"encoder_params: !include:encoders/{encoder_yaml}"),
        (r"^probe_params: !include:.*$", f"probe_params: !include:probes/{PROBE_YAML}"),
        (r"^data_params: !include:.*$", f"data_params: !include:tasks/{task_yaml}"),
    ]
    if warm_cache_override is not None:
        subs.append((r"^warm_cache:.*$", f"warm_cache: {str(warm_cache_override).lower()}"))
    for pattern, replacement in subs:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    # Remove chunk_at line entirely
    text = re.sub(r"^chunk_at:.*\n?", "", text, flags=re.MULTILINE)
    # Override test_only and warm_cache if requested
    if test_only:
        text = re.sub(r"^test_only:.*$", "test_only: True", text, flags=re.MULTILINE)
        text = re.sub(r"^warm_cache:.*$", "warm_cache: false", text, flags=re.MULTILINE)
    # Include model name + SLURM job id so concurrent jobs (same encoder, different
    # datasets) don't overwrite/delete each other's temp config.
    job_suffix = f"_job{os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else ""
    config_path = Path(f"training/config/_tmp_run_{model_name}{job_suffix}{config_id}.yaml")
    config_path.write_text(text)
    return config_path


def run_one(task_stem: str, model_name: str, encoder_yaml: str, device=None,
            config_id: str = "",
            warm_cache_override: bool | None = None, test_only: bool = False) -> tuple[str, bool, float]:
    """Run a single training job. Returns (label, success, elapsed_seconds)."""
    label = f"{task_stem} × {model_name}"
    task_yaml = f"{task_stem}.yaml"
    config_path = make_config(model_name, encoder_yaml, task_yaml, config_id,
                              warm_cache_override=warm_cache_override, test_only=test_only)

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


# ─── Result parsing ────────────────────────────────────────────────────
CLASSIFICATION_METRICS = ["AUROC", "F1", "accuracy"]
REGRESSION_METRICS = ["MAE", "MSE", "PearsonR", "R2"]
# filename -> list of metric keys that belong in that CSV
CSV_LAYOUT = {
    "AUC": "AUROC",
    "F1": "F1",
    "Acc": "accuracy",
    "MAE": "MAE",
    "MSE": "MSE",
    "PearsonR": "PearsonR",
    "R2": "R2",
}


def parse_results_txt(path: Path) -> dict[str, float]:
    """Parse 'key: value' lines from a test_results.txt file."""
    metrics = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        try:
            metrics[key.strip()] = float(val.strip())
        except ValueError:
            pass
    return metrics


def parse_results_yaml(path: Path) -> dict[str, float]:
    """Parse test_results.yaml (trainPerFoldCV) — take summary means."""
    data = yaml.safe_load(path.read_text())
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    return {k: float(v["mean"]) for k, v in summary.items()
            if isinstance(v, dict) and "mean" in v}


def load_metrics(output_folder: Path, task_stem: str) -> dict[str, float] | None:
    results_file = output_folder / get_results_file(task_stem)
    if not results_file.exists():
        return None
    if results_file.suffix == ".yaml":
        return parse_results_yaml(results_file)
    return parse_results_txt(results_file)


def cmd_summary(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    encoders = list(ENCODERS.keys())

    # metric_key -> {task_stem: {encoder: value}}
    matrices: dict[str, dict[str, dict[str, float]]] = {
        k: defaultdict(dict) for k in CSV_LAYOUT.values()
    }
    # Classify tasks by task_type yaml field so empty rows are still emitted
    regression_tasks: set[str] = set()
    classification_tasks: set[str] = set()
    for task_stem in tasks:
        if _load_task_yaml(task_stem).get("task_type") == "R":
            regression_tasks.add(task_stem)
        else:
            classification_tasks.add(task_stem)
    found, missing = 0, 0

    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name, args.tag)
            metrics = load_metrics(folder, task_stem)
            if metrics is None:
                missing += 1
                continue
            found += 1
            for metric_key in CSV_LAYOUT.values():
                if metric_key in metrics:
                    matrices[metric_key][task_stem][model_name] = metrics[metric_key]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for csv_name, metric_key in CSV_LAYOUT.items():
        data = matrices[metric_key]
        # Classification CSVs → only classification tasks; regression → only regression tasks.
        # Emit every task of the matching type, even if all encoders are empty.
        pool = regression_tasks if metric_key in REGRESSION_METRICS else classification_tasks
        task_rows = sorted(pool)
        if not task_rows:
            continue

        csv_path = out_dir / f"{csv_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *encoders])
            for task_stem in task_rows:
                row = [task_stem]
                for enc in encoders:
                    val = data[task_stem].get(enc, "")
                    row.append(f"{val:.4f}" if isinstance(val, float) else "")
                writer.writerow(row)
        written.append(csv_path)

    # Completion matrix — non-exclusion: every task yaml × every encoder yaml
    all_tasks = discover_all_tasks()
    all_encoders = discover_all_encoders()  # stem -> folder_name
    comp_path = out_dir / "completion.csv"
    totals = {stem: 0 for stem in all_encoders}
    with comp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *all_encoders.keys()])
        for task_stem in all_tasks:
            dataset, task = get_task_info(task_stem)
            row = [task_stem]
            for stem, folder_name in all_encoders.items():
                folder = get_output_folder(dataset, task, folder_name, args.tag)
                done = is_complete(folder, task_stem)
                row.append("1" if done else "0")
                if done:
                    totals[stem] += 1
            writer.writerow(row)
        writer.writerow(["TOTAL", *(str(totals[s]) for s in all_encoders)])
    written.append(comp_path)

    print(f"\nParsed {found} result files, {missing} missing")
    print(f"Wrote {len(written)} CSV(s) to {out_dir}/")
    for p in written:
        print(f"  {p}")


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    complete, incomplete = 0, 0

    encoders = list(ENCODERS.keys())
    rows = []
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        row = [task_stem]
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name)
            if is_complete(folder, task_stem):
                row.append("☑")
                complete += 1
            else:
                row.append("☐")
                incomplete += 1
        rows.append(row)

    headers = ["task", *encoders]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))

    total = complete + incomplete
    print(f"\nLegend: ☑ complete, ☐ incomplete")
    print(f"Summary: {complete}/{total} complete, {incomplete} remaining")


def _needed_keys(task_stem: str) -> set[tuple[str, int]]:
    """Cache keys a task's warmup would need: (id, version) for each id × version."""
    ids = task_ids(task_stem)
    num_ver = task_num_ver(task_stem)
    return {(uid, v) for uid in ids for v in range(num_ver)}


def cmd_run(args: argparse.Namespace) -> None:
    tasks = discover_tasks(args.dataset, args.task)
    skipped, completed, failed = 0, 0, []
    test_only = getattr(args, 'test_only', False)

    # Filter encoders by --encoder flag (default: all). Repeatable.
    if args.encoder:
        encoders = {e: ENCODERS[e] for e in args.encoder}
    else:
        encoders = ENCODERS

    # Collect pending jobs + track all tasks (pending + completed per encoder)
    # so we can seed the written-key state from completed tasks.
    pending: list[tuple[str, str, str]] = []  # (task_stem, model_name, encoder_yaml)
    completed_by_ds_enc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name, encoder_yaml in encoders.items():
            folder = get_output_folder(dataset, task, model_name)
            if test_only:
                if not is_trained(folder, task_stem):
                    print(f"SKIP (no trained model): {task_stem} × {model_name}")
                    skipped += 1
                    continue
            elif is_complete(folder, task_stem):
                print(f"SKIP (done): {task_stem} × {model_name}")
                skipped += 1
                completed_by_ds_enc[(dataset, model_name)].append(task_stem)
                continue
            pending.append((task_stem, model_name, encoder_yaml))

    # Pre-generate manifests for every task we intend to run, so the writer/reader
    # classification has accurate ID sets before dispatch. prepare_data_fn is
    # deterministic + cheap; skips if manifests already exist.
    needed_tasks = {ts for ts, _, _ in pending}
    # Also ensure manifests for completed tasks we'll use for seeding the written set.
    needed_tasks |= {ts for lst in completed_by_ds_enc.values() for ts in lst}
    for ts in sorted(needed_tasks):
        try:
            ensure_manifest(ts)
        except Exception as e:
            print(f"⚠ prepare_data for {ts} failed: {e}", flush=True)

    # Seed per-(dataset, encoder) written-key sets from completed tasks.
    written: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for (ds, enc), tlist in completed_by_ds_enc.items():
        for ts in tlist:
            try:
                written[(ds, enc)] |= _needed_keys(ts)
            except Exception:
                pass  # missing manifest; best-effort seeding

    total_jobs = len(pending)
    print(f"\n{total_jobs} pending jobs")
    print()

    # Sort pending jobs so biggest cache-writers go first (more readers unlock sooner).
    pending.sort(key=lambda j: -task_weight(j[0]))

    max_workers = max(1, args.max_workers)
    print(f"Running with up to {max_workers} concurrent workers\n")

    # Per-(dataset, encoder) lock: held only by writer tasks. Readers run free.
    # Pre-create so concurrent lookups all resolve to the same Lock instance.
    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn, _ in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()  # guards `written` mutations

    def dispatch(job: tuple[str, str, str], idx: int) -> tuple[str, bool, float]:
        task_stem, model_name, encoder_yaml = job
        dataset, _ = get_task_info(task_stem)
        key = (dataset, model_name)

        # test_only never writes cache — skip lock logic entirely.
        if test_only:
            return run_one(task_stem, model_name, encoder_yaml, args.device,
                           f"_{idx}", warm_cache_override=False, test_only=True)

        try:
            needed = _needed_keys(task_stem)
        except Exception:
            needed = None  # unknown → treat as writer

        # Reader check (under written_lock to see up-to-date state).
        is_reader = False
        if needed is not None:
            with written_lock:
                is_reader = needed.issubset(written[key])

        if is_reader:
            return run_one(task_stem, model_name, encoder_yaml, args.device,
                           f"_{idx}", warm_cache_override=False, test_only=test_only)

        # Writer path — acquire per-(dataset, encoder) lock. Release early if the
        # re-check shows a concurrent writer satisfied our keys (run as reader).
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
                return run_one(task_stem, model_name, encoder_yaml, args.device,
                               f"_{idx}", warm_cache_override=False, test_only=test_only)
            result = run_one(task_stem, model_name, encoder_yaml, args.device,
                             f"_{idx}", warm_cache_override=None, test_only=test_only)
            _, success, _ = result
            if success and needed is not None:
                with written_lock:
                    written[key] |= needed
            return result
        finally:
            if not released:
                ds_lock.release()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(dispatch, job, i) for i, job in enumerate(pending)]
        for future in as_completed(futures):
            label, success, _ = future.result()
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

    summary_parser = sub.add_parser("summary", help="Collect results into per-metric CSVs")
    summary_parser.add_argument("--out-dir", type=str, default="exps/_summary",
                                help="Directory to write metric CSVs (default: exps/_summary)")
    summary_parser.add_argument("--tag", type=str, default=EXPERIMENT_TAG,
                                help=f"Experiment tag to scan (default: {EXPERIMENT_TAG})")

    run_parser = sub.add_parser("run", help="Execute all incomplete runs")
    run_parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0)")
    run_parser.add_argument("--max-workers", "-j", type=int, default=3, help="Max concurrent tasks (default: 3). Writer tasks still serialize per (dataset, encoder).")
    run_parser.add_argument("--encoder", type=str, default=None,
                            action="append",
                            choices=list(ENCODERS.keys()),
                            help="Run only this encoder (repeatable; default: all in ENCODERS dict)")
    run_parser.add_argument("--dataset", type=str, default=None,
                            action="append",
                            help="Run only tasks whose dataset field matches (repeatable; default: all datasets)")
    run_parser.add_argument("--task", type=str, default=None,
                            action="append",
                            help="Run only these task stems (repeatable; default: all tasks)")
    run_parser.add_argument("--test-only", action="store_true",
                            help="Run inference only (no training), requires prior completed runs")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "summary":
        cmd_summary(args)


if __name__ == "__main__":
    main()
