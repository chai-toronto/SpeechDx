#!/usr/bin/env python3
"""Run training across all cross-task × encoder combinations."""

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

# Keep in sync with run_all.py unless intentionally different.
ENCODERS = {
    "emotion2vec": "emotion2vec.yaml",
}

TASK = []
EXCLUDE_DATASETS = set()

PROBE_NAME = "AvgTProbe"
PROBE_YAML = "Probe.yaml"
EXPERIMENT_TAG = "run1"

BASE_CONFIG = Path("training/config/main_cross.yaml")
TASKS_DIR = Path("training/config/cross_tasks")
ENCODERS_DIR = Path("training/config/encoders")


def discover_all_tasks() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def discover_all_encoders() -> dict[str, str]:
    stem_to_yaml = {
        p.stem: p.name for p in ENCODERS_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    }
    yaml_to_folder = {v: k for k, v in ENCODERS.items()}
    return {
        stem: yaml_to_folder.get(yaml_name, stem)
        for stem, yaml_name in sorted(stem_to_yaml.items())
    }


def discover_tasks(datasets: list[str] | None = None,
                   tasks: list[str] | None = None) -> list[str]:
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
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    dataset_m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    task_m = re.search(r"^task:\s*(\S+)", text, re.MULTILINE)
    if not dataset_m or not task_m:
        raise ValueError(f"Could not parse dataset/task from {task_stem}.yaml")
    return dataset_m.group(1), task_m.group(1)


def get_output_folder(dataset: str, task: str, model_name: str, tag: str = EXPERIMENT_TAG) -> Path:
    return Path(f"./exps/{dataset}_{task}/{model_name}-{PROBE_NAME}-{tag}")


_task_yaml_cache: dict[str, dict] = {}
_task_ids_cache: dict[str, frozenset[str]] = {}
_main_yaml_cache: dict | None = None


class _TolerantLoader(yaml.SafeLoader):
    pass


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
    frozen = frozenset(ids)
    _task_ids_cache[task_stem] = frozen
    return frozen


def task_weight(task_stem: str) -> int:
    return task_num_ver(task_stem) * len(task_ids(task_stem))


def ensure_manifest(task_stem: str) -> None:
    tr, va, te = manifest_paths(task_stem)
    if tr.exists() and va.exists() and te.exists():
        return
    tcfg = _load_task_yaml(task_stem)
    mcfg = _load_main_yaml()
    dataset = tcfg["dataset"]
    task = tcfg["task"]
    train_dataset = tcfg["train_dataset"]
    test_dataset = tcfg["test_dataset"]
    data_folder = mcfg.get("data_folder", "./data/")

    module = importlib.import_module(tcfg["data_io_script"])
    fn = getattr(module, tcfg["prepare_data_fn"])
    print(f"Preparing manifests for {task_stem} …", flush=True)
    tr.parent.mkdir(parents=True, exist_ok=True)

    all_kwargs = {
        "wav_folder_train": f"{data_folder}/{train_dataset}/processed/audio",
        "metadata_path_train": f"{data_folder}/{train_dataset}/processed/{train_dataset}.csv",
        "wav_folder_test": f"{data_folder}/{test_dataset}/processed/audio",
        "metadata_path_test": f"{data_folder}/{test_dataset}/processed/{test_dataset}.csv",
        "manifest_train_path": str(tr),
        "manifest_val_path": str(va),
        "manifest_test_path": str(te),
        "ratio": mcfg.get("ratio"),
        "random_seed": mcfg.get("random_seed"),
        "dataset": dataset,
        "task": task,
    }
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    fn(**kwargs)
    _task_ids_cache.pop(task_stem, None)


def get_train_command() -> str:
    return "python -m training.train"


def get_results_file() -> str:
    return "test_results.txt"


def is_complete(output_folder: Path) -> bool:
    return (output_folder / get_results_file()).exists()


def has_ci_results(output_folder: Path) -> bool:
    results_file = output_folder / get_results_file()
    if not results_file.exists():
        return False
    text = results_file.read_text()
    return ("AUROC_CI_low" in text and "AUROC_CI_high" in text) or ("MAE_CI_low" in text and "MAE_CI_high" in text)


def make_config(model_name: str, encoder_yaml: str, task_yaml: str, config_id: str = "",
                warm_cache_override: bool | None = None, test_only: bool = False,
                cache_only: bool = False) -> Path:
    text = BASE_CONFIG.read_text()
    subs = [
        (r"^model_name:.*$", f"model_name: {model_name}"),
        (r"^probe_name:.*$", f"probe_name: {PROBE_NAME}"),
        (r"^encoder_params: !include:.*$", f"encoder_params: !include:encoders/{encoder_yaml}"),
        (r"^probe_params: !include:.*$", f"probe_params: !include:probes/{PROBE_YAML}"),
        (r"^data_params: !include:.*$", f"data_params: !include:cross_tasks/{task_yaml}"),
    ]
    if warm_cache_override is not None:
        subs.append((r"^warm_cache:.*$", f"warm_cache: {str(warm_cache_override).lower()}"))
    for pattern, replacement in subs:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    text = re.sub(r"^skip_prep:.*$", "skip_prep: True", text, flags=re.MULTILINE)
    if cache_only:
        text = re.sub(r"^warm_cache:.*$", "warm_cache: true", text, flags=re.MULTILINE)
        text += "\ncache_only: True\n"
    elif test_only:
        text = re.sub(r"^test_only:.*$", "test_only: True", text, flags=re.MULTILINE)
        text = re.sub(r"^warm_cache:.*$", "warm_cache: false", text, flags=re.MULTILINE)
    job_suffix = f"_job{os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else ""
    config_path = Path(f"training/config/_tmp_run_cross_{model_name}{job_suffix}{config_id}.yaml")
    config_path.write_text(text)
    return config_path


def run_one(task_stem: str, model_name: str, encoder_yaml: str, device=None,
            config_id: str = "", warm_cache_override: bool | None = None,
            test_only: bool = False, cache_only: bool = False) -> tuple[str, bool, float]:
    label = f"{task_stem} × {model_name}"
    task_yaml = f"{task_stem}.yaml"
    config_path = make_config(model_name, encoder_yaml, task_yaml, config_id,
                              warm_cache_override=warm_cache_override, test_only=test_only,
                              cache_only=cache_only)
    cmd = get_train_command().split()
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


def parse_results_txt(path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        try:
            metrics[key.strip()] = float(val.strip())
        except ValueError:
            pass
    lo, hi = metrics.get("AUROC_CI_low"), metrics.get("AUROC_CI_high")
    if isinstance(lo, float) and isinstance(hi, float):
        metrics["AUC_CI"] = f"({lo:.4f}, {hi:.4f})"
    return metrics


def load_metrics(output_folder: Path) -> dict[str, float] | None:
    results_file = output_folder / get_results_file()
    if not results_file.exists():
        return None
    return parse_results_txt(results_file)


def cmd_summary(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    encoders = list(ENCODERS.keys())
    metrics_map = {"AUC": "AUROC", "AUC_CI": "AUC_CI", "F1": "F1", "Acc": "accuracy"}
    matrices: dict[str, dict[str, dict[str, float]]] = {k: defaultdict(dict) for k in metrics_map.values()}
    found, missing = 0, 0
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name, args.tag)
            metrics = load_metrics(folder)
            if metrics is None:
                missing += 1
                continue
            found += 1
            for metric_key in metrics_map.values():
                if metric_key in metrics:
                    matrices[metric_key][task_stem][model_name] = metrics[metric_key]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for csv_name, metric_key in metrics_map.items():
        csv_path = out_dir / f"{csv_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *encoders])
            for task_stem in sorted(tasks):
                row = [task_stem]
                for enc in encoders:
                    val = matrices[metric_key][task_stem].get(enc, "")
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(val)
                writer.writerow(row)
        written.append(csv_path)
    print(f"\nParsed {found} result files, {missing} missing")
    print(f"Wrote {len(written)} CSV(s) to {out_dir}/")


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    test_only = getattr(args, "test_only", False)
    complete, incomplete, absent = 0, 0, 0
    encoders = list(ENCODERS.keys())
    rows = []
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        row = [task_stem]
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name)
            if test_only:
                if not (folder / get_results_file()).exists():
                    row.append(" ")
                    absent += 1
                elif has_ci_results(folder):
                    row.append("☑")
                    complete += 1
                else:
                    row.append("☐")
                    incomplete += 1
            elif is_complete(folder):
                row.append("☑")
                complete += 1
            else:
                row.append("☐")
                incomplete += 1
        rows.append(row)
    headers = ["task", *encoders]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt(row): return " | ".join(c.ljust(widths[i]) for i, c in enumerate(row))
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))
    if test_only:
        total = complete + incomplete + absent
        print(f"\nSummary: {complete}/{total} with CI, {incomplete} pending CI, {absent} no test result")
    else:
        total = complete + incomplete
        print(f"\nSummary: {complete}/{total} complete, {incomplete} remaining")


def _needed_keys(task_stem: str) -> set[tuple[str, int]]:
    ids = task_ids(task_stem)
    num_ver = task_num_ver(task_stem)
    return {(uid, v) for uid in ids for v in range(num_ver)}


def cmd_run(args: argparse.Namespace) -> None:
    tasks = discover_tasks(args.dataset, args.task)
    skipped, completed, failed = 0, 0, []
    test_only = getattr(args, "test_only", False)
    cache_only = getattr(args, "cache_only", False)
    if args.encoder:
        encoders = {e: ENCODERS[e] for e in args.encoder}
    else:
        encoders = ENCODERS

    pending: list[tuple[str, str, str]] = []
    completed_by_ds_enc: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        for model_name, encoder_yaml in encoders.items():
            folder = get_output_folder(dataset, task, model_name)
            if cache_only:
                pass
            elif test_only:
                if has_ci_results(folder):
                    print(f"SKIP (CI present): {task_stem} × {model_name}")
                    skipped += 1
                    continue
                if not (folder / "best_hparams.yaml").exists():
                    print(f"SKIP (no trained model): {task_stem} × {model_name}")
                    skipped += 1
                    continue
            elif is_complete(folder):
                print(f"SKIP (done): {task_stem} × {model_name}")
                skipped += 1
                completed_by_ds_enc[(dataset, model_name)].append(task_stem)
                continue
            pending.append((task_stem, model_name, encoder_yaml))

    needed_tasks = {ts for ts, _, _ in pending}
    needed_tasks |= {ts for lst in completed_by_ds_enc.values() for ts in lst}
    for ts in sorted(needed_tasks):
        try:
            ensure_manifest(ts)
        except Exception as e:
            print(f"⚠ prepare_data for {ts} failed: {e}", flush=True)

    written: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for (ds, enc), tlist in completed_by_ds_enc.items():
        for ts in tlist:
            try:
                written[(ds, enc)] |= _needed_keys(ts)
            except Exception:
                pass

    pending.sort(key=lambda j: -task_weight(j[0]))
    max_workers = max(1, args.max_workers)
    print(f"\n{len(pending)} pending jobs")
    print(f"Running with up to {max_workers} concurrent workers\n")

    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn, _ in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()

    def dispatch(job: tuple[str, str, str], idx: int) -> tuple[str, bool, float]:
        task_stem, model_name, encoder_yaml = job
        dataset, _ = get_task_info(task_stem)
        key = (dataset, model_name)
        label = f"{task_stem} × {model_name}"
        if test_only:
            return run_one(task_stem, model_name, encoder_yaml, args.device,
                           f"_{idx}", warm_cache_override=False, test_only=True)
        try:
            needed = _needed_keys(task_stem)
        except Exception:
            needed = None
        is_reader = False
        if needed is not None:
            with written_lock:
                is_reader = needed.issubset(written[key])
        if is_reader:
            if cache_only:
                print(f"SKIP (cache covered): {label}", flush=True)
                return label, True, 0.0
            return run_one(task_stem, model_name, encoder_yaml, args.device,
                           f"_{idx}", warm_cache_override=False, test_only=test_only,
                           cache_only=cache_only)
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
                    print(f"SKIP (cache covered): {label}", flush=True)
                    return label, True, 0.0
                return run_one(task_stem, model_name, encoder_yaml, args.device,
                               f"_{idx}", warm_cache_override=False, test_only=test_only,
                               cache_only=cache_only)
            result = run_one(task_stem, model_name, encoder_yaml, args.device,
                             f"_{idx}", warm_cache_override=None, test_only=test_only,
                             cache_only=cache_only)
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

    done_label = "cache warmed" if cache_only else "completed"
    print(f"\n{'='*60}")
    print(f"  Done — {completed} {done_label}, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("  Failed runs:")
        for name in failed:
            print(f"    - {name}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training across cross-task × encoder combos")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show completion status of all runs")
    status_parser.add_argument("--test-only", action="store_true",
                               help="Report CI-presence status instead of completion status")

    summary_parser = sub.add_parser("summary", help="Collect results into per-metric CSVs")
    summary_parser.add_argument("--out-dir", type=str, default="exps/_summary_cross",
                                help="Directory to write metric CSVs (default: exps/_summary_cross)")
    summary_parser.add_argument("--tag", type=str, default=EXPERIMENT_TAG,
                                help=f"Experiment tag to scan (default: {EXPERIMENT_TAG})")

    run_parser = sub.add_parser("run", help="Execute all incomplete runs")
    run_parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0)")
    run_parser.add_argument("--max-workers", "-j", type=int, default=3,
                            help="Max concurrent tasks (default: 3). Writer tasks still serialize per (dataset, encoder).")
    run_parser.add_argument("--encoder", type=str, default=None, action="append",
                            choices=list(ENCODERS.keys()),
                            help="Run only this encoder (repeatable; default: all in ENCODERS dict)")
    run_parser.add_argument("--dataset", type=str, default=None, action="append",
                            help="Run only tasks whose dataset field matches (repeatable; default: all datasets)")
    run_parser.add_argument("--task", type=str, default=None, action="append",
                            help="Run only these task stems (repeatable; default: all tasks)")
    run_parser.add_argument("--test-only", action="store_true",
                            help="Run inference only (no training), requires prior completed runs")
    run_parser.add_argument("--cache-only", action="store_true",
                            help="Warm HDF5 caches via dataio_prep and exit before any training/evaluation")

    args = parser.parse_args()
    if args.command == "run" and getattr(args, "cache_only", False) and getattr(args, "test_only", False):
        parser.error("--cache-only and --test-only are mutually exclusive")
    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "summary":
        cmd_summary(args)


if __name__ == "__main__":
    main()
