#!/usr/bin/env python3
"""Data-efficiency driver: re-trains the paper-benchmark probes at four
reduced training-set sizes (6.25%, 12.5%, 25%, 50% of train participants).

Manifests are produced once by the regular pipeline (under exps/<...>/manifest/),
then copied + subsampled into data_eff_exps/<level>/<...>/manifest/. Test split
(and CV held-out folds) pass through unchanged. The encoder cache at
embeddings_avg_final/<dataset>/<model>/ is shared with the full benchmark and
is unaffected by manifest size.
"""

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
from datetime import datetime
from pathlib import Path

import yaml

from training.dataio.subsample import subsample_and_write

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
PROBE_YAML = "Probe.yaml"
EXPERIMENT_TAG = "run1"

BASE_CONFIG = Path("training/config/main.yaml")
TMP_CONFIG = Path("training/config/_tmp_run.yaml")
TASKS_DIR = Path("training/config/tasks")
ENCODERS_DIR = Path("training/config/encoders")

# ─── Data-efficiency configuration ─────────────────────────────────────
EXP_ROOT_BASE = "data_eff_exps"
# (level_dir, level_value). level_dir becomes a path component, so no '.'.
LEVELS = [
    ("06p25", 0.0625),
    ("12p5",  0.125),
    ("25",    0.25),
    ("50",    0.50),
]
LEVEL_VALUE = {d: v for d, v in LEVELS}

# Tasks listed on the paper "Task Characteristics" sheet — the only set
# evaluated under data-efficiency.
PAPER_TASKS = [
    "edaic_depC", "edaic_phqR",
    "ravdess_emoC", "ravdess_emoBC",
    "iemocap_emoC", "iemocap_emoBC",
    "dbank_adC", "dbank_mmseR",
    "aphasia_pwaC",
    "torgo_dysC", "torgo_sevR",
    "uaspeech_dysC",
    "mvdr_parkC", "mvdr_updrs5R", "mvdr_updrs18R", "mvdr_hyR",
    "ksof_intC", "ksof_stutL",
    "c9s_t1", "c9s_L_t1", "c9s_t2", "c9s_L_t2", "c9s_sympL",
    "coswara_sympC", "coswara_covidC", "coswara_sympL",
    "avfad_pathC",
]

# ─── Per-task log delegation ────────────────────────────────────────────
LOGS_ROOT = Path("logs/run_all_data_eff")
ROLE_W = 7
_terminal_lock = threading.Lock()


def _emit(*lines: str) -> None:
    with _terminal_lock:
        for ln in lines:
            print(ln, flush=True)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _tail(path: Path, n: int = 20) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ["(log not written)"]
    except Exception as e:
        return [f"(log unreadable: {e})"]
    lines = text.splitlines()
    return lines[-n:] if lines else ["(log empty)"]


class _Progress:
    def __init__(self, total: int):
        self.total = total
        self.active = 0
        self.done = 0
        self.failed = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.active += 1

    def finish(self, ok: bool) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if ok:
                self.done += 1
            else:
                self.failed += 1

    def snap(self) -> str:
        with self._lock:
            queued = max(0, self.total - self.active - self.done - self.failed)
            return (f"act={self.active} queued={queued} "
                    f"done={self.done} fail={self.failed}/{self.total}")
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
    """Return sorted PAPER_TASKS, optionally narrowed by `datasets` / `tasks`.
    Tasks not present under TASKS_DIR are dropped silently."""
    available = {p.stem for p in TASKS_DIR.glob("*.yaml")}
    stems = sorted(s for s in PAPER_TASKS if s in available)
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


def get_output_folder(dataset: str, task: str, model_name: str, level_dir: str,
                      tag: str = EXPERIMENT_TAG) -> Path:
    return Path(f"./{EXP_ROOT_BASE}/{level_dir}/{dataset}_{task}/"
                f"{model_name}-{PROBE_NAME}-{tag}")


def src_manifest_paths(task_stem: str) -> tuple[Path, Path, Path]:
    """Original (full) manifest paths under exps/<dataset>_<task>/manifest/."""
    dataset, task = get_task_info(task_stem)
    base = Path(f"./exps/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


# ─── Task metadata (manifests, num_aug_ver) ─────────────────────────────
_task_yaml_cache: dict[str, dict] = {}
_task_ids_cache: dict[tuple[str, str], frozenset[str]] = {}
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


def manifest_paths(task_stem: str, level_dir: str) -> tuple[Path, Path, Path]:
    """Subsampled manifest paths for a (task, level) pair."""
    dataset, task = get_task_info(task_stem)
    base = Path(f"./{EXP_ROOT_BASE}/{level_dir}/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


def task_num_ver(task_stem: str) -> int:
    return int(_load_task_yaml(task_stem).get("num_aug_ver", 1))


def task_ids(task_stem: str, level_dir: str) -> frozenset[str]:
    """Union of uids across the subsampled manifests for (task, level)."""
    cache_key = (task_stem, level_dir)
    if cache_key in _task_ids_cache:
        return _task_ids_cache[cache_key]
    ids: set[str] = set()
    for p in manifest_paths(task_stem, level_dir):
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
    _task_ids_cache[cache_key] = frozen
    return frozen


def task_weight(task_stem: str, level_dir: str) -> int:
    """Sort key: run biggest cache-writers first so readers unlock sooner."""
    return task_num_ver(task_stem) * len(task_ids(task_stem, level_dir))


def _ensure_source_manifest(task_stem: str) -> None:
    """Call the task's prepare_data_fn if the source manifest under
    exps/<dataset>_<task>/manifest/ is missing. Mirrors run_all.ensure_manifest.

    Different prep_*.py modules take slightly different kwargs (e.g. CV tasks
    take `num_fold` and `raw_label_key` instead of `ratio`/`manifest_test_path`),
    so we build a superset kwargs dict and filter to the function's actual
    signature.
    """
    tr, va, te = src_manifest_paths(task_stem)
    if tr.exists() and va.exists():
        return
    tcfg = _load_task_yaml(task_stem)
    mcfg = _load_main_yaml()
    dataset, task = tcfg["dataset"], tcfg["task"]
    data_folder = mcfg.get("data_folder", "./data/")

    module = importlib.import_module(tcfg["data_io_script"])
    fn = getattr(module, tcfg["prepare_data_fn"])
    print(f"Preparing source manifests for {task_stem} …", flush=True)
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


def ensure_manifest(task_stem: str, level_dir: str) -> None:
    """Ensure the subsampled manifests exist at data_eff_exps/<level>/<...>/manifest/.

    Idempotent: skips when the destination already has the expected files.
    Source manifests under exps/<...>/manifest/ are produced on demand.
    """
    src_tr, src_va, src_te = src_manifest_paths(task_stem)
    dst_tr, dst_va, dst_te = manifest_paths(task_stem, level_dir)
    cv = is_cv(task_stem)

    # Idempotent skip: train+valid present (and test if non-CV) ⇒ already done.
    if dst_tr.exists() and dst_va.exists() and (cv or dst_te.exists()):
        return

    _ensure_source_manifest(task_stem)
    seed = int(_load_main_yaml().get("random_seed", 2026))
    level_value = LEVEL_VALUE[level_dir]
    print(f"Subsampling {task_stem} → {level_dir} ({level_value:.4%}) …", flush=True)
    subsample_and_write(
        src_manifest_dir=src_tr.parent,
        dst_manifest_dir=dst_tr.parent,
        task_yaml_path=TASKS_DIR / f"{task_stem}.yaml",
        level=level_value,
        seed=seed,
        task_stem=task_stem,
    )
    _task_ids_cache.pop((task_stem, level_dir), None)


def is_cv(task_stem: str) -> bool:
    """A task is CV iff its yaml sets num_fold."""
    return _load_task_yaml(task_stem).get("num_fold") is not None


def get_train_command(task_stem: str) -> str:
    """CV tasks use trainPerFoldCV; everything else uses train."""
    if is_cv(task_stem):
        return "python -m training.trainPerFoldCV"
    return "python -m training.train"


def get_results_file(task_stem: str) -> str:
    """trainPerFoldCV writes test_results.yaml; train writes test_results.txt."""
    if is_cv(task_stem):
        return "test_results.yaml"
    return "test_results.txt"


def is_complete(output_folder: Path, task_stem: str) -> bool:
    return (output_folder / get_results_file(task_stem)).exists()


def _expected_ci_keys(task_type: str) -> tuple[str, str] | None:
    """CI field names brain.py emits per task_type. None = no CI produced."""
    if task_type == "R":
        return ("MAE_CI_low", "MAE_CI_high")
    if task_type in ("B", "C", "L"):
        return ("AUROC_CI_low", "AUROC_CI_high")
    return None


def has_ci_results(output_folder: Path, task_stem: str) -> bool:
    """test_only skip gate: True iff results already contain CI fields (so a
    re-run would add nothing). mvdr (CV yaml) synthesizes CI from cross-fold
    std and doesn't go through test_only; task types without CI support are
    also treated as already-satisfied."""
    results_file = output_folder / get_results_file(task_stem)
    if not results_file.exists():
        return False
    if results_file.suffix == ".yaml":
        return True
    expected = _expected_ci_keys(_load_task_yaml(task_stem).get("task_type", ""))
    if expected is None:
        return True
    lo, hi = expected
    text = results_file.read_text()
    return lo in text and hi in text


def make_config(model_name: str, encoder_yaml: str, task_yaml: str, level_dir: str,
                config_id: str = "",
                warm_cache_override: bool | None = None, test_only: bool = False,
                cache_only: bool = False) -> Path:
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
    # Redirect ./exps/ → ./data_eff_exps/<level>/ for output_folder and the three
    # manifest paths in main.yaml (no other literal references to ./exps/).
    text = text.replace("./exps/", f"./{EXP_ROOT_BASE}/{level_dir}/")
    # Remove chunk_at line entirely
    text = re.sub(r"^chunk_at:.*\n?", "", text, flags=re.MULTILINE)
    # Force skip_prep: True — ensure_manifest handles this
    text = re.sub(r"^skip_prep:.*$", "skip_prep: True", text, flags=re.MULTILINE)
    # Override test_only and warm_cache if requested. test_only and cache_only
    # are mutually exclusive (enforced at the CLI), so handle them separately.
    if cache_only:
        # Force warm_cache true — otherwise dataio_prep opens read-only and the
        # train script's cache_only exit path warms nothing.
        text = re.sub(r"^warm_cache:.*$", "warm_cache: true", text, flags=re.MULTILINE)
        text += "\ncache_only: True\n"
    elif test_only:
        text = re.sub(r"^test_only:.*$", "test_only: True", text, flags=re.MULTILINE)
        text = re.sub(r"^warm_cache:.*$", "warm_cache: false", text, flags=re.MULTILINE)
    # Include model name + level + SLURM job id so concurrent jobs (same encoder,
    # different datasets/levels) don't overwrite/delete each other's temp config.
    job_suffix = f"_job{os.environ['SLURM_JOB_ID']}" if os.environ.get("SLURM_JOB_ID") else ""
    config_path = Path(
        f"training/config/_tmp_run_de_{model_name}_{level_dir}{job_suffix}{config_id}.yaml"
    )
    config_path.write_text(text)
    return config_path


def run_one(task_stem: str, model_name: str, encoder_yaml: str, level_dir: str,
            device=None, config_id: str = "",
            warm_cache_override: bool | None = None, test_only: bool = False,
            cache_only: bool = False,
            log_path: Path | None = None) -> tuple[str, bool, float]:
    """Run a single training job, redirecting child stdout/stderr to log_path.
    Returns (label, success, elapsed_seconds)."""
    label = f"{task_stem} × {model_name} @ {level_dir}"
    task_yaml = f"{task_stem}.yaml"
    config_path = make_config(model_name, encoder_yaml, task_yaml, level_dir, config_id,
                              warm_cache_override=warm_cache_override, test_only=test_only,
                              cache_only=cache_only)

    cmd = get_train_command(task_stem).split()
    cmd.append(str(config_path))
    if device:
        cmd.append(f"--device={device}")

    if log_path is None:
        log_path = (LOGS_ROOT / "_ad_hoc" / level_dir
                    / f"{_slug(task_stem)}__{_slug(model_name)}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    start = time.time()
    try:
        with log_path.open("w", buffering=1) as f:
            f.write(f"{'='*60}\n")
            f.write(f"  Running: {label}\n")
            f.write(f"  Command: {' '.join(cmd)}\n")
            f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"  test_only={test_only}  cache_only={cache_only}  "
                    f"warm_cache_override={warm_cache_override}  level={level_dir}\n")
            f.write(f"{'='*60}\n")
            f.flush()
            subprocess.run(cmd, check=True, stdout=f,
                           stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - start
        with log_path.open("a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"  ✓ {label} completed in {elapsed/60:.1f} min\n")
            f.write(f"  Finished: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"{'='*60}\n")
        return label, True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        with log_path.open("a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"  ✗ {label} FAILED (exit code {e.returncode}) after {elapsed/60:.1f} min\n")
            f.write(f"  Finished: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"{'='*60}\n")
        return label, False, elapsed
    finally:
        config_path.unlink(missing_ok=True)


# ─── Result parsing ────────────────────────────────────────────────────
CLASSIFICATION_METRICS = ["AUROC", "AUC_CI", "F1", "accuracy"]
REGRESSION_METRICS = ["MAE", "MAE_CI", "MSE", "PearsonR", "R2"]
# filename -> list of metric keys that belong in that CSV
CSV_LAYOUT = {
    "AUC": "AUROC",
    "AUC_CI": "AUC_CI",
    "F1": "F1",
    "Acc": "accuracy",
    "MAE": "MAE",
    "MAE_CI": "MAE_CI",
    "MSE": "MSE",
    "PearsonR": "PearsonR",
    "R2": "R2",
}


def parse_results_txt(path: Path) -> dict[str, float | str]:
    """Parse 'key: value' lines from a test_results.txt file."""
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
    lo, hi = metrics.get("MAE_CI_low"), metrics.get("MAE_CI_high")
    if isinstance(lo, float) and isinstance(hi, float):
        metrics["MAE_CI"] = f"({lo:.4f}, {hi:.4f})"
    return metrics


def parse_results_yaml(path: Path) -> dict[str, float | str]:
    """Parse test_results.yaml (trainPerFoldCV) — take summary means.

    For mvdr (CV) AUROC and MAE, synthesize AUC_CI / MAE_CI as
    (mean - std, mean + std) since per-fold CIs aren't aggregated; cross-fold
    std is the meaningful uncertainty.
    """
    data = yaml.safe_load(path.read_text())
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    metrics: dict[str, float | str] = {
        k: float(v["mean"]) for k, v in summary.items()
        if isinstance(v, dict) and "mean" in v
    }
    auroc = summary.get("AUROC")
    if isinstance(auroc, dict) and "mean" in auroc and "std" in auroc:
        m, s = float(auroc["mean"]), float(auroc["std"])
        metrics["AUC_CI"] = f"({m - s:.4f}, {m + s:.4f})"
    mae = summary.get("MAE")
    if isinstance(mae, dict) and "mean" in mae and "std" in mae:
        m, s = float(mae["mean"]), float(mae["std"])
        metrics["MAE_CI"] = f"({m - s:.4f}, {m + s:.4f})"
    return metrics


def load_metrics(output_folder: Path, task_stem: str) -> dict[str, float] | None:
    results_file = output_folder / get_results_file(task_stem)
    if not results_file.exists():
        return None
    if results_file.suffix == ".yaml":
        return parse_results_yaml(results_file)
    return parse_results_txt(results_file)


def _resolve_levels(arg_levels: list[str] | None) -> list[tuple[str, float]]:
    """Filter LEVELS to the user-specified set. None ⇒ all levels."""
    if not arg_levels:
        return list(LEVELS)
    by_name = {d: v for d, v in LEVELS}
    out = []
    for name in arg_levels:
        if name not in by_name:
            raise SystemExit(
                f"Unknown level {name!r}. Choices: {[d for d, _ in LEVELS]}"
            )
        out.append((name, by_name[name]))
    return out


def cmd_summary(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    encoders = list(ENCODERS.keys())
    levels = _resolve_levels(getattr(args, "level", None))

    # Classify tasks by task_type yaml field so empty rows are still emitted
    regression_tasks: set[str] = set()
    classification_tasks: set[str] = set()
    for task_stem in tasks:
        if _load_task_yaml(task_stem).get("task_type") == "R":
            regression_tasks.add(task_stem)
        else:
            classification_tasks.add(task_stem)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    grand_found, grand_missing, written_paths = 0, 0, []

    for level_dir, _ in levels:
        # metric_key -> {task_stem: {encoder: value}}
        matrices: dict[str, dict[str, dict[str, float]]] = {
            k: defaultdict(dict) for k in CSV_LAYOUT.values()
        }
        found, missing = 0, 0
        for task_stem in tasks:
            dataset, task = get_task_info(task_stem)
            for model_name in encoders:
                folder = get_output_folder(dataset, task, model_name, level_dir, args.tag)
                metrics = load_metrics(folder, task_stem)
                if metrics is None:
                    missing += 1
                    continue
                found += 1
                for metric_key in CSV_LAYOUT.values():
                    if metric_key in metrics:
                        matrices[metric_key][task_stem][model_name] = metrics[metric_key]

        grand_found += found
        grand_missing += missing
        out_dir = out_root / level_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv_name, metric_key in CSV_LAYOUT.items():
            data = matrices[metric_key]
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
                        if isinstance(val, float):
                            row.append(f"{val:.4f}")
                        elif isinstance(val, str):
                            row.append(val)
                        else:
                            row.append("")
                    writer.writerow(row)
            written_paths.append(csv_path)

        # Per-level completion matrix over the paper task set.
        comp_path = out_dir / "completion.csv"
        totals = {enc: 0 for enc in encoders}
        with comp_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *encoders])
            for task_stem in tasks:
                dataset, task = get_task_info(task_stem)
                row = [task_stem]
                for enc in encoders:
                    folder = get_output_folder(dataset, task, enc, level_dir, args.tag)
                    done = is_complete(folder, task_stem)
                    row.append("1" if done else "0")
                    if done:
                        totals[enc] += 1
                writer.writerow(row)
            writer.writerow(["TOTAL", *(str(totals[e]) for e in encoders)])
        written_paths.append(comp_path)

    print(f"\nParsed {grand_found} result files, {grand_missing} missing")
    print(f"Wrote {len(written_paths)} CSV(s) under {out_root}/")
    for p in written_paths:
        print(f"  {p}")


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    test_only = getattr(args, "test_only", False)
    levels = _resolve_levels(getattr(args, "level", None))
    encoders = list(ENCODERS.keys())

    grand_complete, grand_incomplete, grand_absent = 0, 0, 0

    for level_dir, _ in levels:
        complete, incomplete, absent = 0, 0, 0
        rows = []
        for task_stem in tasks:
            dataset, task = get_task_info(task_stem)
            row = [task_stem]
            ci_applicable = (
                _expected_ci_keys(_load_task_yaml(task_stem).get("task_type", "")) is not None
            ) if test_only else True
            for model_name in encoders:
                folder = get_output_folder(dataset, task, model_name, level_dir)
                if test_only:
                    if not ci_applicable or not (folder / get_results_file(task_stem)).exists():
                        row.append(" ")
                        absent += 1
                    elif has_ci_results(folder, task_stem):
                        row.append("☑")
                        complete += 1
                    else:
                        row.append("☐")
                        incomplete += 1
                elif is_complete(folder, task_stem):
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

        def format_row(row, w=widths):
            return " | ".join(cell.ljust(w[idx]) for idx, cell in enumerate(row))

        print(f"\n=== Level {level_dir} ===")
        print(format_row(headers))
        print("-+-".join("-" * width for width in widths))
        for row in rows:
            print(format_row(row))

        if test_only:
            total = complete + incomplete + absent
            print(f"  ({complete}/{total} with CI, {incomplete} pending CI, {absent} no result)")
        else:
            total = complete + incomplete
            print(f"  ({complete}/{total} complete, {incomplete} remaining)")
        grand_complete += complete
        grand_incomplete += incomplete
        grand_absent += absent

    if test_only:
        print(f"\nLegend: ☑ CI present, ☐ tested without CI, blank = no test result")
        print(f"All levels: {grand_complete} CI / {grand_incomplete} pending CI / {grand_absent} no result")
    else:
        print(f"\nLegend: ☑ complete, ☐ incomplete")
        print(f"All levels: {grand_complete}/{grand_complete + grand_incomplete} complete")


def _needed_keys(task_stem: str, level_dir: str) -> set[tuple[str, int]]:
    """Cache keys this (task, level) job would need: (id, version) per uid × ver."""
    ids = task_ids(task_stem, level_dir)
    num_ver = task_num_ver(task_stem)
    return {(uid, v) for uid in ids for v in range(num_ver)}


def cmd_run(args: argparse.Namespace) -> None:
    tasks = discover_tasks(args.dataset, args.task)
    levels = _resolve_levels(getattr(args, "level", None))
    skipped, completed, failed = 0, 0, []
    failed_log_paths: dict[str, Path] = {}
    test_only = getattr(args, 'test_only', False)
    cache_only = getattr(args, 'cache_only', False)

    # Filter encoders by --encoder flag (default: all). Repeatable.
    if args.encoder:
        encoders = {e: ENCODERS[e] for e in args.encoder}
    else:
        encoders = ENCODERS

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = LOGS_ROOT / run_stamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # job tuple: (task_stem, model_name, encoder_yaml, level_dir)
    pending: list[tuple[str, str, str, str]] = []
    completed_by_ds_enc: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for level_dir, _ in levels:
        for task_stem in tasks:
            dataset, task = get_task_info(task_stem)
            for model_name, encoder_yaml in encoders.items():
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
                    completed_by_ds_enc[(dataset, model_name)].append((task_stem, level_dir))
                    continue
                pending.append((task_stem, model_name, encoder_yaml, level_dir))

    # Subsample manifests for every (task, level) we'll touch, plus completed
    # ones used to seed the cache-written set.
    needed_pairs = {(ts, ld) for ts, _, _, ld in pending}
    needed_pairs |= {pair for lst in completed_by_ds_enc.values() for pair in lst}
    for ts, ld in sorted(needed_pairs):
        try:
            ensure_manifest(ts, ld)
        except Exception as e:
            print(f"⚠ subsample for {ts} @ {ld} failed: {e}", flush=True)

    # Seed per-(dataset, encoder) written-key sets from already-completed jobs.
    written: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for (ds, enc), pair_list in completed_by_ds_enc.items():
        for ts, ld in pair_list:
            try:
                written[(ds, enc)] |= _needed_keys(ts, ld)
            except Exception:
                pass

    # Run biggest cache-writers first so readers unlock sooner.
    pending.sort(key=lambda j: -task_weight(j[0], j[3]))
    total_jobs = len(pending)
    progress = _Progress(total_jobs)
    max_workers = max(1, args.max_workers)
    mode = "cache-only" if cache_only else ("test-only" if test_only else "train+test")
    _emit(
        "",
        f"[{_now()}] Run start  : {total_jobs} pending, {skipped} skipped",
        f"           workers : up to {max_workers} concurrent",
        f"           mode    : {mode}",
        f"           levels  : {[d for d, _ in levels]}",
        f"           logs    : {run_log_dir}/  (one file per job, nested by level)",
        "",
    )

    ds_enc_locks: dict[tuple[str, str], threading.Lock] = {}
    for ts, mn, _, _ in pending:
        ds = get_task_info(ts)[0]
        ds_enc_locks.setdefault((ds, mn), threading.Lock())
    written_lock = threading.Lock()

    def _execute(job: tuple[str, str, str, str], idx: int, role: str
                 ) -> tuple[str, bool, float]:
        task_stem, model_name, encoder_yaml, level_dir = job
        label = f"{task_stem} × {model_name} @ {level_dir}"
        log_path = (run_log_dir / level_dir
                    / f"{_slug(task_stem)}__{_slug(model_name)}.log")
        cmd_str = (f"{get_train_command(task_stem)} <tmp.yaml>"
                   + (f" --device={args.device}" if args.device else ""))
        warm = None if role.strip() == "writer" else False

        progress.start()
        _emit(
            f"[{_now()}] START [{idx+1:>3}/{total_jobs}] {role:<{ROLE_W}} {label}",
            f"           cmd : {cmd_str}",
            f"           log : {log_path}",
            f"           prog: {progress.snap()}",
        )

        result = run_one(task_stem, model_name, encoder_yaml, level_dir,
                         args.device, f"_{idx}", warm_cache_override=warm,
                         test_only=test_only, cache_only=cache_only,
                         log_path=log_path)
        _, ok, elapsed = result
        progress.finish(ok)
        status = "✓ OK  " if ok else "✗ FAIL"
        head = (f"[{_now()}] END   [{idx+1:>3}/{total_jobs}] {status:<{ROLE_W}} {label}    "
                f"elapsed={elapsed/60:.1f} min   prog: {progress.snap()}")
        if not ok:
            tail_lines = _tail(log_path, 20)
            extra = [f"           tail of {log_path}:"]
            extra += [f"             | {ln}" for ln in tail_lines]
            _emit(head, *extra)
        else:
            _emit(head)
        return result

    def dispatch(job: tuple[str, str, str, str], idx: int) -> tuple[str, bool, float]:
        task_stem, model_name, encoder_yaml, level_dir = job
        dataset, _ = get_task_info(task_stem)
        key = (dataset, model_name)
        label = f"{task_stem} × {model_name} @ {level_dir}"

        if test_only:
            return _execute(job, idx, "test")

        try:
            needed = _needed_keys(task_stem, level_dir)
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
                ts, mn, _, ld = job
                failed_log_paths[label] = (run_log_dir / ld
                                           / f"{_slug(ts)}__{_slug(mn)}.log")

    # Summary
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-efficiency driver: paper tasks × encoders × subsample levels"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    level_choices = [d for d, _ in LEVELS]

    status_parser = sub.add_parser("status", help="Show completion status of all runs")
    status_parser.add_argument("--test-only", action="store_true",
                               help="Report CI-presence status instead of completion status")
    status_parser.add_argument("--level", type=str, default=None, action="append",
                               choices=level_choices,
                               help="Restrict to these levels (repeatable; default: all)")

    summary_parser = sub.add_parser("summary", help="Collect results into per-metric CSVs")
    summary_parser.add_argument("--out-dir", type=str,
                                default=f"{EXP_ROOT_BASE}/_summary",
                                help=f"Directory to write metric CSVs (default: {EXP_ROOT_BASE}/_summary)")
    summary_parser.add_argument("--tag", type=str, default=EXPERIMENT_TAG,
                                help=f"Experiment tag to scan (default: {EXPERIMENT_TAG})")
    summary_parser.add_argument("--level", type=str, default=None, action="append",
                                choices=level_choices,
                                help="Restrict to these levels (repeatable; default: all)")

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
                            help="Run only these task stems (repeatable; default: all paper tasks)")
    run_parser.add_argument("--level", type=str, default=None, action="append",
                            choices=level_choices,
                            help="Restrict to these levels (repeatable; default: all)")
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
