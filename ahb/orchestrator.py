"""Shared orchestrator helpers — task discovery, completion checks,
cache-key bookkeeping for the writer/reader serialization in
``ahb/run.py``.

Salvaged from ``run_all.py`` (lines 52-156, 200-239, 564-568) with import
paths swapped to ``ahb`` modules. Preserves the legacy behavior so
``ahb status`` / ``ahb summary`` produce byte-identical output to the
legacy ``run_all.py status`` / ``run_all.py summary``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

from ahb.paths import DEFAULT_EXPERIMENT_TAG, DEFAULT_PROBE_NAME
from ahb.registry import encoders as registry_encoders
from ahb.registry import exclude_datasets, paper_tasks
from ahb.results import expected_ci_keys
from ahb.yaml_io import TolerantLoader

_CONFIGS = Path(__file__).resolve().parent / "configs"
TASKS_DIR = _CONFIGS / "tasks"
CROSS_TASKS_DIR = _CONFIGS / "cross_tasks"
ENCODERS_DIR = _CONFIGS / "encoders"
BASE_CONFIG = _CONFIGS / "main.yaml"
LOGS_ROOT = Path("logs/run_all")


def discover_all_tasks() -> list[str]:
    """Every task yaml stem under TASKS_DIR (no exclusion filter)."""
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def discover_all_encoders() -> dict[str, str]:
    """Every non-partial encoder yaml stem mapped to its output-folder name."""
    encs = registry_encoders()
    yaml_to_folder = {v: k for k, v in encs.items()}
    stem_to_yaml = {p.stem: p.name for p in ENCODERS_DIR.glob("*.yaml")
                    if not p.stem.startswith("_")}
    return {stem: yaml_to_folder.get(yaml_name, stem)
            for stem, yaml_name in sorted(stem_to_yaml.items())}


def discover_tasks(datasets: list[str] | None = None,
                   tasks: list[str] | None = None) -> list[str]:
    """Sorted PAPER_TASKS narrowed by --dataset / --task; respects exclusions."""
    available = {p.stem for p in TASKS_DIR.glob("*.yaml")}
    stems = sorted(s for s in paper_tasks() if s in available)
    allowed_ds = set(datasets) if datasets else None
    allowed_tasks = set(tasks) if tasks else None
    excludes = exclude_datasets()
    out = []
    for s in stems:
        ds = get_task_info(s)[0]
        if ds in excludes:
            continue
        if allowed_ds is not None and ds not in allowed_ds:
            continue
        if allowed_tasks is not None and s not in allowed_tasks:
            continue
        out.append(s)
    return out


def get_task_info(task_stem: str) -> tuple[str, str]:
    """Extract (dataset, task) from a task YAML via regex (cheap; no full parse)."""
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    dataset_m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    task_m = re.search(r"^task:\s*(\S+)", text, re.MULTILINE)
    if not dataset_m or not task_m:
        raise ValueError(f"Could not parse dataset/task from {task_stem}.yaml")
    return dataset_m.group(1), task_m.group(1)


def get_output_folder(dataset: str, task: str, model_name: str,
                      tag: str = DEFAULT_EXPERIMENT_TAG) -> Path:
    return Path(f"./exps/single_task/{dataset}_{task}/{model_name}-{DEFAULT_PROBE_NAME}-{tag}")


_task_yaml_cache: dict[str, dict] = {}
_task_ids_cache: dict[str, frozenset[str]] = {}
_main_yaml_cache: dict | None = None


def _load_main_yaml() -> dict:
    global _main_yaml_cache
    if _main_yaml_cache is None:
        _main_yaml_cache = yaml.load(BASE_CONFIG.read_text(),
                                     Loader=TolerantLoader) or {}
    return _main_yaml_cache


def _load_task_yaml(task_stem: str) -> dict:
    if task_stem not in _task_yaml_cache:
        _task_yaml_cache[task_stem] = yaml.load(
            (TASKS_DIR / f"{task_stem}.yaml").read_text(),
            Loader=TolerantLoader,
        ) or {}
    return _task_yaml_cache[task_stem]


def manifest_paths(task_stem: str) -> tuple[Path, Path, Path]:
    dataset, task = get_task_info(task_stem)
    base = Path(f"./exps/single_task/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


def task_num_ver(task_stem: str) -> int:
    return int(_load_task_yaml(task_stem).get("num_aug_ver", 1))


def task_ids(task_stem: str) -> frozenset[str]:
    """All uids across train/valid/test manifests for ``task_stem``."""
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
    """Sort key: biggest cache-writers first so readers unlock sooner."""
    return task_num_ver(task_stem) * len(task_ids(task_stem))


def needed_keys(task_stem: str) -> set[tuple[str, int]]:
    """Cache keys (id, version) a task's warmup would write."""
    ids = task_ids(task_stem)
    nv = task_num_ver(task_stem)
    return {(uid, v) for uid in ids for v in range(nv)}


def is_cv(task_stem: str) -> bool:
    return _load_task_yaml(task_stem).get("num_fold") is not None


def get_results_file(task_stem: str) -> str:
    return "test_results.yaml" if is_cv(task_stem) else "test_results.txt"


def is_complete(output_folder: Path, task_stem: str) -> bool:
    return (output_folder / get_results_file(task_stem)).exists()


def has_ci_results(output_folder: Path, task_stem: str) -> bool:
    """True iff the result file already contains CI fields (so a re-run adds nothing)."""
    results_file = output_folder / get_results_file(task_stem)
    if not results_file.exists():
        return False
    if results_file.suffix == ".yaml":
        return True
    expected = expected_ci_keys(_load_task_yaml(task_stem).get("task_type", ""))
    if expected is None:
        return True
    lo, hi = expected
    text = results_file.read_text()
    return lo in text and hi in text


def reset_caches() -> None:
    """Reset the module-level caches — used by tests + per-task ID refresh."""
    global _main_yaml_cache
    _main_yaml_cache = None
    _task_yaml_cache.clear()
    _task_ids_cache.clear()


def forget_task_ids(task_stem: str) -> None:
    """Drop one task's cached ID set so the next call re-reads from disk."""
    _task_ids_cache.pop(task_stem, None)
