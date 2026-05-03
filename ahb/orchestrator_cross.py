"""Cross-task variants of the orchestrator helpers in ``ahb/orchestrator.py``.

Salvaged from ``run_all_cross.py`` (lines 28-256) — same shape as the
single-dataset orchestrator helpers but reads task yamls from
``training/config/cross_tasks/`` and writes outputs under
``./exps/cross/``. Probe defaults to ``Probe`` instead of ``AvgTProbe``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ahb.registry import cross_pairs, encoders as registry_encoders
from ahb.results import expected_ci_keys
from ahb.yaml_io import TolerantLoader

TASKS_DIR = Path("training/config/cross_tasks")
ENCODERS_DIR = Path("training/config/encoders")
BASE_CONFIG = Path("training/config/main_cross.yaml")
EXPS_ROOT = Path("./exps/cross")
LOGS_ROOT = Path("logs/run_all_cross")

# Cross-task default probe differs from single-dataset.
CROSS_PROBE_NAME = "Probe"
CROSS_PROBE_YAML = "Probe.yaml"
EXPERIMENT_TAG = "run1"


def discover_all_tasks() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def discover_all_encoders() -> dict[str, str]:
    encs = registry_encoders()
    yaml_to_folder = {v: k for k, v in encs.items()}
    stem_to_yaml = {p.stem: p.name for p in ENCODERS_DIR.glob("*.yaml")
                    if not p.stem.startswith("_")}
    return {stem: yaml_to_folder.get(yaml_name, stem)
            for stem, yaml_name in sorted(stem_to_yaml.items())}


def discover_tasks(datasets: list[str] | None = None,
                   tasks: list[str] | None = None,
                   *, include_categories: bool = False) -> list[str]:
    """Sorted cross-pair task stems narrowed by --dataset / --task.

    By default skips ``category_*`` (those belong to the cross-category
    runner). Set ``include_categories=True`` to flip that — used by
    ``ahb run-cross-category``.
    """
    seed = list(cross_pairs())
    if seed:
        stems = seed
    else:
        stems = sorted(
            p.stem for p in TASKS_DIR.glob("*.yaml")
            if include_categories or not p.stem.startswith("category_")
        )
    if include_categories:
        stems = sorted(p.stem for p in TASKS_DIR.glob("category_*.yaml"))
    allowed_ds = set(datasets) if datasets else None
    allowed_tasks = set(tasks) if tasks else None
    out = []
    for s in stems:
        ds = get_task_info(s)[0]
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


def get_output_folder(dataset: str, task: str, model_name: str,
                      tag: str = EXPERIMENT_TAG,
                      *, exps_root: Path | None = None) -> Path:
    return (exps_root or EXPS_ROOT) / f"{dataset}_{task}" / f"{model_name}-{CROSS_PROBE_NAME}-{tag}"


_task_yaml_cache: dict[str, dict] = {}
_task_ids_cache: dict[str, frozenset[str]] = {}


def _load_task_yaml(task_stem: str) -> dict:
    if task_stem not in _task_yaml_cache:
        _task_yaml_cache[task_stem] = yaml.load(
            (TASKS_DIR / f"{task_stem}.yaml").read_text(),
            Loader=TolerantLoader,
        ) or {}
    return _task_yaml_cache[task_stem]


def manifest_paths(task_stem: str, *, exps_root: Path | None = None) -> tuple[Path, Path, Path]:
    dataset, task = get_task_info(task_stem)
    base = (exps_root or EXPS_ROOT) / f"{dataset}_{task}" / "manifest"
    return base / "train.json", base / "valid.json", base / "test.json"


def task_num_ver(task_stem: str) -> int:
    return int(_load_task_yaml(task_stem).get("num_aug_ver", 1))


def task_ids(task_stem: str, *, exps_root: Path | None = None) -> frozenset[str]:
    cache_key = (task_stem, str(exps_root) if exps_root else "")
    if cache_key in _task_ids_cache:
        return _task_ids_cache[cache_key]
    ids: set[str] = set()
    for p in manifest_paths(task_stem, exps_root=exps_root):
        if p.exists():
            with p.open() as f:
                data = json.load(f)
                if isinstance(data, dict):
                    ids |= set(data.keys())
    frozen = frozenset(ids)
    _task_ids_cache[cache_key] = frozen
    return frozen


def task_weight(task_stem: str, *, exps_root: Path | None = None) -> int:
    return task_num_ver(task_stem) * len(task_ids(task_stem, exps_root=exps_root))


def needed_keys(task_stem: str, *, exps_root: Path | None = None) -> set[tuple[str, int]]:
    ids = task_ids(task_stem, exps_root=exps_root)
    nv = task_num_ver(task_stem)
    return {(uid, v) for uid in ids for v in range(nv)}


def get_results_file(task_stem: str | None = None) -> str:
    """Cross runs always emit test_results.txt — no CV variants under cross."""
    return "test_results.txt"


def is_complete(output_folder: Path, task_stem: str | None = None) -> bool:
    return (output_folder / get_results_file(task_stem)).exists()


def has_ci_results(output_folder: Path, task_stem: str) -> bool:
    results_file = output_folder / get_results_file(task_stem)
    if not results_file.exists():
        return False
    expected = expected_ci_keys(_load_task_yaml(task_stem).get("task_type", ""))
    if expected is None:
        return True
    lo, hi = expected
    text = results_file.read_text()
    return lo in text and hi in text


def forget_task_ids(task_stem: str, *, exps_root: Path | None = None) -> None:
    cache_key = (task_stem, str(exps_root) if exps_root else "")
    _task_ids_cache.pop(cache_key, None)
