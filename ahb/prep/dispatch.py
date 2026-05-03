"""Dispatch ``prepare_*`` functions for any single-dataset, cross, or
category-cross task by reading its YAML and routing the call.

Ports the three legacy ``ensure_manifest`` implementations
(``run_all.py:159``, ``run_all_cross.py:159``) into one entry point.
The legacy ``data_io_script: training.dataio.prep_X`` paths in task YAMLs
are mapped to the new ``ahb.prep.X`` modules via
``ahb.prep.OLD_TO_NEW_MODULE`` so old YAMLs work unchanged.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from pathlib import Path

import yaml

from ahb.prep import OLD_TO_NEW_MODULE
from ahb.yaml_io import TolerantLoader

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO / "training" / "config"

TASKS_SINGLE_DIR = CONFIG_DIR / "tasks"
TASKS_CROSS_DIR = CONFIG_DIR / "cross_tasks"

MAIN_SINGLE = CONFIG_DIR / "main.yaml"
MAIN_CROSS = CONFIG_DIR / "main_cross.yaml"
MAIN_CROSS_CATEGORY = CONFIG_DIR / "main_cross_category.yaml"

EXPS_SINGLE_ROOT = Path("./exps/single_task")
EXPS_CROSS_ROOT = Path("./exps/cross")
EXPS_CROSS_CATEGORY_ROOT = Path("./exps/cross_cat")


@lru_cache(maxsize=None)
def _load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=TolerantLoader) or {}


def _find_task_yaml(task_stem: str) -> Path:
    """Locate ``<task_stem>.yaml`` in the single or cross tasks dir."""
    for d in (TASKS_SINGLE_DIR, TASKS_CROSS_DIR):
        p = d / f"{task_stem}.yaml"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No task yaml for {task_stem!r}. Searched {TASKS_SINGLE_DIR}, {TASKS_CROSS_DIR}."
    )


def _classify(task_yaml_path: Path, tcfg: dict) -> str:
    """Return one of: 'single', 'cross', 'category'."""
    if task_yaml_path.parent == TASKS_SINGLE_DIR:
        return "single"
    if isinstance(tcfg.get("train_datasets"), list):
        return "category"
    if "train_dataset" in tcfg and "test_dataset" in tcfg:
        return "cross"
    raise ValueError(
        f"Cannot classify task yaml {task_yaml_path}: not under single dir and "
        "missing both train_datasets (category) and train_dataset (cross)."
    )


def _exps_root(kind: str) -> Path:
    return {
        "single":   EXPS_SINGLE_ROOT,
        "cross":    EXPS_CROSS_ROOT,
        "category": EXPS_CROSS_CATEGORY_ROOT,
    }[kind]


def _main_yaml(kind: str) -> Path:
    return {
        "single":   MAIN_SINGLE,
        "cross":    MAIN_CROSS,
        "category": MAIN_CROSS_CATEGORY,
    }[kind]


def manifest_paths(task_stem: str) -> tuple[Path, Path, Path]:
    """Return ``(train, valid, test)`` manifest paths for a task."""
    tpath = _find_task_yaml(task_stem)
    tcfg = _load_yaml(tpath)
    kind = _classify(tpath, tcfg)
    base = _exps_root(kind) / f"{tcfg['dataset']}_{tcfg['task']}" / "manifest"
    return base / "train.json", base / "valid.json", base / "test.json"


def _resolve_metadata_path(data_folder: str, ds_name: str) -> str:
    processed_dir = Path(data_folder) / ds_name / "processed"
    candidates = [
        processed_dir / f"{ds_name}.csv",
        processed_dir / "metadata.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No metadata CSV found for dataset {ds_name!r}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _resolve_module(legacy_path: str):
    """Look up the new module name for a legacy ``training.dataio.prep_X`` path."""
    new_path = OLD_TO_NEW_MODULE.get(legacy_path, legacy_path)
    return importlib.import_module(new_path)


def _build_kwargs_single(tr, va, te, tcfg, mcfg) -> dict:
    dataset = tcfg["dataset"]
    data_folder = mcfg.get("data_folder", "./data/")
    return {
        "wav_folder":          f"{data_folder}/{dataset}/processed/audio",
        "metadata_path":       f"{data_folder}/{dataset}/processed/{dataset}.csv",
        "manifest_train_path": str(tr),
        "manifest_val_path":   str(va),
        "manifest_test_path":  str(te),
        "ratio":               mcfg.get("ratio"),
        "random_seed":         mcfg.get("random_seed"),
        "dataset":             dataset,
        "task":                tcfg["task"],
        "raw_label_key":       tcfg.get("raw_label_key"),
        "num_fold":            tcfg.get("num_fold"),
    }


def _build_kwargs_cross(tr, va, te, tcfg, mcfg) -> dict:
    train_dataset = tcfg["train_dataset"]
    test_dataset = tcfg["test_dataset"]
    data_folder = mcfg.get("data_folder", "./data/")
    return {
        "wav_folder_train":    f"{data_folder}/{train_dataset}/processed/audio",
        "metadata_path_train": _resolve_metadata_path(data_folder, train_dataset),
        "wav_folder_test":     f"{data_folder}/{test_dataset}/processed/audio",
        "metadata_path_test":  _resolve_metadata_path(data_folder, test_dataset),
        "manifest_train_path": str(tr),
        "manifest_val_path":   str(va),
        "manifest_test_path":  str(te),
        "ratio":               mcfg.get("ratio"),
        "random_seed":         mcfg.get("random_seed"),
        "dataset":             tcfg["dataset"],
        "task":                tcfg["task"],
    }


def _build_kwargs_category(tr, va, te, tcfg, mcfg) -> dict:
    """Category-cross: yaml lists multiple train/test datasets and setting_N entries."""
    train_datasets = list(tcfg["train_datasets"])
    test_datasets = list(tcfg.get("test_datasets", []))
    all_ds = list(dict.fromkeys(train_datasets + test_datasets))
    category_settings = [
        v for _, v in sorted(
            (
                (int(k.split("_", 1)[1]), v)
                for k, v in tcfg.items()
                if k.startswith("setting_")
            ),
            key=lambda kv: kv[0],
        )
    ]
    data_folder = mcfg.get("data_folder", "./data/")
    return {
        "manifest_train_path": str(tr),
        "manifest_val_path":   str(va),
        "manifest_test_path":  str(te),
        "ratio":               mcfg.get("ratio"),
        "random_seed":         mcfg.get("random_seed"),
        "dataset":             tcfg["dataset"],
        "task":                tcfg["task"],
        "train_datasets":      train_datasets,
        "test_datasets":       test_datasets,
        "category_settings":   category_settings,
        "dataset_audio_roots": {
            ds: f"{data_folder}/{ds}/processed/audio" for ds in all_ds
        },
        "dataset_metadata_paths": {
            ds: _resolve_metadata_path(data_folder, ds) for ds in all_ds
        },
    }


def ensure_manifest(task_stem: str) -> None:
    """Build manifests for ``task_stem`` if any are missing.

    Routes to the single / cross / category-cross kwarg builder based on the
    task YAML's location and contents. The legacy
    ``data_io_script: training.dataio.prep_X`` field is mapped to the new
    ``ahb.prep.X`` module so YAMLs need no edits.
    """
    tpath = _find_task_yaml(task_stem)
    tcfg = _load_yaml(tpath)
    kind = _classify(tpath, tcfg)
    mcfg = _load_yaml(_main_yaml(kind))

    tr, va, te = manifest_paths(task_stem)
    # Single-dataset CV prep produces only train+valid; everyone else also test.
    if tr.exists() and va.exists():
        if kind == "single" or te.exists():
            return

    module = _resolve_module(tcfg["data_io_script"])
    fn = getattr(module, tcfg["prepare_data_fn"])
    print(f"Preparing manifests for {task_stem} …", flush=True)
    tr.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "single":   _build_kwargs_single,
        "cross":    _build_kwargs_cross,
        "category": _build_kwargs_category,
    }
    all_kwargs = builders[kind](tr, va, te, tcfg, mcfg)
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
    fn(**kwargs)
