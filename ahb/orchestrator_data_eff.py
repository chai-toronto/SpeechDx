"""Data-efficiency orchestrator helpers.

Salvaged from ``run_all_data_eff.py`` (lines 38-294). Same
single-dataset task discovery as ``ahb.orchestrator`` but adds the
``level_dir`` axis: each (task, encoder, level) tuple is its own job
under ``./data_eff_exps/<level_dir>/``. Manifests are subsampled into
the level dir from the source ``./exps/<...>/manifest/`` tree on demand.
"""

from __future__ import annotations

from pathlib import Path

from ahb.orchestrator import (
    BASE_CONFIG,
    _load_main_yaml,
    _load_task_yaml,
    discover_all_encoders,
    discover_all_tasks,
    get_results_file,
    get_task_info,
    is_complete,
)
from ahb.prep.dispatch import ensure_manifest as ensure_source_manifest
from ahb.prep.subsample import subsample_and_write
from ahb.registry import (
    data_eff_default_encoders,
    data_eff_levels,
    encoders as registry_encoders,
    exclude_datasets,
    paper_tasks,
)
from ahb.results import expected_ci_keys

EXP_ROOT_BASE = "data_eff_exps"
LOGS_ROOT = Path("logs/run_all_data_eff")
TASKS_DIR = Path("training/config/tasks")

# Levels are (level_dir, level_value) — level_dir is a path component (no
# '.' so subdirs work cleanly), level_value is the float fraction in [0, 1].
LEVELS = data_eff_levels()
LEVEL_VALUE = {d: v for d, v in LEVELS}


def encoders_default() -> dict[str, str]:
    """Default encoder subset for data-eff (registry override or all)."""
    all_encs = registry_encoders()
    subset = data_eff_default_encoders()
    if subset is None:
        return all_encs
    return {n: all_encs[n] for n in subset if n in all_encs}


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


def get_output_folder(dataset: str, task: str, model_name: str, level_dir: str,
                      tag: str = "run1", probe_name: str = "AvgTProbe") -> Path:
    return Path(
        f"./{EXP_ROOT_BASE}/{level_dir}/{dataset}_{task}/"
        f"{model_name}-{probe_name}-{tag}"
    )


def src_manifest_paths(task_stem: str) -> tuple[Path, Path, Path]:
    """Original (full) manifest paths under exps/<dataset>_<task>/manifest/."""
    dataset, task = get_task_info(task_stem)
    base = Path(f"./exps/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


def manifest_paths(task_stem: str, level_dir: str) -> tuple[Path, Path, Path]:
    """Subsampled manifest paths for a (task, level) pair."""
    dataset, task = get_task_info(task_stem)
    base = Path(f"./{EXP_ROOT_BASE}/{level_dir}/{dataset}_{task}/manifest")
    return base / "train.json", base / "valid.json", base / "test.json"


def is_cv(task_stem: str) -> bool:
    return _load_task_yaml(task_stem).get("num_fold") is not None


def ensure_manifest_data_eff(task_stem: str, level_dir: str) -> None:
    """Subsample the source manifest into ``data_eff_exps/<level>/.../manifest/``.

    Idempotent: skips when the destination already has the expected files.
    Source manifests under ``exps/<...>/manifest/`` are produced on demand
    via ``ahb.prep.dispatch.ensure_manifest``.
    """
    src_tr, src_va, src_te = src_manifest_paths(task_stem)
    dst_tr, dst_va, dst_te = manifest_paths(task_stem, level_dir)
    cv = is_cv(task_stem)

    if dst_tr.exists() and dst_va.exists() and (cv or dst_te.exists()):
        return

    ensure_source_manifest(task_stem)

    seed = int(_load_main_yaml().get("random_seed", 2026))
    level_value = LEVEL_VALUE[level_dir]
    print(f"Subsampling {task_stem} → {level_dir} ({level_value:.4%}) …",
          flush=True)
    subsample_and_write(
        src_manifest_dir=src_tr.parent,
        dst_manifest_dir=dst_tr.parent,
        task_yaml_path=TASKS_DIR / f"{task_stem}.yaml",
        level=level_value,
        seed=seed,
        task_stem=task_stem,
    )


def has_ci_results(output_folder: Path, task_stem: str) -> bool:
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


__all__ = [
    "EXP_ROOT_BASE",
    "LEVELS",
    "LEVEL_VALUE",
    "LOGS_ROOT",
    "discover_tasks",
    "encoders_default",
    "ensure_manifest_data_eff",
    "get_output_folder",
    "get_results_file",
    "get_task_info",
    "has_ci_results",
    "is_complete",
    "is_cv",
    "manifest_paths",
    "src_manifest_paths",
]
