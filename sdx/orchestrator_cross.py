"""Cross-task variants of the orchestrator helpers in ``sdx/orchestrator.py``.

Salvaged from ``run_all_cross.py`` (lines 28-256) — same shape as the
single-dataset orchestrator helpers but reads task yamls from
``sdx/configs/cross_tasks/`` and writes outputs under
``./exps/cross/``. Probe defaults to ``Probe`` instead of ``AvgTProbe``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from sdx.registry import (
    cross_categories,
    cross_pairs,
    encoders as registry_encoders,
)
from sdx.yaml_io import TolerantLoader

_CONFIGS = Path(__file__).resolve().parent / "configs"
TASKS_DIR = _CONFIGS / "cross_tasks"
ENCODERS_DIR = _CONFIGS / "encoders"
BASE_CONFIG = _CONFIGS / "main_cross.yaml"
EXPS_ROOT = Path("./exps/cross")
LOGS_ROOT = Path("logs/run_all_cross")

# Cross-task default probe differs from single-dataset.
CROSS_PROBE_NAME = "Probe"
CROSS_PROBE_YAML = "Probe.yaml"
EXPERIMENT_TAG = "run1"


def discover_all_tasks() -> list[str]:
    """Every registered cross stem (pairs + categories) whose yaml exists."""
    available = {p.stem for p in TASKS_DIR.glob("*.yaml")}
    return sorted(s for s in (*cross_pairs(), *cross_categories())
                  if s in available)


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
    """Sorted registered cross stems narrowed by --dataset / --task.

    Reads ``cross_pairs`` (default) or ``cross_categories``
    (``include_categories=True``) from the registry. Stems missing a yaml
    under ``cross_tasks/`` are dropped, matching the ``paper_tasks``
    behavior in ``sdx/orchestrator.py``.

    A cross stem matches ``-d X`` if X appears in the yaml's
    ``train_dataset`` / ``test_dataset`` (cross pairs) or
    ``train_datasets`` / ``test_datasets`` (cross-cat) — i.e. X is
    involved on either side. It also matches if X equals the legacy
    combined ``dataset:`` field (e.g. ``edaic_ravdess``).

    A stem matches ``-t Y`` if Y equals the full cross stem (e.g.
    ``T1_T4``) or either of its train/test components (``T1`` or
    ``T4``); for cross-cat, ``c1_c2`` matches ``c1`` or ``c2`` too.
    """
    available = {p.stem for p in TASKS_DIR.glob("*.yaml")}
    seed = cross_categories() if include_categories else cross_pairs()
    stems = sorted(s for s in seed if s in available)
    allowed_ds = set(datasets) if datasets else None
    allowed_tasks = set(tasks) if tasks else None
    out = []
    for s in stems:
        if allowed_ds is not None and not (_stem_datasets(s) & allowed_ds):
            continue
        if allowed_tasks is not None and not (_stem_task_keys(s) & allowed_tasks):
            continue
        out.append(s)
    return out


_STEM_SPLIT_RE = re.compile(r"^([A-Za-z]+\d+)_([A-Za-z]+\d+)$")


def _stem_task_keys(task_stem: str) -> set[str]:
    """Tokens a cross stem matches against ``-t``: the stem itself plus
    its train/test components when the stem follows the paper
    ``<train>_<test>`` pattern (e.g. ``T1_T4`` → ``{T1_T4, T1, T4}``,
    ``c1_c2`` → ``{c1_c2, c1, c2}``)."""
    keys = {task_stem}
    m = _STEM_SPLIT_RE.match(task_stem)
    if m:
        keys.update(m.groups())
    return keys


def _stem_datasets(task_stem: str) -> set[str]:
    """All dataset names a cross stem is "involved with": the legacy
    combined ``dataset:`` plus every entry under
    ``train_dataset(s)`` / ``test_dataset(s)``."""
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    out: set[str] = set()
    m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    if m:
        out.add(m.group(1))
    for key in ("train_dataset", "test_dataset"):
        m = re.search(rf"^{key}:\s*(\S+)", text, re.MULTILINE)
        if m:
            out.add(m.group(1))
    for key in ("train_datasets", "test_datasets"):
        m = re.search(rf"^{key}:\s*\[(.*?)\]", text, re.MULTILINE)
        if m:
            out.update(tok.strip().strip('"').strip("'")
                       for tok in m.group(1).split(",") if tok.strip())
    return out


def get_task_info(task_stem: str) -> tuple[str, str]:
    text = (TASKS_DIR / f"{task_stem}.yaml").read_text()
    dataset_m = re.search(r"^dataset:\s*(\S+)", text, re.MULTILINE)
    task_m = re.search(r"^task:\s*(\S+)", text, re.MULTILINE)
    if not dataset_m or not task_m:
        raise ValueError(f"Could not parse dataset/task from {task_stem}.yaml")
    return dataset_m.group(1), task_m.group(1)


# Paper cross-task stems are ``T<train>_T<test>`` (e.g. ``T9_T7``) and
# paper cross-category stems are ``c<train>_c<test>`` (e.g. ``c2_c3``).
_PAPER_CROSS_RE = re.compile(r"^T\d+_T\d+$")
_PAPER_CROSSCAT_RE = re.compile(r"^c\d+_c\d+$")


def task_label(task_stem: str) -> str:
    """Display label for a cross or cross-category task stem.

    For paper cross pairs renamed to ``T<train>_T<test>``, returns
    ``"T9_T7 (aphasia_dementiabank_pwaC_adC)"`` — both the paper ID (which now
    also names the experiment folder on disk) and the legacy descriptive
    ``<dataset>_<task>`` reconstructed from the YAML body. For paper
    cross-category stems renamed to ``c<train>_c<test>``, returns
    ``"c2_c3 (category_c2_c3)"`` — the legacy ``category_…`` prefix
    spelled out for readability. Falls back to the bare stem for any
    stem that doesn't match the paper patterns or whose YAML can't be
    read, so callers don't need to wrap in try/except.
    """
    if _PAPER_CROSSCAT_RE.match(task_stem):
        return f"{task_stem} (category_{task_stem})"
    if _PAPER_CROSS_RE.match(task_stem):
        try:
            ds, t = get_task_info(task_stem)
            return f"{task_stem} ({ds}_{t})"
        except (FileNotFoundError, ValueError):
            return task_stem
    return task_stem


def get_output_folder(task_stem: str, model_name: str,
                      tag: str = EXPERIMENT_TAG,
                      *, exps_root: Path | None = None) -> Path:
    """``./exps/cross/<task_stem>/<model>-<probe>-<tag>/``.

    For the cross runner ``task_stem`` already equals the legacy
    ``<dataset>_<task>`` folder name (e.g.
    ``aphasia_dementiabank_pwaC_adC``), so behavior is byte-identical to
    before; the signature change is purely cosmetic for consistency
    with single/data-eff.
    """
    return (exps_root or EXPS_ROOT) / task_stem / f"{model_name}-{CROSS_PROBE_NAME}-{tag}"


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
    base = (exps_root or EXPS_ROOT) / task_stem / "manifest"
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


def forget_task_ids(task_stem: str, *, exps_root: Path | None = None) -> None:
    cache_key = (task_stem, str(exps_root) if exps_root else "")
    _task_ids_cache.pop(cache_key, None)
