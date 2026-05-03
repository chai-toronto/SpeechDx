"""Output-folder and manifest-path conventions used by every subcommand.

Centralizes the ``./exps/single_task/<dataset>_<task>/...`` layout that today
is duplicated across ``run_all.py:100-128`` and the various ``run_all_*.py``
scripts. Pure path construction — no YAML reads, no I/O.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROBE_NAME = "AvgTProbe"
DEFAULT_EXPERIMENT_TAG = "run1"


def get_output_folder(
    dataset: str,
    task: str,
    model_name: str,
    *,
    probe_name: str = DEFAULT_PROBE_NAME,
    tag: str = DEFAULT_EXPERIMENT_TAG,
) -> Path:
    """``./exps/single_task/<dataset>_<task>/<model>-<probe>-<tag>/``."""
    return Path(f"./exps/single_task/{dataset}_{task}/{model_name}-{probe_name}-{tag}")


def manifest_dir(dataset: str, task: str) -> Path:
    """``./exps/single_task/<dataset>_<task>/manifest/``."""
    return Path(f"./exps/single_task/{dataset}_{task}/manifest")


def manifest_paths_from_dataset_task(dataset: str, task: str) -> tuple[Path, Path, Path]:
    """``(train.json, valid.json, test.json)`` under the manifest dir."""
    base = manifest_dir(dataset, task)
    return base / "train.json", base / "valid.json", base / "test.json"
