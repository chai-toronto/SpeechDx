"""Output-folder and manifest-path conventions used by every subcommand.

Centralizes the ``./exps/single_task/<task_id>/...`` layout that today is
duplicated across ``run_all.py:100-128`` and the various ``run_all_*.py``
scripts. Pure path construction — no YAML reads, no I/O.

``task_id`` is the task YAML's stem (e.g. ``"T9"`` for paper tasks,
``"avfad_ageR"`` for auxiliary single tasks, ``"aphasia_dementiabank_pwaC_adC"``
for cross tasks). For auxiliary and cross tasks ``task_id`` already
equals the legacy ``<dataset>_<task>`` folder name, so existing folders
are preserved; paper tasks renamed to ``T1``…``T27`` get the ``T<N>``
folder name instead of the descriptive one.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROBE_NAME = "wavrx"
DEFAULT_EXPERIMENT_TAG = "run1"


def get_output_folder(
    task_id: str,
    model_name: str,
    *,
    probe_name: str = DEFAULT_PROBE_NAME,
    tag: str = DEFAULT_EXPERIMENT_TAG,
) -> Path:
    """``./exps/single_task/<task_id>/<model>-<probe>-<tag>/``."""
    return Path(f"./exps/single_task/{task_id}/{model_name}-{probe_name}-{tag}")


def manifest_dir(task_id: str) -> Path:
    """``./exps/single_task/<task_id>/manifest/``."""
    return Path(f"./exps/single_task/{task_id}/manifest")


def manifest_paths_from_task_id(task_id: str) -> tuple[Path, Path, Path]:
    """``(train.json, valid.json, test.json)`` under the manifest dir."""
    base = manifest_dir(task_id)
    return base / "train.json", base / "valid.json", base / "test.json"
