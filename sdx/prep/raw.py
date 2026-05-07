"""``sdx prep`` raw-data + processed-CSV provisioning.

For each dataset a task touches, ``ensure_raw_and_processed`` does:
    1. ``data/<dataset>/raw/`` empty?
       - has ``scripts/download_<dataset>.sh`` → run it.
       - else → raise ``MissingDatasetError`` with the contact info from
         ``registry.yaml``. The orchestrator catches this and skips every
         task tied to the dataset (with a warning), so the run still
         completes for the data the user does have.
    2. ``data/<dataset>/processed/<dataset>.csv`` missing?
       - run ``metadata_script/create_<dataset>_metadata.py`` if present;
         else raise ``MissingDatasetError``.

``dataset_available(name)`` is a cheap pre-flight probe used by the run
loops to filter unavailable datasets out of ``pending`` *before* warming
or training begins, so the dashboard reports "skipped" rather than
"failed" for missing data.

This module is invoked from ``sdx.prep.dispatch.ensure_manifest`` before
the task-specific manifest builder runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from sdx.registry import datasets as registry_datasets

REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingDatasetError(RuntimeError):
    """Raised when a dataset is referenced but cannot be staged from disk
    or via a download script. The orchestrators catch this and convert
    it into a per-task skip so the rest of the run proceeds."""

    def __init__(self, dataset: str, message: str) -> None:
        super().__init__(message)
        self.dataset = dataset


def _raw_is_staged(raw_dir: Path) -> bool:
    """Heuristic: raw is "staged" if the directory exists and isn't empty."""
    return raw_dir.exists() and any(raw_dir.iterdir())


def _processed_is_staged(dataset: str) -> bool:
    """True if the user has anything under ``data/<dataset>/processed/`` —
    in which case we skip the raw check (the user supplied processed data
    directly, possibly without raw)."""
    processed_dir = REPO_ROOT / "data" / dataset / "processed"
    return processed_dir.exists() and any(processed_dir.iterdir())


def _processed_csv_path(dataset: str) -> Path:
    return REPO_ROOT / "data" / dataset / "processed" / f"{dataset}.csv"


def _download_script(dataset: str) -> Path:
    return REPO_ROOT / "scripts" / f"download_{dataset}.sh"


def _metadata_script(dataset: str) -> Path:
    return REPO_ROOT / "metadata_script" / f"create_{dataset}_metadata.py"


def dataset_available(dataset: str) -> bool:
    """Cheap pre-flight: True if `data/<name>/raw/` is staged, the
    processed dir already has content, the processed CSV exists, OR a
    `scripts/download_<name>.sh` exists (i.e. raw can be auto-staged on
    demand). False means every task that touches this dataset will be
    skipped at run time."""
    if _raw_is_staged(REPO_ROOT / "data" / dataset / "raw"):
        return True
    if _processed_is_staged(dataset):
        return True
    if _processed_csv_path(dataset).exists():
        return True
    if _download_script(dataset).exists():
        return True
    return False


def ensure_raw(dataset: str) -> None:
    """Run the dataset's download script if raw/ is empty; otherwise raise
    ``MissingDatasetError`` with the contact info from registry.yaml.

    Short-circuits when ``data/<dataset>/processed/`` already has content —
    the user supplied processed data directly, so raw isn't needed.
    """
    raw_dir = REPO_ROOT / "data" / dataset / "raw"
    if _raw_is_staged(raw_dir):
        return
    if _processed_is_staged(dataset):
        return
    info = registry_datasets().get(dataset, {})
    script = _download_script(dataset)
    if script.exists():
        print(f"Raw data missing for {dataset}; running {script.relative_to(REPO_ROOT)} …")
        try:
            subprocess.run(["bash", str(script)], check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as e:
            raise MissingDatasetError(
                dataset,
                f"Download script for {dataset!r} failed (exit {e.returncode}). "
                f"Stage manually at {raw_dir.relative_to(REPO_ROOT)}/."
            )
        if not _raw_is_staged(raw_dir):
            raise MissingDatasetError(
                dataset,
                f"Download script ran but {raw_dir} is still empty."
            )
        return
    contact = info.get("contact") or "(see README dataset table)"
    access = info.get("access") or "special access"
    raise MissingDatasetError(
        dataset,
        f"Raw data missing for dataset {dataset!r} (access: {access}). "
        f"No scripts/download_{dataset}.sh, so it can't be auto-staged. "
        f"Stage the upstream archive at {raw_dir.relative_to(REPO_ROOT)}/ "
        f"and re-run. Contact: {contact}."
    )


def ensure_processed(dataset: str) -> None:
    """Run ``metadata_script/create_<dataset>_metadata.py`` if the per-dataset
    processed CSV is missing. Idempotent: no-op when the CSV already exists."""
    csv = _processed_csv_path(dataset)
    if csv.exists():
        return
    script = _metadata_script(dataset)
    if not script.exists():
        raise MissingDatasetError(
            dataset,
            f"Processed CSV {csv.relative_to(REPO_ROOT)} is missing and there's "
            f"no {script.relative_to(REPO_ROOT)} to build it. Add a metadata "
            f"script for {dataset!r} or stage the CSV manually."
        )
    print(f"Processed CSV missing for {dataset}; running "
          f"{script.relative_to(REPO_ROOT)} …")
    try:
        subprocess.run(["python", str(script)], check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        raise MissingDatasetError(
            dataset,
            f"Metadata script for {dataset!r} failed (exit {e.returncode})."
        )
    if not csv.exists():
        raise MissingDatasetError(
            dataset,
            f"Metadata script ran but {csv} still doesn't exist."
        )


def ensure_raw_and_processed(datasets_in_scope: list[str]) -> None:
    """Provision every dataset a task needs: raw → processed CSV. Skips
    datasets that are already fully provisioned (cheap idempotent checks).
    Raises ``MissingDatasetError`` on the first dataset that can't be
    staged — callers should catch and skip the affected tasks."""
    for ds in datasets_in_scope:
        ensure_raw(ds)
        ensure_processed(ds)
