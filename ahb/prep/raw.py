"""``ahb prep`` raw-data + processed-CSV provisioning.

For each dataset a task touches, ``ensure_raw_and_processed`` does:
    1. ``data/<dataset>/raw/`` empty?
       - has ``scripts/download_<dataset>.sh`` → run it.
       - else → fail loudly with the contact info from ``registry.yaml``.
    2. ``data/<dataset>/processed/<dataset>.csv`` missing?
       - run ``metadata_script/create_<dataset>_metadata.py`` if present.

This module is invoked from ``ahb.prep.dispatch.ensure_manifest`` before
the task-specific manifest builder runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ahb.registry import datasets as registry_datasets

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def ensure_raw(dataset: str) -> None:
    """Run the dataset's download script if raw/ is empty and the dataset is
    public; otherwise raise SystemExit with the contact info.

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
        subprocess.run(["bash", str(script)], check=True, cwd=REPO_ROOT)
        if not _raw_is_staged(raw_dir):
            raise SystemExit(f"Download script ran but {raw_dir} is still empty.")
        return
    contact = info.get("contact") or "(see README dataset table)"
    access = info.get("access") or "special access"
    raise SystemExit(
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
        raise SystemExit(
            f"Processed CSV {csv.relative_to(REPO_ROOT)} is missing and there's "
            f"no {script.relative_to(REPO_ROOT)} to build it. Add a metadata "
            f"script for {dataset!r} or stage the CSV manually."
        )
    print(f"Processed CSV missing for {dataset}; running "
          f"{script.relative_to(REPO_ROOT)} …")
    subprocess.run(["python", str(script)], check=True, cwd=REPO_ROOT)
    if not csv.exists():
        raise SystemExit(f"Metadata script ran but {csv} still doesn't exist.")


def ensure_raw_and_processed(datasets_in_scope: list[str]) -> None:
    """Provision every dataset a task needs: raw → processed CSV. Skips
    datasets that are already fully provisioned (cheap idempotent checks)."""
    for ds in datasets_in_scope:
        ensure_raw(ds)
        ensure_processed(ds)
