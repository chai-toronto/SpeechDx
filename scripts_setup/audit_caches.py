"""Pre-flight check for `run_all_cross.py run --no-writer`.

For every (cross_task × encoder) pair the runner would touch, verify that:
  - the dataset's metadata CSV exists under `data_folder`
  - the (train_dataset, model_name) and (test_dataset, model_name) caches
    exist under `slurm_tmpdir/<dataset>/<model_name>/{train,val}/single_avg/cache.hdf5`
    and are non-empty + openable as HDF5

Run from the project root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import run_all_cross as rc  # noqa: E402


def _grep_yaml_value(text: str, key: str, default: str) -> str:
    import re
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else default


def main() -> int:
    text = (PROJECT / "training/config/main_cross.yaml").read_text()
    data_folder = Path(_grep_yaml_value(text, "data_folder", "./data/")).resolve()
    slurm_tmpdir = Path(_grep_yaml_value(text, "slurm_tmpdir", "./embeddings_avg_final/")).resolve()
    print(f"data_folder  : {data_folder}")
    print(f"slurm_tmpdir : {slurm_tmpdir}\n")

    encoders = rc.ENCODERS
    print(f"Encoders     ({len(encoders)}): {sorted(encoders)}")

    tasks = sorted(p.stem for p in rc.TASKS_DIR.glob("*.yaml"))
    print(f"Cross-tasks  ({len(tasks)}): {tasks}\n")

    used_datasets: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for ts in tasks:
        cfg = rc._load_task_yaml(ts)
        train_ds = cfg["train_dataset"]
        test_ds = cfg["test_dataset"]
        used_datasets.update({train_ds, test_ds})
        for enc in encoders:
            pairs.add((train_ds, enc))
            pairs.add((test_ds, enc))

    print(f"Unique datasets ({len(used_datasets)}): {sorted(used_datasets)}\n")

    # 1) CSV check
    print("=== checking metadata CSVs ===")
    csv_missing = []
    for ds in sorted(used_datasets):
        ds_dir = data_folder / ds
        candidates = [
            ds_dir / "processed" / f"{ds}.csv",
            ds_dir / "processed" / "metadata.csv",
            ds_dir / f"{ds}.csv",
            ds_dir / "metadata.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            csv_missing.append(ds)
            print(f"  ✗ {ds:>10s}  no CSV (tried: {[str(c) for c in candidates]})")
        else:
            print(f"  ✓ {ds:>10s}  {found}")
    print()

    # 2) Cache HDF5 check
    print("=== checking embedding caches (cache_pool=mean → single_avg/) ===")
    cache_missing: list[tuple[str, str, str, str]] = []
    cache_empty: list[tuple[str, str, str, str]] = []
    cache_ok = 0
    for ds in sorted(used_datasets):
        for enc in sorted(encoders):
            for split in ("train", "val"):
                p = slurm_tmpdir / ds / enc / split / "single_avg" / "cache.hdf5"
                if not p.exists():
                    cache_missing.append((ds, enc, split, str(p)))
                elif p.stat().st_size == 0:
                    cache_empty.append((ds, enc, split, str(p)))
                else:
                    cache_ok += 1

    total_expected = len(used_datasets) * len(encoders) * 2
    print(f"  {cache_ok}/{total_expected} caches present and non-empty")
    if cache_missing:
        print(f"\n  MISSING ({len(cache_missing)}):")
        for ds, enc, split, p in cache_missing[:80]:
            print(f"    - {ds}/{enc}/{split}  ({p})")
        if len(cache_missing) > 80:
            print(f"    ... and {len(cache_missing) - 80} more")
    if cache_empty:
        print(f"\n  EMPTY ({len(cache_empty)}):")
        for ds, enc, split, p in cache_empty[:80]:
            print(f"    - {ds}/{enc}/{split}  ({p})")

    # 3) Spot-check ONE cache file with h5py
    print("\n=== spot-check ===")
    sample = None
    for ds in sorted(used_datasets):
        for enc in sorted(encoders):
            p = slurm_tmpdir / ds / enc / "train" / "single_avg" / "cache.hdf5"
            if p.exists() and p.stat().st_size > 0:
                sample = (ds, enc, p)
                break
        if sample:
            break
    if sample:
        try:
            import h5py
            ds, enc, p = sample
            with h5py.File(p, "r") as f:
                keys = list(f.keys())
                k0 = keys[0]
                obj = f[k0]
                if isinstance(obj, h5py.Group):
                    inner = list(obj.keys())[0]
                    note = f"group→{inner} {obj[inner].shape} {obj[inner].dtype}"
                else:
                    note = f"dataset {obj.shape} {obj.dtype}"
            print(f"  {ds}/{enc}/train: {len(keys)} ids, e.g. {k0!r} {note}")
        except Exception as e:
            print(f"  WARN: could not h5py-open {sample[2]}: {e}")
    else:
        print("  (no cache to spot-check)")

    issues = csv_missing or cache_missing or cache_empty
    print("\n" + ("=" * 60))
    if not issues:
        print("  ✅ All clear — safe to submit.")
        return 0
    print("  ⚠ Issues detected (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
