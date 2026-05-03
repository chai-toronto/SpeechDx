"""Reader-only ``DynamicItemDataset`` builders.

Mirrors the structure of ``master_dataio_prep`` (single-dataset) and
``master_dataio_prep_cross_category`` (the existing reader-only template
in ``preprocessing.py:700-875``) but never instantiates an encoder. The
trainer process consumes only these builders, which is what enforces the
encoder-isolation invariant: ``ahb/warm.py`` is the only place
``model.<encoder>`` is imported.

Pre-flight via :class:`CachedHDF5DynamicItem.uncached_ids` raises a clear
error if any (uid, version) is missing instead of failing mid-training
with an inscrutable HDF5 KeyError.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import speechbrain as sb
import torch

from ahb.dataio.cache import CachedHDF5DynamicItem


def _cache_mode_for(hparams: dict[str, Any]) -> str:
    """Resolve the ``multi_L<N>`` / ``single_avg`` / ``single`` subdir name."""
    speech_encoder = hparams["encoder"]
    num_layers = hparams["num_layers"]
    cache_pool = hparams.get("cache_pool", "none")
    if speech_encoder.output_hidden_states:
        return f"multi_L{num_layers}"
    if cache_pool == "mean":
        return "single_avg"
    return "single"


def _output_vars(hparams: dict[str, Any]) -> list[str]:
    speech_encoder = hparams["encoder"]
    num_layers = hparams["num_layers"]
    n = num_layers if speech_encoder.output_hidden_states else 1
    return [f"emb_{i}" for i in range(n)]


def _label_pipeline_dynitem():
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        if isinstance(label, list):
            label_encoded = torch.tensor(label, dtype=torch.float)
        else:
            label_encoded = label
        yield label_encoded

    return label_pipeline


def _pid_pipeline_dynitem():
    @sb.utils.data_pipeline.takes("Participant_ID")
    @sb.utils.data_pipeline.provides("pid")
    def get_pid(pid):
        return pid

    return get_pid


def _make_cache_reader(cache_dir: Path, num_versions: int, output_vars: list[str]):
    """Build a read-only DynamicItem that loads embeddings from HDF5."""
    @CachedHDF5DynamicItem.cache(cache_dir, file_mode="r", num_version=num_versions)
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides(*output_vars)
    def read_cache(id):
        # The decorator caches via _is_cached → _load. This body only runs
        # when _is_cached returns False (i.e. v{num_versions-1} is missing),
        # which means the caller's num_aug_ver exceeds what's on disk.
        raise RuntimeError(
            f"Cache miss for id={id!r} at {cache_dir} with num_ver={num_versions}. "
            f"Re-warm with `python -m ahb warm <task> <encoder>`."
        )

    return read_cache


def _preflight(reader: CachedHDF5DynamicItem, ids: list[str], num_versions: int,
               kind: str, cache_dir: Path) -> None:
    """Raise if any (uid, version) for ``kind`` is missing — actionable error."""
    missing_total = 0
    for v in range(num_versions):
        missing = reader.uncached_ids(ids, v)
        missing_total += len(missing)
        if missing:
            raise RuntimeError(
                f"{kind} cache miss at {cache_dir} v{v}: {len(missing)}/{len(ids)} "
                f"uids missing (first: {missing[:3]!r}). "
                f"Re-warm with `python -m ahb warm`."
            )


def build_read_datasets_standard(data_dict: dict[str, dict],
                                 hparams: dict[str, Any]) -> dict[str, sb.dataio.dataset.DynamicItemDataset]:
    """Reader-only single-dataset datasets (train/val/test).

    Pre-flights both caches and raises if any (uid, version) is missing.
    """
    cache_mode = _cache_mode_for(hparams)
    train_cache_dir = Path(hparams["train_cache_dir"]) / cache_mode
    val_cache_dir = Path(hparams["val_cache_dir"]) / cache_mode
    num_versions = int(hparams["data_params"].get("num_aug_ver", 1))
    output_vars = _output_vars(hparams)

    train_reader = _make_cache_reader(train_cache_dir, num_versions, output_vars)
    val_reader = _make_cache_reader(val_cache_dir, 1, output_vars)

    label = _label_pipeline_dynitem()
    pid = _pid_pipeline_dynitem()

    output_keys = ["id", "path", "pid", "label_encoded"] + output_vars

    try:
        # Pre-flight: missing entries fail loudly here, not mid-fit.
        _preflight(train_reader, list(data_dict["train"].keys()),
                   num_versions, "train", train_cache_dir)
        for split in ("val", "test"):
            ids = list(data_dict.get(split, {}).keys())
            if ids:
                _preflight(val_reader, ids, 1, split, val_cache_dir)
    except Exception:
        train_reader.close()
        val_reader.close()
        raise

    datasets: dict[str, sb.dataio.dataset.DynamicItemDataset] = {}
    for split in ("train", "val", "test"):
        if split not in data_dict:
            continue
        reader = train_reader if split == "train" else val_reader
        datasets[split] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[split],
            dynamic_items=[pid, label, reader],
            output_keys=output_keys,
        )
    return datasets


def load_manifest_data_standard(hparams: dict[str, Any]) -> dict[str, dict]:
    """Load train/val/test JSON manifests for the standard (non-CV) flow."""
    out: dict[str, dict] = {}
    with open(hparams["train_annotation"]) as f:
        out["train"] = json.load(f)
    with open(hparams["val_annotation"]) as f:
        out["val"] = json.load(f)
    test_path = Path(hparams["test_annotation"])
    if test_path.exists():
        with open(test_path) as f:
            out["test"] = json.load(f)
    return out


def load_manifest_data_cv(hparams: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Load train and val fold lists for CV tasks (mvdr_*)."""
    with open(hparams["train_annotation"]) as f:
        train_folds = json.load(f)
    with open(hparams["val_annotation"]) as f:
        val_folds = json.load(f)
    if not isinstance(train_folds, list):
        raise ValueError(
            f"{hparams['train_annotation']} is not a list-of-folds manifest"
        )
    return train_folds, val_folds


def build_read_datasets_cv(train_fold: dict, val_fold: dict,
                           hparams: dict[str, Any]) -> dict[str, sb.dataio.dataset.DynamicItemDataset]:
    """Reader-only datasets for one CV fold (train + val only)."""
    cache_mode = _cache_mode_for(hparams)
    train_cache_dir = Path(hparams["train_cache_dir"]) / cache_mode
    val_cache_dir = Path(hparams["val_cache_dir"]) / cache_mode
    num_versions = int(hparams["data_params"].get("num_aug_ver", 1))
    output_vars = _output_vars(hparams)

    train_reader = _make_cache_reader(train_cache_dir, num_versions, output_vars)
    val_reader = _make_cache_reader(val_cache_dir, 1, output_vars)

    label = _label_pipeline_dynitem()
    pid = _pid_pipeline_dynitem()
    output_keys = ["id", "path", "pid", "label_encoded"] + output_vars

    try:
        _preflight(train_reader, list(train_fold.keys()),
                   num_versions, "train_fold", train_cache_dir)
        _preflight(val_reader, list(val_fold.keys()),
                   1, "val_fold", val_cache_dir)
    except Exception:
        train_reader.close()
        val_reader.close()
        raise

    return {
        "train": sb.dataio.dataset.DynamicItemDataset(
            data=train_fold,
            dynamic_items=[pid, label, train_reader],
            output_keys=output_keys,
        ),
        "val": sb.dataio.dataset.DynamicItemDataset(
            data=val_fold,
            dynamic_items=[pid, label, val_reader],
            output_keys=output_keys,
        ),
    }


def build_read_datasets_cross(data_dict: dict[str, dict],
                              hparams: dict[str, Any]) -> dict[str, sb.dataio.dataset.DynamicItemDataset]:
    """Reader-only cross-task datasets.

    Cross runs use three cache directories (vs two for single-dataset):
    ``train_cache_dir_1`` (train, augmented N versions) and
    ``val_cache_dir_1`` (val, no aug, 1 version) both under the train
    dataset's cache root, plus ``val_cache_dir_2`` (test, no aug, 1
    version) under the test dataset's root.
    """
    cache_mode = _cache_mode_for(hparams)
    train_cache_dir = Path(hparams["train_cache_dir_1"]) / cache_mode
    val_cache_dir = Path(hparams["val_cache_dir_1"]) / cache_mode
    test_cache_dir = Path(hparams["val_cache_dir_2"]) / cache_mode
    num_versions = int(hparams["data_params"].get("num_aug_ver", 1))
    output_vars = _output_vars(hparams)

    train_reader = _make_cache_reader(train_cache_dir, num_versions, output_vars)
    val_reader = _make_cache_reader(val_cache_dir, 1, output_vars)
    test_reader = _make_cache_reader(test_cache_dir, 1, output_vars)

    label = _label_pipeline_dynitem()
    pid = _pid_pipeline_dynitem()
    output_keys = ["id", "path", "pid", "label_encoded"] + output_vars

    try:
        _preflight(train_reader, list(data_dict["train"].keys()),
                   num_versions, "train", train_cache_dir)
        _preflight(val_reader, list(data_dict["val"].keys()),
                   1, "val", val_cache_dir)
        _preflight(test_reader, list(data_dict["test"].keys()),
                   1, "test", test_cache_dir)
    except Exception:
        train_reader.close()
        val_reader.close()
        test_reader.close()
        raise

    return {
        "train": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["train"],
            dynamic_items=[pid, label, train_reader],
            output_keys=output_keys,
        ),
        "val": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["val"],
            dynamic_items=[pid, label, val_reader],
            output_keys=output_keys,
        ),
        "test": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["test"],
            dynamic_items=[pid, label, test_reader],
            output_keys=output_keys,
        ),
    }


def load_manifest_data_cross(hparams: dict[str, Any]) -> dict[str, dict]:
    """Load train/val/test JSON manifests for cross tasks (same format as standard)."""
    out: dict[str, dict] = {}
    with open(hparams["train_annotation"]) as f:
        out["train"] = json.load(f)
    with open(hparams["val_annotation"]) as f:
        out["val"] = json.load(f)
    with open(hparams["test_annotation"]) as f:
        out["test"] = json.load(f)
    # all_train + all_test mirror the legacy data_dict expected by warm_cross.
    out["all_train"] = out["train"] | out["val"]
    out["all_test"] = out["test"]
    return out


def assert_no_encoder_imports() -> None:
    """Encoder-isolation invariant: trainer must not import any model.<encoder>.

    Allowed imports under model.* are probe, pool, and stub_encoder — those
    don't pull in heavy weights or downstream HF dependencies. Raises with
    the offending list so a regression that re-imports an encoder (e.g. via
    a stale ``from model.x import ...`` in the read path) fails loudly.
    """
    import sys

    allowed = {"model", "model.probe", "model.pool", "model.stub_encoder"}
    forbidden = [
        m for m in sys.modules
        if m.startswith("model.") and m not in allowed
    ]
    if forbidden:
        raise AssertionError(
            f"Trainer must not import encoders; saw: {sorted(forbidden)}"
        )
