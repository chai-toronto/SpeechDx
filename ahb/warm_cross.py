"""Standalone HDF5 cache warmer for cross-task runs.

Same shape as ``ahb.warm.run_warm`` but writes three caches per
(task, encoder) pair:

- ``train_cache_dir_1`` — train + val ids, augmented (num_aug_ver versions)
- ``val_cache_dir_1`` — train + val ids, no augmentation (1 version)
- ``val_cache_dir_2`` — test ids, no augmentation (1 version)

The train and val caches share the SAME id pool (``train | val``) but
differ in augmentation: the augmented copies feed the train DataLoader
during fit; the unaugmented val_cache_dir_1 backs the val split's
metric computation.

Decoupled from training: the encoder lives only here and is dropped
before the warmer returns.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import speechbrain as sb
import torch

from ahb.config import compose_config
from ahb.dataio.cache import CachedHDF5DynamicItem
from ahb.dataio.pipeline import (
    build_augmenters,
    make_audio_pipeline,
    make_augment,
    make_get_pid,
    make_label_pipeline,
    make_passthrough_duration,
    make_process_signal,
    make_split_signal,
)
from ahb.dataio.read import load_manifest_data_cross
from ahb.warm import _cache_mode  # reuse — same logic for both warmers


def _cross_cache_dirs(hparams: dict[str, Any]) -> tuple[Path, Path, Path]:
    mode = _cache_mode(hparams)
    return (
        Path(hparams["train_cache_dir_1"]) / mode,
        Path(hparams["val_cache_dir_1"]) / mode,
        Path(hparams["val_cache_dir_2"]) / mode,
    )


def _is_fully_warm_cross(train_dir: Path, val_dir: Path, test_dir: Path,
                         num_versions: int,
                         all_train_ids: list[str], all_test_ids: list[str]) -> bool:
    """Return True iff all three caches contain every (uid, version) needed."""
    for d in (train_dir, val_dir, test_dir):
        if not (d / "cache.hdf5").exists():
            return False
    train = CachedHDF5DynamicItem(
        train_dir, file_mode="r", num_version=num_versions,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    val = CachedHDF5DynamicItem(
        val_dir, file_mode="r", num_version=1,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    test = CachedHDF5DynamicItem(
        test_dir, file_mode="r", num_version=1,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    try:
        return (
            train.is_fully_cached(all_train_ids)
            and val.is_fully_cached(all_train_ids)
            and test.is_fully_cached(all_test_ids)
        )
    finally:
        train.close()
        val.close()
        test.close()


def _build_warm_dynamic_items(hparams: dict[str, Any], *, augmented: bool) -> list:
    """Audio → augment → label → process → split chain (same as single-dataset)."""
    sample_rate = hparams.get("sample_rate", 16000)
    max_samples = int(hparams.get("max_length", 10e5) * sample_rate)
    min_samples = int(hparams.get("min_length", 3) * sample_rate)
    split_by_boundary = hparams["data_params"].get("split_by_boundary", False)

    items = [make_get_pid(), make_audio_pipeline(sample_rate)]
    if augmented:
        noisifier, reverb, perturbator = build_augmenters(
            noise_folder=os.path.abspath(hparams["noise_folder"]),
            rir_folder=hparams["rir_folder"],
            sample_rate=sample_rate,
            snr_low=hparams["data_params"]["snr_low"],
            snr_high=hparams["data_params"]["snr_high"],
            speeds=hparams["data_params"]["speed"],
        )
        items.append(make_augment(noisifier, reverb, perturbator))
    else:
        items.append(make_passthrough_duration())

    items.append(make_label_pipeline())
    if not split_by_boundary:
        items.append(make_process_signal(min_samples))
    items.append(make_split_signal(
        max_samples=max_samples, min_samples=min_samples,
        sample_rate=sample_rate, split_by_boundary=split_by_boundary,
    ))
    return items


def _make_cache_writer(cache_dir: Path, num_versions: int,
                       speech_encoder, output_vars: list[str], cache_pool: str):
    """Per-uid forward — same as ahb/warm.py:_make_cache_writer."""
    @CachedHDF5DynamicItem.cache(cache_dir, file_mode="a", num_version=num_versions)
    @sb.utils.data_pipeline.takes("id", "signals")
    @sb.utils.data_pipeline.provides(*output_vars)
    def cache_emb(id, raw_signals):
        device = next(speech_encoder.parameters()).device
        with torch.no_grad():
            embs = []
            for chunk in raw_signals:
                embs.append(speech_encoder(chunk.unsqueeze(0).to(device)))
            if speech_encoder.output_hidden_states:
                n_layers = len(embs[0])
                emb = tuple(
                    torch.cat([e[i].squeeze(0) for e in embs], dim=-2).cpu()
                    for i in range(n_layers)
                )
            else:
                emb = torch.cat([e.squeeze(0) for e in embs], dim=-2).cpu()
                if cache_pool == "mean":
                    emb = emb.mean(dim=-2, keepdim=False)
        return emb

    return cache_emb


def run_warm_cross(task: str, encoder: str, *,
                   probe: str = "Probe", probe_yaml: str = "Probe.yaml",
                   device: str | None = None) -> None:
    """Warm the three HDF5 caches for one cross (task, encoder) pair.

    Idempotent: returns without instantiating the encoder if every
    (uid, version) is already on disk.
    """
    hparams_stub = compose_config(task, encoder, probe=probe, probe_yaml=probe_yaml,
                                  mode="read")
    train_dir, val_dir, test_dir = _cross_cache_dirs(hparams_stub)
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_versions = int(hparams_stub["data_params"].get("num_aug_ver", 1))
    data_dict = load_manifest_data_cross(hparams_stub)
    all_train_ids = list(data_dict["all_train"].keys())
    all_test_ids = list(data_dict["all_test"].keys())

    if _is_fully_warm_cross(train_dir, val_dir, test_dir,
                            num_versions, all_train_ids, all_test_ids):
        print(f"Cross caches fully warm for {task} × {encoder}, nothing to do.")
        return

    hparams = compose_config(task, encoder, probe=probe, probe_yaml=probe_yaml,
                             mode="warm")
    train_dir, val_dir, test_dir = _cross_cache_dirs(hparams)
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    speech_encoder = hparams["encoder"]
    speech_encoder.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_encoder = speech_encoder.to(torch.device(device))

    num_layers = hparams["num_layers"]
    num_outputs = num_layers if speech_encoder.output_hidden_states else 1
    output_vars = [f"emb_{i}" for i in range(num_outputs)]
    cache_pool = hparams.get("cache_pool", "none")

    train_writer = _make_cache_writer(train_dir, num_versions, speech_encoder,
                                      output_vars, cache_pool)
    val_writer = _make_cache_writer(val_dir, 1, speech_encoder,
                                    output_vars, cache_pool)
    test_writer = _make_cache_writer(test_dir, 1, speech_encoder,
                                     output_vars, cache_pool)

    train_items = _build_warm_dynamic_items(hparams, augmented=True) + [train_writer]
    val_items = _build_warm_dynamic_items(hparams, augmented=False) + [val_writer]
    test_items = _build_warm_dynamic_items(hparams, augmented=False) + [test_writer]

    output_keys_base = ["id", "path", "pid", "label_encoded"]
    train_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_train"], dynamic_items=train_items,
        output_keys=output_keys_base + output_vars,
    )
    val_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_train"], dynamic_items=val_items,
        output_keys=output_keys_base + output_vars,
    )
    test_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_test"], dynamic_items=test_items,
        output_keys=output_keys_base + output_vars,
    )

    warmup = [
        (train_ds, train_writer, all_train_ids, "train", v)
        for v in range(num_versions)
    ]
    warmup.append((val_ds, val_writer, all_train_ids, "val", 0))
    warmup.append((test_ds, test_writer, all_test_ids, "test", 0))

    try:
        for i, (ds, cache, ids, kind, version) in enumerate(warmup):
            uncached = cache.uncached_ids(ids, version)
            if not uncached:
                print(f"Iteration {i} ({kind} v{version}): already warmed, skipping.")
                continue
            print(f"Iteration {i} ({kind} v{version}): warming "
                  f"{len(uncached)}/{len(ids)} uncached uids.")
            subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, uncached)
            subset.iterate_once()
    finally:
        train_writer.close()
        val_writer.close()
        test_writer.close()
        del speech_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
