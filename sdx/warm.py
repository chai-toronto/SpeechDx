"""Standalone HDF5 cache warmer.

Decouples cache writing from training: the encoder is instantiated once
per (task, encoder) pair, runs forward over every (uid, version) chunk,
and is dropped before the function returns. Training never imports any
encoder module — it reads the cache that this module wrote.

Pre-flight uses ``compose_config(mode="read")`` (stub encoder, no GPU
allocation, no HuggingFace download) so an already-warm cache is detected
without paying the encoder load cost.
"""

from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from filelock import FileLock
import speechbrain as sb
import torch

from sdx.config import compose_config
from sdx.dataio.cache import CachedHDF5DynamicItem
from sdx.dataio.pipeline import (
    build_augmenters,
    make_audio_pipeline,
    make_augment,
    make_label_pipeline,
    make_passthrough_duration,
    make_process_signal,
    make_split_signal,
)
from sdx.log import _JobDashboard, _now, _run_logged_job, _slug

WARM_LOGS_ROOT = Path("logs/warm")


def _cache_mode(hparams: dict[str, Any]) -> str:
    """Resolve the ``cache_mode`` subdirectory used inside the cache root."""
    speech_encoder = hparams["encoder"]
    num_layers = hparams["num_layers"]
    cache_pool = hparams.get("cache_pool", "none")
    if speech_encoder.output_hidden_states:
        return f"multi_L{num_layers}"
    if cache_pool == "mean":
        return "single_avg"
    return "single"


def _cache_dirs(hparams: dict[str, Any]) -> tuple[Path, Path]:
    mode = _cache_mode(hparams)
    return (
        Path(hparams["train_cache_dir"]) / mode,
        Path(hparams["val_cache_dir"]) / mode,
    )


def _cache_root(hparams: dict[str, Any]) -> Path:
    """Shared ``<tmp>/<dataset>/<encoder>`` parent for the train/val caches."""
    return Path(hparams["train_cache_dir"]).parent


def _lock_path(hparams: dict[str, Any]) -> Path:
    """Per-cache writer lock shared by every warm caller/process."""
    return _cache_root(hparams) / ".warm.lock"


def _wipe_cache_dirs(train_cache_dir: Path, val_cache_dir: Path) -> None:
    """Delete the cache payloads while leaving the directory layout intact."""
    for cache_dir in (train_cache_dir, val_cache_dir):
        cache_file = cache_dir / "cache.hdf5"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed {cache_file}")


def _load_manifests(hparams: dict[str, Any]) -> dict[str, dict]:
    """Load train/val/test JSON manifests and produce the ``all`` union used for warming.

    Two supported shapes:
    - Standard (single-dataset, train/val/test as flat dicts).
    - K-fold CV (mdvr): train_annotation and val_annotation are lists of
      per-fold dicts; the cache only needs the first fold's union since
      every uid lands in some (train, val) pair.
    """
    with open(hparams["train_annotation"]) as f:
        train = json.load(f)
    with open(hparams["val_annotation"]) as f:
        val = json.load(f)

    out: dict[str, dict] = {}
    if isinstance(train, list):
        # CV: lists of per-fold dicts. Cache warming only needs unique uids.
        out["train"] = train[0]
        out["val"] = val[0]
        out["all"] = train[0] | val[0]
        return out

    out["train"] = train
    out["val"] = val
    test_path = hparams.get("test_annotation")
    if test_path and Path(test_path).exists():
        with open(test_path) as f:
            out["test"] = json.load(f)
        out["all"] = out["train"] | out["val"] | out["test"]
    else:
        out["test"] = {}
        out["all"] = out["train"] | out["val"]
    return out


def _is_fully_warm(train_cache_dir: Path, val_cache_dir: Path,
                   num_versions: int, all_ids: list[str]) -> bool:
    """Open both caches read-only and report whether every (uid, version) exists."""
    if not (train_cache_dir / "cache.hdf5").exists():
        return False
    if not (val_cache_dir / "cache.hdf5").exists():
        return False
    train = CachedHDF5DynamicItem(
        train_cache_dir, file_mode="r", num_version=num_versions,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    val = CachedHDF5DynamicItem(
        val_cache_dir, file_mode="r", num_version=1,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    try:
        return train.is_fully_cached(all_ids) and val.is_fully_cached(all_ids)
    finally:
        train.close()
        val.close()


def _build_warm_dynamic_items(hparams: dict[str, Any], *, augmented: bool) -> list:
    """Build the audio → augment → label → process → split pipeline."""
    sample_rate = hparams.get("sample_rate", 16000)
    max_samples = int(hparams.get("max_length", 10e5) * sample_rate)
    min_samples = int(hparams.get("min_length", 3) * sample_rate)
    split_by_boundary = hparams["data_params"].get("split_by_boundary", False)

    items = [
        make_audio_pipeline(sample_rate),
    ]

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
        # split_by_boundary handles its own padding; skip the pre-pad step.
        items.append(make_process_signal(min_samples))
    items.append(make_split_signal(
        max_samples=max_samples, min_samples=min_samples,
        sample_rate=sample_rate, split_by_boundary=split_by_boundary,
    ))
    return items


def _make_cache_writer(cache_dir: Path, num_versions: int,
                       speech_encoder, output_vars: list[str],
                       cache_pool: str):
    """Build the ``cache_emb`` DynamicItem that runs the encoder and writes HDF5.

    Per-uid forward (one chunk at a time) — matches the legacy
    ``preprocessing.py:246-266`` byte-for-byte. Cross-uid batching is a
    perf improvement deferred to a follow-up; the structural decoupling
    is the goal here.
    """
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


def _warm_uncached(task: str, encoder: str, *,
                   probe: str, device: str | None,
                   data_dict: dict[str, dict],
                   all_ids: list[str],
                   num_versions: int) -> None:
    """Load the real encoder and fill whichever cache entries are still missing."""
    hparams = compose_config(task, encoder, probe=probe, mode="warm")
    train_cache_dir, val_cache_dir = _cache_dirs(hparams)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    speech_encoder = hparams["encoder"]
    speech_encoder.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_encoder = speech_encoder.to(torch.device(device))

    num_layers = hparams["num_layers"]
    num_outputs = num_layers if speech_encoder.output_hidden_states else 1
    output_vars = [f"emb_{i}" for i in range(num_outputs)]
    cache_pool = hparams.get("cache_pool", "none")

    train_writer = _make_cache_writer(
        train_cache_dir, num_versions, speech_encoder, output_vars, cache_pool,
    )
    val_writer = _make_cache_writer(
        val_cache_dir, 1, speech_encoder, output_vars, cache_pool,
    )

    train_items = _build_warm_dynamic_items(hparams, augmented=True) + [train_writer]
    val_items = _build_warm_dynamic_items(hparams, augmented=False) + [val_writer]

    output_keys_base = ["id", "path", "Participant_ID", "label_encoded"]
    train_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"],
        dynamic_items=train_items,
        output_keys=output_keys_base + output_vars,
    )
    val_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"],
        dynamic_items=val_items,
        output_keys=output_keys_base + output_vars,
    )

    warmup = [
        (train_ds, train_writer, "train", v) for v in range(num_versions)
    ]
    warmup.append((val_ds, val_writer, "val", 0))

    try:
        for i, (ds, cache, kind, version) in enumerate(warmup):
            uncached = cache.uncached_ids(all_ids, version)
            if not uncached:
                print(f"Iteration {i} ({kind} v{version}): already warmed, skipping.")
                continue
            print(f"Iteration {i} ({kind} v{version}): warming "
                  f"{len(uncached)}/{len(all_ids)} uncached uids.")
            subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, uncached)
            subset.iterate_once()
    finally:
        train_writer.close()
        val_writer.close()
        del speech_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_warm(task: str, encoder: str, *,
             probe: str = "AvgTProbe", device: str | None = None,
             num_aug_ver: int | None = None,
             overwrite: bool = False) -> None:
    """Warm the HDF5 cache for one (task, encoder) pair.

    Idempotent: if every (uid, version) for both train and val is already
    cached, returns without instantiating the encoder. The per-cache writer
    lock lives here so every caller (direct ``warm``, ``run``, cross warm
    delegation, separate processes) shares the same serialization point.

    ``num_aug_ver`` (optional) overrides the task yaml's value — used by
    ``cross warm`` / ``cross-cat warm`` to extend the per-(dataset, encoder)
    cache to whatever aug count the cross task needs. The append-mode
    HDF5 writer extends the cache; existing aug versions are untouched.
    """
    # Phase 1 — pre-flight using the stub encoder so a warm cache check
    # never pays the real encoder's load cost.
    hparams_stub = compose_config(task, encoder, probe=probe, mode="read")
    train_cache_dir, val_cache_dir = _cache_dirs(hparams_stub)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    base_num_versions = int(hparams_stub["data_params"].get("num_aug_ver", 1))
    num_versions = max(base_num_versions, int(num_aug_ver)) if num_aug_ver else base_num_versions
    data_dict = _load_manifests(hparams_stub)
    all_ids = list(data_dict["all"].keys())

    if not overwrite and _is_fully_warm(train_cache_dir, val_cache_dir, num_versions, all_ids):
        print(f"Cache fully warm for {task} × {encoder}, nothing to do.")
        return

    lock_path = _lock_path(hparams_stub)
    print(f"Acquiring warm writer lock: {lock_path}")
    with FileLock(str(lock_path)):
        if overwrite:
            _wipe_cache_dirs(train_cache_dir, val_cache_dir)
        if _is_fully_warm(train_cache_dir, val_cache_dir, num_versions, all_ids):
            print(f"Cache fully warm for {task} × {encoder}, nothing to do.")
            return
        _warm_uncached(
            task, encoder,
            probe=probe, device=device,
            data_dict=data_dict, all_ids=all_ids, num_versions=num_versions,
        )


def _execute_warm_job(task_stem: str, model_name: str, *,
                      device: str | None,
                      num_aug_ver: int | None,
                      overwrite: bool,
                      log_path: Path) -> tuple[str, bool, float]:
    """Run one warm job and capture its stdout/stderr into a per-pair log file."""
    label = f"{task_stem} × {model_name}"
    detail_lines = [f"overwrite={overwrite}"]
    if num_aug_ver is not None:
        detail_lines.append(f"num_aug_ver={num_aug_ver}")
    success, elapsed = _run_logged_job(
        label,
        log_path,
        lambda: run_warm(
            task_stem, model_name,
            device=device, num_aug_ver=num_aug_ver,
            overwrite=overwrite,
        ),
        detail_lines=detail_lines,
    )
    return label, success, elapsed


def cmd_warm_jobs(jobs: list[tuple[str, str, int | None]], *,
                  device: str | None,
                  overwrite: bool = False,
                  max_workers: int = 1,
                  logs_root: Path = WARM_LOGS_ROOT) -> None:
    """Run one or more warm jobs with the same progress/log UX as ``run``."""
    total_jobs = len(jobs)
    max_workers = max(1, max_workers)
    dashboard = _JobDashboard(
        total_jobs=total_jobs,
        logs_root=logs_root,
        header_lines=(
            f"[{_now()}] Warm start : {total_jobs} pending",
            f"           workers : up to {max_workers} concurrent",
        ),
    )

    def _execute(job: tuple[str, str, int | None], idx: int) -> tuple[str, bool, float]:
        task_stem, model_name, num_aug_ver = job
        label = f"{task_stem} × {model_name}"
        log_path = dashboard.run_log_dir / f"{_slug(task_stem)}__{_slug(model_name)}.log"

        dashboard.start_job(idx, "writer", label, log_path)
        result = _execute_warm_job(
            task_stem, model_name,
            device=device, num_aug_ver=num_aug_ver,
            overwrite=overwrite, log_path=log_path,
        )
        _, ok, elapsed = result
        dashboard.finish_job(idx, label, ok=ok, elapsed=elapsed, log_path=log_path)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_execute, job, i): job for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            future.result()

    dashboard.print_summary(done_label="cache warmed")
