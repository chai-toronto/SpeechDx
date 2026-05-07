"""Standalone HDF5 cache warmer for cross tasks.

Self-contained: composes from ``main_cross.yaml`` + ``cross_tasks/<stem>.yaml``
only — no single-task yaml or single-task manifest is read. Writes three
caches per (cross_task, encoder) pair:

- ``<train_dataset>/<encoder>/train`` — train + val ids of train_dataset,
  ``num_aug_ver`` augmented versions, chunked using ``train_split_by_boundary``.
- ``<train_dataset>/<encoder>/val`` — train + val ids of train_dataset,
  1 unaugmented version, chunked using ``train_split_by_boundary``.
- ``<test_dataset>/<encoder>/val`` — test ids of test_dataset, 1
  unaugmented version, chunked using ``test_split_by_boundary``.

The trainer's ``test_reader`` ([dataio/read.py:219]) reads from the
test dataset's val cache, never from a train cache, so there is no
``<test_dataset>/train`` warmup.
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from filelock import FileLock
import speechbrain as sb
import torch

from sdx.config import compose_config
from sdx.dataio.cache import CachedHDF5DynamicItem
from sdx.dataio.pipeline import (
    ID,
    LABEL_ENCODED,
    SIGNALS,
    build_augmenters,
    make_audio_pipeline,
    make_augment,
    make_label_pipeline,
    make_passthrough_duration,
    make_process_signal,
    make_split_signal,
)
from sdx.dataio.read import load_manifest_data_cross
from sdx.log import _JobDashboard, _now, _run_logged_job, _slug
from sdx.warm import _cache_mode

WARM_CROSS_LOGS_ROOT = Path("logs/warm_cross")


def _cross_cache_dirs(hparams: dict[str, Any]) -> tuple[Path, Path, Path]:
    mode = _cache_mode(hparams)
    return (
        Path(hparams["train_cache_dir_1"]) / mode,
        Path(hparams["val_cache_dir_1"]) / mode,
        Path(hparams["val_cache_dir_2"]) / mode,
    )


def _is_fully_cached(cache_dir: Path, num_versions: int, ids: list[str]) -> bool:
    if not (cache_dir / "cache.hdf5").exists():
        return False
    reader = CachedHDF5DynamicItem(
        cache_dir, file_mode="r", num_version=num_versions,
        takes=[ID], func=lambda *_: None, provides=["_"],
    )
    try:
        return reader.is_fully_cached(ids)
    finally:
        reader.close()


def _build_items(*, sample_rate: int, max_samples: int, min_samples: int,
                 augmented: bool, split_by_boundary: bool,
                 noise_folder: str | None = None,
                 rir_folder: str | None = None,
                 snr_low: float | None = None, snr_high: float | None = None,
                 speeds=None) -> list:
    items = [make_audio_pipeline(sample_rate)]
    if augmented:
        noisifier, reverb, perturbator = build_augmenters(
            noise_folder=os.path.abspath(noise_folder),
            rir_folder=rir_folder,
            sample_rate=sample_rate,
            snr_low=snr_low, snr_high=snr_high, speeds=speeds,
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
    @CachedHDF5DynamicItem.cache(cache_dir, file_mode="a", num_version=num_versions)
    @sb.utils.data_pipeline.takes(ID, SIGNALS)
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


def _all_warm(train_dir: Path, val_dir: Path, test_dir: Path,
              num_versions: int,
              train_ids: list[str], test_ids: list[str]) -> bool:
    return (
        _is_fully_cached(train_dir, num_versions, train_ids)
        and _is_fully_cached(val_dir, 1, train_ids)
        and _is_fully_cached(test_dir, 1, test_ids)
    )


def run_warm_cross(task: str, encoder_name: str, *,
                   probe: str = "AvgTProbe", probe_yaml: str = "Probe.yaml",
                   device: str | None = None, overwrite: bool = False) -> None:
    """Warm the three HDF5 caches for one cross (task, encoder) pair.

    Idempotent: returns without instantiating the encoder when every
    (uid, version) is already cached. Acquires per-cache-root file locks
    (one for train_dataset, one for test_dataset) so concurrent warm
    callers serialize cache writes.
    """
    # Phase 1 — pre-flight via stub encoder (no GPU alloc, no HF download).
    hparams_stub = compose_config(task, encoder_name, probe=probe,
                                  probe_yaml=probe_yaml, mode="read")
    train_dir, val_dir, test_dir = _cross_cache_dirs(hparams_stub)
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    data_params = hparams_stub["data_params"]
    num_versions = int(data_params.get("num_aug_ver", 1) or 1)
    train_sbb = bool(data_params.get("train_split_by_boundary", False))
    test_sbb = bool(data_params.get("test_split_by_boundary", False))

    data_dict = load_manifest_data_cross(hparams_stub)
    train_ids = list(data_dict["all_train"].keys())
    test_ids = list(data_dict["all_test"].keys())

    if not overwrite and _all_warm(train_dir, val_dir, test_dir,
                                   num_versions, train_ids, test_ids):
        print(f"Cross caches fully warm for {task} × {encoder_name}, nothing to do.")
        return

    # Phase 2 — file locks per dataset cache root, acquired in stable order.
    train_root = train_dir.parent.parent  # <tmp>/<train_dataset>/<encoder>
    test_root = test_dir.parent.parent
    lock_paths = sorted({train_root / ".warm.lock", test_root / ".warm.lock"})
    print(f"Acquiring warm-cross locks: {[str(p) for p in lock_paths]}")
    with ExitStack() as stack:
        for lp in lock_paths:
            lp.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(FileLock(str(lp)))

        if overwrite:
            for d in (train_dir, val_dir, test_dir):
                f = d / "cache.hdf5"
                if f.exists():
                    f.unlink()
                    print(f"Removed {f}")
        if _all_warm(train_dir, val_dir, test_dir, num_versions,
                     train_ids, test_ids):
            print(f"Cross caches fully warm under lock for {task} × {encoder_name}.")
            return
        _warm_uncached(
            task, encoder_name, probe=probe, probe_yaml=probe_yaml,
            device=device,
            data_dict=data_dict, train_ids=train_ids, test_ids=test_ids,
            num_versions=num_versions,
            train_sbb=train_sbb, test_sbb=test_sbb,
            cache_dirs=(train_dir, val_dir, test_dir),
        )


def _warm_uncached(task: str, encoder_name: str, *,
                   probe: str, probe_yaml: str, device: str | None,
                   data_dict: dict[str, dict],
                   train_ids: list[str], test_ids: list[str],
                   num_versions: int,
                   train_sbb: bool, test_sbb: bool,
                   cache_dirs: tuple[Path, Path, Path]) -> None:
    train_dir, val_dir, test_dir = cache_dirs
    hparams = compose_config(task, encoder_name, probe=probe,
                             probe_yaml=probe_yaml, mode="warm")

    speech_encoder = hparams["encoder"]
    speech_encoder.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_encoder = speech_encoder.to(torch.device(device))

    sample_rate = hparams.get("sample_rate", 16000)
    max_samples = int(hparams.get("max_length", 10e5) * sample_rate)
    min_samples = int(hparams.get("min_length", 3) * sample_rate)
    num_layers = hparams["num_layers"]
    num_outputs = num_layers if speech_encoder.output_hidden_states else 1
    output_vars = [f"emb_{i}" for i in range(num_outputs)]
    cache_pool = hparams.get("cache_pool", "none")

    data_params = hparams["data_params"]
    snr_low = data_params.get("snr_low")
    snr_high = data_params.get("snr_high")
    speeds = data_params.get("speed")
    noise_folder = hparams["noise_folder"]
    rir_folder = hparams["rir_folder"]

    train_writer = _make_cache_writer(train_dir, num_versions, speech_encoder,
                                      output_vars, cache_pool)
    val_writer = _make_cache_writer(val_dir, 1, speech_encoder,
                                    output_vars, cache_pool)
    test_writer = _make_cache_writer(test_dir, 1, speech_encoder,
                                     output_vars, cache_pool)

    train_items = _build_items(
        sample_rate=sample_rate, max_samples=max_samples, min_samples=min_samples,
        augmented=True, split_by_boundary=train_sbb,
        noise_folder=noise_folder, rir_folder=rir_folder,
        snr_low=snr_low, snr_high=snr_high, speeds=speeds,
    ) + [train_writer]
    val_items = _build_items(
        sample_rate=sample_rate, max_samples=max_samples, min_samples=min_samples,
        augmented=False, split_by_boundary=train_sbb,
    ) + [val_writer]
    test_items = _build_items(
        sample_rate=sample_rate, max_samples=max_samples, min_samples=min_samples,
        augmented=False, split_by_boundary=test_sbb,
    ) + [test_writer]

    keys_base = [ID, "path", "Participant_ID", LABEL_ENCODED]
    train_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_train"], dynamic_items=train_items,
        output_keys=keys_base + output_vars,
    )
    val_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_train"], dynamic_items=val_items,
        output_keys=keys_base + output_vars,
    )
    test_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all_test"], dynamic_items=test_items,
        output_keys=keys_base + output_vars,
    )

    warmup = [
        (train_ds, train_writer, train_ids, "train", v)
        for v in range(num_versions)
    ]
    warmup.append((val_ds, val_writer, train_ids, "val", 0))
    warmup.append((test_ds, test_writer, test_ids, "test", 0))

    try:
        for i, (ds, cache, ids, kind, v) in enumerate(warmup):
            uncached = cache.uncached_ids(ids, v)
            if not uncached:
                print(f"Iteration {i} ({kind} v{v}): already warmed, skipping.")
                continue
            print(f"Iteration {i} ({kind} v{v}): warming "
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


def _execute_warm_cross_job(task: str, encoder: str, *,
                            device: str | None, overwrite: bool,
                            log_path: Path) -> tuple[str, bool, float]:
    label = f"{task} × {encoder}"
    detail_lines = [f"overwrite={overwrite}"]
    success, elapsed = _run_logged_job(
        label, log_path,
        lambda: run_warm_cross(task, encoder, device=device, overwrite=overwrite),
        detail_lines=detail_lines,
    )
    return label, success, elapsed


def cmd_warm_cross_jobs(jobs: list[tuple[str, str]], *,
                        device: str | None,
                        overwrite: bool = False,
                        max_workers: int = 1,
                        logs_root: Path = WARM_CROSS_LOGS_ROOT) -> None:
    """Run warm_cross for each (cross_task, encoder) pair under the dashboard."""
    total_jobs = len(jobs)
    max_workers = max(1, max_workers)
    dashboard = _JobDashboard(
        total_jobs=total_jobs,
        logs_root=logs_root,
        header_lines=(
            f"[{_now()}] Warm-cross start: {total_jobs} pending",
            f"           workers : up to {max_workers} concurrent",
        ),
    )

    def _execute(job: tuple[str, str], idx: int) -> tuple[str, bool, float]:
        task, encoder = job
        label = f"{task} × {encoder}"
        log_path = dashboard.run_log_dir / f"{_slug(task)}__{_slug(encoder)}.log"
        dashboard.start_job(idx, "writer", label, log_path)
        result = _execute_warm_cross_job(
            task, encoder, device=device, overwrite=overwrite, log_path=log_path,
        )
        _, ok, elapsed = result
        dashboard.finish_job(idx, label, ok=ok, elapsed=elapsed, log_path=log_path)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_execute, job, i): job for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            future.result()

    dashboard.print_summary(done_label="cache warmed")
