"""Standalone HDF5 cache warmer for cross-category tasks.

Self-contained: composes from ``main_cross_category.yaml`` +
``cross_tasks/<stem>.yaml`` only. No single-task yaml or single-task
manifest is read at runtime — every per-train-dataset knob lives in the
cat yaml's ``setting_<N>`` block.

Cross-cat caches are per-(dataset, encoder), keyed by the bare
``cache_uid`` (the part after ``::`` in category manifest ids). The
trainer's category readers ([dataio/read.py:_make_category_cache_reader])
dispatch by id prefix and look up
``<slurm_tmpdir>/<dataset>/<encoder>/<split>/<cache_mode>/cache.hdf5``.

Per-train-dataset settings come from ``setting_<N>`` (idx-aligned with
``train_datasets``) and carry ``num_aug_ver``, ``split_by_boundary``,
``snr_low``, ``snr_high``, ``speed``. Per-test-dataset chunking comes
from ``test_setting_<N>`` (idx-aligned with ``test_datasets``). Test
side is never augmented (always 1 version, no aug).
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
from sdx.warm import _cache_mode

WARM_CROSSCAT_LOGS_ROOT = Path("logs/warm_cross_cat")


def _split_id(uid: str) -> tuple[str, str]:
    """``edaic_t1::A123`` -> ``("edaic", "A123")``. Matches the convention
    in ``sdx/dataio/read.py:_make_category_cache_reader._split_id``.
    """
    prefix, bare = uid.split("::", 1)
    dataset = prefix.split("_", 1)[0]
    return dataset, bare


def _per_dataset_subset(manifest_dict: dict, dataset: str) -> tuple[dict, list[str]]:
    """Filter category manifest rows to one contributing dataset and re-key
    by bare cache_uid. Returns (subset_dict, list_of_bare_ids)."""
    out: dict = {}
    for uid, row in manifest_dict.items():
        try:
            ds, bare = _split_id(uid)
        except ValueError:
            continue
        if ds != dataset:
            continue
        # Cache writer keys on ``id``; rewrite the row's id to the bare
        # cache_uid so HDF5 keys match the category reader's expectation.
        row = dict(row)
        row["id"] = bare
        out[bare] = row
    return out, list(out.keys())


def _cat_cache_dirs(cache_root: Path, dataset: str, encoder: str,
                    cache_mode: str) -> tuple[Path, Path]:
    base = cache_root / dataset / encoder
    return base / "train" / cache_mode, base / "val" / cache_mode


def _is_fully_cached(cache_dir: Path, num_versions: int, ids: list[str]) -> bool:
    if not (cache_dir / "cache.hdf5").exists():
        return False
    reader = CachedHDF5DynamicItem(
        cache_dir, file_mode="r", num_version=num_versions,
        takes=["id"], func=lambda *_: None, provides=["_"],
    )
    try:
        return reader.is_fully_cached(ids)
    finally:
        reader.close()


def _build_items(*, sample_rate: int, max_samples: int, min_samples: int,
                 augmented: bool, split_by_boundary: bool,
                 noise_folder: str | None = None,
                 rir_folder: str | None = None,
                 snr_low=None, snr_high=None, speeds=None) -> list:
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


def _load_cat_manifests(hparams: dict[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for split, key in (("train", "train_annotation"),
                       ("val", "val_annotation"),
                       ("test", "test_annotation")):
        with open(hparams[key]) as f:
            out[split] = json.load(f)
    return out


def _gather_setting(data_params: dict, prefix: str, idx: int) -> dict:
    """Read ``setting_<idx>`` (1-based) or ``test_setting_<idx>`` from data_params."""
    return data_params.get(f"{prefix}_{idx}", {}) or {}


def run_warm_cross_cat(task: str, encoder_name: str, *,
                       probe: str = "AvgTProbe", probe_yaml: str = "Probe.yaml",
                       device: str | None = None,
                       overwrite: bool = False) -> None:
    """Warm per-dataset HDF5 caches for one cross-cat (task, encoder) pair.

    For each train dataset: writes ``<dataset>/<encoder>/train`` (N
    versions, aug) and ``<dataset>/<encoder>/val`` (1 version, no aug).
    For each test dataset: writes ``<dataset>/<encoder>/val`` only.

    Cache file lock is per dataset cache root so concurrent warm callers
    serialize correctly even when warming multiple datasets sequentially.
    """
    hparams_stub = compose_config(task, encoder_name, probe=probe,
                                  probe_yaml=probe_yaml, mode="read")
    data_params = hparams_stub["data_params"]
    train_datasets: list[str] = list(data_params.get("train_datasets") or [])
    test_datasets: list[str] = list(data_params.get("test_datasets") or [])

    cache_root = Path(hparams_stub["train_cache_dir_1"])
    cache_mode = _cache_mode(hparams_stub)

    manifests = _load_cat_manifests(hparams_stub)
    all_train = {**manifests["train"], **manifests["val"]}

    # Build per-dataset jobs (deduping by dataset since the same dataset
    # cannot be both train and test in a single cat yaml).
    train_jobs: list[dict] = []
    test_jobs: list[dict] = []
    for i, ds in enumerate(train_datasets):
        cfg = _gather_setting(data_params, "setting", i + 1)
        sub, ids = _per_dataset_subset(all_train, ds)
        if not ids:
            print(f"  warn: no train+val ids for {ds!r} in {task} manifest, skipping")
            continue
        if "snr_low" not in cfg or "snr_high" not in cfg or "speed" not in cfg:
            raise RuntimeError(
                f"setting_{i+1} for dataset {ds!r} in {task} is missing "
                "snr_low / snr_high / speed. Cross-cat warm requires every "
                "per-train-dataset block to carry its own augmentation params."
            )
        train_jobs.append({
            "dataset": ds,
            "manifest": sub,
            "ids": ids,
            "num_versions": int(cfg.get("num_aug_ver", 1) or 1),
            "split_by_boundary": bool(cfg.get("split_by_boundary", False)),
            "snr_low": cfg["snr_low"],
            "snr_high": cfg["snr_high"],
            "speed": cfg["speed"],
        })
    for j, ds in enumerate(test_datasets):
        cfg = _gather_setting(data_params, "test_setting", j + 1)
        sub, ids = _per_dataset_subset(manifests["test"], ds)
        if not ids:
            print(f"  warn: no test ids for {ds!r} in {task} manifest, skipping")
            continue
        test_jobs.append({
            "dataset": ds,
            "manifest": sub,
            "ids": ids,
            "split_by_boundary": bool(cfg.get("split_by_boundary", False)),
        })

    # Pre-flight: check each per-dataset cache and short-circuit if all warm.
    needs_work = False
    for job in train_jobs:
        train_dir, val_dir = _cat_cache_dirs(cache_root, job["dataset"],
                                             encoder_name, cache_mode)
        if (overwrite or
                not _is_fully_cached(train_dir, job["num_versions"], job["ids"]) or
                not _is_fully_cached(val_dir, 1, job["ids"])):
            needs_work = True
            break
    if not needs_work:
        for job in test_jobs:
            _, val_dir = _cat_cache_dirs(cache_root, job["dataset"],
                                          encoder_name, cache_mode)
            if overwrite or not _is_fully_cached(val_dir, 1, job["ids"]):
                needs_work = True
                break
    if not needs_work:
        print(f"Cross-cat caches fully warm for {task} × {encoder_name}, "
              f"nothing to do.")
        return

    # Load real encoder once for all per-dataset writes in this task.
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

    noise_folder = hparams["noise_folder"]
    rir_folder = hparams["rir_folder"]

    keys_base = ["id", "path", "Participant_ID", "label_encoded"]

    try:
        for job in train_jobs:
            ds = job["dataset"]
            train_dir, val_dir = _cat_cache_dirs(cache_root, ds, encoder_name,
                                                 cache_mode)
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            lock_path = (cache_root / ds / encoder_name / ".warm.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Warming train ({ds}): num_aug_ver={job['num_versions']}, "
                  f"split_by_boundary={job['split_by_boundary']}, "
                  f"snr=[{job['snr_low']}, {job['snr_high']}]")
            with FileLock(str(lock_path)):
                if overwrite:
                    for d in (train_dir, val_dir):
                        f = d / "cache.hdf5"
                        if f.exists():
                            f.unlink()
                            print(f"Removed {f}")
                _warm_dataset_split(
                    cache_dir=train_dir, num_versions=job["num_versions"],
                    augmented=True, split_by_boundary=job["split_by_boundary"],
                    manifest=job["manifest"], ids=job["ids"],
                    speech_encoder=speech_encoder, output_vars=output_vars,
                    cache_pool=cache_pool, sample_rate=sample_rate,
                    max_samples=max_samples, min_samples=min_samples,
                    keys_base=keys_base, noise_folder=noise_folder,
                    rir_folder=rir_folder,
                    snr_low=job["snr_low"], snr_high=job["snr_high"],
                    speeds=job["speed"],
                    label=f"{ds}/train",
                )
                _warm_dataset_split(
                    cache_dir=val_dir, num_versions=1,
                    augmented=False, split_by_boundary=job["split_by_boundary"],
                    manifest=job["manifest"], ids=job["ids"],
                    speech_encoder=speech_encoder, output_vars=output_vars,
                    cache_pool=cache_pool, sample_rate=sample_rate,
                    max_samples=max_samples, min_samples=min_samples,
                    keys_base=keys_base, label=f"{ds}/val",
                )

        for job in test_jobs:
            ds = job["dataset"]
            _, val_dir = _cat_cache_dirs(cache_root, ds, encoder_name, cache_mode)
            val_dir.mkdir(parents=True, exist_ok=True)
            lock_path = (cache_root / ds / encoder_name / ".warm.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Warming test ({ds}): split_by_boundary={job['split_by_boundary']}")
            with FileLock(str(lock_path)):
                if overwrite:
                    f = val_dir / "cache.hdf5"
                    if f.exists():
                        f.unlink()
                        print(f"Removed {f}")
                _warm_dataset_split(
                    cache_dir=val_dir, num_versions=1,
                    augmented=False, split_by_boundary=job["split_by_boundary"],
                    manifest=job["manifest"], ids=job["ids"],
                    speech_encoder=speech_encoder, output_vars=output_vars,
                    cache_pool=cache_pool, sample_rate=sample_rate,
                    max_samples=max_samples, min_samples=min_samples,
                    keys_base=keys_base, label=f"{ds}/val (test)",
                )
    finally:
        del speech_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _warm_dataset_split(*, cache_dir: Path, num_versions: int,
                        augmented: bool, split_by_boundary: bool,
                        manifest: dict, ids: list[str],
                        speech_encoder, output_vars: list[str], cache_pool: str,
                        sample_rate: int, max_samples: int, min_samples: int,
                        keys_base: list[str],
                        noise_folder: str | None = None,
                        rir_folder: str | None = None,
                        snr_low=None, snr_high=None, speeds=None,
                        label: str = "") -> None:
    writer = _make_cache_writer(cache_dir, num_versions, speech_encoder,
                                output_vars, cache_pool)
    items = _build_items(
        sample_rate=sample_rate, max_samples=max_samples, min_samples=min_samples,
        augmented=augmented, split_by_boundary=split_by_boundary,
        noise_folder=noise_folder, rir_folder=rir_folder,
        snr_low=snr_low, snr_high=snr_high, speeds=speeds,
    ) + [writer]
    ds = sb.dataio.dataset.DynamicItemDataset(
        data=manifest, dynamic_items=items, output_keys=keys_base + output_vars,
    )
    try:
        for v in range(num_versions if augmented else 1):
            uncached = writer.uncached_ids(ids, v)
            if not uncached:
                print(f"  {label} v{v}: warm, skipping.")
                continue
            print(f"  {label} v{v}: warming {len(uncached)}/{len(ids)}.")
            subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, uncached)
            subset.iterate_once()
    finally:
        writer.close()


def _execute_warm_crosscat_job(task: str, encoder: str, *,
                               device: str | None, overwrite: bool,
                               log_path: Path) -> tuple[str, bool, float]:
    label = f"{task} × {encoder}"
    detail_lines = [f"overwrite={overwrite}"]
    success, elapsed = _run_logged_job(
        label, log_path,
        lambda: run_warm_cross_cat(task, encoder, device=device,
                                   overwrite=overwrite),
        detail_lines=detail_lines,
    )
    return label, success, elapsed


def cmd_warm_crosscat_jobs(jobs: list[tuple[str, str]], *,
                           device: str | None,
                           overwrite: bool = False,
                           max_workers: int = 1,
                           logs_root: Path = WARM_CROSSCAT_LOGS_ROOT) -> None:
    total_jobs = len(jobs)
    max_workers = max(1, max_workers)
    dashboard = _JobDashboard(
        total_jobs=total_jobs,
        logs_root=logs_root,
        header_lines=(
            f"[{_now()}] Warm-cross-cat start: {total_jobs} pending",
            f"           workers : up to {max_workers} concurrent",
        ),
    )

    def _execute(job: tuple[str, str], idx: int) -> tuple[str, bool, float]:
        task, encoder = job
        label = f"{task} × {encoder}"
        log_path = dashboard.run_log_dir / f"{_slug(task)}__{_slug(encoder)}.log"
        dashboard.start_job(idx, "writer", label, log_path)
        result = _execute_warm_crosscat_job(
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
