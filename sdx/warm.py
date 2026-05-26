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
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

from filelock import FileLock
import h5py
import speechbrain as sb
import torch

from sdx.config import compose_config
from sdx.dataio.cache import CachedHDF5DynamicItem, _to_ndarray
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
        takes=[ID], func=lambda *_: None, provides=["_"],
    )
    val = CachedHDF5DynamicItem(
        val_cache_dir, file_mode="r", num_version=1,
        takes=[ID], func=lambda *_: None, provides=["_"],
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


def _compute_emb_from_signals(raw_signals, speech_encoder, cache_pool: str):
    """Run the encoder over a per-uid chunk list and produce the cache payload.

    Encodes one chunk at a time and concatenates — the ``batch_size == 1``
    equivalent of the batched driver. Used by ``_drive_warm_api`` and by the
    cross-warm writers (via ``_make_cache_writer``).
    """
    _p = next(speech_encoder.parameters(), None)
    if _p is None:
        _p = next(speech_encoder.buffers(), None)
    device = _p.device if _p is not None else torch.device("cpu")
    with torch.no_grad():
        embs = [speech_encoder(chunk.unsqueeze(0).to(device)) for chunk in raw_signals]
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


def _make_cache_writer(cache_dir: Path, num_versions: int,
                       speech_encoder, output_vars: list[str],
                       cache_pool: str, cache_dtype: str = "fp16"):
    """Build the ``cache_emb`` DynamicItem that runs the encoder and writes HDF5."""
    @CachedHDF5DynamicItem.cache(cache_dir, file_mode="a",
                                 num_version=num_versions, dtype=cache_dtype)
    @sb.utils.data_pipeline.takes(ID, SIGNALS)
    @sb.utils.data_pipeline.provides(*output_vars)
    def cache_emb(_id, raw_signals):
        return _compute_emb_from_signals(raw_signals, speech_encoder, cache_pool)

    return cache_emb


def _drive_warm_api(warmup, speech_encoder, cache_pool: str,
                    all_ids: list[str], max_workers: int) -> None:
    """Parallel-fanout driver for API encoders. Fetches raw signals in the
    main thread (cheap CPU I/O + augment), runs the encoder forward in a
    thread pool, and writes HDF5 in the main thread (h5py isn't thread-safe)."""
    from sdx.dataio.pipeline import SIGNALS

    cap = max(1, max_workers * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (ds, writer, kind, version) in enumerate(warmup):
            uncached = writer.uncached_ids(all_ids, version)
            if not uncached:
                print(f"Iteration {i} ({kind} v{version}): already warmed, skipping.")
                continue
            print(f"Iteration {i} ({kind} v{version}): warming "
                  f"{len(uncached)}/{len(all_ids)} via API (workers={max_workers}).")

            # speechbrain's __getitem__ wants a positional index, not a uid.
            uid_to_idx = {uid: idx for idx, uid in enumerate(ds.data_ids)}
            in_flight: dict[Future, str] = {}
            idx = 0
            done_n = 0
            t0 = time.time()
            while idx < len(uncached) or in_flight:
                while idx < len(uncached) and len(in_flight) < cap:
                    uid = uncached[idx]
                    idx += 1
                    sample = ds[uid_to_idx[uid]]
                    fut = ex.submit(
                        _compute_emb_from_signals,
                        sample[SIGNALS], speech_encoder, cache_pool,
                    )
                    in_flight[fut] = uid
                if not in_flight:
                    break
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    uid = in_flight.pop(fut)
                    try:
                        writer._cache(fut.result(), uid)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warn] {kind} v{version} uid={uid} failed: "
                              f"{type(e).__name__}: {str(e)[:160]}")
                    done_n += 1
                    if done_n % 50 == 0 or done_n == len(uncached):
                        rate = (time.time() - t0) / done_n
                        eta = rate * (len(uncached) - done_n)
                        print(f"  [{kind} v{version}] {done_n}/{len(uncached)}  "
                              f"({rate:.2f}s/uid, eta {eta:.0f}s)")


def _combine_chunk_embs(per_chunk: list, output_hidden_states: bool,
                        cache_pool: str):
    """Concatenate one uid's per-chunk encoder outputs into a cache payload.

    ``per_chunk`` holds the uid's chunk outputs in order; each item is a
    ``(frames, dim)`` tensor, or — for hidden-state encoders — a tuple of
    one such tensor per layer. Produces the same payload as
    ``_compute_emb_from_signals`` so the batched and serial drivers write
    byte-compatible HDF5.
    """
    if output_hidden_states:
        n_layers = len(per_chunk[0])
        return tuple(
            torch.cat([c[i] for c in per_chunk], dim=-2).cpu()
            for i in range(n_layers)
        )
    emb = torch.cat(list(per_chunk), dim=-2).cpu()
    if cache_pool == "mean":
        emb = emb.mean(dim=-2, keepdim=False)
    return emb


def _encode_chunk_batch(buffer: list, speech_encoder, tmp: h5py.File,
                        ohs: bool, device, cache_dtype: str = "fp16") -> None:
    """Encode one batch of ``(chunk_key, signal)`` and write each output to tmp.

    A single-chunk batch uses the plain ``forward`` (no padding). A
    multi-chunk batch is right-zero-padded to a ``(B, T)`` tensor here in
    the warmer and handed to the encoder with relative ``lengths``; the
    encoder masks the padded frames, and each ``(B, T', D)`` row is cropped
    back to its true frame count — exactly via ``encoder.feature_lengths``
    when the encoder provides it, else proportionally to the relative
    length (a good approximation for fixed-stride encoders).
    """
    sigs = [s for _, s in buffer]
    with torch.no_grad():
        if len(sigs) == 1:
            out = speech_encoder(sigs[0].unsqueeze(0).to(device))
            crops = [tuple(o[0] for o in out)] if ohs else [out[0]]
        else:
            lens = torch.tensor([float(s.shape[-1]) for s in sigs])
            padded = torch.nn.utils.rnn.pad_sequence(sigs, batch_first=True)
            out = speech_encoder(padded.to(device), lengths=lens / lens.max())
            n_frames = (out[0] if ohs else out).shape[1]
            if hasattr(speech_encoder, "feature_lengths"):
                feat_len = speech_encoder.feature_lengths(lens).tolist()
            else:
                # No exact samples->frames map: crop proportionally by length.
                feat_len = [min(n_frames, max(1, round(float(L) * n_frames
                                                       / float(lens.max()))))
                            for L in lens]
            if ohs:
                crops = [tuple(layer[i, :feat_len[i]] for layer in out)
                         for i in range(len(sigs))]
            else:
                crops = [out[i, :feat_len[i]] for i in range(len(sigs))]
    # Scratch HDF5 stores one chunk's per-layer activations between Phase 1
    # (encode) and Phase 2 (concat + final write). On the bigger datasets
    # this scratch can grow to TB scale (24 layers × ~256-frame chunks ×
    # ~100k chunks for c19sounds), so it follows the same cache_dtype the
    # final writer uses — picking fp32 doubles scratch I/O as well as
    # final on-disk size, but stays internally consistent.
    for (key, _), o in zip(buffer, crops):
        if ohs:
            for li, layer in enumerate(o):
                tmp.create_dataset(
                    f"{key}/L{li}", data=_to_ndarray(layer, cache_dtype),
                )
        else:
            tmp.create_dataset(key, data=_to_ndarray(o, cache_dtype))


def _is_cuda_oom(exc: BaseException) -> bool:
    """Heuristic: a real CUDA OOM vs any other RuntimeError.

    ``torch.cuda.OutOfMemoryError`` only exists in recent PyTorch; pre-2.5 it
    is a plain ``RuntimeError`` whose message starts with ``CUDA out of
    memory``. Check both shapes so the retry logic works either way.
    """
    if torch.cuda.is_available():
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _encode_chunk_batch_oom_safe(buffer: list, speech_encoder, tmp: h5py.File,
                                 ohs: bool, device,
                                 cache_dtype: str = "fp16") -> None:
    """Run ``_encode_chunk_batch`` with an OOM-driven halve-and-retry.

    The gated relative position bias in WavLMAttention (and similar encoders)
    materializes a ``(B, H, T, T)`` tensor before SDPA can stream the scores
    — so a batch that happens to bundle several max-length chunks can still
    OOM even with SDPA enabled. Rather than crash the whole warm cycle, peel
    the batch in half and retry; the recursion bottoms out at ``len==1``,
    which is the per-chunk serial path (and if a single chunk OOMs there's
    nothing left to shrink, so the error propagates).
    """
    try:
        _encode_chunk_batch(buffer, speech_encoder, tmp, ohs, device, cache_dtype)
        return
    except BaseException as exc:  # noqa: BLE001 — narrowed inside
        if not _is_cuda_oom(exc) or len(buffer) <= 1:
            raise
        # The exception object holds a traceback, which holds frames, which
        # hold locals — including the giant GPU tensors that just OOM'd.
        # `empty_cache()` won't reclaim them while those frames are alive.
        # Drop the traceback, drop the local, gc, *then* empty the cache so
        # the retry actually starts from a clean allocator state. Without
        # this the cascade can OOM at B=1 with PyTorch still holding the
        # B=32 working set in unreachable traceback frames.
        exc.__traceback__ = None
        mid = len(buffer) // 2
        msg = f"  [warn] OOM at batch={len(buffer)} — splitting to {mid}+{len(buffer) - mid} and retrying."
        del exc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(msg)
    _encode_chunk_batch_oom_safe(buffer[:mid], speech_encoder, tmp, ohs, device,
                                 cache_dtype)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _encode_chunk_batch_oom_safe(buffer[mid:], speech_encoder, tmp, ohs, device,
                                 cache_dtype)


def _read_chunk(tmp: h5py.File, key: str, ohs: bool):
    """Read one chunk's per-frame embedding back from the tmp HDF5."""
    if ohs:
        grp = tmp[key]
        return tuple(torch.from_numpy(grp[f"L{li}"][:]) for li in range(len(grp)))
    return torch.from_numpy(tmp[key][:])


def _drive_warm_batched(warmup, speech_encoder, cache_pool: str,
                        all_ids: list[str], batch_size: int,
                        cache_dtype: str = "fp16") -> None:
    """Batched driver — the main warm path for local encoders.

    Two phases per (kind, version):

    1. Stream chunks from the pipeline in uid order (no sorting). As soon
       as ``batch_size`` chunks have accumulated, run them through the
       encoder in one ``forward`` call and write each chunk's per-frame
       embedding to a scratch HDF5, keyed ``uid/c{i}``. Variable-length
       chunks are passed as a list with relative lengths so the encoder
       pads + masks them; ``batch_size == 1`` falls back to the plain
       tensor forward, so non-batched encoders behave exactly as before.

    2. Read the scratch HDF5 back per uid, concatenate the uid's chunks in
       order and pool (``_combine_chunk_embs``), then write the cache entry.

    The scratch file keeps per-chunk activations off the heap, so memory
    is bounded by one batch + one uid's chunks regardless of dataset size.
    """
    from sdx.dataio.pipeline import SIGNALS

    ohs = speech_encoder.output_hidden_states
    p = next(speech_encoder.parameters(), None)
    device = p.device if p is not None else torch.device("cpu")

    for i, (ds, writer, kind, version) in enumerate(warmup):
        uncached = writer.uncached_ids(all_ids, version)
        if not uncached:
            print(f"Iteration {i} ({kind} v{version}): already warmed, skipping.")
            continue
        print(f"Iteration {i} ({kind} v{version}): warming "
              f"{len(uncached)}/{len(all_ids)} uncached uids "
              f"(batched, batch_size={batch_size}).")

        uid_to_idx = {uid: idx for idx, uid in enumerate(ds.data_ids)}
        # Scratch HDF5 for Phase 1; under multi_L24 with the bigger datasets
        # this can grow to hundreds of GB. Honor SLURM_TMPDIR (node-local
        # NVMe, hundreds of GB) when present so we never spill to a tiny
        # /tmp tmpfs. tempfile.mkdtemp's default ``dir`` is None which
        # falls back to TMPDIR / /tmp — fine off-cluster.
        scratch_root = os.environ.get("SLURM_TMPDIR") or None
        tmp_dir = Path(tempfile.mkdtemp(prefix="warm_chunks_", dir=scratch_root))
        tmp_path = tmp_dir / "chunks.hdf5"
        n_chunks: dict[str, int] = {}
        t0 = time.time()
        done_n = 0
        try:
            # Phase 1: stream chunks -> batched forward -> per-chunk scratch HDF5.
            with h5py.File(tmp_path, "w") as tmp:
                buffer: list[tuple[str, Any]] = []
                for uid in uncached:
                    signals = ds[uid_to_idx[uid]][SIGNALS]
                    n_chunks[uid] = len(signals)
                    for ci, sig in enumerate(signals):
                        buffer.append((f"{uid}/c{ci}", sig))
                        if len(buffer) >= batch_size:
                            _encode_chunk_batch_oom_safe(buffer, speech_encoder, tmp,
                                                         ohs, device, cache_dtype)
                            done_n += len(buffer)
                            buffer = []
                            rate = (time.time() - t0) / done_n
                            print(f"  [{kind} v{version}] {done_n} chunks encoded "
                                  f"({rate:.2f}s/chunk)")
                            # Periodically return cached allocator blocks to
                            # the OS — long sweeps of variable-size batches
                            # (post-OOM-shrink) fragment the cache until OOM
                            # even at small batch. PYTORCH_CUDA_ALLOC_CONF=
                            # expandable_segments:True is the primary fix;
                            # this is a belt for the suspenders.
                            if torch.cuda.is_available() and done_n % 256 == 0:
                                torch.cuda.empty_cache()
                if buffer:
                    _encode_chunk_batch_oom_safe(buffer, speech_encoder, tmp,
                                                 ohs, device, cache_dtype)
                    done_n += len(buffer)

            # Phase 2: concat each uid's chunks from scratch, pool, write cache.
            with h5py.File(tmp_path, "r") as tmp:
                for uid in uncached:
                    try:
                        per_chunk = [_read_chunk(tmp, f"{uid}/c{ci}", ohs)
                                     for ci in range(n_chunks[uid])]
                        writer._cache(
                            _combine_chunk_embs(per_chunk, ohs, cache_pool), uid)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warn] {kind} v{version} uid={uid} failed: "
                              f"{type(e).__name__}: {str(e)[:160]}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _warm_uncached(task: str, encoder: str, *,
                   probe: str, device: str | None,
                   data_dict: dict[str, dict],
                   all_ids: list[str],
                   num_versions: int) -> None:
    """Load the real encoder and fill whichever cache entries are still missing.

    Builds the writers and datasets once, then dispatches to either the
    batched driver (local encoders) or the API parallel-fanout driver
    (``is_api_encoder = True``). The batched driver is the main warm path;
    ``warm_batch_size`` (encoder yaml, default 1) sets how many chunks go
    through the encoder per ``forward`` call — 1 reproduces the old
    per-chunk behaviour, >1 batches.
    """
    from sdx.dataio.pipeline import SIGNALS

    hparams = compose_config(task, encoder, probe=probe, mode="warm")
    train_cache_dir, val_cache_dir = _cache_dirs(hparams)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    speech_encoder = hparams["encoder"]
    speech_encoder.eval()
    is_api = getattr(speech_encoder, "is_api_encoder", False)
    # warm_batch_size lives in the encoder yaml (so each encoder picks its own
    # batch sane default). main.yaml does not re-export it at top level, so
    # consult encoder_params before falling back to 1. Keeping the top-level
    # check first lets callers override via load_hyperpyyaml overrides=.
    warm_batch_size = int(
        hparams.get("warm_batch_size")
        or hparams.get("encoder_params", {}).get("warm_batch_size", 1)
    )

    cache_pool = hparams.get("cache_pool", "none")
    cache_dtype = hparams.get("cache_dtype", "fp16")

    if not is_api:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        speech_encoder = speech_encoder.to(torch.device(device))
        # Drive the encoder's compute dtype from main.yaml: cache_dtype so
        # the warm chain is end-to-end one precision (weights, autocast,
        # input wav, scratch + final HDF5). Encoder yamls intentionally
        # don't carry compute_dtype anymore — single source of truth.
        # .to(dtype) recasts every parameter/buffer; setting compute_dtype
        # makes WavLM.forward (and other encoders that read this attr)
        # autocast and cast the input wav to the same dtype.
        from sdx.dataio.cache import _CACHE_TORCH_DTYPES
        th_dt = _CACHE_TORCH_DTYPES[cache_dtype.lower()]
        speech_encoder.to(th_dt)
        if hasattr(speech_encoder, "compute_dtype"):
            speech_encoder.compute_dtype = th_dt

    num_layers = hparams["num_layers"]
    num_outputs = num_layers if speech_encoder.output_hidden_states else 1
    # output_vars names the writer's `provides`. Both drivers here call
    # writer._cache directly, so it is only load-bearing when a writer is
    # used as an inline pipeline DynamicItem (cross-warm) — keep it.
    output_vars = [f"emb_{i}" for i in range(num_outputs)]

    train_writer = _make_cache_writer(
        train_cache_dir, num_versions, speech_encoder, output_vars, cache_pool,
        cache_dtype,
    )
    val_writer = _make_cache_writer(
        val_cache_dir, 1, speech_encoder, output_vars, cache_pool, cache_dtype,
    )

    # Both drivers call the encoder + writer themselves, so the datasets
    # stop at SIGNALS (the writer is not an inline pipeline DynamicItem).
    output_keys = [ID, "path", "Participant_ID", LABEL_ENCODED, SIGNALS]
    train_items = _build_warm_dynamic_items(hparams, augmented=True)
    val_items = _build_warm_dynamic_items(hparams, augmented=False)

    train_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"], dynamic_items=train_items, output_keys=output_keys,
    )
    val_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"], dynamic_items=val_items, output_keys=output_keys,
    )

    warmup = [(train_ds, train_writer, "train", v) for v in range(num_versions)]
    warmup.append((val_ds, val_writer, "val", 0))

    try:
        if is_api:
            max_workers = int(hparams.get("api_max_workers", 50))
            _drive_warm_api(warmup, speech_encoder, cache_pool, all_ids, max_workers)
        else:
            _drive_warm_batched(warmup, speech_encoder, cache_pool, all_ids,
                                warm_batch_size, cache_dtype)
    finally:
        train_writer.close()
        val_writer.close()
        del speech_encoder
        gc.collect()
        if not is_api and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_warm(task: str, encoder: str, *,
             probe: str = "wavrx", device: str | None = None,
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
    # Pre-flight config (stub encoder — no GPU/download cost).
    hparams_stub = compose_config(task, encoder, probe=probe, mode="read")
    train_cache_dir, val_cache_dir = _cache_dirs(hparams_stub)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    base_num_versions = int(hparams_stub["data_params"].get("num_aug_ver", 1))
    num_versions = max(base_num_versions, int(num_aug_ver)) if num_aug_ver else base_num_versions
    data_dict = _load_manifests(hparams_stub)
    all_ids = list(data_dict["all"].keys())

    # Acquire the per-(dataset, encoder) lock BEFORE any HDF5 access so the
    # preflight read and the write are serialized together. Previously the
    # preflight ran outside the lock, causing h5py corruption when concurrent
    # workers warmed tasks sharing the same dataset (e.g. T5/T6 both iemocap).
    lock_path = _lock_path(hparams_stub)
    print(f"Acquiring warm writer lock: {lock_path}")
    with FileLock(str(lock_path)):
        if not overwrite and _is_fully_warm(train_cache_dir, val_cache_dir, num_versions, all_ids):
            print(f"Cache fully warm for {task} × {encoder}, nothing to do.")
            return
        if overwrite:
            _wipe_cache_dirs(train_cache_dir, val_cache_dir)
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

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_execute, job, i): job for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            label, ok, _ = future.result()
            if not ok:
                failed.append(label)

    dashboard.print_summary(done_label="cache warmed")

    # Surface per-pair failures as a non-zero process exit — otherwise the
    # rolling sbatch's "exit 0 → all warm, stop" branch interprets a fully
    # broken cycle as success. ``_run_logged_job`` already swallowed each
    # exception for the dashboard, so we have to re-raise here.
    if failed:
        sample = failed[0] if len(failed) == 1 else f"{failed[0]} ... +{len(failed)-1} more"
        raise SystemExit(
            f"warm: {len(failed)}/{len(jobs)} pair(s) failed ({sample}). "
            f"See per-pair logs under {dashboard.run_log_dir}/."
        )
