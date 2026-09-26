"""Build the frozen-prefix boundary cache.

Reuses ``sdx.warm._build_warm_dynamic_items`` verbatim so the audio →
augment → chunk pipeline is identical to the one that produces the frozen
benchmark caches. The only difference from a normal warm is *where* the
encoder is tapped: :func:`sdx.lora.tails.capture_boundary` stops the forward at
the first adapted block instead of letting it run to the end.

Because the prefix is all that runs, this is cheaper than a normal warm for
every encoder. The cache is larger, though: it stores every frame of the
boundary activation (like the ASP ``single/`` cache), not a pooled vector.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import speechbrain as sb
import torch

from sdx.lora.cache import BoundaryCache, cache_mode
from sdx.lora.tails import capture_boundary

ID = "id"
LABEL_ENCODED = "label_encoded"
SIGNALS = "signals"


# The <dataset>/<encoder>/<split> tail of a cache dir. Both the reader and the
# stager derive the node-local layout from this ONE function, so they cannot
# drift into disagreeing about where a staged file lives.
CACHE_TAIL_DEPTH = 3


def rerooted(cache_dir: str | Path, root: str | Path) -> Path:
    """``<root>/<dataset>/<encoder>/<split>`` for a committed cache dir."""
    return Path(root) / Path(*Path(cache_dir).parts[-CACHE_TAIL_DEPTH:])


def boundary_cache_paths(hparams: dict[str, Any], cut: int,
                         root: str | Path | None = None) -> tuple[Path, Path]:
    """``(train, val)`` boundary cache files.

    ``root`` re-roots the pair onto another filesystem for reading — e.g. a
    node-local SSD you copied the cache to. Training reads every recording's
    full frame sequence every epoch, and random reads of a multi-hundred-GB
    HDF5 file over a network filesystem can dominate the run.

    It is an explicit ARGUMENT and deliberately NOT an env lookup inside this
    function: this is also the WRITE path (``scripts/lora_warm.py``,
    :func:`warm_boundary`), and a warm that silently picked up a scratch root
    would write the cache somewhere temporary. Only readers pass ``root``.
    """
    mode = cache_mode(cut)
    train, val = Path(hparams["train_cache_dir"]), Path(hparams["val_cache_dir"])
    if root:
        train, val = rerooted(train, root), rerooted(val, root)
    return train / mode / "cache.hdf5", val / mode / "cache.hdf5"


def build_meta(hparams: dict[str, Any], model_name: str, plan,
               precision: str = "fp32") -> dict:
    dp = hparams["data_params"]
    enc = hparams["encoder"]
    source = (getattr(enc, "ssl_encoder_source", None)
              or getattr(enc, "source", None) or "<unknown>")
    revision = os.environ.get("SDX_LORA_PIN_REVISION", "<unpinned>")
    return {
        "encoder": model_name,
        "source": str(source),
        "revision": revision,
        "cut_index": plan.cut_index,
        "num_blocks": plan.num_blocks,
        "width": plan.width,
        "rank": plan.rank,
        "budget": plan.budget,
        "target_paths": plan.target_paths(),
        "dataset": hparams["dataset"],
        "sample_rate": hparams.get("sample_rate"),
        "max_length": hparams.get("max_length"),
        "min_length": hparams.get("min_length"),
        "split_by_boundary": bool(dp.get("split_by_boundary", False)),
        "num_aug_ver": int(dp.get("num_aug_ver", 1)),
        "snr_low": dp.get("snr_low"),
        "snr_high": dp.get("snr_high"),
        "speed": str(dp.get("speed")),
        "precision": precision,
        "dtype": "float16",
    }


def _autocast(device, precision: str):
    """fp16 autocast on CUDA when asked; otherwise run in the weights' dtype.

    The benchmark's own warm runs the encoder in fp32, which is the default
    here too, so boundary activations sit on the same numerical footing as the
    frozen caches they are compared against. ``"fp16"`` halves warm time on a
    GPU; the leaderboard's LoRA rows were warmed that way.
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")
    if precision == "fp16" and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _encode_uid(encoder, model_name: str, cut: int,
                signals: list[torch.Tensor], device,
                precision: str = "fp32") -> list[torch.Tensor]:
    """Prefix-forward each chunk separately, exactly as the frozen warm does."""
    out = []
    with torch.no_grad(), _autocast(device, precision):
        for chunk in signals:
            x = chunk.unsqueeze(0).to(device)
            hidden = capture_boundary(encoder, model_name, cut, lambda: encoder(x))
            out.append(hidden.squeeze(0).detach().float().cpu())
    return out


def warm_boundary(hparams: dict[str, Any], data_dict: dict, all_ids: list[str],
                  model_name: str, plan, *, device: str = "cpu",
                  num_versions: int | None = None,
                  limit: int | None = None,
                  precision: str = "fp32",
                  progress_every: int = 200) -> dict[str, int]:
    """Fill the train (augmented, N versions) and val (clean, 1 version) caches."""
    from sdx.warm import _build_warm_dynamic_items

    cut = plan.cut_index
    train_path, val_path = boundary_cache_paths(hparams, cut)
    meta = build_meta(hparams, model_name, plan, precision)
    n_ver = num_versions or meta["num_aug_ver"]

    encoder = hparams["encoder"].to(device).eval()
    output_keys = [ID, "path", "Participant_ID", LABEL_ENCODED, SIGNALS]

    train_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"],
        dynamic_items=_build_warm_dynamic_items(hparams, augmented=True),
        output_keys=output_keys,
    )
    val_ds = sb.dataio.dataset.DynamicItemDataset(
        data=data_dict["all"],
        dynamic_items=_build_warm_dynamic_items(hparams, augmented=False),
        output_keys=output_keys,
    )

    ids = all_ids[:limit] if limit else all_ids
    written = {"train": 0, "val": 0, "chunks": 0}
    passes = [(train_ds, train_path, "train", v) for v in range(n_ver)]
    passes.append((val_ds, val_path, "val", 0))

    for ds, path, kind, version in passes:
        with BoundaryCache(path, mode="a", meta=meta) as cache:
            todo = cache.uncached(ids, version)
            if not todo:
                print(f"[boundary] {kind} v{version}: already complete", flush=True)
                continue
            print(f"[boundary] {kind} v{version}: {len(todo)}/{len(ids)} uncached",
                  flush=True)
            subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, todo)
            with torch.no_grad():
                for i, sample in enumerate(subset):
                    chunks = _encode_uid(
                        encoder, model_name, cut, sample[SIGNALS], device,
                        precision)
                    cache.write(sample[ID], version, chunks)
                    written[kind] += 1
                    written["chunks"] += len(chunks)
                    if progress_every and (i + 1) % progress_every == 0:
                        print(f"[boundary] {kind} v{version}: {i + 1}/{len(todo)} "
                              f"uids, {written['chunks']} chunks encoded",
                              flush=True)
    return written
