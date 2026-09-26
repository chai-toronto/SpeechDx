"""Verify the batched cache warmer against the per-chunk serial path.

Encodes mdvr (T16) through wavlm two ways — the legacy per-chunk serial
forward and the new ``_drive_warm_batched`` path — and reports per-uid
agreement. The val pipeline applies no augmentation, so the two paths see
identical signals and must agree numerically; the multi-chunk subtest
lowers ``max_length`` so recordings split, exercising the cross-chunk
concat.

Usage:
  python scripts/compare_warm_batched.py quick   # bridge + subset + multichunk (~5 min)
  python scripts/compare_warm_batched.py full    # all val uids + train subset vs on-disk
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import speechbrain as sb
import torch

from sdx.config import compose_config
from sdx.dataio.pipeline import ID, LABEL_ENCODED, SIGNALS
from sdx.warm import (
    _build_warm_dynamic_items,
    _compute_emb_from_signals,
    _drive_warm_batched,
    _load_manifests,
    _make_cache_writer,
)

TASK, ENCODER = "T16", "wavlm"
BATCH_SIZE = 4
EXISTING_VAL = Path("embeddings_avg_finalv2/mdvr/wavlm/val/single_avg/cache.hdf5")
EXISTING_TRAIN = Path("embeddings_avg_finalv2/mdvr/wavlm/train/single_avg/cache.hdf5")


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = np.abs(a - b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel": float(diff.max() / (np.abs(a).max() + 1e-12)),
        "cos": cos,
    }


def _build_ds(hparams: dict, *, augmented: bool, max_length: int | None = None):
    """Dataset stopping at SIGNALS (the batched driver pipeline shape)."""
    hp = dict(hparams)
    if max_length is not None:
        hp["max_length"] = max_length
    items = _build_warm_dynamic_items(hp, augmented=augmented)
    return sb.dataio.dataset.DynamicItemDataset(
        data=_load_manifests(hp)["all"],
        dynamic_items=items,
        output_keys=[ID, "path", "Participant_ID", LABEL_ENCODED, SIGNALS],
    )


def _batched_into_tmp(ds, uids, encoder, cache_pool, *,
                      num_versions: int = 1, kind: str = "val") -> dict:
    """Warm ``uids`` via the real _drive_warm_batched into a temp HDF5; read back."""
    tmp = Path(tempfile.mkdtemp(prefix="warm_batched_"))
    writer = _make_cache_writer(tmp, num_versions, encoder, ["emb_0"], cache_pool)
    warmup = [(ds, writer, kind, v) for v in range(num_versions)]
    _drive_warm_batched(warmup, encoder, cache_pool, uids, BATCH_SIZE)
    writer.close()
    with h5py.File(tmp / "cache.hdf5", "r") as f:
        return {u: {v: f[u][v][:] for v in f[u]} for u in uids}


def _report(title: str, rows: list[tuple[str, dict]]) -> bool:
    print(f"\n=== {title} ===")
    worst_cos = min(r[1]["cos"] for r in rows)
    worst_rel = max(r[1]["rel"] for r in rows)
    for uid, s in sorted(rows, key=lambda r: r[1]["cos"])[:5]:
        print(f"  uid={uid:<6} max_abs={s['max_abs']:.2e}  rel={s['rel']:.2e}  "
              f"cos={s['cos']:.9f}")
    ok = worst_cos > 0.9999 and worst_rel < 1e-3
    print(f"  -> {len(rows)} uids | worst cos={worst_cos:.9f} | worst rel={worst_rel:.2e}"
          f" | {'PASS' if ok else 'FAIL'}")
    return ok


def bridge(hparams, encoder) -> bool:
    """A padded (B, T) batch row, cropped, must equal that signal alone."""
    print("\n=== bridge: padded (B, T) forward vs single forward ===")
    sig = _build_ds(hparams, augmented=False)[0][SIGNALS][0]
    sigs = [sig, sig[: sig.shape[-1] // 2]]  # full + half -> forces padding
    lens = torch.tensor([float(s.shape[-1]) for s in sigs])
    padded = torch.nn.utils.rnn.pad_sequence(sigs, batch_first=True)
    with torch.no_grad():
        singles = [encoder(s.unsqueeze(0))[0].cpu().numpy() for s in sigs]
        out = encoder(padded, lengths=lens / lens.max())  # (2, T', D)
        feat_len = encoder.feature_lengths(lens).tolist()
    ok = True
    for row, sg in enumerate(singles):
        cropped = out[row, : feat_len[row]].cpu().numpy()
        s = _stats(sg, cropped)
        rok = s["cos"] > 0.9999999 and s["max_abs"] < 1e-2
        ok &= rok
        tag = "unpadded" if row == 0 else "padded"
        print(f"  row {row} ({tag}): shapes {sg.shape} vs {cropped.shape}  "
              f"max_abs={s['max_abs']:.2e}  cos={s['cos']:.12f}  "
              f"-> {'PASS' if rok else 'FAIL'}")
    return ok


def quick(hparams, encoder) -> bool:
    cache_pool = hparams.get("cache_pool", "none")
    ok = bridge(hparams, encoder)

    # --- val subset: single-chunk recordings, batched across uids ---
    ds = _build_ds(hparams, augmented=False)
    uids = list(ds.data_ids)[:6]
    idx = {u: i for i, u in enumerate(ds.data_ids)}
    t0 = time.time()
    serial = {u: np.asarray(_compute_emb_from_signals(ds[idx[u]][SIGNALS], encoder, cache_pool))
              for u in uids}
    print(f"\n[serial val subset done in {time.time() - t0:.0f}s]")
    batched = _batched_into_tmp(ds, uids, encoder, cache_pool)
    ok &= _report(
        "val subset (6 uids, 1 chunk each, no augmentation)",
        [(u, _stats(serial[u], batched[u]["v0"])) for u in uids],
    )

    # --- multichunk subtest: lower max_length so recordings split ---
    ds_mc = _build_ds(hparams, augmented=False, max_length=30)
    uids_mc = list(ds_mc.data_ids)[:3]
    idx_mc = {u: i for i, u in enumerate(ds_mc.data_ids)}
    nchunks = {u: len(ds_mc[idx_mc[u]][SIGNALS]) for u in uids_mc}
    print(f"\n[multichunk: max_length=30s -> chunk counts {nchunks}]")
    serial_mc = {u: np.asarray(_compute_emb_from_signals(ds_mc[idx_mc[u]][SIGNALS], encoder, cache_pool))
                 for u in uids_mc}
    batched_mc = _batched_into_tmp(ds_mc, uids_mc, encoder, cache_pool)
    ok &= _report(
        "multichunk (3 uids, split recordings — exercises cross-chunk concat)",
        [(u, _stats(serial_mc[u], batched_mc[u]["v0"])) for u in uids_mc],
    )
    return ok


def full(hparams, encoder) -> bool:
    """Warm every val uid + a train subset batched, diff vs the on-disk cache."""
    cache_pool = hparams.get("cache_pool", "none")

    # --- val: every uid, full length, must match the on-disk serial cache ---
    ds = _build_ds(hparams, augmented=False)
    uids = list(ds.data_ids)
    print(f"\n[full: warming {len(uids)} val uids batched (full length)]")
    batched = _batched_into_tmp(ds, uids, encoder, cache_pool)
    with h5py.File(EXISTING_VAL, "r") as f:
        existing = {u: f[u]["v0"][:] for u in uids if u in f}
    ok = _report(
        f"val vs existing on-disk cache ({EXISTING_VAL})",
        [(u, _stats(existing[u], batched[u]["v0"])) for u in uids if u in existing],
    )

    # --- train: subset, 5 aug versions; correlated (not equal) to on-disk ---
    ds_tr = _build_ds(hparams, augmented=True)
    uids_tr = list(ds_tr.data_ids)[:16]
    print(f"\n[full: warming {len(uids_tr)} train uids batched x5 augmented versions]")
    batched_tr = _batched_into_tmp(ds_tr, uids_tr, encoder, cache_pool,
                                   num_versions=5, kind="train")
    with h5py.File(EXISTING_TRAIN, "r") as f:
        same = [float(np.corrcoef(batched_tr[u]["v0"], f[u]["v0"][:])[0, 1])
                for u in uids_tr if u in f]
        # cross-uid baseline: a fresh v0 vs a *different* uid's on-disk v0.
        ondisk = {u: f[u]["v0"][:] for u in uids_tr if u in f}
    cross = [float(np.corrcoef(batched_tr[u]["v0"], ondisk[w])[0, 1])
             for u in uids_tr for w in uids_tr if u != w and u in ondisk and w in ondisk]
    print(f"\n=== train subset vs existing on-disk cache ({EXISTING_TRAIN}) ===")
    print(f"  same recording (fresh aug vs on-disk aug): "
          f"mean r={np.mean(same):.4f}  min r={np.min(same):.4f}  n={len(same)}")
    print(f"  different recording (baseline):            "
          f"mean r={np.mean(cross):.4f}  n={len(cross)}")
    train_ok = np.mean(same) > 0.9 and np.mean(same) - np.mean(cross) > 0.05
    print(f"  -> train cache correlates | {'PASS' if train_ok else 'CHECK'}")
    return ok and train_ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    t0 = time.time()
    print(f"composing {TASK} x {ENCODER}, loading encoder ...")
    hparams = compose_config(TASK, ENCODER, mode="warm")
    enc = hparams["encoder"].eval()
    print(f"[encoder ready in {time.time() - t0:.0f}s]")

    if mode == "bridge":
        passed = bridge(hparams, enc)
    elif mode == "full":
        passed = full(hparams, enc)
    else:
        passed = quick(hparams, enc)

    print(f"\n{'=' * 60}\n{mode.upper()}: {'ALL PASS' if passed else 'FAILURES'}  "
          f"(total {time.time() - t0:.0f}s)")
    sys.exit(0 if passed else 1)
