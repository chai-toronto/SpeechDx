"""Frozen-prefix boundary cache.

Stores, per (uid, augmentation version), the activation entering the first
adapted block — concatenated across the recording's chunks, plus the chunk
lengths so the tail can re-split and run **per chunk** exactly as the frozen
encoder did.

Layout, deliberately a sibling of the existing modes rather than a variant of
them::

    <root>/<dataset>/<encoder>/{train,val}/boundary_L<cut>/cache.hdf5
        "<uid>/v<version>"            (sum_T, d) fp16
            .attrs["chunk_lengths"]   int32[n_chunks]

``boundary_L<cut>`` must never collide with the benchmark's ``single_avg/`` or
``single/`` caches — hence a distinct mode name carrying the cut index, so two
different cut layers cannot share a directory either.

File-level attrs carry the cache's identity: encoder + pinned revision, preprocessing/augmentation identity, cut layer, dtype, adapter target
spec, schema version, and a hash over every field that determines the stored
values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

SCHEMA_VERSION = 1
DTYPE = "float16"


def _resolve_dtype(meta: dict | None) -> str:
    return (meta or {}).get("dtype", DTYPE)


def cache_mode(cut: int) -> str:
    return f"boundary_L{cut}"


def identity_hash(meta: dict[str, Any]) -> str:
    """Hash over exactly the fields that determine activation values.

    Excludes bookkeeping (schema version, build host, timestamps) so a rebuild
    with the same inputs is recognisably the same cache.
    """
    # NOTE: num_aug_ver is deliberately NOT here. It is a COUNT of versions
    # written, not a property of any stored value: the reader draws one of
    # its own task's versions, so a task with num_aug_ver=1 always reads v0
    # and is indifferent to whether v1/v2 also exist. A 3-version cache is a
    # superset that serves a 1-version reader correctly — which is how the
    # standard protocol already treats c19sounds (T19/T21 use 3, T20/T22/T23
    # use 1, one dataset), and it avoids building a second multi-TB cache.
    #
    # The augmentation PARAMETERS (snr_low/high, speed) do stay: they change
    # what a given version contains.
    keys = ("encoder", "source", "revision", "cut_index", "num_blocks",
            "sample_rate", "max_length", "min_length", "split_by_boundary",
            "snr_low", "snr_high", "speed", "dataset", "precision", "dtype")
    payload = json.dumps({k: meta.get(k) for k in keys}, sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class BoundaryCache:
    """Read/write access to one ``boundary_L<cut>`` file."""

    def __init__(self, path: Path, mode: str = "r", meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5: h5py.File | None = None
        self._mode = mode
        self._meta = meta

    # -- handle -----------------------------------------------------------
    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, self._mode)
            if self._mode != "r" and self._meta is not None:
                self._write_meta(self._meta)
        return self._h5

    def _write_meta(self, meta: dict) -> None:
        existing = dict(self.h5.attrs)
        want = dict(meta)
        want["schema_version"] = SCHEMA_VERSION
        want["identity"] = identity_hash(meta)
        if existing.get("identity") and existing["identity"] != want["identity"]:
            raise ValueError(
                f"{self.path} was built with a different identity "
                f"({existing['identity']} != {want['identity']}). Refusing to "
                f"mix incompatible activations in one cache."
            )
        for k, v in want.items():
            self.h5.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v

    @property
    def meta(self) -> dict:
        return dict(self.h5.attrs)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- keys -------------------------------------------------------------
    @staticmethod
    def key(uid: str, version: int) -> str:
        return f"{uid}/v{version}"

    def has(self, uid: str, version: int) -> bool:
        k = self.key(uid, version)
        return k in self.h5 and self.h5[k].attrs.get("complete", False)

    def uncached(self, uids: list[str], version: int) -> list[str]:
        return [u for u in uids if not self.has(u, version)]

    def covers(self, uids: list[str], versions: int) -> tuple[bool, int]:
        """Does this cache hold ``versions`` versions for every uid?

        A cache may legitimately hold MORE versions than a reader needs (see
        ``identity_hash``), and may hold different counts for different uids —
        e.g. c19sounds where the subset tasks wrote 3 versions and the
        full-cohort tasks only need 1. Coverage is therefore "at least N", per
        uid, not an equality check on the file.
        """
        missing = sum(1 for u in uids for v in range(versions)
                      if not self.has(u, v))
        return missing == 0, missing

    # -- payload ----------------------------------------------------------
    def write(self, uid: str, version: int, chunks: list[torch.Tensor]) -> None:
        if not chunks:
            raise ValueError(f"{uid}: no chunks to write")
        arrays = [c.detach().to(torch.float32).cpu().numpy() for c in chunks]
        lengths = np.array([a.shape[-2] for a in arrays], dtype=np.int32)
        stacked = np.concatenate(arrays, axis=-2).astype(
            _resolve_dtype(self._meta))
        k = self.key(uid, version)
        if k in self.h5:
            del self.h5[k]
        ds = self.h5.create_dataset(k, data=stacked)
        ds.attrs["chunk_lengths"] = lengths
        ds.attrs["complete"] = True

    def read(self, uid: str, version: int = 0) -> list[torch.Tensor]:
        """Return the per-chunk tensors, re-split at the stored boundaries."""
        k = self.key(uid, version)
        if k not in self.h5:
            raise KeyError(f"{k} not in {self.path}")
        ds = self.h5[k]
        flat = torch.from_numpy(ds[()])
        lengths = list(ds.attrs["chunk_lengths"])
        if sum(lengths) != flat.shape[-2]:
            raise ValueError(
                f"{k}: chunk_lengths sum {sum(lengths)} != stored frames "
                f"{flat.shape[-2]} — cache is corrupt"
            )
        out, off = [], 0
        for n in lengths:
            out.append(flat[off:off + n])
            off += n
        return out

    def num_versions(self, uid: str) -> int:
        return len(self.h5[uid]) if uid in self.h5 else 0
