"""HDF5-backed ``CachedDynamicItem``.
"""

from __future__ import annotations

import os
import random

import h5py
import numpy as np
import torch

from speechbrain.utils.data_pipeline import DynamicItem, CachedDynamicItem


def _to_fp16_ndarray(t):
    """Coerce torch.Tensor or ndarray to a fp16 numpy array for cache write.

    Cache reads always pass through to fp32 in the trainer (the brain casts
    on .to(device)), so on-disk fp16 is a pure space/time optimization with
    no math impact. bf16 → fp16 conversion has the same exponent range
    truncation a downstream .half() would do anyway.
    """
    if hasattr(t, "detach"):
        t = t.detach().to(torch.float16).cpu().numpy()
    else:
        t = np.asarray(t, dtype=np.float16)
    return t


class CachedHDF5DynamicItem(CachedDynamicItem):
    """``CachedDynamicItem`` that uses an HDF5 file for storage.

    A single HDF5 file per cache directory may be faster or more
    space-efficient than the default storage (one torch file per id).
    ``num_version`` enables multiple cached versions per id, useful for
    augmentation variety; one is sampled uniformly at load time.
    """

    def __init__(self, cache_location, file_mode="a", num_version=1, *args, **kwargs):
        super().__init__(cache_location, *args, **kwargs)

        self.file_mode = file_mode
        self.cache_location /= "cache.hdf5"
        self.num_version = num_version
        print(f"Opening HDF5 cache at {self.cache_location} with mode {file_mode}")
        open_kwargs = {"locking": False} if file_mode == "r" else {}
        self.hdf5file = h5py.File(self.cache_location, file_mode, **open_kwargs)
        # Track the pid that owns this handle. h5py's library state is NOT
        # fork-safe — a PyTorch DataLoader worker (default fork start
        # method on Linux) that inherits this handle from the parent will
        # SIGSEGV on the first dataset access. We detect "first call from
        # a new pid" and reopen lazily; see _ensure_handle.
        self._pid = os.getpid()

    def __getstate__(self):
        self.hdf5file.close()
        state = self.__dict__.copy()
        del state["hdf5file"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        open_kwargs = {"locking": False} if self.file_mode == "r" else {}
        self.hdf5file = h5py.File(self.cache_location, self.file_mode, **open_kwargs)

    def __deepcopy__(self, memo):
        # FilteredSortedDynamicItemDataset deepcopies the pipeline; we want the
        # cache shared, not duplicated. Otherwise __getstate__ closes our handle
        # and subsequent uncached_ids() silently returns False for every key,
        # making warm_cache iterate every uid for every version slot.
        return self

    def _ensure_handle(self):
        """Reopen the HDF5 handle if we're now in a different process than
        the one that opened it (e.g. forked DataLoader worker). h5py keeps
        per-process library state; using the parent's handle from a forked
        child segfaults on the first dataset access.
        """
        pid = os.getpid()
        if pid != self._pid:
            open_kwargs = {"locking": False} if self.file_mode == "r" else {}
            self.hdf5file = h5py.File(self.cache_location, self.file_mode, **open_kwargs)
            self._pid = pid

    def _version_key(self, uid, version):
        return f"{uid}/v{version}"

    def is_fully_cached(self, ids):
        self._ensure_handle()
        return all(self._is_cached(uid) for uid in ids)

    def uncached_ids(self, ids, version):
        """Subset of ``ids`` whose ``v{version}`` key is missing, in input order."""
        self._ensure_handle()
        return [uid for uid in ids if self._version_key(uid, version) not in self.hdf5file]

    def _is_cached(self, uid):
        self._ensure_handle()
        return all([self._version_key(uid, v) in self.hdf5file for v in range(self.num_version)])

    def _load(self, uid):
        self._ensure_handle()
        version = random.randint(0, self.num_version - 1)
        key = self._version_key(uid, version)
        grp = self.hdf5file[key]
        # Multi-layer caches store L sibling datasets under ``<uid>/v{v}/L{li}``
        # (h5py group); single-layer caches store one ndarray dataset at
        # ``<uid>/v{v}``. Dispatch on the on-disk type so both shapes
        # round-trip without the caller needing to know which mode we're in.
        if isinstance(grp, h5py.Group):
            n = len(grp)
            return tuple(grp[f"L{li}"][:] for li in range(n))
        return grp[:]

    def _cache(self, result, uid):
        self._ensure_handle()
        for v in range(self.num_version):
            key = self._version_key(uid, v)
            if key in self.hdf5file:
                continue
            if isinstance(result, (tuple, list)):
                # Multi-layer write: store one dataset per layer so each
                # layer keeps its native (T, D) shape (T can vary across
                # layers in principle, though WavLM's layers all share T).
                # fp16 on disk halves the size vs default fp32; activations
                # have plenty of fp16 headroom for use as a read-only cache.
                # Storage budget is the binding constraint for multi_L24
                # (~9TB at fp16 across all 12 datasets vs ~18TB at fp32).
                grp = self.hdf5file.create_group(key)
                for li, t in enumerate(result):
                    arr = _to_fp16_ndarray(t)
                    grp.create_dataset(f"L{li}", data=arr)
            else:
                # Single-layer write: keep the existing fp32 on disk so the
                # multi-GB ``single/`` caches already on disk stay compatible
                # bit-for-bit (no re-warm needed). Multi-layer is the only
                # case driving the fp16 switch.
                self.hdf5file.create_dataset(key, data=result)
            return

    def change_file_mode(self, new_file_mode):
        """Reopen the underlying HDF5 file in a different mode."""
        self.hdf5file.close()
        self.file_mode = new_file_mode
        self.hdf5file = h5py.File(self.cache_location, new_file_mode)

    def close(self):
        self.hdf5file.close()

    @classmethod
    def cache(cls, cache_location, file_mode="a", num_version=1):
        """Decorator: wrap a ``DynamicItem`` factory into a cached one."""

        def decorator(obj):
            if not isinstance(obj, DynamicItem):
                raise ValueError("Can only cache a DynamicItem")
            return cls(
                cache_location,
                file_mode,
                num_version,
                takes=obj.takes,
                func=obj.func,
                provides=obj.provides,
            )

        return decorator
