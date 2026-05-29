"""HDF5-backed ``CachedDynamicItem``.
"""

from __future__ import annotations

import os
import random

import h5py
import numpy as np
import torch

from speechbrain.utils.data_pipeline import DynamicItem, CachedDynamicItem


# Numpy dtypes the cache layer can write to h5py. bf16 is intentionally
# absent — numpy has no native bf16 dtype, so h5py can't write it.
_CACHE_NP_DTYPES = {
    "fp16": np.float16, "float16": np.float16, "half": np.float16,
    "fp32": np.float32, "float32": np.float32,
}
_CACHE_TORCH_DTYPES = {
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def _to_ndarray(t, dtype: str):
    """Coerce torch.Tensor or ndarray to a numpy array of the requested
    cache dtype (``"fp16"`` or ``"fp32"``)."""
    try:
        np_dt = _CACHE_NP_DTYPES[dtype.lower()]
        th_dt = _CACHE_TORCH_DTYPES[dtype.lower()]
    except KeyError as e:
        raise ValueError(
            f"cache_dtype: unsupported {dtype!r}; expected one of "
            f"{sorted(_CACHE_NP_DTYPES)}. (bf16 is not supported — numpy "
            f"has no bf16 dtype.)"
        ) from e
    if hasattr(t, "detach"):
        return t.detach().to(th_dt).cpu().numpy()
    return np.asarray(t, dtype=np_dt)


class CachedHDF5DynamicItem(CachedDynamicItem):
    """``CachedDynamicItem`` that uses an HDF5 file for storage.

    A single HDF5 file per cache directory may be faster or more
    space-efficient than the default storage (one torch file per id).
    ``num_version`` enables multiple cached versions per id, useful for
    augmentation variety; one is sampled uniformly at load time.
    """

    def __init__(self, cache_location, file_mode="a", num_version=1,
                 dtype: str = "fp16", read_max_frames: int = 0,
                 random_crop: bool = False,
                 *args, **kwargs):
        super().__init__(cache_location, *args, **kwargs)

        self.file_mode = file_mode
        self.cache_location /= "cache.hdf5"
        self.num_version = num_version
        # Read-time crop along the time axis. Each cached uid stores its
        # full pre-encoded length (up to ~70k frames for a long edaic
        # interview); without a cap the dataloader collator pads the batch
        # to that max-T and the WavRx modulation-block STFT then balloons
        # the activation/autograd memory (~40 GB at B=16, the source of the
        # CPU OOM + GPU batch=4 limits). Bounding T at the reader caps the
        # whole pipeline. 0 = disabled (default; cross/warm paths unchanged).
        #
        # ``random_crop`` reproduces the paper's augmentation: a 500-frame
        # window (10 s @ WavLM's 50 fps; clamp_length=160000 samples) sampled
        # at a RANDOM offset per read during training, FIRST-N at eval. This
        # matches model/wavrx.py's in-probe crop semantics but does it at
        # h5py read time, so it costs nothing (a contiguous slice reads only
        # the kept rows) AND keeps the augmentation that a deterministic
        # first-N crop would destroy — critical because most clips exceed
        # 500 frames (aphasia 94%, dementiabank/edaic 100%), so first-N would
        # train every epoch on only the clip's opening. random.randint is
        # already seeded per-trial via main.yaml's random_seed.
        self.read_max_frames = int(read_max_frames or 0)
        self.random_crop = bool(random_crop)
        # Validates the dtype string here so a typo fails fast at warm-time,
        # not 30 min later in the middle of a forward pass.
        if dtype.lower() not in _CACHE_NP_DTYPES:
            raise ValueError(
                f"cache_dtype: unsupported {dtype!r}; expected one of "
                f"{sorted(_CACHE_NP_DTYPES)}."
            )
        self.dtype = dtype.lower()
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
        """Subset of ``ids`` whose ``v{version}`` key is missing or partial.

        "Partial" applies only to the multi-layer write path: a SIGKILL or
        SIGTERM mid-`_cache()` can leave a uid's group with the create_group
        call already committed but only some of the L<i> datasets written.
        Treating that as cached makes warm idempotency lie and forces the
        downstream reader to silently load a tuple of the wrong length.
        Here we report the entry as uncached so warm re-encodes it.
        """
        self._ensure_handle()
        return [uid for uid in ids
                if not self._uid_version_complete(uid, version)]

    def _is_cached(self, uid):
        self._ensure_handle()
        return all(self._uid_version_complete(uid, v) for v in range(self.num_version))

    def _uid_version_complete(self, uid, version):
        """True iff ``<uid>/v<v>`` exists and isn't a partial multi-layer write."""
        key = self._version_key(uid, version)
        if key not in self.hdf5file:
            return False
        obj = self.hdf5file[key]
        if isinstance(obj, h5py.Group):
            # Multi-layer entries are marked with attrs['complete']=True at
            # the end of _cache(). Legacy entries from before the marker
            # was added are accepted if they look fully populated.
            if obj.attrs.get("complete", False):
                return True
            # Fall back to a content sniff: pick another group from the file
            # as a reference for "how many layers should this have" and
            # require this group's L<i> count to match. Empty file → no
            # reference → conservatively treat as incomplete (will trigger
            # a re-encode, which is the safe side of the trade-off).
            ref_n = self._reference_layer_count()
            if ref_n is None:
                return False
            return len(obj) >= ref_n
        return True

    def _reference_layer_count(self):
        """Number of L<i> datasets in any complete sibling group, or None.

        Cached after first compute; the layer count for a given encoder is
        fixed across the whole cache file.
        """
        n = getattr(self, "_ref_n", None)
        if n is not None:
            return n
        for top_uid in self.hdf5file:
            for v_name in self.hdf5file[top_uid]:
                inner = self.hdf5file[top_uid][v_name]
                if isinstance(inner, h5py.Group) and len(inner) > 0:
                    n = len(inner)
                    self._ref_n = n
                    return n
        return None

    def _crop_slice(self, full_T):
        """Time-axis slice for one read. No-op when read_max_frames is 0 or
        the clip is already short enough. With random_crop (train), sample a
        random window start so each epoch sees a different 500-frame window
        (paper augmentation); otherwise first-N (eval, reproducible)."""
        m = self.read_max_frames
        if m <= 0 or full_T <= m:
            return slice(None)
        start = random.randint(0, full_T - m) if self.random_crop else 0
        return slice(start, start + m)

    def _load(self, uid):
        self._ensure_handle()
        version = random.randint(0, self.num_version - 1)
        key = self._version_key(uid, version)
        grp = self.hdf5file[key]
        # Multi-layer caches store L sibling datasets under ``<uid>/v{v}/L{li}``
        # (h5py group); single-layer caches store one ndarray dataset at
        # ``<uid>/v{v}``. Dispatch on the on-disk type so both shapes
        # round-trip without the caller needing to know which mode we're in.
        # The crop slice is computed from .shape[0] (h5py metadata, no read)
        # then applied as a contiguous h5py index, so only the kept rows are
        # paged off disk. One slice per uid so all L layers share the same
        # random window.
        if isinstance(grp, h5py.Group):
            n = len(grp)
            t_slice = self._crop_slice(grp["L0"].shape[0])
            return tuple(grp[f"L{li}"][t_slice] for li in range(n))
        t_slice = self._crop_slice(grp.shape[0])
        return grp[t_slice]

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
                # On-disk dtype follows self.dtype (main.yaml: cache_dtype);
                # fp16 (default) halves the storage footprint and matches
                # training.yaml: precision=fp16, so no conversion happens
                # between cache read and the autocast region.
                grp = self.hdf5file.create_group(key)
                for li, t in enumerate(result):
                    arr = _to_ndarray(t, self.dtype)
                    grp.create_dataset(f"L{li}", data=arr)
                # Marker: only present once every L<i> dataset is written.
                # _uid_version_complete looks for this so a partial write
                # (warm killed by TIMEOUT mid-uid) is correctly classified
                # as "uncached" and re-encoded on the next warm cycle.
                grp.attrs["complete"] = True
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
    def cache(cls, cache_location, file_mode="a", num_version=1,
              dtype: str = "fp16", read_max_frames: int = 0,
              random_crop: bool = False):
        """Decorator: wrap a ``DynamicItem`` factory into a cached one."""

        def decorator(obj):
            if not isinstance(obj, DynamicItem):
                raise ValueError("Can only cache a DynamicItem")
            return cls(
                cache_location,
                file_mode,
                num_version,
                dtype,
                read_max_frames,
                random_crop,
                takes=obj.takes,
                func=obj.func,
                provides=obj.provides,
            )

        return decorator
