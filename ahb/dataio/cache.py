"""HDF5-backed ``CachedDynamicItem``.

Salvaged verbatim from ``training/dataio/cache_dynamic_item.py``. The
on-disk schema (``<uid>/v<version>``) is unchanged — caches written by the
legacy ``master_dataio_prep`` are byte-compatible with this reader.

Author: Peter Plantinga
"""

from __future__ import annotations

import os
import random

import h5py

from speechbrain.utils.data_pipeline import DynamicItem, CachedDynamicItem


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

    def _version_key(self, uid, version):
        return f"{uid}/v{version}"

    def is_fully_cached(self, ids):
        return all(self._is_cached(uid) for uid in ids)

    def uncached_ids(self, ids, version):
        """Subset of ``ids`` whose ``v{version}`` key is missing, in input order."""
        return [uid for uid in ids if self._version_key(uid, version) not in self.hdf5file]

    def _is_cached(self, uid):
        return self._version_key(uid, self.num_version - 1) in self.hdf5file

    def _load(self, uid):
        version = random.randint(0, self.num_version - 1)
        return self.hdf5file[self._version_key(uid, version)][:]

    def _cache(self, result, uid):
        for v in range(self.num_version):
            key = self._version_key(uid, v)
            if key not in self.hdf5file:
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
