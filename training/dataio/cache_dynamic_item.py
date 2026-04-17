"""A pipeline for caching data transformations into hdf5 files.

Author:
 * Peter Plantinga
"""
import os
import random

import h5py

from speechbrain.utils.data_pipeline import DynamicItem, CachedDynamicItem

class CachedHDF5DynamicItem(CachedDynamicItem):
    """CachedDynamicItem that uses HDF5 to store the cache. This performant
    data storage format only creates a single file, which may be faster or
    more efficient than the default storage (one torch file per id).

    Arguments
    ---------
    cache_location : os.PathLike
        Storage folder for containing HDF5 cached output file.
    file_mode : str
        The mode to use when opening the HDF5 file. When creating the
        cache, writing must be allowed, but when reading from multiple
        processes, writing should not be allowed.
    num_version : int
        The number of file version to be saved. @load, uniformly sample one. Useful for variety in Augment/Random Crop
    *args
    **kwargs
        Forwarded to DynamicItem constructor
    """

    def __init__(self, cache_location, file_mode="a", num_version=1, *args, **kwargs):
        super().__init__(cache_location, *args, **kwargs)

        # Open connection to HDF5 file
        self.file_mode = file_mode
        self.cache_location /= "cache.hdf5"
        self.num_version = num_version
        print(f"Opening HDF5 cache at {self.cache_location} with mode {file_mode}")
        open_kwargs = {"locking": False} if file_mode == "r" else {}
        self.hdf5file = h5py.File(self.cache_location, file_mode, **open_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["hdf5file"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        open_kwargs = {"locking": False} if self.file_mode == "r" else {}
        self.hdf5file = h5py.File(self.cache_location, self.file_mode, **open_kwargs)

    def _version_key(self, uid, version):
        """Return the HDF5 key for a specific version of uid."""
        return f"{uid}/v{version}"

    def is_fully_cached(self, ids):
        """Check if all ids have all versions cached.

        Arguments
        ---------
        ids : iterable
            Collection of uid strings to check.

        Returns
        -------
        bool
            True if every uid has all versions cached.
        """
        return all(self._is_cached(uid) for uid in ids)

    def uncached_ids(self, ids, version):
        """Return ids whose given version slot is not yet cached.

        Arguments
        ---------
        ids : iterable
            Collection of uid strings to check.
        version : int
            Version index to check (0-based).

        Returns
        -------
        list
            Subset of ``ids`` whose ``v{version}`` key is missing from
            the HDF5 file, preserving input order.
        """
        return [uid for uid in ids if self._version_key(uid, version) not in self.hdf5file]

    def _is_cached(self, uid):
        """Test whether all versions of uid are cached."""
        return self._version_key(uid, self.num_version - 1) in self.hdf5file

    def _load(self, uid):
        """Load one version uniformly at random from cache."""
        version = random.randint(0, self.num_version - 1)
        return self.hdf5file[self._version_key(uid, version)][:]

    def _cache(self, result, uid):
        """Save result to the next available version slot."""
        for v in range(self.num_version):
            key = self._version_key(uid, v)
            if key not in self.hdf5file:
                self.hdf5file.create_dataset(key, data=result)
                return

    def change_file_mode(self, new_file_mode):
        """Change mode that the hdf5 file is opened with. Usually used to convert from
        writing format (building cache) to read-only format (multi-process loading)."""
        self.hdf5file.close()
        self.file_mode = new_file_mode
        self.hdf5file = h5py.File(self.cache_location, new_file_mode)

    def close(self):
        """Used to initialize another DynamicItem on the same cache"""
        self.hdf5file.close()

    @classmethod
    def cache(cls, cache_location, file_mode="a", num_version=1):
        """Decorator which takes a DynamicItem and creates a CachedHDF5DynamicItem

        Arguments
        ---------
        cache_location : os.PathLike
            Storage folder for containing HDF5 cached output file.
        file_mode : str
            The mode to use when opening the HDF5 file. When creating the
            cache, writing must be allowed, but when reading from multiple
            processes, writing should not be allowed.
        num_version : int
            The number of file version to be saved. @load, uniformly sample one. Useful for variety in Augment/Random Crop
        Example
        -------
        >>> import os, numpy
        >>> from speechbrain.utils.data_pipeline import takes, provides
        >>> tempdir = getfixture("tmpdir")
        >>> @CachedHDF5DynamicItem.cache(tempdir)
        ... @takes("id", "text")
        ... @provides("tokenized")
        ... def count_to(id, limit):
        ...     return numpy.arange(limit)
        >>> "utt_id" in count_to.hdf5file
        False
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> "utt_id" in count_to.hdf5file
        True
        >>> # The output shouldn't change on the second call
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> # NOTE: NO INVALID CACHE DETECTION
        >>> count_to("utt_id", 10)
        array([0, 1, 2, 3, 4])
        """

        def decorator(obj):
            """Decorator definition."""
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
