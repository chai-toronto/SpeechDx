"""A pipeline for caching data transformations into hdf5 files.

Author:
 * Peter Plantinga
"""
import os
import pathlib
import torch

import h5py

from speechbrain.utils.data_pipeline import DynamicItem, CachedDynamicItem

class CachedPersistDynamicItem(CachedDynamicItem):
    """ Not recommended for large datasets. Caches all data in a single pt file, Nor multiprocess safe.

    Arguments
    ---------
    cache_location : os.PathLike
        Storage folder for containing HDF5 cached output file.
    *args
    **kwargs
        Forwarded to DynamicItem constructor
    """

    def __init__(self, cache_location, dataset_size, *args, **kwargs):
        super().__init__(cache_location, *args, **kwargs)

        self.cache_location /= "cache.pt"

        self.cache = torch.load(self.cache_location, weights_only=False, map_location="cpu") if self.cache_location.exists() else {}
        self.num_cached = len(self.cache)
        self.dataset_size = dataset_size


        self.read_only = False
        if self.num_cached == self.dataset_size:
            print(f"Loaded full cache from {self.cache_location}")
            self.read_only = True

    def _is_cached(self, uid):
        """Test whether uid is cached."""
        return uid in self.cache

    def _load(self, uid):
        """Load result from cache"""
        return self.cache[uid].clone()

    def _cache(self, result, uid):
        """Save the result to the cache"""
        self.cache[uid] = result.clone()
        self.num_cached += 1
        if self.num_cached == self.dataset_size and not self.read_only:
            torch.save(self.cache, self.cache_location)

    @classmethod
    def cache(cls, cache_location, dataset_size=None):
        """Decorator which takes a DynamicItem and creates a CachedPersistDynamicItem

        Arguments
        ---------
        cache_location : os.PathLike
            Storage folder for containing pt cached output file.
        dataset_size: int
            Size of the dataset being cached. Used to save at the end.

        Example
        -------
        >>> import os, numpy
        >>> from speechbrain.utils.data_pipeline import takes, provides
        >>> tempdir = getfixture("tmpdir")
        >>> @CachedPersisDynamicItem.cache(tempdir)
        ... @takes("id", "text")
        ... @provides("tokenized")
        ... def count_to(id, limit):
        ...     return numpy.arange(limit)
        >>> "utt_id" in count_to.cache
        False
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> "utt_id" in count_to.cache
        True
        >>> # The output shouldn't change on the second call
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> # NOTE: NO INVALID CACHE DETECTION
        >>> count_to("utt_id", 10)
        array([0, 1, 2, 3, 4])
        """
        if dataset_size is None:
            raise ValueError("dataset_size must be provided to CachedPersistDynamicItem.cache")

        def decorator(obj):
            """Decorator definition."""
            if not isinstance(obj, DynamicItem):
                raise ValueError("Can only cache a DynamicItem")
            return cls(
                cache_location,
                dataset_size,
                takes=obj.takes,
                func=obj.func,
                provides=obj.provides,
            )

        return decorator

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
    *args
    **kwargs
        Forwarded to DynamicItem constructor
    """

    def __init__(self, cache_location, file_mode="a", *args, **kwargs):
        super().__init__(cache_location, *args, **kwargs)

        # Open connection to HDF5 file
        self.file_mode = file_mode
        self.cache_location /= "cache.hdf5"
        print(f"Opening HDF5 cache at {self.cache_location} with mode {file_mode}")
        self.hdf5file = h5py.File(self.cache_location, file_mode)

    def _is_cached(self, uid):
        """Test whether uid is cached."""
        return uid in self.hdf5file

    def _load(self, uid):
        """Load result from cache"""
        return self.hdf5file[uid][:]

    def _cache(self, result, uid):
        """Save the result to the cache"""
        self.hdf5file.create_dataset(uid, data=result)

    def change_file_mode(self, new_file_mode):
        """Change mode that the hdf5 file is opened with. Usually used to convert from
        writing format (building cache) to read-only format (multi-process loading)."""
        self.hdf5file.close()
        self.file_mode = new_file_mode
        self.hdf5file = h5py.File(self.cache_location, new_file_mode)

    @classmethod
    def cache(cls, cache_location, file_mode="a"):
        """Decorator which takes a DynamicItem and creates a CachedHDF5DynamicItem

        Arguments
        ---------
        cache_location : os.PathLike
            Storage folder for containing HDF5 cached output file.
        file_mode : str
            The mode to use when opening the HDF5 file. When creating the
            cache, writing must be allowed, but when reading from multiple
            processes, writing should not be allowed.

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
                takes=obj.takes,
                func=obj.func,
                provides=obj.provides,
            )

        return decorator
