"""HDF5 cache + audio-pipeline primitives shared by warm and read paths.

``cache.py`` is the storage backend; ``pipeline.py`` is the factory of
SpeechBrain ``DynamicItem``s for audio loading, augmentation, label
encoding, and chunking. Neither imports any encoder module.
"""
