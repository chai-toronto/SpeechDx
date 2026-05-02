"""Tolerant YAML loader for hyperpyyaml-tagged config files.

The benchmark's main config files use hyperpyyaml tags (``!new:``, ``!ref``,
``!include:``, ``!apply:``, ``!name:``) that the standard SafeLoader
rejects. We need to *read* these files for orchestration (extracting
``data_folder``, ``num_aug_ver``, etc.) without instantiating any of the
tagged objects. ``TolerantLoader`` parses the structure and ignores tags.
"""

from __future__ import annotations

import yaml


class TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores hyperpyyaml tags (!new:, !ref, !include:, !apply:, !name:)."""


def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return None


TolerantLoader.add_multi_constructor("!", _ignore_unknown)
TolerantLoader.add_multi_constructor("tag:", _ignore_unknown)
