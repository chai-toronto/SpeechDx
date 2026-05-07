"""JSON encoding helpers used by the manifest writers.

Salvaged from ``training/dataio/utils.py``. Only the helpers actually used
by the prep modules are carried over (``ensure_dir``, ``PathEncoder``);
``locate_bad`` and ``proc_length_vec`` are not used in the manifest path.
"""

from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: Path) -> None:
    if path.is_file:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        # numpy scalars (int64/float64/bool_) aren't JSON-native
        if hasattr(obj, "item") and hasattr(obj, "dtype"):
            return obj.item()
        return super().default(obj)
