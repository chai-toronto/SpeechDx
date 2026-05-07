"""Single source of truth for orchestrator task/encoder enumeration.

Reads ``sdx/configs/registry.yaml``. The harness subcommands call into
here so the same ``paper_tasks``, ``data_eff_levels``, and exclusion lists
are not forked across files.

Salvaged from ``training/registry.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "configs" / "registry.yaml"


@lru_cache(maxsize=1)
def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def encoders() -> dict[str, str]:
    """Return ``{model_name: encoder_yaml_filename}`` for all known encoders."""
    return dict(load_registry().get("encoders", {}))


def paper_tasks() -> list[str]:
    return list(load_registry().get("paper_tasks", []))


def data_eff_levels() -> list[tuple[str, float]]:
    return [
        (d["name"], float(d["fraction"]))
        for d in load_registry().get("data_eff_levels", [])
    ]


def data_eff_default_encoders() -> list[str] | None:
    """Encoder subset for data-efficiency default. ``None`` = all encoders."""
    val = load_registry().get("data_eff_default_encoders")
    return list(val) if val is not None else None


def cross_pairs() -> list[str]:
    return list(load_registry().get("cross_pairs", []))


def cross_categories() -> list[str]:
    return list(load_registry().get("cross_categories", []))


def exclude_datasets() -> set[str]:
    return set(load_registry().get("exclude_datasets", []))


def datasets() -> dict[str, dict]:
    """Per-dataset access metadata (``access``, ``contact``, …)."""
    return dict(load_registry().get("datasets", {}))
