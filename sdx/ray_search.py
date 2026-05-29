"""Ray Tune search-space + path-stamping helpers used by ``sdx/train.py``.

``parse_hp_search_space`` is salvaged verbatim from
``training/hp_utils.py``. ``_PATH_STAMP_KEYS``, ``_collect_resolved_paths``,
and ``_as_override_dict`` are extracted from
``training/train.py:49-90`` — they belong in this module because both the
single-split and CV trainers need them.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from ray import tune


def ray_init_kwargs() -> dict:
    """Kwargs for every ``ray.init`` in the trainers.

    Hardening for many leaf processes (sdx ... --workers N) and the
    per-fold ray.init/shutdown churn of the CV trainer on shared Slurm nodes:

    * ``num_cpus`` (from ``SDX_RAY_NUM_CPUS``): without it ray.init detects
      every core (192 on Fir) and prestarts ~one worker per core; several
      leaves cold-starting at once → hundreds of spawns → GCS startup
      timeout. A trial needs only its resources_per_trial[cpu] scheduling
      slots (compute is OMP threads, not Ray workers). Unset → all cores.
    * ``_temp_dir`` (per-pid, under SLURM_TMPDIR when set): isolates each
      leaf's Ray session dir + GCS sockets. The default ``/tmp/ray`` is
      shared, so concurrent leaves — and the CV trainer's repeated
      init/shutdown per fold — race on session files/ports and time out
      ("node timed out during startup"). A unique dir per process removes it.
    * ``include_dashboard=False``: unused headless; its port/agent startup
      is pure overhead and another contention point.
    """
    kwargs: dict = {"ignore_reinit_error": True, "include_dashboard": False}
    ncpus = os.environ.get("SDX_RAY_NUM_CPUS")
    if ncpus:
        kwargs["num_cpus"] = int(ncpus)
    tmp_root = os.environ.get("SLURM_TMPDIR", "/tmp")
    kwargs["_temp_dir"] = os.path.join(tmp_root, f"ray_{os.getpid()}")
    return kwargs


# Top-level source paths in main.yaml whose absolute values get stamped
# into each trial's forked config. Derived paths (save_folder, wav_folder,
# train_cache_dir, ...) are deliberately excluded — they use
# !ref <source_key>/... in the yaml, so stamping the source is enough for
# the chain to resolve to an absolute value automatically. In particular,
# stamping save_folder here would shadow the !ref <output_folder>/save chain
# and break per-trial nesting.
_PATH_STAMP_KEYS = (
    "data_folder",
    "slurm_tmpdir",
    "output_folder",
    "train_annotation",
    "val_annotation",
    "test_annotation",
)


def collect_resolved_paths(hparams, project_root):
    """Pre-resolve top-level source paths to absolute strings.

    These get stamped into the forked main.yaml so Ray workers never see a
    relative path (they may run from a different CWD). Only SOURCE keys
    (:data:`_PATH_STAMP_KEYS`) are resolved — derived paths reach the worker
    via the yaml's ``!ref`` chains.
    """
    resolved = {}
    for key in _PATH_STAMP_KEYS:
        val = hparams.get(key)
        if not isinstance(val, str):
            continue
        p = Path(val)
        resolved[key] = str(p if p.is_absolute() else (project_root / p).resolve())
    return resolved


def as_override_dict(overrides):
    """Normalize overrides (YAML string, dict, or None) to a dict."""
    if overrides is None or overrides == "":
        return {}
    if isinstance(overrides, dict):
        return dict(overrides)
    return yaml.safe_load(overrides) or {}


def parse_hp_search_space(hparams):
    """Build the Ray Tune search space dict from hparams.

    Reads ``hpopt_config`` from the yaml; falls back to a default
    ``lr_start`` / ``dp`` / ``num_fc_neurons`` space when absent.
    """
    search_space = {}

    tune_mapping = {
        "!tune.choice":     tune.choice,
        "!tune.uniform":    tune.uniform,
        "!tune.loguniform": tune.loguniform,
        "!tune.randint":    tune.randint,
        "!tune.quniform":   tune.quniform,
    }

    if hparams.get("hpopt_config"):
        for param, config in hparams["hpopt_config"].items():
            if isinstance(config, dict) and "type" in config:
                tune_fn = tune_mapping.get(config["type"])
                if tune_fn == tune.choice:
                    search_space[param] = tune_fn(config["values"])
                else:
                    search_space[param] = tune_fn(*config["values"])
    else:
        search_space = {
            "lr_start":        tune.loguniform(1e-5, 1e-2),
            "dp":              tune.uniform(0.1, 0.5),
            "num_fc_neurons":  tune.choice([512, 768, 1024]),
        }

    return search_space
