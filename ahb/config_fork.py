"""Per-trial config forking for Ray Tune.

Instead of mutating the loaded ``hparams`` dict per-trial (the old pattern in
``train_with_ray`` / ``train_fold_with_ray`` / ``brain.__init__``), this module
writes a self-contained copy of the config tree into ``<trial_dir>/config/``
and stamps trial-specific values into the forked ``main.yaml``. The worker
then loads the fork with no overrides.

Why:
  1. The real config a trial ran under used to only exist in worker RAM.
  2. On SLURM timeout/resubmit the Ray trainable re-read the *original*
     main.yaml, so any edits to ``ahb/configs/*.yaml`` between runs would
     silently leak into resumed trials. With a frozen on-disk fork, resume is
     drift-proof: the fork is authoritative and never regenerated.

CAVEAT — editing a task yaml does NOT propagate to existing trial forks.
If you change ``ahb/configs/tasks/<task>.yaml`` (e.g. drop ``num_aug_ver``
from 3 → 1), every pre-existing ``<exp>/<trial_id>/config/tasks/<task>.yaml``
still carries the old value. For in-flight HPOs this can desync the fork from
the shared on-disk cache (``tmp/<dataset>/<encoder>/...``) and produce an
inscrutable ``h5py.create_dataset(..., data=None)`` TypeError when a fork's
``num_aug_ver`` exceeds the cache's fill depth. Wipe the affected trial dirs
(or the whole ``<exp>/results/`` HPO state) after editing a task yaml.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

_INCLUDE_RE = re.compile(r'!include:(\S+)')

# Matches the full 5-line ``epoch_counter: !new:...EpochCounterWithStopper``
# block (header + all indented children). Ray Tune handles trial stopping, so
# the stopper variant is replaced by a plain EpochCounter in every forked yaml.
_EPOCH_COUNTER_RE = re.compile(
    r'^epoch_counter:[ \t]*!new:speechbrain\.utils\.epoch_loop\.EpochCounterWithStopper[ \t]*\n'
    r'(?:[ \t]+.*\n)*',
    re.MULTILINE,
)
_PLAIN_EPOCH_COUNTER = (
    'epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter\n'
    '  limit: !ref <number_of_epochs>\n'
)


def _yaml_scalar(value: Any) -> str:
    """Render a Python value as an inline YAML scalar suitable for stamping."""
    s = yaml.safe_dump(value, default_flow_style=True).rstrip('\n')
    if s.endswith('\n...'):
        s = s[:-4].rstrip()
    return s


def _copy_include_tree(base_yaml: Path, dest_dir: Path) -> None:
    """Walk ``!include:`` references reachable from *base_yaml* and copy each
    source file into *dest_dir*, preserving relative layout so the ``!include:``
    lines in the copied yaml resolve unchanged."""
    visited: set[Path] = set()

    def walk(src: Path) -> None:
        if src in visited:
            return
        visited.add(src)
        text = src.read_text()
        for rel in _INCLUDE_RE.findall(text):
            if rel.startswith('/'):
                # Absolute include — already self-contained, nothing to copy.
                continue
            src_dep = (src.parent / rel).resolve()
            if not src_dep.exists():
                continue
            try:
                rel_to_base = src_dep.relative_to(base_yaml.parent)
            except ValueError:
                # Include path escapes base_yaml.parent (unusual). Skip —
                # the fork won't be closed under the include graph, but
                # there's no sane default for where to place it.
                continue
            dst_dep = dest_dir / rel_to_base
            dst_dep.parent.mkdir(parents=True, exist_ok=True)
            if not dst_dep.exists():
                shutil.copy2(src_dep, dst_dep)
            walk(src_dep)

    walk(base_yaml)


def _stamp_overrides(text: str, overrides: Dict[str, Any]) -> str:
    """Line-replace top-level ``key: ...`` entries with stamped values.

    Keys that don't already exist at the top level are appended. Matches
    HyperPyYAML override semantics: shadowing the top-level line means
    downstream ``!ref <key>`` chains pick up the new literal.
    """
    for key, value in overrides.items():
        new_line = f'{key}: {_yaml_scalar(value)}'
        pattern = re.compile(rf'^{re.escape(key)}:.*$', re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(lambda _m, n=new_line: n, text)
        else:
            if not text.endswith('\n'):
                text += '\n'
            text += new_line + '\n'
    return text


def fork_trial_config(
    base_yaml: Path,
    trial_dir: Path,
    overrides: Dict[str, Any],
) -> Path:
    """Fork the config tree into ``<trial_dir>/config/`` and stamp *overrides*.

    On resume (when ``<trial_dir>/config/main.yaml`` already exists) this is a
    no-op — the existing fork is returned unchanged so the trial sees exactly
    the config it was launched with.

    Parameters
    ----------
    base_yaml:
        Path to the source main yaml (typically the ``_tmp_run_<model>.yaml``
        produced by ``run_all.make_config``, or directly ``main.yaml``).
    trial_dir:
        Absolute trial output folder. Will contain ``config/main.yaml`` plus
        the copied include tree after this call.
    overrides:
        Top-level scalar overrides to stamp into the forked main yaml —
        already-resolved absolute paths, Ray-sampled hyperparameters,
        ``warm_cache: False``, etc. Keys that already exist in ``main.yaml``
        have their line replaced; new keys are appended.

    Returns
    -------
    Path to the forked ``main.yaml``.
    """
    trial_dir = Path(trial_dir)
    # Resolve base_yaml so relative_to() against src.parent works regardless
    # of the caller's CWD.
    base_yaml = Path(base_yaml).resolve()
    dest_config = trial_dir / 'config'
    dest_main = dest_config / 'main.yaml'

    if dest_main.exists():
        # Resume path — trust the existing fork exactly.
        return dest_main

    dest_config.mkdir(parents=True, exist_ok=True)

    # Copy all reachable !include: files (shallow in practice — the leaf yamls
    # under tasks/, encoders/, probes/ don't nest further — but the walker
    # recurses for safety if that ever changes).
    _copy_include_tree(base_yaml, dest_config)

    text = base_yaml.read_text()
    # Swap EpochCounterWithStopper → plain EpochCounter before stamping
    # so the epoch_counter line itself isn't a regex target for overrides.
    text = _EPOCH_COUNTER_RE.sub(_PLAIN_EPOCH_COUNTER, text)
    text = _stamp_overrides(text, overrides)

    dest_main.write_text(text)
    return dest_main
