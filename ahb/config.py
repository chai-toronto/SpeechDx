"""Compose the per-job ``main.yaml`` text and load it via HyperPyYAML.

Replaces the regex-driven ``run_all.make_config`` (lines 242-287). The
algorithm is line-oriented (``str.replace`` on whole-line prefixes, no
regex), which makes the substitutions auditable and order-independent.

Three modes, matching the legacy ``warm_cache``/``test_only``/``cache_only``
flag combinations:

- ``mode="warm"``    — encoder yaml included; warm_cache=True. Used by ``ahb warm``.
- ``mode="read"``    — encoder yaml replaced by stub; warm_cache=False. Used by ``ahb train``.
- ``mode="test"``    — encoder yaml included; warm_cache=False, test_only=True. Used by the final-eval pass.
- ``mode="cache"``   — encoder yaml included; warm_cache=True, cache_only=True. Used by ``ahb run --cache-only`` (legacy parity).

The composed text is written to a temp file under ``ahb/configs/`` so
HyperPyYAML's ``!include:`` directives resolve relative to it.
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hyperpyyaml import load_hyperpyyaml

from ahb.encoder_stub import build_stub_encoder_params
from ahb.registry import encoders as registry_encoders

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
ENCODERS_DIR = CONFIG_DIR / "encoders"
PROBES_DIR = CONFIG_DIR / "probes"
TASKS_DIR = CONFIG_DIR / "tasks"
CROSS_TASKS_DIR = CONFIG_DIR / "cross_tasks"

MAIN_SINGLE = CONFIG_DIR / "main.yaml"
MAIN_CROSS = CONFIG_DIR / "main_cross.yaml"
MAIN_CROSS_CATEGORY = CONFIG_DIR / "main_cross_category.yaml"

_MODE_FLAGS = {
    # mode -> (warm_cache_override, test_only, cache_only)
    "warm":  (True,  False, False),
    "read":  (False, False, False),
    "test":  (False, True,  False),
    "cache": (True,  False, True),
}


def _resolve_task_subdir(task: str) -> tuple[Path, Path]:
    """Return ``(base_main_yaml, task_subdir)`` based on which dir holds the task."""
    if (TASKS_DIR / f"{task}.yaml").exists():
        return MAIN_SINGLE, TASKS_DIR
    if (CROSS_TASKS_DIR / f"{task}.yaml").exists():
        # Category-cross yamls have a list ``train_datasets:``; everything
        # else under cross_tasks/ is the regular cross runner.
        text = (CROSS_TASKS_DIR / f"{task}.yaml").read_text()
        if "\ntrain_datasets:" in text or text.startswith("train_datasets:"):
            return MAIN_CROSS_CATEGORY, CROSS_TASKS_DIR
        return MAIN_CROSS, CROSS_TASKS_DIR
    raise FileNotFoundError(
        f"No task yaml for {task!r}. Searched {TASKS_DIR}, {CROSS_TASKS_DIR}."
    )


def _task_yaml_include(task_subdir: Path, task_yaml: str) -> str:
    """Return the include path used in main.yaml's ``data_params:`` line.

    Single-dataset main.yaml uses ``tasks/<task>.yaml``; cross main yamls
    use ``cross_tasks/<task>.yaml``. The path is *relative to the main yaml's
    parent directory* (i.e. ``ahb/configs/``).
    """
    return f"{task_subdir.name}/{task_yaml}"


def compose_yaml_text(
    model_name: str,
    encoder_yaml: str,
    task_yaml: str,
    *,
    probe_yaml: str = "Probe.yaml",
    probe_name: str = "AvgTProbe",
    experiment_tag: str | None = None,
    warm_cache_override: bool | None = None,
    test_only: bool = False,
    cache_only: bool = False,
    base_main_yaml: Path | None = None,
    task_include_prefix: str = "tasks",
    level_dir: str | None = None,
) -> str:
    """Compose the per-job main yaml text.

    Mirrors the line-by-line substitutions ``run_all.make_config`` performs
    via ``re.sub``, but using whole-line ``str.startswith`` matching so the
    resulting text can be diffed against ``make_config``'s output.
    """
    base = base_main_yaml or MAIN_SINGLE
    text = base.read_text()

    use_stub_encoder = (
        warm_cache_override is False and not test_only and not cache_only
    )
    if use_stub_encoder:
        encoder_block = build_stub_encoder_params(ENCODERS_DIR / encoder_yaml)
    else:
        encoder_block = f"encoder_params: !include:encoders/{encoder_yaml}"

    # Whole-line substitutions keyed by line prefix. Order doesn't matter
    # since each prefix is unique to one line in main.yaml.
    line_subs: dict[str, str] = {
        "model_name:":     f"model_name: {model_name}",
        "probe_name:":     f"probe_name: {probe_name}",
        "encoder_params:": encoder_block,
        "probe_params:":   f"probe_params: !include:probes/{probe_yaml}",
        "data_params:":    f"data_params: !include:{task_include_prefix}/{task_yaml}",
        "skip_prep:":      "skip_prep: True",
    }
    if experiment_tag is not None:
        line_subs["experiment_tag:"] = f"experiment_tag: {experiment_tag}"
    if warm_cache_override is not None and not cache_only:
        line_subs["warm_cache:"] = f"warm_cache: {str(warm_cache_override).lower()}"
    if cache_only:
        # Cache-only forces warm_cache true so dataio_prep does the writing.
        line_subs["warm_cache:"] = "warm_cache: true"
    if test_only:
        line_subs["test_only:"] = "test_only: True"
        line_subs["warm_cache:"] = "warm_cache: false"

    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        # Drop chunk_at: lines entirely (legacy field, no longer used).
        if stripped.startswith("chunk_at:"):
            continue
        replaced = False
        for prefix, new_line in line_subs.items():
            if stripped.startswith(prefix):
                out_lines.append(new_line)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    composed = "\n".join(out_lines)
    if level_dir:
        # Data-efficiency runs reroute every ./exps/single_task/ literal in
        # main.yaml to ./exps/data_eff/<level_dir>/. Affects output_folder,
        # train/val/test_annotation — the only literals starting with
        # ./exps/single_task/ in main.yaml. Cache paths (slurm_tmpdir/...)
        # are unchanged: data-eff reuses the full-benchmark encoder cache.
        composed = composed.replace("./exps/single_task/", f"./exps/data_eff/{level_dir}/")
    if cache_only:
        composed += "\ncache_only: True\n"
    return composed


@contextmanager
def _temp_main_yaml(text: str, prefix: str = "_tmp_run_ahb_") -> Iterator[Path]:
    """Write composed text to a temp file under CONFIG_DIR and clean up."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".yaml", dir=str(CONFIG_DIR))
    os.close(fd)
    p = Path(path)
    try:
        p.write_text(text)
        yield p
    finally:
        p.unlink(missing_ok=True)


def compose_config(
    task: str,
    encoder: str,
    *,
    probe: str = "AvgTProbe",
    probe_yaml: str = "Probe.yaml",
    tag: str | None = None,
    mode: str = "read",
    overrides: dict | None = None,
    level_dir: str | None = None,
) -> dict:
    """Produce the resolved hparams dict for one (task, encoder, probe) job.

    ``mode`` selects the warm/read/test/cache flag combination (see module
    docstring). ``overrides`` is forwarded to ``load_hyperpyyaml`` and can
    re-stamp scalar fields like ``output_folder`` for Ray-trial forks.
    """
    if mode not in _MODE_FLAGS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(_MODE_FLAGS)}")
    warm_override, test_only, cache_only = _MODE_FLAGS[mode]

    encoder_yaml = registry_encoders().get(encoder)
    if encoder_yaml is None:
        raise KeyError(f"encoder {encoder!r} not found in ahb/configs/registry.yaml")

    base_main_yaml, task_subdir = _resolve_task_subdir(task)
    task_yaml = f"{task}.yaml"
    text = compose_yaml_text(
        model_name=encoder,
        encoder_yaml=encoder_yaml,
        task_yaml=task_yaml,
        probe_yaml=probe_yaml,
        probe_name=probe,
        experiment_tag=tag,
        warm_cache_override=warm_override,
        test_only=test_only,
        cache_only=cache_only,
        base_main_yaml=base_main_yaml,
        task_include_prefix=task_subdir.name,
        level_dir=level_dir,
    )
    overrides_str = _format_overrides(overrides) if overrides else ""
    with _temp_main_yaml(text) as path:
        with open(path) as f:
            return load_hyperpyyaml(f, overrides=overrides_str)


def _format_overrides(overrides: dict) -> str:
    """Convert dict overrides to the YAML string form ``load_hyperpyyaml`` expects."""
    import yaml as _yaml

    return _yaml.safe_dump(overrides, default_flow_style=False, sort_keys=False)
