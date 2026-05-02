# `bench/` — unified orchestration entry point

```bash
python -m bench <mode> <subcommand> [flags...]
```

`<mode>` is one of `single`, `data-eff`, `cross`, `cross-category`. The
subcommand and remaining flags are forwarded verbatim to the matching
legacy script (`run_all.py`, `run_all_data_eff.py`, `run_all_cross.py`,
`run_all_cross_category.py`). Both invocations are equivalent today:

```bash
python -m bench single run --encoder wavlm --task c9s_t1 -j 4
python run_all.py            run --encoder wavlm --task c9s_t1 -j 4
```

The legacy scripts exist because SLURM templates and tooling reference
them by path; do not delete them.

## Modules in this package

| Module              | Purpose                                                                  |
|---------------------|--------------------------------------------------------------------------|
| `__main__.py`       | The dispatcher above (`python -m bench`).                                |
| `logutil.py`        | Shared `_emit / _slug / _now / _tail / _Progress / _terminal_lock`.      |
| `yaml_io.py`        | `TolerantLoader` — read hyperpyyaml-tagged configs without instantiating. |
| `encoder_params.py` | Build a stub `encoder_params:` block for reader-only jobs.               |
| `results.py`        | `expected_ci_keys(task_type)` — names of CI fields brain.py emits.       |

These four modules are the deduplicated leaves of the three orchestrators.
Higher-level concerns (config mutation, dispatch, manifest handling,
result aggregation) still live inside each `run_all*.py` script.

## Future work — full BaseOrchestrator consolidation

The dispatch / scheduling / manifest / result-parsing logic is still
duplicated (~1000 LOC) across `run_all.py`, `run_all_data_eff.py`, and
`run_all_cross.py`. Folding them into a `BaseOrchestrator` with variant
subclasses is the obvious next step but is deferred — it is a 3000-LOC
diff with subtle concurrency, cache-locking, and Ray-tmpdir constraints
that benefit from interactive review rather than a one-shot session.

When that work happens, the dispatcher in `__main__.py` becomes the
permanent home for the unified CLI; the legacy `run_all*.py` scripts
collapse to 5-line shims.
