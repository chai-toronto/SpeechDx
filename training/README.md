# `training/` — pipeline, configs, data prep

This directory holds the actual training stack: the SpeechBrain-based
training loop, the hyper-parameter optimization layer (Ray Tune +
Optuna), per-dataset preparation modules, and the YAML configuration
hierarchy that ties them together.

## Entry points

The orchestrators in `bench/` and `run_all*.py` shell out to one of:

| Script                | When                                                            |
|-----------------------|-----------------------------------------------------------------|
| `train.py`            | Standard probe training. Default for almost everything.         |
| `trainCV.py`          | Cross-validation; folds may be trained concurrently.            |
| `trainPerFoldCV.py`   | Cross-validation; one fold at a time. Used when num_fold is set. |

A task yaml selects between them implicitly: tasks that declare
`num_fold:` go through `trainPerFoldCV.py`; the rest use `train.py`.

`brain.py` defines `DiagnosticsBrain` (the SpeechBrain `Brain` subclass)
along with metric definitions, DeLong / bootstrap confidence intervals,
and automatic class-weight computation. `brains.py` exposes a `Brains`
manager + `DiagnosticsCVBrain` used by both CV variants.

## Configuration hierarchy

Every run is parameterized by a single root yaml (`main.yaml`,
`main_cross.yaml`, or `main_cross_category.yaml`) that pulls in four
sub-configs via `!include:`:

```
main.yaml
├── data_params:    !include:tasks/<task>.yaml         # task identity, loss, label key
├── encoder_params: !include:encoders/<enc>.yaml       # sample rate, dims, encoder ctor
├── probe_params:   !include:probes/<probe>.yaml       # probe head + pooler
├── training_params:!include:training.yaml             # batch size, LR, epochs
└── hpopt_params:   !include:hpopt.yaml                # Ray Tune search space
```

Task / encoder / probe yamls are tagged with hyperpyyaml constructors
(`!new:`, `!ref`, `!include:`); reading them safely from outside
training requires `bench.yaml_io.TolerantLoader`.

### `config/registry.yaml` — single source of truth

The orchestrators no longer hard-code task or encoder lists. Everything
lives in one file:

```yaml
encoders: { qwen3voice: qwen3_voice.yaml, wavlm: wavlm.yaml, … }
paper_tasks: [edaic_depC, edaic_phqR, …]            # used by single + data-eff
data_eff_levels: [{name: 06p25, fraction: 0.0625}, …]
data_eff_default_encoders: [qwen3voice, wavlm, ast, whisper]
cross_pairs: []                                       # cross-mode default tasks
exclude_datasets: [daic_woz]
```

`training/registry.py` is the reader. Adding a task / encoder / level is
a single-file edit here plus the matching yaml under
`config/{tasks,encoders,probes}/`.

## Warm cache mechanics

Encoder forward passes dominate wall time. The cache amortizes them.

1. The first run for a `(dataset, encoder)` pair walks every audio file
   once, runs the encoder, and stores the output in an HDF5 file under
   `embeddings_avg_finalv*/<dataset>/<encoder>/{train,val}/…`.
2. `cache_pool: mean` (set in `main.yaml`) writes `(B, D)` mean-pooled
   features under a `single_avg/` subdirectory; without it, the full
   `(B, T, D)` tensor is written.
3. Subsequent runs read the cache and skip the encoder — the
   orchestrator inlines a stub encoder via `bench/encoder_params.py` so
   the heavy weights are never loaded.

`warm_cache: true` in `main.yaml` enables the pre-pass; `--cache-only`
on `bench …  run` stops after the warm-up so multiple training trials
can later share the cache.

The cache is keyed by `(dataset, encoder, min_length, max_length)`.
Changing any of those requires invalidating the affected entries —
edit and run `invalidate_caches.sh` at the repo root.

## `dataio/`

| File                    | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `prep_<dataset>.py`     | One per dataset (and per cross-pair). Builds JSON manifests.     |
| `preprocessing.py`      | Audio loading, resampling, augmentation, cache wrap.             |
| `subsample.py`          | Participant-level subsampling for `data-eff` mode.               |
| `cache_dynamic_item.py` | HDF5 cache adapter over SpeechBrain's `DynamicItem`.             |
| `invalidate_cache.py`   | Implements the per-`(encoder, dataset)` invalidator script.      |
| `stratified_group_k_fold.py` | Speaker-stratified group k-fold split.                       |

A new task adds (1) an entry in `prep_<dataset>.py` (or a new module if
it's a new dataset) and (2) a yaml in `config/tasks/`. See the
top-level README's "Adding a task" section for the full recipe.
