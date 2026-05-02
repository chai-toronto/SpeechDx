# Audio Health Benchmark

A reproducible benchmark for self-supervised audio encoders on health-related
tasks (depression, dementia, dysarthria, COVID-19, emotion, …). Each task
trains a lightweight probe on top of a frozen encoder and reports AUROC /
macro-AUROC / MAE with bootstrap confidence intervals.

## Modes

The benchmark exposes four orchestration modes through a single entry point:

| Mode             | What it does                                                                |
|------------------|-----------------------------------------------------------------------------|
| `single`         | Full benchmark — every paper task × every encoder, single dataset per task. |
| `data-eff`       | Same tasks at 4 reduced training-set sizes (6.25 %, 12.5 %, 25 %, 50 %).    |
| `cross`          | Zero-shot cross-task — train on dataset A, evaluate on dataset B.           |
| `cross-category` | Multi-source cross-task variant for category-level transfer.                |

All four are invoked as:

```bash
python -m bench <mode> <subcommand> [flags]
```

where `<subcommand>` is one of `run`, `status`, `summary`. The legacy
`run_all*.py` scripts continue to work and are equivalent — `python -m
bench single …` is exactly `python run_all.py …`. SLURM submission
scripts that reference `run_all.py` directly are unaffected.

## Install

```bash
pip install -r requirements.txt
```

The benchmark assumes a CUDA GPU; CPU works for the smallest tasks but is
not exercised in CI.

## Quickstart — the warm cache flow

The recommended way to run the benchmark is **warm cache first, then
train**. Encoder forward passes dominate wall time; pre-computing them
once and reusing the cache across runs is what makes the benchmark
tractable.

```bash
# 1. Pre-compute (warm) the encoder cache for one (task, encoder) pair.
python -m bench single run --task c9s_t1 --encoder wavlm --cache-only

# 2. Run training. Subsequent runs of the same task/encoder reuse the cache.
python -m bench single run --task c9s_t1 --encoder wavlm

# 3. Inspect results.
python -m bench single status
python -m bench single summary
```

`status` prints a task × encoder grid of `☑ / ☐` completion marks.
`summary` writes per-metric CSVs under `exps/_summary/`.

## Repository layout

```
.
├── bench/                  Unified entry point (python -m bench …) + shared helpers
├── training/               Training pipeline (SpeechBrain + Ray Tune)
│   ├── train.py            Standard probe training (called by orchestrators)
│   ├── trainCV.py          Cross-validation training (concurrent folds)
│   ├── trainPerFoldCV.py   Cross-validation training (sequential folds)
│   ├── brain.py            Metrics, confidence intervals, class weighting
│   ├── brains.py           Multi-fold brain runner used by trainCV variants
│   ├── registry.py         Reads training/config/registry.yaml
│   ├── config/             YAML hierarchy: tasks/, encoders/, probes/, registry.yaml
│   └── dataio/             Per-dataset prep_*.py + caching, subsampling
├── model/                  Encoder + probe + pooling implementations
├── metadata_script/        One create_<dataset>_metadata.py per dataset
├── script/                 Operational helpers (SLURM tracker, cache utilities)
├── data/                   Audio + per-dataset CSVs (gitignored, large)
├── exps/                   Single-mode results — one folder per task per encoder
├── exps_cross/             cross-mode results
├── cross_cat_exps/         cross-category-mode results
├── data_eff_exps/          data-eff results, one per (level, task, encoder)
├── embeddings_avg_finalv*/ Pre-computed encoder caches (HDF5)
├── logs/                   Per-run training logs
├── run_all.py              Legacy entry: single mode (still works; SLURM-friendly)
├── run_all_data_eff.py     Legacy entry: data-eff mode
├── run_all_cross.py        Legacy entry: cross mode
├── run_all_cross_category.py  Legacy entry: cross-category mode
└── invalidate_caches.sh    Edit-and-run cache invalidator (see "Cache" below)
```

## Adding a task

A task is one (dataset, label) pair — e.g. *c9s_t1* trains binary COVID
classification on the COVID-19 Sounds dataset. To add a new task:

1. **Make sure the dataset is staged.** Put audio under
   `data/<dataset>/processed/audio/<...>` and a metadata CSV at
   `data/<dataset>/processed/<dataset>.csv`. Required CSV columns:
   `uid, Participant_ID, split, label, path`. The CSV row's `path` is
   relative to `processed/audio/`. See
   [`metadata_script/`](metadata_script/) for per-dataset builders.
2. **Write a `prep_<task>` function** under
   `training/dataio/prep_<dataset>.py`. It builds train/valid/test
   manifests from the metadata CSV. Existing modules (e.g.
   `prep_c9s.py`, `prep_torgo.py`) are the templates.
3. **Add a task yaml** at `training/config/tasks/<dataset>_<task>.yaml`
   pointing at the prep function (`data_io_script`,
   `prepare_data_fn`), the label column (`label_key`), the loss, and
   `task_type` (B / C / R / L for binary, multiclass, regression,
   multilabel).
4. **Register the task** by adding its stem to `paper_tasks` in
   `training/config/registry.yaml`. That's the single source of truth
   used by every orchestrator.

## Adding an encoder

1. Implement the encoder under `model/<name>.py`. It should be an
   `nn.Module` whose `forward(waveform, lengths=...)` returns either
   `(B, T, D)` or a tuple-of-layers. See `model/wavlm.py` for the
   minimal pattern.
2. Add an encoder yaml at `training/config/encoders/<name>.yaml` with
   metadata fields (`sample_rate`, `feature_dim`, `num_layers`,
   `layer_dim`, `max_length`, `min_length`) and an `encoder: !new:…`
   construction.
3. Register the encoder in `training/config/registry.yaml` under
   `encoders:`. Pick a short `model_name` — that's what shows up in
   experiment folder names and is what `--encoder` accepts.

You do **not** need to edit the orchestrator scripts.

## Reading results

Each completed `(task, encoder)` job writes:

- `exps/<task>/<encoder>-AvgTProbe-run1/test_results.txt` — flat `key:
  value` pairs (AUROC, F1, accuracy, AUROC_CI_low/high, MAE, …).
  Cross-validation tasks write `test_results.yaml` with per-fold detail.
- `events.out.tfevents.*` — TensorBoard scalar logs.
- `best_hparams.yaml` — winning hyperparameters from the Ray Tune search.

Aggregate everything into per-metric CSVs:

```bash
python -m bench single summary
ls exps/_summary/    # AUROC.csv, MAE.csv, completion.csv, …
```

## Cache invalidation

The encoder cache is keyed by `(dataset, encoder, min_length,
max_length)`. If you change any of those, stale entries need to go.

```bash
./invalidate_caches.sh           # dry-run; prints what would be deleted
./invalidate_caches.sh --apply   # actually delete
```

Edit the `ENCODERS` and `DATASETS` arrays at the top of the script to
choose the scope.

## Concurrency

`-j N` in `run` controls how many trainings run in parallel. There is a
per-`(dataset, encoder)` write lock so the warm-cache step never races;
reader-only jobs (cache already populated) run unblocked and will
short-circuit the encoder load via a stub. CLAP requires `-j 1` because
of 48 kHz memory pressure.

## Multi-host / SLURM

`run_all_slurm.sh` is the SLURM submission template; it sets
`HF_HOME`, the Ray temp dir, and forwards `ENCODER`, `DATASET`, `TASK`,
`JOBS`, `TEST_ONLY` env vars to the orchestrator. Examples:

```bash
ENCODER=wavlm,ast DATASET=torgo,ravdess JOBS=4 sbatch run_all_slurm.sh
```

## Troubleshooting

| Symptom                                       | Likely cause / fix                                                                        |
|-----------------------------------------------|-------------------------------------------------------------------------------------------|
| Job hangs on "warming cache"                  | Another process holds the write lock; check `logs/` for the active warmer.                |
| `KeyError` on a probe field                   | Probe yaml's `feature_dim` / `num_layers` doesn't match the encoder yaml.                 |
| Cache miss reported after an encoder upgrade  | Run `./invalidate_caches.sh --apply` for the affected `(encoder, dataset)` rows.          |
| `python run_all.py` fails after pulling main  | `pip install -r requirements.txt` — registry/yaml deps may have moved.                    |
