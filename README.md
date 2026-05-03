# Audio Health Benchmark

A reproducible benchmark for self-supervised audio encoders on health-related
tasks (depression, dementia, dysarthria, COVID-19, emotion, …). Each task
trains a lightweight probe on top of a frozen encoder and reports AUROC /
macro-AUROC / MAE with bootstrap confidence intervals.

## Quickstart

```bash
git clone <this repo> && cd Audio-Health-Benchmark
pip install -r requirements.txt
pip install -e .

# 1. Stage one dataset (audio + metadata CSV) under data/<dataset>/processed/.
#    See data/README.md for required CSV columns.

# 2. Build manifests for one task, warm the encoder cache, then train.
python -m ahb prep   c9s_t1
python -m ahb warm   c9s_t1 wavlm
python -m ahb train  c9s_t1 wavlm

# 3. Or run the full benchmark — every paper task × every encoder.
python -m ahb run -j 4

# 4. Inspect.
python -m ahb status                # task × encoder grid of ☑ / ☐
python -m ahb summary               # per-metric CSVs under exps/single_task/_summary/
```

The recommended flow is **warm cache first, then train** — encoder forward
passes dominate wall time, and `ahb run` already serializes warm vs read
correctly per `(dataset, encoder)` pair so multiple jobs share the cache.

## Modes

A single CLI entry point (`python -m ahb <subcommand>`) covers four modes:

| Mode             | Subcommands                                            | What it does                                                                |
|------------------|--------------------------------------------------------|-----------------------------------------------------------------------------|
| single           | `prep` / `warm` / `train` / `train-cv` / `run`         | Full benchmark — every paper task × every encoder, single dataset per task. |
| data-eff         | `run-data-eff`                                         | Same tasks at 4 reduced training-set sizes (6.25 %, 12.5 %, 25 %, 50 %).    |
| cross            | `warm-cross` / `train-cross` / `run-cross`             | Zero-shot cross-task — train on dataset A, evaluate on dataset B.           |
| cross-category   | `run-cross-category`                                   | Multi-source cross-task variant for category-level transfer.                |

`harness.py` is a thin shim equivalent to `python -m ahb`.

## Encoders

12 frozen backbones ship by default. The `Source` column is the identifier
passed to `from_pretrained(...)`; the underlying weight files are downloaded
on first use.

| Name (`--encoder`) | Source                                                     | Hub          | Notes                                                |
|--------------------|------------------------------------------------------------|--------------|------------------------------------------------------|
| `wavlm`            | `microsoft/wavlm-large`                                    | HF           | 1024-dim, 24 layers                                  |
| `w2v2`             | `facebook/wav2vec2-large-960h-lv60-self`                   | HF           | 1024-dim, 24 layers                                  |
| `hubert`           | `facebook/hubert-large-ls960-ft`                           | HF           | 1024-dim, 24 layers                                  |
| `whisper`          | `openai/whisper-large-v3`                                  | HF           | 30 s audio, 1280-dim, 32 layers                      |
| `ast`              | `MIT/ast-finetuned-audioset-10-10-0.4593`                  | HF           | 10 s audio, 768-dim                                  |
| `audiomae`         | `hance-ai/audiomae`                                        | HF           | 10 s audio, 768-dim                                  |
| `clap`             | `laion/larger_clap_general`                                | HF           | 48 kHz, 10 s, 1024-dim                               |
| `mms`              | `facebook/mms-1b`                                          | HF           | 1280-dim, 48 layers                                  |
| `wavjepa`          | `labhamlet/wavjepa-nat-base`                               | HF           | 768-dim, 12 layers                                   |
| `qwen3voice`       | `Qwen/Qwen3-TTS-Tokenizer-12Hz`                            | HF           | 24 kHz, 512-dim                                      |
| `emotion2vec`      | `iic/emotion2vec_plus_large`                               | ModelScope   | Loaded via FunASR                                    |
| `opera_gt`         | `evelyn0414/OPERA` → `encoder-operaGT.ckpt`                | HF (ckpt)    | Vendored loader at `third_party/OPERA/`              |

The yaml at `ahb/configs/encoders/<name>.yaml` is the source of truth — it
holds the default `ssl_encoder_source`, sample rate, feature dim, layer count,
and length window. Edit those to swap to a smaller variant (e.g. `wavlm-base`
in place of `wavlm-large`).

### Pinning a revision

`from_pretrained(...)` calls accept a `revision=` kwarg, but the encoder
classes today don't forward one — they pull HEAD. To pin a specific commit
for a backbone, either edit the encoder wrapper in `model/<name>.py` to pass
`revision="<sha>"` through, or set `HF_HUB_REVISION` and use a per-repo
override file. For `opera_gt` the checkpoint is downloaded once into
`cks/model/` and reused; replace it manually to pin.

## Repository layout

```
.
├── ahb/                    Harness — CLI, orchestrators, prep, dataio, brain
│   ├── cli.py              python -m ahb …
│   ├── orchestrator*.py    Task discovery, completion checks, run scheduling
│   ├── train*.py           Probe training (single / cross / per-fold CV)
│   ├── warm*.py            Encoder cache warmers
│   ├── run*.py             Top-level run loops per mode
│   ├── prep/               Per-dataset manifest builders
│   ├── dataio/             HDF5 cache + speechbrain pipeline glue
│   └── configs/            YAML hierarchy (tasks/, encoders/, probes/, registry.yaml)
├── model/                  Encoder + probe + pooling implementations
├── metadata_script/        One create_<dataset>_metadata.py per dataset
├── script/                 Operational helpers (slurm tracker, prune, layer-weight viz)
├── slurm/                  SLURM job templates
├── third_party/OPERA/      Vendored OPERA encoder loader
├── data/                   Audio + per-dataset CSVs (gitignored, large)
├── exps/                   All experiment results, grouped by mode
│   ├── single_task/        single-mode results (one folder per task per encoder)
│   ├── cross/              cross-mode results
│   ├── cross_cat/          cross-category-mode results
│   ├── data_eff/           data-eff results, one subdir per level
│   └── slurm_logs/         SLURM stdout/stderr (top-level; spans modes)
├── embeddings_avg_finalv*/ Pre-computed encoder caches (HDF5, gitignored)
├── logs/                   Per-run training logs
├── harness.py              Convenience shim: equivalent to python -m ahb
├── run_all_slurm.sh        SLURM submission template
└── invalidate_caches.sh    Edit-and-run cache invalidator
```

## Adding a task

A task is one (dataset, label) pair — e.g. *c9s_t1* trains binary COVID
classification on the COVID-19 Sounds dataset.

1. **Stage the dataset.** Audio under `data/<dataset>/processed/audio/`,
   metadata CSV at `data/<dataset>/processed/<dataset>.csv`. Required CSV
   columns: `uid, Participant_ID, split, label, path` (`path` is relative to
   `processed/audio/`). See [`metadata_script/`](metadata_script/) for
   per-dataset builders.
2. **Write a `prepare_*` function** under `ahb/prep/<dataset>.py`. It builds
   train/valid/test manifests from the metadata CSV. `ahb/prep/c9s.py` and
   `ahb/prep/torgo.py` are the templates.
3. **Add a task yaml** at `ahb/configs/tasks/<dataset>_<task>.yaml` pointing
   at the prep function (`data_io_script`, `prepare_data_fn`), the label
   column (`label_key`), the loss, and `task_type` (B / C / R / L for binary,
   multiclass, regression, multilabel).
4. **Register the task** by adding its stem to `paper_tasks` in
   `ahb/configs/registry.yaml`. That's the single source of truth used by
   every orchestrator.

## Adding an encoder

1. Implement the encoder under `model/<name>.py`. Contract: an `nn.Module`
   whose `forward(waveform, lengths=...)` returns `(B, T, D)` (or a tuple of
   per-layer `(B, T, D)` when `output_hidden_states=True`). See `model/wavlm.py`
   for the minimal pattern.
2. Add an encoder yaml at `ahb/configs/encoders/<name>.yaml` with
   `sample_rate`, `feature_dim`, `num_layers`, `layer_dim`, `max_length`,
   `min_length` and an `encoder: !new:…` construction.
3. Register the encoder in `ahb/configs/registry.yaml` under `encoders:`.
   The key is the `--encoder` value and shows up in experiment folder names.

## Reading results

Each completed `(task, encoder)` job writes:

- `exps/single_task/<task>/<encoder>-AvgTProbe-run1/test_results.txt` — flat
  `key: value` pairs (AUROC, F1, accuracy, AUROC_CI_low/high, MAE, …).
  Cross-validation tasks write `test_results.yaml` with per-fold detail.
- `events.out.tfevents.*` — TensorBoard scalar logs.
- `best_hparams.yaml` — winning hyperparameters from the Ray Tune search.

Aggregate everything into per-metric CSVs:

```bash
python -m ahb summary
ls exps/single_task/_summary/    # AUROC.csv, MAE.csv, completion.csv, …
```

## Cache invalidation

The encoder cache is keyed by `(dataset, encoder, min_length, max_length)`.
If you change any of those, stale entries need to go.

```bash
./invalidate_caches.sh           # dry-run; prints what would be deleted
./invalidate_caches.sh --apply   # actually delete
```

Edit the `ENCODERS` and `DATASETS` arrays at the top of the script to pick
the scope.

## Concurrency

`-j N` on `ahb run` controls how many jobs run in parallel. There is a
per-`(dataset, encoder)` write lock so warm-cache never races; reader-only
jobs (cache already populated) run unblocked and short-circuit the encoder
load via a stub. CLAP requires `-j 1` because of 48 kHz memory pressure.

## Multi-host / SLURM

`run_all_slurm.sh` is the submission template; it sets `HF_HOME`, the Ray
temp dir, and forwards `ENCODER`, `DATASET`, `TASK`, `JOBS`, `TEST_ONLY` env
vars to the orchestrator:

```bash
ENCODER=wavlm,ast DATASET=torgo,ravdess JOBS=4 sbatch run_all_slurm.sh
```

Per-cluster job specs live in `slurm/` (e.g. `slurm/trillium.slurm`).

## Troubleshooting

| Symptom                                       | Likely cause / fix                                                                |
|-----------------------------------------------|-----------------------------------------------------------------------------------|
| Job hangs on "warming cache"                  | Another process holds the write lock; check `logs/` for the active warmer.        |
| `KeyError` on a probe field                   | Probe yaml's `feature_dim` / `num_layers` doesn't match the encoder yaml.         |
| Cache miss after an encoder upgrade           | Run `./invalidate_caches.sh --apply` for the affected `(encoder, dataset)` rows.  |
| `python -m ahb …` fails after pulling main    | `pip install -r requirements.txt` — registry/yaml deps may have moved.            |
