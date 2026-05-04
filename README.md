## Larry's TODO: 
  1. Drop the level/single_avg/single cache tag + class-level prune of probe/pool unreachables (cache → v3, drop output_hidden_state outside model wrappers)                                                                                                                            
  2. Do a test only thru run2.
  3. Paralel cache gen
  4. Rerun regression experiments for run1 (Optional, not run2) 
                                                 

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

## Modes

A single CLI entry point (`python -m ahb <subcommand>`) covers four modes:

| Mode             | Subcommands                                            | What it does                                                                |
|------------------|--------------------------------------------------------|-----------------------------------------------------------------------------|
| single           | `prep` / `warm` / `train` / `train-cv` / `run`         | Full benchmark — every paper task × every encoder, single dataset per task. |
| data-eff         | `run-data-eff`                                         | Same tasks at 4 reduced training-set sizes (6.25 %, 12.5 %, 25 %, 50 %).    |
| cross            | `warm-cross` / `train-cross` / `run-cross`             | Zero-shot cross-task — train on dataset A, evaluate on dataset B.           |
| cross-category   | `run-cross-category`                                   | Multi-source cross-task variant for category-level transfer.                |

**Cache warming is a hard prerequisite for every mode** — training reads from
the per-`(dataset, encoder)` HDF5 cache and will fail on miss. Encoder forward
passes dominate wall time, so `ahb run` serializes warm vs read per
`(dataset, encoder)` pair to let multiple jobs share the cache. `run` and
`run-cross` chain warming and training internally (`warm`+`train`,
`warm-cross`+`train-cross`). `run-cross-category` does not: `warm-cross`
short-circuits for category tasks, so the constituent single-task caches must
already be populated via `ahb warm <dataset-task> <encoder>` (or a prior
single-mode `run`) before invoking it. Running the full benchmark via
`ahb run-all` handles the ordering automatically (single → cross → cross-cat
→ data-eff).

## Encoders

12 frozen backbones ship by default. The `Source` column is the identifier
passed to `from_pretrained(...)`; the underlying weight files are downloaded
on first use. Sizes / layer counts / sample rates live in the per-encoder yaml
under [`ahb/configs/encoders/`](ahb/configs/encoders/) — that's the source of
truth, not the table below.

| Name (`--encoder`) | Source                                          | Hub        | Notes                                              |
|--------------------|-------------------------------------------------|------------|----------------------------------------------------|
| `wavlm`            | `microsoft/wavlm-large`                         | HF         | 1024-dim, 24 layers                                |
| `w2v2`             | `facebook/wav2vec2-large-960h-lv60-self`        | HF         | 1024-dim, 24 layers                                |
| `hubert`           | `facebook/hubert-large-ls960-ft`                | HF         | 1024-dim, 24 layers                                |
| `whisper`          | `openai/whisper-large-v3`                       | HF         | 30 s audio, 1280-dim, 32 layers                    |
| `ast`              | `MIT/ast-finetuned-audioset-10-10-0.4593`       | HF         | 10 s audio, 768-dim                                |
| `audiomae`         | `hance-ai/audiomae`                             | HF         | 10 s audio, 768-dim                                |
| `clap`             | `laion/larger_clap_general`                     | HF         | 48 kHz, 10 s, 1024-dim                             |
| `mms`              | `facebook/mms-1b`                               | HF         | 1280-dim, 48 layers                                |
| `wavjepa`          | `labhamlet/wavjepa-nat-base`                    | HF         | 768-dim, 12 layers                                 |
| `qwen3voice`       | `Qwen/Qwen3-TTS-Tokenizer-12Hz`                 | HF         | 24 kHz, 512-dim                                    |
| `emotion2vec`      | `emotion2vec/emotion2vec_plus_large`            | HF + FunASR| Loaded via FunASR; HF mirror of `iic/...` on ModelScope |
| `opera_gt`         | `evelyn0414/OPERA` → `encoder-operaGT.ckpt`     | HF (ckpt)  | Vendored loader at `third_party/OPERA/`            |

### Model revision

| Encoder       | Repo                                              | HEAD commit (as of 2026-05-03)             | Last commit |
|---------------|---------------------------------------------------|--------------------------------------------|-------------|
| `wavlm`       | `microsoft/wavlm-large`                           | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | 2022-02-02  |
| `w2v2`        | `facebook/wav2vec2-large-960h-lv60-self`          | `54074b1c16f4de6a5ad59affb4caa8f2ea03a119` | 2022-05-23  |
| `hubert`      | `facebook/hubert-large-ls960-ft`                  | `ece5fabbf034c1073acae96d5401b25be96709d8` | 2022-05-24  |
| `whisper`     | `openai/whisper-large-v3`                         | `06f233fe06e710322aca913c1bc4249a0d71fce1` | 2024-08-12  |
| `ast`         | `MIT/ast-finetuned-audioset-10-10-0.4593`         | `f826b80d28226b62986cc218e5cec390b1096902` | 2023-09-06  |
| `audiomae`    | `hance-ai/audiomae`                               | `c1379969532da421855d2f225f40c9c7b4959188` | 2024-08-16  |
| `clap`        | `laion/larger_clap_general`                       | `ada0c23a36c4e8582805bb38fec3905903f18b41` | 2023-10-31  |
| `mms`         | `facebook/mms-1b`                                 | `0d2f7adb9903d98894d70ae11f7fbdfc8cb71a69` | 2023-06-05  |
| `wavjepa`     | `labhamlet/wavjepa-nat-base`                      | `15d95ff67fa98117b17e83a1653bbca97877ff6f` | 2025-11-06  |
| `qwen3voice`  | `Qwen/Qwen3-TTS-Tokenizer-12Hz`                   | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` | 2026-01-29  |
| `emotion2vec` | `emotion2vec/emotion2vec_plus_large` (HF)         | `6c303ba987b86b93193de93e34bb2b077a6bedc4` | 2024-06-24  |
| `opera_gt`    | `evelyn0414/OPERA`                                | `d8de4322870b596f0a6ff6ea907b9a6996cd243a` | 2024-11-15  |

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
├── scripts/                Per-dataset download scripts (open-access corpora)
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
├── run_all_slurm.sh        SLURM submission template
└── invalidate_caches.sh    Edit-and-run cache invalidator
```

## Datasets

13 health-speech corpora ship with prep modules and metadata builders. Most
require a license / DTA / EULA — only RAVDESS, Coswara, and MDVR-KCL can be
fetched without contacting the authors. The "Local" column is the directory
name under `data/` and the prefix used in task ids; the "Upstream" column is
the canonical name in the literature.

| Local        | Upstream                                       | Access | Source                                                                                                                        |
|--------------|------------------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------|
| `ravdess`    | RAVDESS (Speech)                               | open   | https://zenodo.org/records/1188976 — `scripts/download_ravdess.sh`                                                             |
| `coswara`    | Project Coswara (IISc)                         | open   | https://github.com/iiscleap/Coswara-Data — `scripts/download_coswara.sh`                                                       |
| `mvdr`       | MDVR-KCL (King's College London + Fraunhofer)  | open   | https://zenodo.org/records/2867216 — `scripts/download_mvdr.sh` (CC BY 4.0)                                                    |
| `ksof`       | Kassel State of Fluency                        | EULA   | https://zenodo.org/records/6801844 — sign EULA at https://th-nuernberg.github.io/kassel-state-of-fluency/                     |
| `torgo`      | TORGO Database of Dysarthric Articulation      | open   | http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html (pending cluster shutdown to verify)                             |
| `uaspeech`   | UASpeech                                       | email  | https://speechtechnology.web.illinois.edu/uaspeech/ — request via uaspeech-requests@lists.illinois.edu                        |
| `iemocap`    | IEMOCAP                                        | release form | https://sail.usc.edu/iemocap/ — academic release form to USC SAIL                                                             |
| `dbank`      | DementiaBank ADReSS-M (ICASSP 2023 SPGC)       | DTA    | https://luzs.gitlab.io/madress-2023/ — request via madress2023@ed.ac.uk; data on TalkBank                                     |
| `aphasia`    | AphasiaBank (TalkBank)                         | registration | https://aphasia.talkbank.org/ — TalkBank account; some sub-corpora (APROCSA, Dysphagia) require extra approval                |
| `edaic`      | E-DAIC (AVEC 2019 / DAIC-WOZ extended)         | DTA    | https://dcapswoz.ict.usc.edu/ — academic form to USC ICT                                                                      |
| `c9s`        | COVID-19 Sounds (Cambridge)                    | DTA    | https://covid-19-sounds.org/ — DTA via covid-19-sounds@cl.cam.ac.uk                                                           |
| `avfad`      | Advanced Voice Function Assessment Database    | email  | https://acsa.web.ua.pt/AVFAD.htm — request via ieeta-acsa@ua.pt                                                               |

**Staging contract.** Once raw data is on disk, a metadata script copies the
audio into `data/<name>/processed/audio/` and writes the CSV to
`data/<name>/processed/<name>.csv`. The CSV must have at least
`uid, Participant_ID, split, label, path` — see
[`metadata_script/README.md`](metadata_script/README.md).

All `create_<name>_metadata.py` scripts read from `data/<name>/raw/` by
default. Drop the upstream archive there (or run the matching
`scripts/download_*.sh` for the open ones), then run the metadata script —
it stages audio into `processed/audio/` and writes the CSV.


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

For per-dataset staging conventions and the full CSV schema, see
[`metadata_script/README.md`](metadata_script/README.md).

## Adding an encoder

1. Implement the encoder under `model/<name>.py`. Contract: an `nn.Module`
   whose `forward(waveform, lengths=...)` returns `(B, T, D)` (or a tuple of
   per-layer `(B, T, D)` when `output_hidden_states=True`). `lengths` is a
   `(B,)` tensor of **relative** lengths in `[0, 1]` (fraction of the padded
   batch length), matching the SpeechBrain convention. See `model/wavlm.py`
   for the minimal pattern.
2. Add an encoder yaml at `ahb/configs/encoders/<name>.yaml` with
   `sample_rate`, `feature_dim`, `num_layers`, `layer_dim`, `max_length`,
   `min_length` and an `encoder: !new:…` construction.
3. Register the encoder in `ahb/configs/registry.yaml` under `encoders:`.
   The key is the `--encoder` value and shows up in experiment folder names.

For the full encoder / probe / pool contracts and additional examples, see
[`model/README.md`](model/README.md).

## Reading results

Each completed `(task, encoder)` job writes:

- `exps/single_task/<task>/<encoder>-AvgTProbe-run1/test_results.txt` — flat
  `key: value` pairs (AUROC, F1, accuracy, AUROC_CI_low/high, MAE, …).
  Cross-validation tasks (see below) write `test_results.yaml` with per-fold detail.
- `events.out.tfevents.*` — TensorBoard scalar logs.
- `best_hparams.yaml` — winning hyperparameters from the Ray Tune search.

Aggregate everything into per-metric CSVs:

```bash
python -m ahb summary
ls exps/single_task/_summary/    # AUROC.csv, MAE.csv, completion.csv, …
```

### Cross-validation tasks

Tasks whose yaml sets `num_fold:` (in `ahb/configs/tasks/<stem>.yaml`) are
routed through `ahb train-cv` instead of `ahb train` — `ahb run` dispatches
automatically based on that field (`ahb/orchestrator.py:is_cv`). Results are
aggregated (mean ± std across folds) into `test_results.yaml`. 

Currently CV-routed (5-fold each):

| Dataset    | Tasks                                            |
|------------|--------------------------------------------------|
| `iemocap`  | `iemocap_emoC`, `iemocap_emoBC`                  |
| `ravdess`  | `ravdess_emoC`, `ravdess_emoBC`                  |
| `torgo`    | `torgo_dysC`, `torgo_sevR`                       |
| `uaspeech` | `uaspeech_dysC`                                  |
| `mvdr`     | `mvdr_parkC`, `mvdr_hyR`, `mvdr_updrs5R`, `mvdr_updrs18R` |
| `ksof`     | `ksof_intC`, `ksof_stutL`                        |

To add or remove a task from this set, toggle `num_fold` in its task yaml —
no orchestrator code changes required.


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

