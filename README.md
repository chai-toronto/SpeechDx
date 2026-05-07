
# Audio Health Benchmark

📊 [**Leaderboard**](./leaderboard.csv)

A reproducible benchmark for self-supervised audio encoders on health-related
tasks (depression, dementia, dysarthria, COVID-19, emotion, …). Each task
trains a lightweight probe on top of a frozen encoder and reports AUROC /
macro-AUROC / MAE with bootstrap confidence intervals.

## Quickstart

```bash
git clone <this repo> && cd Audio-Health-Benchmark
uv sync
```

If you don't have uv, a pinned `requirements.txt` is checked in:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 1. Control what dataset + task to work on. `prep` will:
#    - download raw data if the dataset is public (mvdr, ravdess, coswara,
#      torgo) and a `scripts/download_<name>.sh` exists,
#    - else fail loudly with the contact info from registry.yaml,
#    - run metadata_script/create_<name>_metadata.py if processed/<name>.csv
#      is missing,
#    - build the train/valid/test manifests.
#    Filters: `-t/--task`, `-e/--encoder`, `-d/--dataset` (repeatable;
#    empty = "every match in scope"). `single` is the default mode, so
#    `ahb prep ...` ≡ `ahb single prep ...`.
uv run python -m ahb single prep  -t T19
# Drop uv run if run without uv
uv run python -m ahb single warm  -t T19 -e wavlm
uv run python -m ahb single train -t T19 -e wavlm -j 4

# 2. Or `run` does the whole pipeline for matching pairs:
#    download → prep → warm → train → summary. Idempotent — skips
#    pairs whose results already exist; pass --overwrite to redo.
uv run python -m ahb single run -j 4

# 3. … or chain every mode (single → cross → cross-cat → data-eff). 
# When the datasets are in place, this will completely reproduce. 
uv run python -m ahb all run -j 4

# 4. Inspect.
uv run python -m ahb single status         # task × encoder grid of ☑ / ☐ for --tag
uv run python -m ahb single summary        # per-metric CSVs + stdout tables
```

## Modes

The CLI surface is `python -m ahb <mode> <command> [flags]`. If the first
argument is a command rather than a mode, mode defaults to `single` — so
`python -m ahb run` is shorthand for `python -m ahb single run`.

| Mode        | What it does                                                                                                           |
|-------------|------------------------------------------------------------------------------------------------------------------------|
| `single`    | Full benchmark — every paper task × every encoder, single dataset per task. Default mode.                              |
| `cross`     | Zero-shot cross-task — train on dataset A, evaluate on dataset B.                                                      |
| `cross-cat` | Multi-source cross-category variant (reuses single-mode caches under the hood).                                        |
| `data-eff`  | Same tasks at 4 reduced training-set sizes (6.25 %, 12.5 %, 25 %, 50 %); reuses single-mode prep / warm.                |
| `all`       | Chain the command across single → cross → cross-cat → data-eff. Adds `--skip-mode` and `--stop-on-failure` (continues past failures by default). `all train` runs each mode's train phase in parallel via the orchestrator (caches must be warm); `all summary` drops `--out-dir` (collides across modes). |

**Filters** (every command): `-t/--task`, `-e/--encoder`, `-d/--dataset`.
Repeatable; empty = "every match in scope".

## Commands

`run` is a wrapper that chains `warm → train → summary` for matching
pairs. It takes the **union** of those phases' flags so you can drive the
whole pipeline from one call.

### `prep`
Download raw data (if open-access) or fail with the contact info from
`registry.yaml`; run the metadata script if the processed CSV is missing;
build manifests. Idempotent.
- `--overwrite` — rebuild manifests even if they already exist.

### `warm`
Extract embeddings into the per-`(dataset, encoder)` HDF5 cache for every
matching pair. Idempotent.
- `--device` — torch device override (e.g. `cuda:0`).
- `--workers` — concurrent warm workers (default `1`).
- `--overwrite` — wipe `cache.hdf5` and re-extract.

### `train`
Ray-Tune HP search on top of warmed caches. Auto-routes to per-fold CV
when the task yaml sets `num_fold`. Skips pairs whose results already
exist.
- `--tag` — experiment tag (default `run1`); names the output folder.
- `--overrides` — extra YAML string forwarded to `load_hyperpyyaml`.
- `--workers` / `-j` — concurrent train workers (default `3`).
- `--overwrite` — wipe the experiment folder and retrain.
- `--test-only` — re-evaluate the saved best trial without retraining;
  pairs without a trained model are skipped. Mutually exclusive with
  `--overwrite`.
- `single train` only: `--level-dir` — reroute output into
  `exps/data_eff/<level_dir>/`.
- `data-eff train` only: `--level` — restrict to specific levels
  (repeatable; default: every level in `registry.yaml`).

### `run` — `warm` + `train` + `summary`
Strict phases: warm everything → train everything → summarize. Skips
already-complete pairs. Takes every flag from the underlying phases plus
orchestration flags.
- Filters; `--tag`; `--device`.
- Forwarded to train: `--overrides`, `--overwrite`.
- Forwarded to summary: `--out-dir`.
- Concurrency: `--warm-workers` (default `1`), `--train-workers` / `-j`
  (default `3`). Independent pools — warm dominates GPU, train dominates
  CPU.
- `--dry-run` — print the plan and exit without doing work.
- `--overwrite` redo prompts unless `-y`/`--yes`.
- `data-eff run` / `all run` add `--level`.

### `status`
Per-task × per-encoder ☑/☐ completion grid for `--tag`. data-eff and all
add a `level` axis.
- `--tag`.
- `data-eff status` / `all status` only: `--level`.

### `summary`
Aggregate `test_results.{txt,yaml}` into per-metric CSVs at
`<mode-root>/_summary_<tag>/` and pretty-print to stdout.
- `--out-dir` (default `<mode-root>/_summary_<tag>`), `--tag`.
- `data-eff summary` only: `--level`.

### Notes

**Auto-resume:** if `train` finds partial Ray Tune state on disk
(`storage/`, `best_hparams.yaml`) and no completed `test_results`, it
auto-resumes rather than wiping. Pass `--overwrite` to force a fresh
start.

**Cache sharing:** training reads from the per-`(dataset, encoder)` HDF5
cache and will fail on miss. cross / cross-cat warm delegate to
`single warm` for every task on each listed dataset (propagating
`num_aug_ver`), so `single` / `cross` / `cross-cat` runs are each
self-sufficient. `all run`'s single → cross → cross-cat → data-eff
ordering also keeps later phases' cache work cheap because earlier
phases populated the caches.

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
name under `data/` and the prefix used in task ids.

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

**Staging contract.** Once raw data is on disk at `data/<name>/raw/`, a
metadata script copies the audio into `data/<name>/processed/audio/` and
writes the CSV to `data/<name>/processed/<name>.csv`. The CSV must have at
least `uid, Participant_ID, split, label, path` — see
[`metadata_script/README.md`](metadata_script/README.md).

`ahb prep` automates the whole chain: it runs `scripts/download_<name>.sh`
when raw is missing and the dataset is open, then `metadata_script/create_<name>_metadata.py`
when the processed CSV is missing, then builds manifests. Datasets with
restricted access fail loudly with the contact info from
`ahb/configs/registry.yaml` (`datasets.<name>.contact`). Drop the upstream
archive at `data/<name>/raw/` and re-run.


## Adding a task

A task is one (dataset, label) pair — e.g. *T19* (`c9s_t1` internally) trains
binary COVID classification on the COVID-19 Sounds dataset. Paper tasks are
identified by ID (T1…T27) — see `paper_tasks` in `ahb/configs/registry.yaml`
for the full ID-to-(dataset, label) mapping.

1. **Stage the dataset.** Audio under `data/<dataset>/processed/audio/`,
   metadata CSV at `data/<dataset>/processed/<dataset>.csv`. Required CSV
   columns: `uid, Participant_ID, split, label, path` (`path` is relative to
   `processed/audio/`). See [`metadata_script/`](metadata_script/) for
   per-dataset builders.
2. **Write a `prepare_*` function** under `ahb/prep/<dataset>.py`. It builds
   train/valid/test manifests from the metadata CSV. `ahb/prep/c9s.py` and
   `ahb/prep/torgo.py` are the templates.
3. **Add a task yaml** at `ahb/configs/tasks/T<N>.yaml` (next free ID;
   non-paper / scratch tasks may keep descriptive `<dataset>_<task>.yaml`
   stems instead) pointing at the prep function (`data_io_script`,
   `prepare_data_fn`), the label column (`label_key`), the loss, and
   `task_type` (B / C / R / L for binary, multiclass, regression, multilabel).
   The yaml's **stem** drives the experiment folder name
   (`exps/single_task/<stem>/...`), so paper tasks land at
   `exps/single_task/T<N>/` and auxiliary tasks at the descriptive name —
   renaming a stem moves the on-disk results with it.
4. **Register the task** by adding its stem (e.g. `T28`) to `paper_tasks` in
   `ahb/configs/registry.yaml`. That's the single source of truth used by
   every orchestrator.

For per-dataset staging conventions and the full CSV schema, see
[`metadata_script/README.md`](metadata_script/README.md).

## Adding a cross task

A cross task trains on one dataset and evaluates on another (`cross`) or on
multi-source train/test groups (`cross-cat`). The shape mirrors single-mode but
configs live under `ahb/configs/cross_tasks/` and registration goes into a
different list.

1. **Stage both datasets.** Same contract as single — audio under
   `data/<dataset>/processed/audio/`, CSV at
   `data/<dataset>/processed/<dataset>.csv` for *each* dataset the cross task
   touches.
2. **Write a `prepare_*` function** under `ahb/prep/cross_<train>_<test>.py`
   (or extend `ahb/prep/category.py` for cross-cat). It builds the manifests
   by joining the per-dataset CSVs. `ahb/prep/cross_aphasia_dbank.py` and
   `ahb/prep/category.py` are the templates.
3. **Add a cross-task yaml** at `ahb/configs/cross_tasks/<stem>.yaml`. Stems
   follow the paper IDs of the underlying single tasks: pair tasks use
   `T<train>_T<test>` (e.g. `T9_T7`), category tasks use `c<train>_c<test>`
   (e.g. `c2_c3`). The stem also names the experiment folder on disk
   (`exps/cross/T9_T7/...`, `exps/cross_cat/c2_c3/...`).
   - Pair tasks use singular `train_dataset` / `test_dataset` and a combined
     `dataset:` field (e.g. `aphasia_dbank`).
     See [`T9_T7.yaml`](ahb/configs/cross_tasks/T9_T7.yaml).
   - Category tasks use plural `train_datasets` / `test_datasets` lists and
     `setting_1/2/3` blocks; `dataset:` should be `cross_tasks`.
     See [`c1_c2.yaml`](ahb/configs/cross_tasks/c1_c2.yaml).
   Both schemas point at the prep function via `data_io_script` /
   `prepare_data_fn` and set `label_key`, `loss`, and `task_type`.
4. **Register the task** in `ahb/configs/registry.yaml`:
   - Pair tasks → append the stem to `cross_pairs:`.
   - Category tasks → append the stem to `cross_categories:`.
   Unregistered yamls are ignored by `cross run` / `cross-cat run` /
   `status` / `summary`, matching the single-mode `paper_tasks` rule.

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

Aggregate everything into per-metric CSVs (and pretty-print the same
matrices to stdout):

```bash
uv run python -m ahb single summary
ls exps/single_task/_summary_run1/   # AUC.csv, F1.csv, MAE.csv, completion.csv, …
```

`run` invokes the matching mode's `summary` automatically as its third
phase, so a single `ahb single run` ends with the metric tables for the
pairs it just trained printed to stdout.

### Cross-validation tasks

Tasks whose yaml sets `num_fold:` (in `ahb/configs/tasks/<stem>.yaml`) are
routed through per-fold CV training automatically — `ahb single train` and
`ahb single run` both dispatch based on that field
(`ahb/orchestrator.py:is_cv`). Results are aggregated (mean ± std across
folds) into `test_results.yaml`. 

Currently CV-routed (5-fold each):

| Dataset    | Task IDs (paper name)                                                                            |
|------------|--------------------------------------------------------------------------------------------------|
| `iemocap`  | `T5` (iemocap_emoC), `T6` (iemocap_emoBC)                                                        |
| `ravdess`  | `T3` (ravdess_emoC), `T4` (ravdess_emoBC)                                                        |
| `torgo`    | `T10` (torgo_dysC), `T11` (torgo_sevR)                                                           |
| `uaspeech` | `T12` (uaspeech_dysC)                                                                            |
| `mvdr`     | `T13` (mvdr_parkC), `T14` (mvdr_updrs5R), `T15` (mvdr_updrs18R), `T16` (mvdr_hyR)                |
| `ksof`     | `T17` (ksof_intC), `T18` (ksof_stutL)                                                            |

To add or remove a task from this set, toggle `num_fold` in its task yaml —
no orchestrator code changes required.


## Concurrency

`train` owns the short worker flag `--workers` / `-j` (default `3`).
`run` also accepts `-j` as the short alias for `--train-workers`, while
keeping `--warm-workers` for the warm phase. Warm dominates GPU; train
dominates CPU; strict warm-then-train phases let each saturate its
bottleneck. `warm` itself owns a per-`(dataset, encoder)` file lock, so
direct `warm`, delegated cross warm, and separate `run` processes all
serialize on the same cache boundary. Reader-only jobs (cache already
populated) run unblocked and short-circuit the encoder load via a stub.
CLAP requires `--warm-workers 1` because of 48 kHz memory pressure
(already the default).

## Multi-host / SLURM

`run_all_slurm.sh` is the submission template; it sets `HF_HOME`, the Ray
temp dir, and forwards `ENCODER`, `DATASET`, `TASK`, `JOBS`, `TEST_ONLY` env
vars to the orchestrator:

```bash
ENCODER=wavlm,ast DATASET=torgo,ravdess JOBS=4 sbatch run_all_slurm.sh
```

Per-cluster job specs live in `slurm/` (e.g. `slurm/trillium.slurm`).

## TODO: 
  1. Drop the level/single_avg/single cache tag + class-level prune of probe/pool unreachables (cache → v3, drop output_hidden_state outside model wrappers
  2. Paralel cache gen
