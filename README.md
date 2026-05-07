
# SpeechDx

[**Leaderboard**](./leaderboard.csv)

A reproducible benchmark for self-supervised audio encoders on health-related
tasks (depression, dementia, dysarthria, COVID-19, emotion, …). Each task
trains a lightweight probe on top of a frozen encoder and reports AUROC /
macro-AUROC / MAE with bootstrap confidence intervals.

## Contents

- [Install](#install)
- [Reproducing the full benchmark](#reproducing-the-full-benchmark)
- [Modes](#modes)
- [Commands](#commands) — [`prep`](#prep), [`warm`](#warm), [`train`](#train), [`run`](#run--warm--train--summary), [`status`](#status), [`summary`](#summary)
- [Examples](#examples)
- [Encoders](#encoders)
- [Datasets](#datasets)
- [Adding a task](#adding-a-task) — [Cross-validation tasks](#cross-validation-tasks)
- [Adding a cross task](#adding-a-cross-task)
- [Adding an encoder](#adding-an-encoder)
- [Notes](#notes)
- [Concurrency](#concurrency)
- [Multi-host / SLURM](#multi-host--slurm)
- [Repository layout](#repository-layout)

## Install

```bash
git clone <this repo> && cd SpeechDx
uv sync
```

If you don't have uv, a pinned `requirements.txt` is checked in:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Drop the `uv run` prefix from every command below if you went the venv route.)

## Reproducing the full benchmark

End-to-end, single command, all four [modes](#modes):

```bash
# Stage every dataset the registry knows about (open ones auto-download;
# restricted ones print a warning + skip
uv run python -m sdx all prep

# Run every (task, encoder) pair across single → cross → cross-cat → data-eff.
# Skips pairs whose results already exist. -j is train workers.
uv run python -m sdx all run -j 4
```

`all run` chains `warm → train → summary` for each of the four modes in
order. It survives partial failures by default (pass
`--stop-on-failure` to abort on the first one).

### Where the results land

Each `(task, encoder)` writes into `exps/<mode-root>/<task>/<encoder>-AvgTProbe-run1/`:

- `test_results.txt` — flat `key: value` (AUROC, F1, accuracy, AUROC_CI_low/high, MAE, …).
- `test_results.yaml` — per-fold detail for CV tasks (see [Cross-validation tasks](#cross-validation-tasks)).
- `events.out.tfevents.*` — TensorBoard scalar logs.
- `best_hparams.yaml` — winning Ray-Tune trial.

Mode roots: `exps/single_task/`, `exps/cross/`, `exps/cross_cat/`, `exps/data_eff/<level>/`.

### Aggregating and inspecting results

`run` already invokes `summary` as its last phase, but you can re-aggregate any time:

```bash
uv run python -m sdx single   summary    # → exps/single_task/_summary_run1/
uv run python -m sdx cross    summary    # → exps/cross/_summary_run1/
uv run python -m sdx cross-cat summary   # → exps/cross_cat/_summary_run1/
uv run python -m sdx data-eff summary    # → exps/data_eff/_summary_run1/<level>/
uv run python -m sdx all      summary    # all four, stacked

uv run python -m sdx all status          # ☑/☐ completion grid for --tag (default run1)
```

Each `_summary_<tag>/` directory holds per-metric CSVs (`AUC.csv`,
`AUC_CI.csv`, `Acc.csv`, `F1.csv`, `MAE.csv`, `MAE_CI.csv`, `MSE.csv`,
`PearsonR.csv`, `R2.csv`, `completion.csv`) — rows are tasks, columns are
encoders. The same matrices are pretty-printed to stdout at the end of
the command.

## Modes

The CLI surface is `python -m sdx <mode> <command> [flags]`. If the first
argument is a command rather than a mode, mode defaults to `single` — so
`python -m sdx run` is shorthand for `python -m sdx single run`.

| Mode        | What it does                                                                                                           |
|-------------|------------------------------------------------------------------------------------------------------------------------|
| `single`    | Full benchmark — every paper task × every encoder, single dataset per task. Default mode.                              |
| `cross`     | Zero-shot cross-task — train on dataset A, evaluate on dataset B.                                                      |
| `cross-cat` | Multi-source cross-category variant.                                                                                   |
| `data-eff`  | Same tasks at 4 reduced training-set sizes (6.25 %, 12.5 %, 25 %, 50 %).                                               |
| `all`       | Chain the command across single → cross → cross-cat → data-eff. Adds `--skip-mode` and `--stop-on-failure` (continues past failures by default). `all train` runs each mode's train phase in parallel via the orchestrator (caches must be warm); `all summary` drops `--out-dir` (collides across modes). |

**Filters** (every command): `-t/--task`, `-e/--encoder`, `-d/--dataset`.
Repeatable; empty = "every match in scope".

## Commands

`run` is a wrapper that chains `warm → train → summary` for matching
pairs. It takes the **union** of those phases' flags so you can drive the
whole pipeline from one call.

All commands skip pairs whose output already
exists, so reruns only do the missing work. Pass `--overwrite` to force
a redo.

### `prep`
Download raw data (if open-access) or fail with the contact info from
`registry.yaml`; run the metadata script if the processed CSV is missing;
build manifests.
- `--overwrite` — rebuild manifests even if they already exist.

### `warm`
Extract embeddings into the per-`(dataset, encoder)` HDF5 cache for every
matching pair.
- `--device` — torch device override (e.g. `cuda:0`).
- `--workers` — concurrent warm workers (default `1`).
- `--overwrite` — wipe `cache.hdf5` and re-extract.

### `train`
Ray-Tune HP search on top of warmed caches. Auto-routes to per-fold CV
when the task yaml sets `num_fold`.
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
Strict phases: warm everything → train everything → summarize. Takes
every flag from the underlying phases plus orchestration flags.
- Filters; `--tag`; `--device`.
- Forwarded to train: `--overrides`, `--overwrite`.
- Forwarded to summary: `--out-dir`.
- Concurrency: `--warm-workers` (default `1`), `--train-workers` / `-j`
  (default `3`). Independent pools — see [Concurrency](#concurrency).
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

## Examples

Each block below shows a typical (mode, command) combination, the call,
and a representative slice of its output.

### Single task × single encoder

Stage one dataset, warm one cache, train one probe — fastest way to
sanity-check a fresh checkout.

```bash
uv run python -m sdx single run -t T19 -e wavlm -j 4
```

Final stdout (the `summary` phase):

```
=== AUC.csv ===
task                | wavlm
--------------------+-------
T19 (c19sounds_T19) | 0.6810

=== AUC_CI.csv ===
task                | wavlm
--------------------+-----------------
T19 (c19sounds_T19) | (0.6533, 0.7116)

=== F1.csv ===
task                | wavlm
--------------------+-------
T19 (c19sounds_T19) | 0.6528

=== Acc.csv ===
task                | wavlm
--------------------+-------
T19 (c19sounds_T19) | 0.6237

Parsed 1 result files, 0 missing
Wrote 5 CSV(s) to exps/single_task/_summary_run1/
```

The same numbers land in [exps/single_task/T19/wavlm-AvgTProbe-run1/test_results.txt](exps/single_task/T19/wavlm-AvgTProbe-run1/test_results.txt):

```
AUROC: 0.6810
AUROC_CI_low: 0.6533
AUROC_CI_high: 0.7116
F1: 0.6528
accuracy: 0.6237
```

### Full single benchmark (all paper tasks × all 12 encoders)

```bash
uv run python -m sdx single run -j 4
```

The summary phase prints the same per-metric tables as the above test —
just wider (one column per encoder, one row per task). With every paper
task and encoder available the trailing line reads:

```
Parsed 324 result files, 0 missing
Wrote 10 CSV(s) to exps/single_task/_summary_run1/
```

(9 metric CSVs — `AUC.csv`, `AUC_CI.csv`, `F1.csv`, `Acc.csv`,
`MAE.csv`, `MAE_CI.csv`, `MSE.csv`, `PearsonR.csv`, `R2.csv` — plus
`completion.csv`. Classification metrics only get rows for B/C/M tasks;
regression metrics only get rows for R tasks.)

### Regression task (verbatim regression-side output)

A B-task only fills `AUC.csv` / `AUC_CI.csv` / `F1.csv` / `Acc.csv`. An
R-task fills `MAE.csv` / `MAE_CI.csv` / `MSE.csv` / `PearsonR.csv` /
`R2.csv` instead. Real run, two encoders against `T8` (DementiaBank
MMSE regression):

```bash
uv run python -m sdx single run -t T8 -e wavlm whisper -j 4
```

Final stdout:

```
=== MAE.csv ===
task                 | wavlm  | whisper
---------------------+--------+--------
T8 (dementiabank_T8) | 8.5847 | 9.9750

=== MAE_CI.csv ===
task                 | wavlm            | whisper
---------------------+------------------+-------------------
T8 (dementiabank_T8) | (7.2261, 9.8711) | (8.6044, 11.3337)

=== MSE.csv ===
task                 | wavlm   | whisper
---------------------+---------+---------
T8 (dementiabank_T8) | 93.1460 | 120.2364

=== PearsonR.csv ===
task                 | wavlm   | whisper
---------------------+---------+---------
T8 (dementiabank_T8) | -0.2148 | -0.2244

=== R2.csv ===
task                 | wavlm   | whisper
---------------------+---------+---------
T8 (dementiabank_T8) | -2.3907 | -3.3769

Parsed 2 result files, 0 missing
Wrote 6 CSV(s) to exps/single_task/_summary_run1/
```

### Cross-task (zero-shot transfer)

Train on one dataset, evaluate on another. Stems are `T<train>_T<test>`
for pairs and `c<train>_c<test>` for category groups.

```bash
uv run python -m sdx cross run -t T13_T10 -e wavlm hubert
```

Final stdout — `T13` (mdvr_parkC) trains the probe, `T10` (torgo_dysC)
evaluates it:

```
=== AUC.csv ===
task                         | wavlm  | hubert
-----------------------------+--------+-------
T13_T10 (mdvr_torgo_T13_T10) | 0.4828 | 0.5642

=== AUC_CI.csv ===
task                         | wavlm            | hubert
-----------------------------+------------------+-----------------
T13_T10 (mdvr_torgo_T13_T10) | (0.3732, 0.5720) | (0.4878, 0.6269)

=== F1.csv ===
task                         | wavlm  | hubert
-----------------------------+--------+-------
T13_T10 (mdvr_torgo_T13_T10) | 0.0225 | 0.0559

=== Acc.csv ===
task                         | wavlm  | hubert
-----------------------------+--------+-------
T13_T10 (mdvr_torgo_T13_T10) | 0.6589 | 0.6594

Parsed 2 result files, 0 missing
Wrote 5 CSV(s) to exps/cross/_summary_run1/
```

### Cross-category (multi-source train/test groups)

```bash
uv run python -m sdx cross-cat run -t c2_c3 -e wavlm
```

The probe is trained on the union of one category's datasets and
evaluated on another's. Stdout matches `cross`, with `cN_cM`-style
stems:

```
=== AUC.csv ===
task                   | wavlm
-----------------------+-------
c2_c3 (category_c2_c3) | 0.7288

=== AUC_CI.csv ===
task                   | wavlm
-----------------------+-----------------
c2_c3 (category_c2_c3) | (0.7228, 0.7347)

=== F1.csv ===
task                   | wavlm
-----------------------+-------
c2_c3 (category_c2_c3) | 0.6845

=== Acc.csv ===
task                   | wavlm
-----------------------+-------
c2_c3 (category_c2_c3) | 0.5419

Parsed 1 result files, 0 missing
Wrote 5 CSV(s) to exps/cross_cat/_summary_run1/
```

### Data-efficiency sweep

Re-runs `train` four times per pair at 6.25 / 12.5 / 25 / 50 % of
training data.

```bash
uv run python -m sdx data-eff run -t T19 -e wavlm --level 06p25 50
```

Unlike the other modes, `data-eff summary` does not print the metric
tables — only one parsed/missing line per active level, then the
totals (per-level matrices land on disk):

```
  level 06p25: parsed 1, missing 0
  level 50: parsed 1, missing 0

Parsed 2 result files, 0 missing
Wrote 15 CSV(s) under exps/data_eff/_summary_run1/
```

(5 CSVs per level — `AUC.csv`, `AUC_CI.csv`, `F1.csv`, `Acc.csv`,
`completion.csv` — × 2 levels = 10, plus 5 combined progression CSVs at
the root.) The combined progression CSV (e.g.
`exps/data_eff/_summary_run1/AUC.csv`) has columns
`(encoder × level)` so you can read accuracy vs. training-set size off
a single row.

### Status grid

```bash
uv run python -m sdx single status -t T19 -e wavlm whisper
```

```
task                | wavlm | whisper
--------------------+-------+--------
T19 (c19sounds_T19) | ☑     | ☑

Legend: ☑ complete, ☐ incomplete
Summary (single): 2/2 complete, 0 remaining
```

### Re-evaluate without retraining

Loads the saved best trial, runs the test loop, overwrites
`test_results.txt`. Pairs without a trained model are skipped.

```bash
uv run python -m sdx single train --test-only -t T19 -e wavlm
```

```
sdx.train - Test stats: AUROC: 6.81e-01 - F1: 6.53e-01 - accuracy: 6.24e-01
```

[exps/single_task/T19/wavlm-AvgTProbe-run1/test_results.txt](exps/single_task/T19/wavlm-AvgTProbe-run1/test_results.txt)

```
AUROC: 0.6810
AUROC_CI_low: 0.6533
AUROC_CI_high: 0.7116
F1: 0.6528
accuracy: 0.6237
```

## Encoders

12 frozen backbones ship by default. The `Source` column is the identifier
passed to `from_pretrained(...)`; the underlying weight files are downloaded
on first use. Sizes / layer counts / sample rates live in the per-encoder yaml
under [`sdx/configs/encoders/`](sdx/configs/encoders/) — that's the source of
truth, not the table below.

| Name (`--encoder`) | Source                                          | HEAD commit (as of 2026-05-03)             | Last commit |
|--------------------|-------------------------------------------------|--------------------------------------------|-------------|
| `wavlm`            | `microsoft/wavlm-large`                         | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | 2022-02-02  |
| `w2v2`             | `facebook/wav2vec2-large-960h-lv60-self`        | `54074b1c16f4de6a5ad59affb4caa8f2ea03a119` | 2022-05-23  |
| `hubert`           | `facebook/hubert-large-ls960-ft`                | `ece5fabbf034c1073acae96d5401b25be96709d8` | 2022-05-24  |
| `whisper`          | `openai/whisper-large-v3`                       | `06f233fe06e710322aca913c1bc4249a0d71fce1` | 2024-08-12  |
| `ast`              | `MIT/ast-finetuned-audioset-10-10-0.4593`       | `f826b80d28226b62986cc218e5cec390b1096902` | 2023-09-06  |
| `audiomae`         | `hance-ai/audiomae`                             | `c1379969532da421855d2f225f40c9c7b4959188` | 2024-08-16  |
| `clap`             | `laion/larger_clap_general`                     | `ada0c23a36c4e8582805bb38fec3905903f18b41` | 2023-10-31  |
| `mms`              | `facebook/mms-1b`                               | `0d2f7adb9903d98894d70ae11f7fbdfc8cb71a69` | 2023-06-05  |
| `wavjepa`          | `labhamlet/wavjepa-nat-base`                    | `15d95ff67fa98117b17e83a1653bbca97877ff6f` | 2025-11-06  |
| `qwen3voice`       | `Qwen/Qwen3-TTS-Tokenizer-12Hz`                 | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` | 2026-01-29  |
| `emotion2vec`      | `emotion2vec/emotion2vec_plus_large`            | `6c303ba987b86b93193de93e34bb2b077a6bedc4` | 2024-06-24  |
| `opera_gt`         | `evelyn0414/OPERA` → `encoder-operaGT.ckpt`     | `d8de4322870b596f0a6ff6ea907b9a6996cd243a` | 2024-11-15  |

For hub source, dimensions, layer counts, and per-encoder loading notes, see [models.md](models.md).

## Datasets

13 health-speech corpora ship with prep modules and metadata builders. Most
require a license / DTA / EULA — only RAVDESS, Coswara, and MDVR-KCL can be
fetched without contacting the authors. The "Local" column is the directory
name under `data/` and the prefix used in task ids.

| Local        | Upstream                                       | Access | Source                                                                                                                        |
|--------------|------------------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------|
| `ravdess`    | RAVDESS (Speech)                               | open   | https://zenodo.org/records/1188976 — `scripts/download_ravdess.sh`                                                             |
| `coswara`    | Project Coswara (IISc)                         | open   | https://github.com/iiscleap/Coswara-Data — `scripts/download_coswara.sh`                                                       |
| `mdvr`       | MDVR-KCL (King's College London + Fraunhofer)  | open   | https://zenodo.org/records/2867216 — `scripts/download_mdvr.sh` (CC BY 4.0)                                                    |
| `ksof`       | Kassel State of Fluency                        | EULA   | https://zenodo.org/records/6801844 — sign EULA at https://th-nuernberg.github.io/kassel-state-of-fluency/                     |
| `torgo`      | TORGO Database of Dysarthric Articulation      | open   | http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html (pending cluster shutdown to verify)                             |
| `uaspeech`   | UASpeech                                       | email  | https://speechtechnology.web.illinois.edu/uaspeech/ — request via uaspeech-requests@lists.illinois.edu                        |
| `iemocap`    | IEMOCAP                                        | release form | https://sail.usc.edu/iemocap/ — academic release form to USC SAIL                                                             |
| `dementiabank`      | DementiaBank ADReSS-M (ICASSP 2023 SPGC)       | DTA    | https://luzs.gitlab.io/madress-2023/ — request via madress2023@ed.ac.uk; data on TalkBank                                     |
| `aphasia`    | AphasiaBank (TalkBank)                         | registration | https://aphasia.talkbank.org/ — TalkBank account; some sub-corpora (APROCSA, Dysphagia) require extra approval                |
| `edaic`      | E-DAIC (AVEC 2019 / DAIC-WOZ extended)         | DTA    | https://dcapswoz.ict.usc.edu/ — academic form to USC ICT                                                                      |
| `c19sounds`        | COVID-19 Sounds (Cambridge)                    | DTA    | https://covid-19-sounds.org/ — DTA via covid-19-sounds@cl.cam.ac.uk                                                           |
| `avfad`      | Advanced Voice Function Assessment Database    | email  | https://acsa.web.ua.pt/AVFAD.htm — request via ieeta-acsa@ua.pt                                                               |

**Staging contract.** Once raw data is on disk at `data/<name>/raw/`, a
metadata script copies the audio into `data/<name>/processed/audio/` and
writes the CSV to `data/<name>/processed/<name>.csv`. The CSV must have at
least `uid, Participant_ID, split, label, path` — see
[`metadata_script/README.md`](metadata_script/README.md).

`sdx prep` automates the whole chain: it runs `scripts/download_<name>.sh`
when raw is missing and the dataset is open, then `metadata_script/create_<name>_metadata.py`
when the processed CSV is missing, then builds manifests.

**Missing data → soft skip with a warning.** Every paper task and cross
task in [`sdx/configs/registry.yaml`](sdx/configs/registry.yaml) is
enabled. Each `run` / `prep` invocation runs a pre-flight check
([`sdx/prep/raw.py:dataset_available`](sdx/prep/raw.py)) — a dataset
counts as "available" if `data/<name>/raw/` is staged, the processed
CSV is on disk, *or* a `scripts/download_<name>.sh` exists. Tasks whose
datasets are unavailable are dropped from the run with one warning per
missing dataset:

```
[12:34:56] SKIP (no data)         : T7 (missing: dementiabank)
…
  ⚠ 'dementiabank': not staged. Contact: madress2023@ed.ac.uk (https://luzs.gitlab.io/madress-2023/)
```

`sdx all run` therefore proceeds against whatever is staged — out of
the box that's the four open-access corpora (`ravdess`, `coswara`,
`mdvr`, `torgo`). To bring a restricted dataset into scope: obtain it
via the contacts in the table above, drop the archive at
`data/<name>/raw/` (or a pre-built CSV at `data/<name>/processed/<name>.csv`),
and re-run — no registry edits required. The `exclude_datasets:` list
in `registry.yaml` is an emergency hatch for forcing skips even when
data *is* staged, and is empty by default.


## Adding a task

A task is one (dataset, label) pair — e.g. *T19* (`c19sounds_t1` internally) trains
binary COVID classification on the COVID-19 Sounds dataset. Paper tasks are
identified by ID (T1…T27) — see `paper_tasks` in `sdx/configs/registry.yaml`
for the full ID-to-(dataset, label) mapping.

1. **Stage the dataset.** See the [Datasets](#datasets) section for the
   on-disk paths and CSV schema; per-dataset builders live in
   [`metadata_script/`](metadata_script/).
2. **Write a `prepare_*` function** under `sdx/prep/<dataset>.py`. It builds
   train/valid/test manifests from the metadata CSV. `sdx/prep/c19sounds.py` and
   `sdx/prep/torgo.py` are the templates.
3. **Add a task yaml** at `sdx/configs/tasks/T<N>.yaml` (next free ID;
   non-paper / scratch tasks may keep descriptive `<dataset>_<task>.yaml`
   stems instead) pointing at the prep function (`data_io_script`,
   `prepare_data_fn`), the label column (`label_key`), the loss, and
   `task_type` (B / C / R / L for binary, multiclass, regression, multilabel).
   The yaml's **stem** drives the experiment folder name
   (`exps/single_task/<stem>/...`), so paper tasks land at
   `exps/single_task/T<N>/` and auxiliary tasks at the descriptive name —
   renaming a stem moves the on-disk results with it.
4. **Register the task** by adding its stem (e.g. `T28`) to `paper_tasks` in
   `sdx/configs/registry.yaml`. That's the single source of truth used by
   every orchestrator.

For per-dataset staging conventions and the full CSV schema, see
[`metadata_script/README.md`](metadata_script/README.md).

### Cross-validation tasks

Tasks whose yaml sets `num_fold:` (in `sdx/configs/tasks/<stem>.yaml`) are
routed through per-fold CV training automatically — `sdx single train` and
`sdx single run` both dispatch based on that field
([sdx/orchestrator.py](sdx/orchestrator.py) `is_cv`). Results are
aggregated (mean ± std across folds) into `test_results.yaml`.

Currently CV-routed (5-fold each):

| Dataset    | Task IDs (paper name)                                                                            |
|------------|--------------------------------------------------------------------------------------------------|
| `iemocap`  | `T5` (iemocap_emoC), `T6` (iemocap_emoBC)                                                        |
| `ravdess`  | `T3` (ravdess_emoC), `T4` (ravdess_emoBC)                                                        |
| `torgo`    | `T10` (torgo_dysC), `T11` (torgo_sevR)                                                           |
| `uaspeech` | `T12` (uaspeech_dysC)                                                                            |
| `mdvr`     | `T13` (mdvr_parkC), `T14` (mdvr_updrs5R), `T15` (mdvr_updrs18R), `T16` (mdvr_hyR)                |
| `ksof`     | `T17` (ksof_intC), `T18` (ksof_stutL)                                                            |

To add or remove a task from this set, toggle `num_fold` in its task yaml —
no orchestrator code changes required.

## Adding a cross task

A cross task trains on one dataset and evaluates on another (`cross`) or on
multi-source train/test groups (`cross-cat`). The shape mirrors single-mode but
configs live under `sdx/configs/cross_tasks/`, registration goes into a
different list, and the cross / cross-cat yamls now own every warm-time
knob themselves (no fall-through to single-task yamls).

1. **Stage every dataset** the cross task touches (see [Datasets](#datasets)
   for the staging contract).
2. **Write a `prepare_*` function** under `sdx/prep/cross_<train>_<test>.py`
   (or extend `sdx/prep/category.py` for cross-cat). It builds the manifests
   by joining the per-dataset CSVs. `sdx/prep/cross_aphasia_dementiabank.py`
   and `sdx/prep/category.py` are the templates.
3. **Add a cross-task yaml** at `sdx/configs/cross_tasks/<stem>.yaml`. Stems
   follow the paper IDs of the underlying single tasks: pair tasks use
   `T<train>_T<test>` (e.g. `T9_T7`), category tasks use `c<train>_c<test>`
   (e.g. `c2_c3`). The stem names the experiment folder on disk
   (`exps/cross/T9_T7/...`, `exps/cross_cat/c2_c3/...`).

   **Cross pair schema** ([`T9_T7.yaml`](sdx/configs/cross_tasks/T9_T7.yaml)):
   - `train_dataset`, `test_dataset` (singular), combined `dataset:` field
     (e.g. `aphasia_dementiabank`).
   - `data_io_script`, `prepare_data_fn`, `label_key`, `loss`, `task_type`.
   - **Train-side warm knobs** (only the train cache is augmented):
     `num_aug_ver`, `snr_low`, `snr_high`, `speed`,
     `train_split_by_boundary`.
   - **Test-side warm knob:** `test_split_by_boundary`. Test cache is
     always 1 version, no augmentation.

   **Cross-cat schema** ([`c1_c2.yaml`](sdx/configs/cross_tasks/c1_c2.yaml)):
   - `train_datasets` / `test_datasets` (plural lists); `dataset: cross_tasks`.
   - `data_io_script`, `prepare_data_fn`, `label_key`, `loss`, `task_type`.
   - One **`setting_<N>` block per train dataset** (idx-aligned with
     `train_datasets`), each carrying `num_aug_ver`, `split_by_boundary`,
     `snr_low`, `snr_high`, `speed` for that dataset's caches.
   - One **`test_setting_<N>` block per test dataset** (idx-aligned with
     `test_datasets`), each carrying `split_by_boundary`. Test caches are
     always 1 version, no augmentation.

   The cross / cross-cat warmers read these blocks directly — they no longer
   fall back to `tasks/T<n>.yaml`. If you want to mirror a single task's
   augmentation defaults (`num_aug_ver`, `snr_*`, `speed`), copy them into
   the cross yaml when authoring it.
4. **Register the task** in `sdx/configs/registry.yaml`:
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
2. Add an encoder yaml at `sdx/configs/encoders/<name>.yaml` with
   `sample_rate`, `feature_dim`, `num_layers`, `layer_dim`, `max_length`,
   `min_length` and an `encoder: !new:…` construction.
3. Register the encoder in `sdx/configs/registry.yaml` under `encoders:`.
   The key is the `--encoder` value and shows up in experiment folder names.

For the full encoder / probe / pool contracts and additional examples, see
[`model/README.md`](model/README.md).

## Notes

**Auto-resume:** if `train` finds partial Ray Tune state on disk
(`storage/`, `best_hparams.yaml`) and no completed `test_results`, it
auto-resumes rather than wiping. Pass `--overwrite` to force a fresh
start.

**Cache sharing.** Training reads from the per-`(dataset, encoder)` HDF5
cache and will fail on miss. The on-disk layout is
`<slurm_tmpdir>/<dataset>/<encoder>/{train,val}/single_avg/cache.hdf5`,
shared across all modes that touch that `(dataset, encoder)` pair.

Each mode's warmer is self-contained — it reads its own task yaml +
mode-specific `main*.yaml` only:
- `single warm` reads `tasks/<stem>.yaml` + `main.yaml`. Writes train
  (`num_aug_ver` augmented versions) and val (1 unaugmented).
- `cross warm` reads `cross_tasks/<stem>.yaml` + `main_cross.yaml`. Per
  cross pair, writes 3 caches: `<train_dataset>/{train,val}` (using
  `num_aug_ver` and `train_split_by_boundary`) and `<test_dataset>/val`
  (1 version, `test_split_by_boundary`). Test side is never augmented.
- `cross-cat warm` reads `cross_tasks/<stem>.yaml` +
  `main_cross_category.yaml`. Per train dataset (`setting_<N>`) writes
  `<dataset>/{train,val}`; per test dataset (`test_setting_<N>`) writes
  `<dataset>/val`.

Caches append-extend: if a previous mode wrote 3 versions and a later
mode wants 5, only the missing 2 versions get computed. `all run`'s
`single → cross → cross-cat → data-eff` ordering takes advantage of
this — later phases skip whatever earlier phases populated.

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


## Repository layout

```
.
├── sdx/                    Harness — CLI, orchestrators, prep, dataio, brain
│   ├── cli.py              python -m sdx …
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
├── embeddings_avg_final/   Pre-computed encoder caches (HDF5, gitignored)
├── logs/                   Per-run training logs
```
<!-- ## TODO: 
  1. Drop the level/single_avg/single cache tag + class-level prune of probe/pool unreachables (cache → v3, drop output_hidden_state outside model wrappers
  2. Paralel cache gen -->
