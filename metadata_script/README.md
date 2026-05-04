# `metadata_script/` — per-dataset metadata builders

Each dataset has a one-shot script that consumes the upstream CSVs and
audio, copies the audio into the canonical layout, and writes a single
metadata CSV that the training pipeline understands.

| Script                                | Dataset       | Output CSV                                      |
|---------------------------------------|---------------|-------------------------------------------------|
| `create_aphasia_metadata.py`          | aphasia       | `data/aphasia/processed/aphasia.csv`            |
| `create_audio_metadata.py`            | (generic)     | helper for ad-hoc audio folders                 |
| `create_avfad_metadata.py`            | avfad         | `data/avfad/processed/avfad.csv`                |
| `create_c9s_metadata.py`              | c9s           | `data/c9s/processed/c9s.csv` (unified t1+t2 splits) |
| `create_coswara_metadata.py`          | coswara       | `data/coswara/processed/coswara.csv`            |
| `create_dbank_metadata.py`            | DementiaBank (ADReSS-M) | `data/dbank/processed/dbank.csv`      |
| `create_daic_woz_metadata.py`         | DAIC-WOZ      | unused — superseded by EDAIC                    |
| `create_edaic_metadata.py`            | EDAIC-WOZ     | `data/edaic/processed/edaic.csv`                |
| `create_iemocap_metadata.py`          | IEMOCAP       | `data/iemocap/processed/iemocap.csv`            |
| `create_ksof_metadata.py`             | KSoF          | `data/ksof/processed/ksof.csv`                  |
| `create_mvdr_metadata.py`             | MVDR          | `data/mvdr/processed/mvdr.csv`                  |
| `create_nemours_metadata.py`          | Nemours       | `data/nemours/processed/nemours.csv`            |
| `create_ravdess_metadata.py`          | RAVDESS       | `data/ravdess/processed/ravdess.csv`            |
| `create_torgo_metadata.py`            | TORGO         | `data/torgo/processed/torgo.csv`                |
| `create_uaspeech_csv.py`              | UASpeech      | `data/uaspeech/processed/uaspeech.csv`          |

`convert_to_soundfile_compat.py` is helpful for audio formats incompatible with soundfile.

## Required output schema

Every metadata CSV emitted by these scripts must have at least these
columns; downstream `prep_<dataset>` modules and `master_dataio_prep`
assume they exist.

| Column           | Meaning                                                                    |
|------------------|----------------------------------------------------------------------------|
| `uid`            | Unique row identifier — also used as the manifest key for caching.         |
| `Participant_ID` | Speaker / subject id. Used for speaker-stratified splits and bootstraps.   |
| `split`          | `train` / `val` / `test`. Honored unless a `prep_*` overrides via `ratio`. |
| `label`          | Raw label — encoded by the prep module per `task_type`.                    |
| `path`           | Audio path **relative to** `data/<dataset>/processed/audio/`.              |

## Dataset folder layout

After running the matching script, the dataset must look like:

```
data/<dataset>/
├── processed/
│   ├── <dataset>.csv          ← required output above
│   └── audio/
│       └── ...                ← actual wav/flac files, paths reference this
└── (raw upstream files; never read by the training pipeline)
```

