# dementiabank

DementiaBank **ADReSS-M** challenge (ICASSP-SPGC 2023) — multilingual
Alzheimer's disease detection from speech.

- Upstream name: **ADReSS-M** (also referenced in the literature as
  "ADReSS-M: Multilingual Alzheimer's Dementia Recognition through
  Spontaneous Speech").
- Paper: https://talkbank.org/dementia/ADReSS-M/IEEE.pdf
- Distribution: https://dementia.talkbank.org/ADReSS-M/ (upon request)
- Internal short name in this repo: `dementiabank`.

## Layout

```
dementiabank/
  processed/
    audio/             # 291 files, flat (mp3 for English train, wav for Greek)
    dementiabank.csv   # manifest
```

## CSV columns

`uid, Participant_ID, age, gender, educ, dx, mmse, language, split, label, path`

- `label`: Control = 0, ProbableAD = 1 (binary AD classification)
- `split`: 0 = train (English, `adrso*`), 1 = val (Greek `sample-gr`),
  2 = test (Greek held-out `test-gr`)
- `language`: `en` or `el`
- `path`: relative to `processed/audio/`

## Splits

| split | count | cohort                            |
|-------|-------|-----------------------------------|
| 0     | 237   | English train (`adrso*.mp3`)      |
| 1     | 8     | Greek dev (`sample-gr/madrs*.wav`)|
| 2     | 46    | Greek test (`test-gr/madrs*.wav`) |

Participant independence is guaranteed by the organizers: the paper
explicitly states the dataset was constructed to eliminate "repeated
occurrences of speech from the same participant" (Conclusion). Each row
is a unique `Participant_ID` (verified: 291 unique IDs / 291 rows,
no overlap across splits).

## Dropped from the release

The upstream `ADReSS-M-train/sample/` folder ships 8 `madrs-smpl*.mp3`
files — these are **Spanish samples from an earlier draft release**,
later replaced by the Greek `sample-gr` files. They are excluded by
`metadata_script/create_dementiabank_metadata.py`.

## Regenerating

```
python metadata_script/create_dementiabank_metadata.py
```

Source path is set at the top of that script; point it at an
ADReSS-M extract with the standard subfolders
(`ADReSS-M-train/`, `ADReSS-M-sample-gr/`, `test-gr/`, `test-gr-groundtruth.csv`).
