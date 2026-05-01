"""Shared prep logic for category-cross tasks.

Each per-task module (`prep_category_<a>_<b>.py`) is a thin shim that
re-exports `prepare_category` under the function names referenced by the
matching YAML configs in `training/config/cross_tasks/`.

The single `prepare_category` function loads any combination of supported
(dataset, task) pairs, harmonizes columns, optionally OR-merges multi-task
rows for datasets that produce one row per task per uid, then writes the
train/val/test manifests via the standard `prep_utils` helpers.
"""

from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import (
    save_task_csv,
    speaker_stratified_split_train_val,
    to_sb_dict_and_save,
)
from training.dataio.prep_aphasia_dbank import _aphasia_pwa_df, _dbank_ad_df
from training.dataio.prep_avfad import _avfad_stratify_cols
from training.dataio.prep_c9s import _c9s_stratify_cols
from training.dataio.prep_coswara import _coswara_stratify_cols, COSWARA_SYMPTOM_COLS
from training.dataio.prep_edaic_iemocap import _edaic_dep_df
from training.dataio.prep_mvdr_ksof import _build_ksof_int_labels, _ksof_stratify_cols
from training.dataio.prep_ravdess_iemocap import _ravdess_emo_bc_df, _iemocap_emo_bc_df
from training.dataio.prep_torgo_uaspeech import _torgo_stratify_cols, _uaspeech_stratify_cols


# Tasks whose label distribution correlates with age — included in the
# stratified split key when *every* train task qualifies. Mixing with a
# non-age-stratified task drops age_bin to avoid empty strata.
AGE_STRATIFIED_TASKS = {
    ("aphasia", "pwaC"),
    ("c9s", "t1"),
    ("c9s", "t2"),
    ("coswara", "sympC"),
    ("coswara", "covidC"),
    ("avfad", "pathC"),
}

# Datasets where the same uid appears in multiple tasks; collapse to one row
# per uid by OR-ing the task labels (positive in any → positive overall).
OR_MERGE_DATASETS = {"c9s", "coswara"}


def _csv_with_resolved_paths(metadata_path, wav_folder):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    return df


def _load_c9s_t1(wav, meta):
    df = _csv_with_resolved_paths(meta, wav)
    df = df[df["split_t1"].notna()]
    df["label"] = (df["Symptoms"] != "None") & (df["Symptoms"].notna())
    return _c9s_stratify_cols(df)


def _load_c9s_t2(wav, meta):
    df = _csv_with_resolved_paths(meta, wav)
    df = df[df["label_t2"].notna()]
    df["label"] = (df["label_t2"] == 1.0).astype(int)
    return _c9s_stratify_cols(df)


def _load_coswara_symp(wav, meta):
    df = _csv_with_resolved_paths(meta, wav)
    m = df[COSWARA_SYMPTOM_COLS].fillna(False)
    for col in COSWARA_SYMPTOM_COLS:
        m[col] = m[col].apply(
            lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False
        )
    df["label"] = m.any(axis=1).astype(int)
    return _coswara_stratify_cols(df)


def _load_coswara_covid(wav, meta):
    df = _csv_with_resolved_paths(meta, wav)
    pos = ["positive_asymp", "positive_mild", "positive_moderate"]
    df = df[df["covid_status"].isin(pos + ["healthy"])].copy()
    df["label"] = df["covid_status"].isin(pos).astype(int)
    return _coswara_stratify_cols(df)


def _load_csv_passthrough(wav, meta):
    return _csv_with_resolved_paths(meta, wav)


def _load_torgo_dys(wav, meta):
    return _torgo_stratify_cols(_csv_with_resolved_paths(meta, wav))


def _load_uaspeech_dys(wav, meta):
    return _uaspeech_stratify_cols(_csv_with_resolved_paths(meta, wav))


def _load_avfad_path(wav, meta):
    return _avfad_stratify_cols(_csv_with_resolved_paths(meta, wav))


def _load_ksof_int(wav, meta):
    return _ksof_stratify_cols(_build_ksof_int_labels(_csv_with_resolved_paths(meta, wav)))


# (dataset, task) -> loader(wav_folder, metadata_path) -> DataFrame.
TASK_LOADERS = {
    ("edaic",    "depC"):   lambda w, m: _edaic_dep_df(str(w), str(m)),
    ("ravdess",  "emoBC"):  lambda w, m: _ravdess_emo_bc_df(str(w), str(m)),
    ("iemocap",  "emoBC"):  lambda w, m: _iemocap_emo_bc_df(str(w), str(m)),
    ("aphasia",  "pwaC"):   lambda w, m: _aphasia_pwa_df(str(w), str(m)),
    ("dbank",    "adC"):    lambda w, m: _dbank_ad_df(str(w), str(m)),
    ("torgo",    "dysC"):   _load_torgo_dys,
    ("uaspeech", "dysC"):   _load_uaspeech_dys,
    ("mvdr",     "parkC"):  _load_csv_passthrough,
    ("ksof",     "intC"):   _load_ksof_int,
    ("c9s",      "t1"):     _load_c9s_t1,
    ("c9s",      "t2"):     _load_c9s_t2,
    ("coswara",  "sympC"):  _load_coswara_symp,
    ("coswara",  "covidC"): _load_coswara_covid,
    ("avfad",    "pathC"):  _load_avfad_path,
}

# Per-prep-module subset of TASK_LOADERS, expressed as dataset -> [tasks].
DATASET_TASKS = {
    "edaic":    ["depC"],
    "ravdess":  ["emoBC"],
    "iemocap":  ["emoBC"],
    "aphasia":  ["pwaC"],
    "dbank":    ["adC"],
    "torgo":    ["dysC"],
    "uaspeech": ["dysC"],
    "mvdr":     ["parkC"],
    "ksof":     ["intC"],
    "c9s":      ["t1", "t2"],
    "coswara":  ["sympC", "covidC"],
    "avfad":    ["pathC"],
}


def _paths_for_dataset(dataset, audio_roots, meta_paths):
    wav = (audio_roots or {}).get(dataset)
    meta = (meta_paths or {}).get(dataset)
    if not wav or not meta:
        raise ValueError(
            f"Missing dataset path mapping for '{dataset}'. "
            "Expected both dataset_audio_roots and dataset_metadata_paths entries."
        )
    return Path(wav), Path(meta)


def _load_task_df(dataset, task, wav_folder, metadata_path):
    loader = TASK_LOADERS.get((dataset, task))
    if loader is None:
        raise ValueError(f"Unsupported dataset/task for category prep: {dataset}/{task}")
    return loader(wav_folder, metadata_path)


def _merge_multitask_rows(dataset, frames):
    """OR-merge label across tasks for datasets that share uids across tasks."""
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if dataset not in OR_MERGE_DATASETS or "uid" not in df.columns:
        return df
    if df["label"].dtype == bool:
        df["label"] = df["label"].astype(int)
    elif pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(float).round().astype(int)
    keep_cols = [c for c in df.columns if c not in {"uid", "label", "source_task"}]
    merged = (
        df.groupby("uid", as_index=False)
        .agg({"label": "max", **{c: "first" for c in keep_cols}})
    )
    merged["source_task"] = "or_union"
    return merged


def _build_split_frame(datasets, settings, audio_roots, meta_paths, *, is_train):
    frames = []
    for idx, ds in enumerate(datasets):
        wav, meta = _paths_for_dataset(ds, audio_roots, meta_paths)
        ds_frames = []
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav, meta).copy()
            df["source_dataset"] = ds
            df["source_task"] = tk
            if is_train:
                cfg = settings[idx] if idx < len(settings) else {}
                df["setting_idx"] = idx
                df["split_by_boundary"] = bool(cfg.get("split_by_boundary", False))
                df["num_aug_ver"] = int(cfg.get("num_aug_ver", 1))
            else:
                df["split_by_boundary"] = False
                df["num_aug_ver"] = 1
            ds_frames.append(df)
        frames.append(_merge_multitask_rows(ds, ds_frames))
    return pd.concat(frames, ignore_index=True)


def _harmonize(df):
    if "gender" not in df.columns:
        df["gender"] = "unknown"
    if "age_bin" not in df.columns:
        df["age_bin"] = "unknown"
    df["gender"] = df["gender"].astype(str).fillna("unknown")
    df["age_bin"] = df["age_bin"].astype(str).fillna("unknown")
    if df["label"].dtype == bool:
        df["label"] = df["label"].astype(int)
    elif pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(float).round().astype(int)
    df["cache_uid"] = df["uid"].astype(str)
    df["uid"] = df["source_dataset"] + "_" + df["source_task"] + "::" + df["cache_uid"]
    df["Participant_ID"] = (
        df["source_dataset"] + "_" + df["source_task"] + "::"
        + df["Participant_ID"].astype(str)
    )
    return df


def prepare_category(
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
    train_datasets=None,
    test_datasets=None,
    category_settings=None,
    dataset_audio_roots=None,
    dataset_metadata_paths=None,
):
    train_datasets = list(train_datasets or [])
    test_datasets = list(test_datasets or [])
    settings = list(category_settings or [])

    src = _build_split_frame(
        train_datasets, settings, dataset_audio_roots, dataset_metadata_paths,
        is_train=True,
    )
    tgt = _build_split_frame(
        test_datasets, settings, dataset_audio_roots, dataset_metadata_paths,
        is_train=False,
    )

    src = _harmonize(src)
    tgt = _harmonize(tgt)

    stratify_cols = ["label", "gender", "source_dataset"]
    src_tasks = {
        (r.source_dataset, r.source_task)
        for r in src[["source_dataset", "source_task"]].drop_duplicates().itertuples(index=False)
    }
    if src_tasks and all(t in AGE_STRATIFIED_TASKS for t in src_tasks):
        stratify_cols.append("age_bin")

    df_train, df_val = speaker_stratified_split_train_val(
        src, ratio, random_seed, stratify_cols=stratify_cols,
    )
    df_test = tgt.copy()

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(
        df_train, manifest_train_path,
        df_val, manifest_val_path,
        df_test, manifest_test_path,
    )
