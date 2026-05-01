from pathlib import Path
import pandas as pd

from training.dataio.prep_utils import (
    save_task_csv,
    speaker_stratified_split_train_val,
    to_sb_dict_and_save,
)
from training.dataio.prep_edaic_iemocap import _edaic_dep_df
from training.dataio.prep_ravdess_iemocap import _ravdess_emo_bc_df, _iemocap_emo_bc_df
from training.dataio.prep_torgo_uaspeech import _torgo_stratify_cols, _uaspeech_stratify_cols
from training.dataio.prep_mvdr_ksof import _build_ksof_int_labels, _ksof_stratify_cols


DATASET_TASKS = {
    "edaic": ["depC"],
    "ravdess": ["emoBC"],
    "iemocap": ["emoBC"],
    "torgo": ["dysC"],
    "uaspeech": ["dysC"],
    "mvdr": ["parkC"],
    "ksof": ["intC"],
}
AGE_STRATIFIED_TASKS = set()


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
    if dataset == "edaic" and task == "depC":
        return _edaic_dep_df(str(wav_folder), str(metadata_path))
    if dataset == "ravdess" and task == "emoBC":
        return _ravdess_emo_bc_df(str(wav_folder), str(metadata_path))
    if dataset == "iemocap" and task == "emoBC":
        return _iemocap_emo_bc_df(str(wav_folder), str(metadata_path))
    if dataset == "torgo" and task == "dysC":
        df = pd.read_csv(metadata_path)
        df["path"] = Path(wav_folder).resolve() / df["path"]
        return _torgo_stratify_cols(df)
    if dataset == "uaspeech" and task == "dysC":
        df = pd.read_csv(metadata_path)
        df["path"] = Path(wav_folder).resolve() / df["path"]
        return _uaspeech_stratify_cols(df)
    if dataset == "mvdr" and task == "parkC":
        df = pd.read_csv(metadata_path)
        df["path"] = Path(wav_folder).resolve() / df["path"]
        return df
    if dataset == "ksof" and task == "intC":
        df = pd.read_csv(metadata_path)
        df["path"] = Path(wav_folder).resolve() / df["path"]
        return _ksof_stratify_cols(_build_ksof_int_labels(df))
    raise ValueError(f"Unsupported dataset/task for category prep: {dataset}/{task}")


def _prep_category_split(
    train_manifest, val_manifest, test_manifest, ratio, random_seed, dataset, task,
    train_datasets=None, test_datasets=None, category_settings=None, dataset_audio_roots=None, dataset_metadata_paths=None
):
    train_datasets = list(train_datasets or [])
    test_datasets = list(test_datasets or [])
    settings = list(category_settings or [])

    train_frames = []
    for idx, ds in enumerate(train_datasets):
        wav, meta = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav, meta).copy()
            df["source_dataset"], df["source_task"], df["setting_idx"] = ds, tk, idx
            cfg = settings[idx] if idx < len(settings) else {}
            df["split_by_boundary"] = bool(cfg.get("split_by_boundary", False))
            df["num_aug_ver"] = int(cfg.get("num_aug_ver", 1))
            train_frames.append(df)

    test_frames = []
    for ds in test_datasets:
        wav, meta = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav, meta).copy()
            df["source_dataset"], df["source_task"] = ds, tk
            df["split_by_boundary"], df["num_aug_ver"] = False, 1
            test_frames.append(df)

    src = pd.concat(train_frames, ignore_index=True)
    tgt = pd.concat(test_frames, ignore_index=True)
    for df in (src, tgt):
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

    for df in (src, tgt):
        df["cache_uid"] = df["uid"].astype(str)
        df["uid"] = df["source_dataset"] + "_" + df["source_task"] + "::" + df["cache_uid"]
        df["Participant_ID"] = df["source_dataset"] + "_" + df["source_task"] + "::" + df["Participant_ID"].astype(str)

    strat = ["label", "gender", "source_dataset"]
    train_df, val_df = speaker_stratified_split_train_val(src, ratio, random_seed, stratify_cols=strat)
    test_df = tgt.copy()
    save_task_csv(train_df, val_df, test_df, dataset, task)
    to_sb_dict_and_save(train_df, train_manifest, val_df, val_manifest, test_df, test_manifest)


def prepare_category_c1_c3(*args, **kwargs):
    return _prep_category_split(*args, **kwargs)


def prepare_category_c3_c1(*args, **kwargs):
    return _prep_category_split(*args, **kwargs)
