from pathlib import Path
import pandas as pd

from training.dataio.prep_utils import (
    save_task_csv,
    speaker_stratified_split_train_val,
    to_sb_dict_and_save,
)
from training.dataio.prep_edaic_iemocap import _edaic_dep_df
from training.dataio.prep_ravdess_iemocap import _ravdess_emo_bc_df, _iemocap_emo_bc_df
from training.dataio.prep_aphasia_dbank import _aphasia_pwa_df, _dbank_ad_df
 


DATASET_TASKS = {
    "edaic": ["depC"],
    "ravdess": ["emoBC"],
    "iemocap": ["emoBC"],
    "aphasia": ["pwaC"],
    "dbank": ["adC"],
}

AGE_STRATIFIED_TASKS = {
    ("aphasia", "pwaC"),
}


def _paths_for_dataset(dataset, dataset_audio_roots, dataset_metadata_paths):
    wav = (dataset_audio_roots or {}).get(dataset)
    meta = (dataset_metadata_paths or {}).get(dataset)
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
    if dataset == "aphasia" and task == "pwaC":
        return _aphasia_pwa_df(str(wav_folder), str(metadata_path))
    if dataset == "dbank" and task == "adC":
        return _dbank_ad_df(str(wav_folder), str(metadata_path))
    raise ValueError(f"Unsupported dataset/task for category prep: {dataset}/{task}")


def _prep_category_split(
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
    train_datasets=None, test_datasets=None, category_settings=None,
    dataset_audio_roots=None, dataset_metadata_paths=None,
):
    train_datasets = list(train_datasets or [])
    test_datasets = list(test_datasets or [])
    category_settings = list(category_settings or [])

    train_frames = []
    for idx, ds in enumerate(train_datasets):
        wav_root, meta_path = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav_root, meta_path).copy()
            df["source_dataset"] = ds
            df["source_task"] = tk
            df["setting_idx"] = idx
            if idx < len(category_settings):
                cfg = category_settings[idx]
                df["split_by_boundary"] = bool(cfg.get("split_by_boundary", False))
                df["num_aug_ver"] = int(cfg.get("num_aug_ver", 1))
            train_frames.append(df)

    test_frames = []
    for ds in test_datasets:
        wav_root, meta_path = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav_root, meta_path).copy()
            df["source_dataset"] = ds
            df["source_task"] = tk
            df["split_by_boundary"] = False
            df["num_aug_ver"] = 1
            test_frames.append(df)

    df_src = pd.concat(train_frames, ignore_index=True)
    df_tgt = pd.concat(test_frames, ignore_index=True)

    for df in (df_src, df_tgt):
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

    df_src["cache_uid"] = df_src["uid"].astype(str)
    df_src["uid"] = df_src["source_dataset"] + "_" + df_src["source_task"] + "::" + df_src["cache_uid"]
    df_src["Participant_ID"] = df_src["source_dataset"] + "_" + df_src["source_task"] + "::" + df_src["Participant_ID"].astype(str)

    df_tgt["cache_uid"] = df_tgt["uid"].astype(str)
    df_tgt["uid"] = df_tgt["source_dataset"] + "_" + df_tgt["source_task"] + "::" + df_tgt["cache_uid"]
    df_tgt["Participant_ID"] = df_tgt["source_dataset"] + "_" + df_tgt["source_task"] + "::" + df_tgt["Participant_ID"].astype(str)

    stratify_cols = ["label", "gender", "source_dataset"]
    src_tasks = {(r.source_dataset, r.source_task) for r in df_src[["source_dataset", "source_task"]].drop_duplicates().itertuples(index=False)}
    if src_tasks and all(t in AGE_STRATIFIED_TASKS for t in src_tasks):
        stratify_cols.append("age_bin")

    df_train, df_val = speaker_stratified_split_train_val(df_src, ratio, random_seed, stratify_cols=stratify_cols)
    df_test = df_tgt.copy()

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_category_c1_c2(*args, **kwargs):
    return _prep_category_split(*args, **kwargs)


def prepare_category_c2_c1(*args, **kwargs):
    return _prep_category_split(*args, **kwargs)
