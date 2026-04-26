import json
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val


RAVDESS_BINARY_EMOTIONS = {
    0: 0,  # neu
    1: 0,  # calm
    2: 0,  # happy
    3: 1,  # sad
    4: 1,  # angry
    5: 1,  # fearful
    6: 1,  # disgust
    7: 0,  # surprised
}


def _safe_parse_boundaries(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _edaic_stratify_cols(df):
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower()
        df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
        df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    else:
        df["gender"] = "unknown"
    return df


def _ravdess_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    return df


def _edaic_dep_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    if "boundaries" in df.columns:
        df["boundaries"] = df["boundaries"].apply(_safe_parse_boundaries)
    df["label"] = df["label"].astype(int)
    return _edaic_stratify_cols(df)


def _ravdess_emo_bc_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["label"].map(RAVDESS_BINARY_EMOTIONS).astype(int)
    return _ravdess_stratify_cols(df)


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_edaic_ravdess_depC_emoBC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_edaic = _edaic_dep_df(wav_folder_train, metadata_path_train)
    df_ravdess = _ravdess_emo_bc_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_edaic, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_ravdess.copy()
    _log_split_stats("EDAIC", "RAVDESS", df_edaic, df_ravdess, df_train, df_val, df_test)
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_ravdess_edaic_emoBC_depC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_ravdess = _ravdess_emo_bc_df(wav_folder_train, metadata_path_train)
    df_edaic = _edaic_dep_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_ravdess, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_edaic.copy()
    _log_split_stats("RAVDESS", "EDAIC", df_ravdess, df_edaic, df_train, df_val, df_test)
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
