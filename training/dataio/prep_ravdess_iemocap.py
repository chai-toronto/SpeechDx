from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val


RAVDESS_BINARY_EMOTIONS = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0,
}

IEMOCAP_BINARY_EMOTIONS = {
    "neu": 0, "fru": 1, "ang": 1, "sad": 1, "hap": 0, "exc": 0,
}


def _ravdess_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    return df


def _iemocap_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    return df


def _ravdess_emo_bc_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["label"].map(RAVDESS_BINARY_EMOTIONS).astype(int)
    return _ravdess_stratify_cols(df)


def _iemocap_emo_bc_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = df[df["label"].isin(IEMOCAP_BINARY_EMOTIONS.keys())].reset_index(drop=True)
    df["label"] = df["label"].map(IEMOCAP_BINARY_EMOTIONS).astype(int)
    return _iemocap_stratify_cols(df)


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_ravdess_iemocap_emoBC_emoBC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_ravdess = _ravdess_emo_bc_df(wav_folder_train, metadata_path_train)
    df_iemocap = _iemocap_emo_bc_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_ravdess, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_iemocap.copy()
    _log_split_stats("RAVDESS", "IEMOCAP", df_ravdess, df_iemocap, df_train, df_val, df_test)
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_iemocap_ravdess_emoBC_emoBC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_iemocap = _iemocap_emo_bc_df(wav_folder_train, metadata_path_train)
    df_ravdess = _ravdess_emo_bc_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_iemocap, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_ravdess.copy()
    _log_split_stats("IEMOCAP", "RAVDESS", df_iemocap, df_ravdess, df_train, df_val, df_test)
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
