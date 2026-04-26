from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val


def _torgo_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].map({"F": "female", "M": "male"}).fillna("unknown")
    return df


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_torgo_mvdr_dysC_parkC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_torgo = pd.read_csv(metadata_path_train)
    df_torgo["path"] = Path(wav_folder_train).resolve() / df_torgo["path"]
    df_torgo = _torgo_stratify_cols(df_torgo)

    df_mvdr = pd.read_csv(metadata_path_test)
    df_mvdr["path"] = Path(wav_folder_test).resolve() / df_mvdr["path"]

    df_train, df_val = speaker_stratified_split_train_val(
        df_torgo, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_mvdr.copy()
    _log_split_stats("Torgo", "MVDR", df_torgo, df_mvdr, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_mvdr_torgo_parkC_dysC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_mvdr = pd.read_csv(metadata_path_train)
    df_mvdr["path"] = Path(wav_folder_train).resolve() / df_mvdr["path"]

    df_torgo = pd.read_csv(metadata_path_test)
    df_torgo["path"] = Path(wav_folder_test).resolve() / df_torgo["path"]
    df_torgo = _torgo_stratify_cols(df_torgo)

    df_train, df_val = speaker_stratified_split_train_val(
        df_mvdr, ratio, random_seed, stratify_cols=["label"]
    )
    df_test = df_torgo.copy()
    _log_split_stats("MVDR", "Torgo", df_mvdr, df_torgo, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
