from pathlib import Path
import pandas as pd
from sdx.prep.utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val


STUT_DISFL_COLS = [
    "Block",
    "Prolongation",
    "Sound Repetition",
    "Word / Phrase Repetition",
    "Modified/ Speech technique",
    "Interjection",
]
STUT_MAJORITY = 2


def _uaspeech_stratify_cols(df):
    df = df.copy()
    df["gender"] = (
        df["gender"].astype(str).str[-1].map({"F": "female", "M": "male"}).fillna("unknown")
    )
    return df


def _ksof_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower().replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    return df


def _build_ksof_int_labels(df):
    pos = pd.concat([(df[c] >= STUT_MAJORITY) for c in STUT_DISFL_COLS], axis=1).any(axis=1)
    neg = df["No dysfluencies"] >= STUT_MAJORITY
    keep = pos | neg
    out = df[keep].reset_index(drop=True).copy()
    out["label"] = pos[keep].reset_index(drop=True).astype(int).values
    return out


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_uaspeech_ksof_dysC_intC(
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
    df_uaspeech = pd.read_csv(metadata_path_train)
    df_uaspeech["path"] = Path(wav_folder_train).resolve() / df_uaspeech["path"]
    df_uaspeech = _uaspeech_stratify_cols(df_uaspeech)

    df_ksof = pd.read_csv(metadata_path_test)
    df_ksof["path"] = Path(wav_folder_test).resolve() / df_ksof["path"]
    df_ksof = _build_ksof_int_labels(df_ksof)
    df_ksof = _ksof_stratify_cols(df_ksof)

    df_train, df_val = speaker_stratified_split_train_val(
        df_uaspeech, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_ksof.copy()
    _log_split_stats("UASpeech", "KSoF", df_uaspeech, df_ksof, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_ksof_uaspeech_intC_dysC(
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
    df_ksof = pd.read_csv(metadata_path_train)
    df_ksof["path"] = Path(wav_folder_train).resolve() / df_ksof["path"]
    df_ksof = _build_ksof_int_labels(df_ksof)
    df_ksof = _ksof_stratify_cols(df_ksof)

    df_uaspeech = pd.read_csv(metadata_path_test)
    df_uaspeech["path"] = Path(wav_folder_test).resolve() / df_uaspeech["path"]
    df_uaspeech = _uaspeech_stratify_cols(df_uaspeech)

    df_train, df_val = speaker_stratified_split_train_val(
        df_ksof, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_uaspeech.copy()
    _log_split_stats("KSoF", "UASpeech", df_ksof, df_uaspeech, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
