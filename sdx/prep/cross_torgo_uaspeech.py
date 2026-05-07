from pathlib import Path
import pandas as pd
from sdx.prep.utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val

def _torgo_stratify_cols(df):
    df = df.copy()

    df["gender"] = df["gender"].map({
        "F": "female",
        "M": "male",
    }).fillna("unknown")

    return df

def _uaspeech_stratify_cols(df):
    df = df.copy()

    df["gender"] = (
        df["gender"].astype(str).str[-1]
        .map({"F": "female", "M": "male"})
        .fillna("unknown")
    )

    return df

def prepare_torgo_uaspeech_dysC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset dysarthria classification task:
    - train/val from Torgo
    - test from all of UASpeech

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for Torgo train/val split.
    """

    df_torgo = pd.read_csv(metadata_path_train)
    df_torgo["path"] = Path(wav_folder_train).resolve() / df_torgo["path"]

    df_uaspeech = pd.read_csv(metadata_path_test)
    df_uaspeech["path"] = Path(wav_folder_test).resolve() / df_uaspeech["path"]


    df_torgo = _torgo_stratify_cols(df_torgo)
    df_uaspeech = _uaspeech_stratify_cols(df_uaspeech)


    print("Torgo overall distribution:", df_torgo["label"].value_counts().to_dict())
    print("UASpeech overall distribution:", df_uaspeech["label"].value_counts().to_dict())

    # Split only Torgo into train/val
    df_train, df_val = speaker_stratified_split_train_val(
        df_torgo,
        ratio,
        random_seed,
        stratify_cols=["label", "gender"],
    )

    # Use all of UASpeech as test
    df_test = df_uaspeech.copy()

    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())

    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(
        df_train, manifest_train_path,
        df_val, manifest_val_path,
        df_test, manifest_test_path,
    )

    print("--- prepare_torgo_uaspeech_dysC finished ---")

def prepare_uaspeech_torgo_dysC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset dysarthria classification task:
    - train/val from UASpeech
    - test from all of Torgo
    """

    df_uaspeech = pd.read_csv(metadata_path_train)
    df_uaspeech["path"] = Path(wav_folder_train).resolve() / df_uaspeech["path"]

    df_torgo = pd.read_csv(metadata_path_test)
    df_torgo["path"] = Path(wav_folder_test).resolve() / df_torgo["path"]

    df_uaspeech = _uaspeech_stratify_cols(df_uaspeech)
    df_torgo = _torgo_stratify_cols(df_torgo)

    print("UASpeech overall distribution:", df_uaspeech["label"].value_counts().to_dict())
    print("Torgo overall distribution:", df_torgo["label"].value_counts().to_dict())

    # Split only UASpeech into train/val
    df_train, df_val = speaker_stratified_split_train_val(
        df_uaspeech,
        ratio,
        random_seed,
        stratify_cols=["label", "gender"],
    )

    # Use all of Torgo as test
    df_test = df_torgo.copy()

    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())

    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(
        df_train, manifest_train_path,
        df_val, manifest_val_path,
        df_test, manifest_test_path,
    )

    print("--- prepare_uaspeech_torgo_dysC finished ---")

