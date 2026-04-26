from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val

AGE_MIDPOINTS = {
    "0-19": 10, "16-19": 17.5, "20-29": 24.5, "30-39": 34.5,
    "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70-79": 74.5,
    "80-89": 84.5, "90-": 95, "Unter 20": 17.5,
}

def _c9s_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Missing/invalid -> 'unknown' (no row drops)."""
    df = df.copy()
    df["gender"] = df["Sex"].map({"Male": "male", "Female": "female"}).fillna("unknown")
    ages = df["Age"].map(AGE_MIDPOINTS)
    age_bin = pd.cut(ages, bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df

def _avfad_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Missing/invalid -> 'unknown' (no row drops)."""
    df = df.copy()
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["gender"] = df["Sex"].map({"M": "male", "F": "female"}).fillna("unknown")
    age_bin = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["<30", "30-49", "50-69", "70+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df

def prepare_c9s_avfad_t1_pathC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from C9S symptomatic classification
    - test from all of AVFAD

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for C9S train/val split.
    """

    df_c9s = pd.read_csv(metadata_path_train)
    df_c9s["path"] = Path(wav_folder_train).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["split_t1"].notna()]
    df_c9s['label'] = (df_c9s['Symptoms'] != 'None') & (df_c9s['Symptoms'].notna())

    df_avfad = pd.read_csv(metadata_path_test)
    df_avfad["path"] = Path(wav_folder_test).resolve() / df_avfad["path"]

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_avfad = _avfad_stratify_cols(df_avfad)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("AVFAD overall distribution:", df_avfad["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_c9s, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    df_test = df_avfad.copy()

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

    print("--- prepare_c9s_avfad_t1_pathC finished ---")

def prepare_avfad_c9s_pathC_t1(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from AVFAD symptomatic classification
    - test from all of C9S

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for AVFAD train/val split.
    """

    df_avfad = pd.read_csv(metadata_path_train)
    df_avfad["path"] = Path(wav_folder_train).resolve() / df_avfad["path"]

    df_c9s = pd.read_csv(metadata_path_test)
    df_c9s["path"] = Path(wav_folder_test).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["split_t1"].notna()]
    df_c9s['label'] = (df_c9s['Symptoms'] != 'None') & (df_c9s['Symptoms'].notna())

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_avfad = _avfad_stratify_cols(df_avfad)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("AVFAD overall distribution:", df_avfad["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_avfad, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    df_test = df_c9s.copy()

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

    print("--- prepare_avfad_c9s_pathC_t1 finished ---")


def prepare_c9s_avfad_t2_pathC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from C9S t2 task
    - test from all of AVFAD pathC
    """
    df_c9s = pd.read_csv(metadata_path_train)
    df_c9s["path"] = Path(wav_folder_train).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["label_t2"].notna()]
    df_c9s["label"] = (df_c9s["label_t2"] == 1.0).astype(int)

    df_avfad = pd.read_csv(metadata_path_test)
    df_avfad["path"] = Path(wav_folder_test).resolve() / df_avfad["path"]

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_avfad = _avfad_stratify_cols(df_avfad)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("AVFAD overall distribution:", df_avfad["label"].value_counts().to_dict())

    df_train, df_val = speaker_stratified_split_train_val(
        df_c9s, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_avfad.copy()

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

    print("--- prepare_c9s_avfad_t2_pathC finished ---")


def prepare_avfad_c9s_pathC_t2(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from AVFAD pathC
    - test from C9S t2 task
    """
    df_avfad = pd.read_csv(metadata_path_train)
    df_avfad["path"] = Path(wav_folder_train).resolve() / df_avfad["path"]

    df_c9s = pd.read_csv(metadata_path_test)
    df_c9s["path"] = Path(wav_folder_test).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["label_t2"].notna()]
    df_c9s["label"] = (df_c9s["label_t2"] == 1.0).astype(int)

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_avfad = _avfad_stratify_cols(df_avfad)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("AVFAD overall distribution:", df_avfad["label"].value_counts().to_dict())

    df_train, df_val = speaker_stratified_split_train_val(
        df_avfad, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_c9s.copy()

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

    print("--- prepare_avfad_c9s_pathC_t2 finished ---")