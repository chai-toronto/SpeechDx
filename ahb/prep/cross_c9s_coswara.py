
from pathlib import Path
import pandas as pd
from ahb.prep.utils import save_task_csv, speaker_stratified_split_train_val, to_sb_dict_and_save

AGE_MIDPOINTS = {
    "0-19": 10, "16-19": 17.5, "20-29": 24.5, "30-39": 34.5,
    "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70-79": 74.5,
    "80-89": 84.5, "90-": 95, "Unter 20": 17.5,
}

COSWARA_SYMPTOM_COLS = [
    "cold", "cough", "fever", "diarrhoea", "loss_of_smell",
    "muscularpain", "breathing_difficulty", "fatigue", "sore_throat", "others_resp"
]

def _coswara_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Missing/invalid -> 'unknown' (no row drops)."""
    df = df.copy()
    df["gender"] = df["gender"].where(df["gender"].isin(["male", "female"]), "unknown")
    age_bin = pd.cut(df["age"], bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df

def _c9s_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Missing/invalid -> 'unknown' (no row drops)."""
    df = df.copy()
    df["gender"] = df["Sex"].map({"Male": "male", "Female": "female"}).fillna("unknown")
    ages = df["Age"].map(AGE_MIDPOINTS)
    age_bin = pd.cut(ages, bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df

def prepare_c9s_coswara_t1_sympC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from C9S symptomatic classification
    - test from all of Coswara

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for C9S train/val split.
    """

    df_c9s = pd.read_csv(metadata_path_train)
    df_c9s["path"] = Path(wav_folder_train).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["split_t1"].notna()]
    df_c9s['label'] = (df_c9s['Symptoms'] != 'None') & (df_c9s['Symptoms'].notna())

    df_coswara = pd.read_csv(metadata_path_test)
    df_coswara["path"] = Path(wav_folder_test).resolve() / df_coswara["path"]

    # Any symptom column True -> symptomatic
    symp_matrix = df_coswara[COSWARA_SYMPTOM_COLS].fillna(False)
    # Handle mixed types: 'True'/True -> True
    for col in COSWARA_SYMPTOM_COLS:
        symp_matrix[col] = symp_matrix[col].apply(lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False)
    df_coswara["label"] = symp_matrix.any(axis=1).astype(int)

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_coswara = _coswara_stratify_cols(df_coswara)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("Coswara overall distribution:", df_coswara["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_c9s, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    df_test = df_coswara.copy()

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

    print("--- prepare_c9s_coswara_t1_sympC finished ---")

def prepare_coswara_c9s_sympC_t1(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from Coswara symptomatic classification
    - test from all of C9S

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for Coswara train/val split.
    """
    df_coswara = pd.read_csv(metadata_path_train)
    df_coswara["path"] = Path(wav_folder_train).resolve() / df_coswara["path"]

    # Any symptom column True -> symptomatic
    symp_matrix = df_coswara[COSWARA_SYMPTOM_COLS].fillna(False)
    # Handle mixed types: 'True'/True -> True
    for col in COSWARA_SYMPTOM_COLS:
        symp_matrix[col] = symp_matrix[col].apply(lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False)
    df_coswara["label"] = symp_matrix.any(axis=1).astype(int)

    df_c9s = pd.read_csv(metadata_path_test)
    df_c9s["path"] = Path(wav_folder_test).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["split_t1"].notna()]
    df_c9s['label'] = (df_c9s['Symptoms'] != 'None') & (df_c9s['Symptoms'].notna())

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_coswara = _coswara_stratify_cols(df_coswara)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("Coswara overall distribution:", df_coswara["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_coswara, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
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

    print("--- prepare_coswara_c9s_sympC_t1 finished ---")

def prepare_c9s_coswara_t2_covidC(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from C9S symptomatic classification
    - test from all of Coswara

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for C9S train/val split.
    """
    df_c9s = pd.read_csv(metadata_path_train)
    df_c9s["path"] = Path(wav_folder_train).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["label_t2"].notna()]
    df_c9s['label'] = (df_c9s['label_t2'] == 1.0).astype(int)

    df_coswara = pd.read_csv(metadata_path_test)
    df_coswara["path"] = Path(wav_folder_test).resolve() / df_coswara["path"]

    positive = ["positive_asymp", "positive_mild", "positive_moderate"]
    df_coswara = df_coswara[df_coswara["covid_status"].isin(positive + ["healthy"])].copy()
    df_coswara["label"] = df_coswara["covid_status"].isin(positive).astype(int)

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_coswara = _coswara_stratify_cols(df_coswara)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("Coswara overall distribution:", df_coswara["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_c9s, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    df_test = df_coswara.copy()

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

    print("--- prepare_c9s_coswara_t2_covidC finished ---")

def prepare_coswara_c9s_covidC_t2(wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,):
    """
    Prepare cross-dataset classification task:
    - train/val from Coswara symptomatic classification
    - test from all of C9S

    Args:
        ratio: List of 2 ints summing to 100, e.g. [90, 10]
               used only for Coswara train/val split.
    """

    df_coswara = pd.read_csv(metadata_path_train)
    df_coswara["path"] = Path(wav_folder_train).resolve() / df_coswara["path"]

    positive = ["positive_asymp", "positive_mild", "positive_moderate"]
    df_coswara = df_coswara[df_coswara["covid_status"].isin(positive + ["healthy"])].copy()
    df_coswara["label"] = df_coswara["covid_status"].isin(positive).astype(int)

    df_c9s = pd.read_csv(metadata_path_test)
    df_c9s["path"] = Path(wav_folder_test).resolve() / df_c9s["path"]
    df_c9s = df_c9s[df_c9s["label_t2"].notna()]
    df_c9s['label'] = (df_c9s['label_t2'] == 1.0).astype(int)

    df_c9s = _c9s_stratify_cols(df_c9s)
    df_coswara = _coswara_stratify_cols(df_coswara)

    print("C9S overall distribution:", df_c9s["label"].value_counts().to_dict())
    print("Coswara overall distribution:", df_coswara["label"].value_counts().to_dict())

    # Speaker-independent stratified split
    df_train, df_val = speaker_stratified_split_train_val(df_coswara, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
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

    print("--- prepare_coswara_c9s_covidC_t2 finished ---")




    


    