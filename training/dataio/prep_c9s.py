"""
Preparing the metadata to go into preprocessing pipeline. This is for using with default
split, label, classic model fitting
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, speaker_stratified_split, to_sb_dict_and_save
from training.dataio.utils import ensure_dir, PathEncoder


def prepare_c9s_t1(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
        dataset,
        task,
):
    """
    This function is task-specific.
    """
    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["split_t1"].notna()]
    df['label'] = (df['Symptoms'] != 'None') & (df['Symptoms'].notna())

    # Speaker-independent stratified split
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train,
                        manifest_train_path,
                        df_val,
                        manifest_val_path,
                        df_test,
                        manifest_test_path)

    print("Manifests created.")
    print("--- prepare_data finished ---")

def prepare_c9s_t2(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
        dataset,
        task,
):
    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["label_t2"].notna()]
    df['label'] = (df['label_t2'] == 1.0).astype(int)
    print("Distribution")
    print(df['label'].value_counts())

    # Use included split
    df_train = df[df["split_t2"] == 0]
    df_val = df[df["split_t2"] == 1]
    df_test = df[df["split_t2"] == 2]

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train,
                        manifest_train_path,
                        df_val,
                        manifest_val_path,
                        df_test,
                        manifest_test_path)

    print("Manifests created.")
    print("--- prepare_data finished ---")


def prepare_c9s_L_t1(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
        dataset,
        task,
):
    """
    This function is task-specific.
    """
    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df['label'] = (df['Symptoms'] != 'None') & (df['Symptoms'].notna())

    # Speaker-independent stratified split
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train,
                        manifest_train_path,
                        df_val,
                        manifest_val_path,
                        df_test,
                        manifest_test_path)

    print("Manifests created.")
    print("--- prepare_data finished ---")


# ---- Helpers ----
AGE_MIDPOINTS = {
    "0-19": 10, "16-19": 17.5, "20-29": 24.5, "30-39": 34.5,
    "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70-79": 74.5,
    "80-89": 84.5, "90-": 95, "Unter 20": 17.5,
}

C9S_SYMPTOMS = [
    "drycough", "wetcough", "fever", "headache", "muscleache", "dizziness",
    "sorethroat", "shortbreath", "tightness", "runnyblockednose", "smelltasteloss", "runny"
]


def _c9s_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Filters to valid rows."""
    df = df[df["Sex"].isin(["Male", "Female"])].copy()
    df["gender"] = df["Sex"].map({"Male": "male", "Female": "female"})
    ages = df["Age"].map(AGE_MIDPOINTS)
    df["age_bin"] = pd.cut(ages, bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df = df.dropna(subset=["age_bin"])
    return df


def prepare_c9s_sexC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = _c9s_stratify_cols(df)
    df["label"] = (df["Sex"] == "Male").astype(int)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_c9s_sexC finished ---")


def prepare_c9s_smokerC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    active = ["1to10", "11to20", "21+", "ecig"]
    df = df[df["Smoking"].isin(active + ["never"])].copy()
    df["label"] = df["Smoking"].isin(active).astype(int)

    df = _c9s_stratify_cols(df)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_c9s_smokerC finished ---")


def prepare_c9s_ageR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = df[df["Age"].isin(AGE_MIDPOINTS.keys())].copy()
    df["label"] = df["Age"].map(AGE_MIDPOINTS).astype(float)

    df = _c9s_stratify_cols(df)

    print("Distribution:", df["label"].describe().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_c9s_ageR finished ---")


def prepare_c9s_sympL(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Filter to symptomatic only
    df = df[df["Symptoms"].notna() & (df["Symptoms"] != "None")].copy()

    # One-hot encode symptoms
    def encode_symptoms(s):
        parts = set(x.strip() for x in s.split(","))
        if "tighness" in parts:
            parts.discard("tighness"); parts.add("tightness")
        return [int(sym in parts) for sym in C9S_SYMPTOMS]

    df["label"] = df["Symptoms"].apply(encode_symptoms)
    df = _c9s_stratify_cols(df)

    print("Samples:", len(df))
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_c9s_sympL finished ---")
