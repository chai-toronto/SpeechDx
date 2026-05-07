"""
Preparing the metadata for Coswara dataset tasks.
Speaker-stratified splits on gender + age.
"""
from pathlib import Path
import pandas as pd
from sdx.prep.utils import save_task_csv, speaker_stratified_split, to_sb_dict_and_save

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


def prepare_coswara_covidC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    positive = ["positive_asymp", "positive_mild", "positive_moderate"]
    df = df[df["covid_status"].isin(positive + ["healthy"])].copy()
    df["label"] = df["covid_status"].isin(positive).astype(int)

    df = _coswara_stratify_cols(df)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_covidC finished ---")


def prepare_coswara_sympC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Binary symptomatic classification: any symptom True -> 1, else 0."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Any symptom column True -> symptomatic
    symp_matrix = df[COSWARA_SYMPTOM_COLS].fillna(False)
    # Handle mixed types: 'True'/True -> True
    for col in COSWARA_SYMPTOM_COLS:
        symp_matrix[col] = symp_matrix[col].apply(lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False)
    df["label"] = symp_matrix.any(axis=1).astype(int)

    df = _coswara_stratify_cols(df)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_sympC finished ---")


def prepare_coswara_sexC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    # Label requires known gender — drop unknown.
    df = df[df["gender"].isin(["male", "female"])].copy()
    df["label"] = (df["gender"] == "male").astype(int)

    # Age missing/out-of-range -> 'unknown' bin (no row drops for stratification).
    age_bin = pd.cut(df["age"], bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_sexC finished ---")


def prepare_coswara_smokerC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # smoker column has: 'y', 'True', 'n', 'False'
    smoker_map = {"y": 1, "True": 1, "n": 0, "False": 0}
    df = df[df["smoker"].isin(smoker_map.keys())].copy()
    df["label"] = df["smoker"].map(smoker_map)

    df = _coswara_stratify_cols(df)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_smokerC finished ---")


def prepare_coswara_ageR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["age"].notna()].copy()
    df["label"] = df["age"].astype(float)

    df = _coswara_stratify_cols(df)
    # For regression, stratify on gender + age_bin
    df["age_bin"] = pd.cut(df["label"], bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])

    print("Distribution:", df["label"].describe().to_dict())
    # label=age (continuous) -> use age_bin as proxy. Order: label_proxy, sex.
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["age_bin", "gender"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_ageR finished ---")


def prepare_coswara_sympL(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Multi-label symptom classification."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Build multi-label vector from symptom columns
    def encode_symptoms(row):
        vec = []
        for col in COSWARA_SYMPTOM_COLS:
            val = row.get(col)
            vec.append(1 if str(val).strip().lower() == "true" else 0)
        return vec

    df["label"] = df.apply(encode_symptoms, axis=1)
    # Filter out rows with no symptoms at all
    df = df[df["label"].apply(lambda x: sum(x) > 0)].copy()

    df = _coswara_stratify_cols(df)

    print("Samples:", len(df))
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_coswara_sympL finished ---")
