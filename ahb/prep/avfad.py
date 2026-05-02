"""
Preparing the metadata for AVFAD dataset tasks.
Speaker-stratified splits on gender + age.
"""
from pathlib import Path
import pandas as pd
from ahb.prep.utils import save_task_csv, speaker_stratified_split, to_sb_dict_and_save


def _avfad_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Missing/invalid -> 'unknown' (no row drops)."""
    df = df.copy()
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["gender"] = df["Sex"].map({"M": "male", "F": "female"}).fillna("unknown")
    age_bin = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["<30", "30-49", "50-69", "70+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df


def prepare_avfad_pathC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = _avfad_stratify_cols(df)
    # label column already exists (0=healthy, 1=pathological)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_pathC finished ---")


def prepare_avfad_sexC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = _avfad_stratify_cols(df)
    # Label requires known gender — drop unknown.
    df = df[df["gender"].isin(["male", "female"])].copy()
    df["label"] = (df["gender"] == "male").astype(int)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_sexC finished ---")


def prepare_avfad_smokerC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = _avfad_stratify_cols(df)
    # Smoking: 0=non-smoker, 1=smoker, 2=ex-smoker -> drop ex-smokers, binary: 0 vs 1
    df = df[df["Smoking"].isin([0, 1])].copy()
    df["label"] = (df["Smoking"] == 1).astype(int)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_smokerC finished ---")


def prepare_avfad_ageR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = _avfad_stratify_cols(df)
    # Label requires valid age.
    df = df[df["Age"].notna()].copy()
    df["label"] = df["Age"].astype(float)

    print("Distribution:", df["label"].describe().to_dict())
    # label=age (continuous) -> use age_bin as proxy. Order: label_proxy, sex.
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["age_bin", "gender"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_ageR finished ---")


def prepare_avfad_bmiR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    # Drop rows where BMI is "." or NaN before numeric conversion
    df = df[df["BMI"].astype(str).str.strip() != "."]
    df = df.dropna(subset=["BMI"])
    df["BMI"] = df["BMI"].astype(float)
    df["label"] = df["BMI"]

    df = _avfad_stratify_cols(df)

    # BMI categories (WHO): underweight <18.5, normal 18.5-24.9, overweight 25-29.9, obese >=30
    df["bmi_cat"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 200],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False,
    )

    print("Distribution:", df["label"].describe().to_dict())
    print("BMI categories:", df["bmi_cat"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["bmi_cat", "gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_bmiR finished ---")
