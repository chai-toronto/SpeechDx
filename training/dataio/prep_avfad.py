"""
Preparing the metadata for AVFAD dataset tasks.
Speaker-stratified splits on gender + age.
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, speaker_stratified_split, to_sb_dict_and_save


def _avfad_stratify_cols(df):
    """Add gender + age_bin columns for stratification. Filters to valid rows."""
    df["Sex"] = df["Sex"].str.strip()
    df = df[df["Sex"].isin(["M", "F"])].copy()
    df["gender"] = df["Sex"].map({"M": "male", "F": "female"})
    df["age_bin"] = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["<30", "30-49", "50-69", "70+"])
    df = df.dropna(subset=["age_bin"])
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
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
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
    df["label"] = (df["Sex"] == "M").astype(int)

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
    # Smoking: 0=non-smoker, 1=smoker, 2=ex-smoker -> binary: 0 vs 1+2
    df["label"] = (df["Smoking"] > 0).astype(int)

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
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
    df["label"] = df["Age"].astype(float)

    print("Distribution:", df["label"].describe().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
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
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df = df[df["BMI"].notna()].copy()
    df["label"] = df["BMI"].astype(float)

    df = _avfad_stratify_cols(df)

    print("Distribution:", df["label"].describe().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=["gender", "age_bin"])
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_avfad_bmiR finished ---")
