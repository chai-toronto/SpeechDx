"""
Preparing the metadata for AphasiaBank tasks.
No official split — use speaker-stratified split.
"""
from pathlib import Path
import pandas as pd
from sdx.prep.utils import save_task_csv, speaker_stratified_split, to_sb_dict_and_save


def prepare_aphasia_pwaC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Binary: PWA (1) vs Control (0)."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    # label column already exists (0=Control, 1=PWA)

    # Bin age (at speaker-level first value) into deciles for stratification
    age = pd.to_numeric(df["age_at_testing"], errors="coerce")
    df["age_bin"] = pd.qcut(age, q=5, labels=False, duplicates="drop")
    df["age_bin"] = df["age_bin"].fillna(-1).astype(int)

    stratify_cols = ["label", "gender", "age_bin"]

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=stratify_cols)
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_aphasia_pwaC finished ---")