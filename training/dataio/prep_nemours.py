"""
Preparing the metadata for Nemours dataset tasks.
Speaker-stratified split (only 12 subjects).
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import speaker_stratified_split, to_sb_dict_and_save


def prepare_nemours_dysC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed,
):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    # label column already exists (0=healthy, 1=dysarthric)

    # Only 12 subjects — try gender stratification, fall back to label-only
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip().str.lower()
        stratify_cols = ["label", "gender"]
    else:
        stratify_cols = ["label"]

    print("Distribution:", df["label"].value_counts().to_dict())
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed, stratify_cols=stratify_cols)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_nemours_dysC finished ---")
