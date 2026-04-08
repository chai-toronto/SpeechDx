"""
Preparing the metadata for Torgo dataset tasks.
Uses official split column.
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import to_sb_dict_and_save


def prepare_torgo_sevR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed,
):
    """Severity regression: predict severity (0-3) for all speakers."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["severity"].astype(float)

    # Use official split
    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].value_counts().to_dict())
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_torgo_sevR finished ---")
