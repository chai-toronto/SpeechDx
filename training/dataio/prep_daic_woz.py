"""
Preparing the metadata for DAIC-WOZ dataset tasks.
Uses official split column.
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import to_sb_dict_and_save


def prepare_daic_woz_phqR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset=None, task=None,
):
    """PHQ-8 score regression."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["PHQ8_Score"].astype(float)

    # Use official split
    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].describe().to_dict())
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_daic_woz_phqR finished ---")
