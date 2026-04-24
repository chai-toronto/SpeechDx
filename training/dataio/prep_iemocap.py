"""
Preparing metadata for IEMOCAP tasks.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save


IEMOCAP_EMOTIONS = {
    "neu": 0, "fru": 1, "ang": 1, "sad": 2, "hap": 3, "exc":3,

}

BINARY_EMOTIONS = {
    "neu": 0, "fru": 1, "ang": 1, "sad": 1, "hap": 0, "exc":0,

}

def prepare_iemocap_emoC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Multiclass emotion classification. Uses session-based split."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["label"].isin(IEMOCAP_EMOTIONS.keys())].copy()
    df["label"] = df["label"].map(IEMOCAP_EMOTIONS).astype(int)

    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_iemocap_emoC finished ---")


def prepare_iemocap_emoBC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Binary emotion classification. Uses session-based split."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["label"].isin(BINARY_EMOTIONS.keys())].copy()
    df["label"] = df["label"].map(BINARY_EMOTIONS).astype(int)

    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_iemocap_emoBC finished ---")