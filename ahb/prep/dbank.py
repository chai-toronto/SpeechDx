"""
Preparing the metadata for ADReSS-M / dbank tasks.
Uses the official split column shipped with the challenge.
"""
import json
from pathlib import Path

import pandas as pd

from ahb.prep.utils import save_task_csv, to_sb_dict_and_save


def prepare_dbank_mmseR(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """MMSE score regression."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df.dropna(subset=["mmse"]).copy()
    df["label"] = df["mmse"].astype(float)

    if "boundaries" in df.columns:
        df["boundaries"] = df["boundaries"].apply(json.loads)

    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].describe().to_dict())
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_dbank_mmseR finished ---")
