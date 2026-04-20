"""
Preparing the metadata to go into preprocessing pipeline. This is for using with default
split, label, classic model fitting
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save


def prepare_data(
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
    This function is dataset-specific.
    It takes raw data info (like metadata_path and wav_folder) and
    creates the SpeechBrain manifest JSON files (train, valid, test).

    Replace the body of this function with the logic needed for your
    specific dataset (e.g., reading a CSV/TSV file and writing JSONs).

    It first takes the test split out of metadata. Then it splits the
    rest deterministically with seed. Each train manifest now contains
    k subdict for each fold. Each is like the original format.
    """
    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    if "boundaries" in df.columns:
        df["boundaries"] = df["boundaries"].apply(json.loads)

    df_test = df[df['split'] == 2]
    df_train = df[df['split'] == 0]
    df_val = df[df['split'] == 1]

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)

    print("Manifests created.")
    print("--- prepare_data finished ---")



