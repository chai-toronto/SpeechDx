"""
Preparing metadata for UASpeech dataset tasks.
Speaker-disjoint k-fold CV.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import (
    kfold_split, save_kfold_task_csv, to_sb_kfold_dict_and_save,
)


def prepare_data(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """Binary dysarthria classification, speaker-disjoint k-fold."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    folds = kfold_split(
        df, stratify_cols=[raw_label_key, "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_data ({dataset}_{task}) finished ---")
