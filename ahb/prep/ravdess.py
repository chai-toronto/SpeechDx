"""
Preparing metadata for RAVDESS tasks.
Speaker-disjoint k-fold CV.
"""
from pathlib import Path

import pandas as pd

from ahb.prep.utils import (
    kfold_split, save_kfold_task_csv, to_sb_kfold_dict_and_save,
)

BINARY_EMOTIONS = {
    0: 0, # Neu
    1: 0, # Calm
    2: 0, # happy
    3: 1, # sad
    4: 1, # angry
    5: 1, # fearful
    6: 1, # disgust
    7: 0  # surprised
}


def prepare_emoBC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """Binary emotion classification, speaker-disjoint k-fold.

    Stratifies on the original 8-emotion label (richer balance) before
    collapsing the saved ``label`` column to the binary target.
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    folds = kfold_split(
        df, stratify_cols=[raw_label_key, "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    df["label"] = df["label"].map(BINARY_EMOTIONS).astype(int)
    folds = [(tr.assign(label=tr["label"].map(BINARY_EMOTIONS).astype(int)),
              va.assign(label=va["label"].map(BINARY_EMOTIONS).astype(int)))
             for tr, va in folds]

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_emoBC ({dataset}_{task}) finished ---")


def prepare_emoC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """8-class emotion classification, speaker-disjoint k-fold."""
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
    print(f"--- prepare_emoC ({dataset}_{task}) finished ---")
