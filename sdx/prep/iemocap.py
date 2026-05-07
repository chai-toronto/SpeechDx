"""
Preparing metadata for IEMOCAP tasks.
Speaker-disjoint k-fold CV.
"""
from pathlib import Path

import pandas as pd

from sdx.prep.utils import (
    kfold_split, save_kfold_task_csv, to_sb_kfold_dict_and_save,
)


IEMOCAP_EMOTIONS = {
    "neu": 0, "fru": 1, "ang": 1, "sad": 2, "hap": 3, "exc": 3,
}

BINARY_EMOTIONS = {
    "neu": 0, "fru": 1, "ang": 1, "sad": 1, "hap": 0, "exc": 0,
}


def _remap_label(folds, mapping):
    out = []
    for tr, va in folds:
        tr = tr.copy(); va = va.copy()
        tr["label"] = tr["label"].map(mapping).astype(int)
        va["label"] = va["label"].map(mapping).astype(int)
        out.append((tr, va))
    return out


def prepare_iemocap_emoC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """4-class emotion classification, speaker-disjoint k-fold (session-aware via Participant_ID)."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = df[df["label"].isin(IEMOCAP_EMOTIONS.keys())].reset_index(drop=True)

    folds = kfold_split(
        df, stratify_cols=[raw_label_key, "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    folds = _remap_label(folds, IEMOCAP_EMOTIONS)
    df["label"] = df["label"].map(IEMOCAP_EMOTIONS).astype(int)

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_iemocap_emoC ({dataset}_{task}) finished ---")


def prepare_iemocap_emoBC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """Binary emotion classification, speaker-disjoint k-fold.

    Stratifies on the original emotion string (richer balance) before
    collapsing ``label`` to 0/1.
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df = df[df["label"].isin(BINARY_EMOTIONS.keys())].reset_index(drop=True)

    folds = kfold_split(
        df, stratify_cols=[raw_label_key, "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    folds = _remap_label(folds, BINARY_EMOTIONS)
    df["label"] = df["label"].map(BINARY_EMOTIONS).astype(int)

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_iemocap_emoBC ({dataset}_{task}) finished ---")
