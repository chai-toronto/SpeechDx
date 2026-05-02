"""Shared manifest-building utilities.

Salvaged from ``training/dataio/prep_utils.py`` with internal imports
rewritten to point at ``ahb.prep.common_utils``. Each prep module imports
some subset of these helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

from ahb.prep.common_utils import ensure_dir, PathEncoder


def kfold_split(df, stratify_cols, group_col, n_splits, random_seed, shuffle=True):
    """Speaker-disjoint k-fold with stratification fallback.

    Joins ``stratify_cols`` into a single key and uses ``StratifiedGroupKFold``
    so no participant straddles folds. If sklearn rejects the key (e.g. a
    stratum has fewer than ``n_splits`` members), drops the right-most column
    and retries; final fallback is plain ``GroupKFold``.
    """
    work = df.reset_index(drop=True).copy()
    groups = work[group_col].to_numpy()
    X = np.zeros(len(work))
    cols = list(stratify_cols)
    while cols:
        try:
            key = work[cols].astype(str).agg("_".join, axis=1).to_numpy()
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_seed,
            )
            return [(work.iloc[tr].copy(), work.iloc[va].copy())
                    for tr, va in sgkf.split(X, key, groups)]
        except ValueError:
            cols = cols[:-1]
    gkf = GroupKFold(n_splits=n_splits)
    return [(work.iloc[tr].copy(), work.iloc[va].copy())
            for tr, va in gkf.split(X, groups=groups)]


def save_task_csv(df_train, df_val, df_test, dataset, task):
    """Save a combined CSV with updated label and split columns to metadata/<dataset>/<task>.csv.

    Split encoding: 0=train, 1=val, 2=test.
    """
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["split"] = 0
    df_val["split"] = 1
    df_test["split"] = 2

    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)

    out_dir = Path("metadata") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Task CSV saved to {out_path} ({len(df_combined)} rows)")


def save_kfold_task_csv(df, folds, dataset, task):
    """Save a CSV with per-fold split columns to metadata/<dataset>/<task>.csv."""
    df = df.copy()
    for i, (train_df, val_df) in enumerate(folds):
        df.loc[train_df.index, f"split_{i}"] = 0
        df.loc[val_df.index, f"split_{i}"] = 1

    out_dir = Path("metadata") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}.csv"
    df.to_csv(out_path, index=False)
    print(f"Task CSV saved to {out_path} ({len(df)} rows)")


def to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path):
    """Convert k-fold splits to SpeechBrain manifests (list of dicts per fold)."""
    import json

    train_dicts = []
    valid_dicts = []
    for train_df, val_df in folds:
        train_dicts.append(train_df.set_index('uid').to_dict(orient='index'))
        valid_dicts.append(val_df.set_index('uid').to_dict(orient='index'))

    print("Train og size:", len(train_dicts[-1]))
    print("Val og size:", len(valid_dicts[-1]))

    manifest_train_path = Path(manifest_train_path)
    manifest_val_path = Path(manifest_val_path)

    ensure_dir(manifest_train_path)
    with open(manifest_train_path, 'w') as f:
        json.dump(train_dicts, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_val_path)
    with open(manifest_val_path, 'w') as f:
        json.dump(valid_dicts, f, indent=5, cls=PathEncoder)

    print("Manifests created.")


def _make_stratify_key(speaker_df, stratify_cols):
    """Combine multiple columns into a single stratification key.
    Falls back to fewer columns if any combination has <2 members."""
    cols = list(stratify_cols)
    while cols:
        key = speaker_df[cols].astype(str).agg("_".join, axis=1)
        if key.value_counts().min() >= 2:
            return key
        cols = cols[:-1]
    return None


def speaker_stratified_split_train_val(df, ratio, random_seed, stratify_cols=None):
    """Speaker-independent stratified train/val split.

    Args:
        df: DataFrame with 'Participant_ID' and 'label' columns.
        ratio: List of 2 ints summing to 100, e.g. [90, 10].
        random_seed: Random seed for reproducibility.
        stratify_cols: List of column names to stratify on (at the speaker level).
            If None, defaults to ["label"]. Columns must exist in df.
    """
    if stratify_cols is None:
        stratify_cols = ["label"]

    if len(ratio) != 2:
        raise ValueError(f"ratio must have length 2 for train/val split, got {ratio}")

    if sum(ratio) != 100:
        raise ValueError(f"ratio must sum to 100, got {ratio}")

    speakers = df.groupby("Participant_ID")[stratify_cols].first().reset_index()
    stratify_key = _make_stratify_key(speakers, stratify_cols)

    val_size = ratio[1] / 100

    train_spk, val_spk = train_test_split(
        speakers["Participant_ID"],
        test_size=val_size,
        stratify=stratify_key,
        random_state=random_seed,
    )

    df_train = df[df["Participant_ID"].isin(train_spk)].copy()
    df_val = df[df["Participant_ID"].isin(val_spk)].copy()

    return df_train, df_val


def speaker_stratified_split(df, ratio, random_seed, stratify_cols=None):
    """Speaker-independent stratified train/val/test split.

    Args:
        df: DataFrame with 'Participant_ID' and 'label' columns.
        ratio: List of 3 ints summing to 100, e.g. [75, 10, 15].
        random_seed: Random seed for reproducibility.
        stratify_cols: List of column names to stratify on (at the speaker level).
            For regression tasks, pass a pre-binned column.
    """
    if stratify_cols is None:
        stratify_cols = ["label"]

    speakers = df.groupby("Participant_ID")[stratify_cols].first().reset_index()
    stratify_key = _make_stratify_key(speakers, stratify_cols)

    train_val_spk, test_spk = train_test_split(
        speakers["Participant_ID"],
        test_size=ratio[2] / 100,
        stratify=stratify_key,
        random_state=random_seed,
    )

    val_relative = ratio[1] / (ratio[0] + ratio[1])
    remaining = speakers.set_index("Participant_ID").loc[train_val_spk]
    remaining_key = _make_stratify_key(remaining, stratify_cols)

    train_spk, val_spk = train_test_split(
        train_val_spk,
        test_size=val_relative,
        stratify=remaining_key,
        random_state=random_seed,
    )

    df_train = df[df["Participant_ID"].isin(train_spk)]
    df_val = df[df["Participant_ID"].isin(val_spk)]
    df_test = df[df["Participant_ID"].isin(test_spk)]

    return df_train, df_val, df_test


def to_sb_dict_and_save(df_train: Series | DataFrame | Any,
                        manifest_train_path: str,
                        df_val: Series | DataFrame | Any,
                        manifest_val_path: str,
                        df_test: Series | DataFrame | Any,
                        manifest_test_path: str):

    manifest_train_path = Path(manifest_train_path)
    manifest_val_path = Path(manifest_val_path)
    manifest_test_path = Path(manifest_test_path)

    train = df_train.set_index('uid').to_dict(orient='index')
    val = df_val.set_index('uid').to_dict(orient='index')
    test = df_test.set_index("uid").to_dict(orient='index')

    print("Train og size:", len(train), "| subjects:", df_train["Participant_ID"].nunique())
    print("Val og size:", len(val), "| subjects:", df_val["Participant_ID"].nunique())
    print("Test size:", len(test), "| subjects:", df_test["Participant_ID"].nunique())

    import json
    ensure_dir(manifest_train_path)
    with open(manifest_train_path, 'w') as f:
        json.dump(train, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_val_path)
    with open(manifest_val_path, 'w') as f:
        json.dump(val, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_test_path)
    with open(manifest_test_path, 'w') as f:
        json.dump(test, f, indent=4, cls=PathEncoder)
