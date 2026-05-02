"""Speaker-stratified group k-fold split helper.

Salvaged verbatim from ``training/dataio/stratified_group_k_fold.py``.
Used by the mvdr and ksof k-fold prep modules.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_kfold_df(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    group_col: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Stratified group k-fold over a DataFrame.

    Returns: list of (train_df, val_df) for each fold.
    - Stratifies by ``label_col``.
    - Keeps groups (by ``group_col``) intact across splits.
    """
    work_df = df.copy()

    y = work_df[label_col].to_numpy()
    groups = work_df[group_col].to_numpy()
    X = np.zeros(len(work_df))

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_seed,
    )

    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        train_df = work_df.iloc[train_idx].copy()
        val_df = work_df.iloc[val_idx].copy()
        folds.append((train_df, val_df))
    return folds
