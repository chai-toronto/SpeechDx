"""
Preparing metadata for MDVR tasks.
"""
from pathlib import Path

import pandas as pd

from sdx.prep.utils import save_kfold_task_csv, to_sb_kfold_dict_and_save
from sdx.prep.stratified_group_k_fold import stratified_group_kfold_df


def prepare_data(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold,
        dataset, task,
):
    """Generic MDVR task prep with k-fold CV."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Use the per-task raw_label_key as the training target.
    if raw_label_key and raw_label_key != "label" and raw_label_key in df.columns:
        df["label"] = df[raw_label_key]

    folds = list(stratified_group_kfold_df(
        df, 'uid', raw_label_key, "Participant_ID",
        random_seed=random_seed, n_splits=num_fold,
    ))

    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_data ({dataset}_{task}) finished ---")
