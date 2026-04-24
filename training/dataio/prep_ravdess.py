"""
Preparing metadata for RAVDESS tasks.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save

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
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Binary emotion classification"""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df["label"] = df["label"].map(BINARY_EMOTIONS).astype(int)

    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_emoBC finished ---")