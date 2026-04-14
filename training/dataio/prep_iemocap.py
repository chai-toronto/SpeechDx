"""
Preparing metadata for IEMOCAP tasks.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save


IEMOCAP_EMOTIONS = [
    "neu", "fru", "ang", "sad", "hap", "exc", "sur", "fea", "dis", "oth",
]
IEMOCAP_EMO_TO_IDX = {emo: i for i, emo in enumerate(IEMOCAP_EMOTIONS)}


def prepare_iemocap_emoC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Multiclass emotion classification. Excludes 'xxx'. Uses session-based split."""
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["label"].isin(IEMOCAP_EMOTIONS)].copy()
    df["label"] = df["label"].map(IEMOCAP_EMO_TO_IDX).astype(int)

    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]
    df_test = df[df["split"] == 2]

    print("Distribution:", df["label"].value_counts().sort_index().to_dict())
    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
    print("--- prepare_iemocap_emoC finished ---")
