from pathlib import Path
from typing import Any

from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

from training.dataio.utils import ensure_dir, PathEncoder


def speaker_stratified_split(df, ratio, random_seed):
    """
    Split a DataFrame into train/val/test with speaker-independent,
    label-stratified partitioning.

    Args:
        df: DataFrame with 'Participant_ID' and 'label' columns.
        ratio: List of 3 ints summing to 100, e.g. [75, 10, 15].
        random_seed: Random seed for reproducibility.

    Returns:
        (df_train, df_val, df_test)
    """
    speakers = df.groupby("Participant_ID")["label"].first().reset_index()

    train_val_spk, test_spk = train_test_split(
        speakers["Participant_ID"],
        test_size=ratio[2] / 100,
        stratify=speakers["label"],
        random_state=random_seed,
    )

    val_relative = ratio[1] / (ratio[0] + ratio[1])
    train_spk, val_spk = train_test_split(
        train_val_spk,
        test_size=val_relative,
        stratify=speakers.set_index("Participant_ID").loc[train_val_spk, "label"],
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

    print("Train og size:", len(train))
    print("Val og size:", len(val))
    print("Test size:", len(test))

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