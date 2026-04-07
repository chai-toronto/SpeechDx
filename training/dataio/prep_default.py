"""
Preparing the metadata to go into preprocessing pipeline. This is for using with default
split, label, classic model fitting
"""
from pathlib import Path
import pandas as pd
from training.dataio.utils import ensure_dir, PathEncoder


def prepare_data(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
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
    manifest_train_path = Path(manifest_train_path)
    manifest_val_path = Path(manifest_val_path)
    manifest_test_path = Path(manifest_test_path)

    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df_test = df[df['split'] == 2]
    df_train_og = df[df['split'] == 0]
    df_val_og = df[df['split'] == 1]

    train_dicts = []
    train_dicts.append(df_train_og.set_index('uid').to_dict(orient='index'))
    # Append dataset_all to list of datasets
    train_dicts.append(df.set_index('uid').to_dict(orient='index'))

    val = df_val_og.set_index('uid').to_dict(orient='index')

    test = df_test.set_index("uid").to_dict(orient='index')

    print("Train og size:", len(df_train_og))
    print("Val og size:", len(val))
    print("Test size:", len(test))

    import json
    ensure_dir(manifest_train_path)
    with open(manifest_train_path, 'w') as f:
        json.dump(train_dicts, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_val_path)
    with open(manifest_val_path, 'w') as f:
        json.dump(val, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_test_path)
    with open(manifest_test_path, 'w') as f:
        json.dump(test, f, indent=4, cls=PathEncoder)

    print("Manifests created.")
    print("--- prepare_data finished ---")
