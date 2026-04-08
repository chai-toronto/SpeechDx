from pathlib import Path

import pandas as pd

from training.dataio.stratified_group_k_fold import stratified_group_kfold_df
from training.dataio.utils import ensure_dir, PathEncoder


def prepare_data(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        random_seed,
        raw_label_key,
        num_fold,
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

    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # split train data into folds, each goes by manifest id
    folds = stratified_group_kfold_df(df,
                                      'uid',
                                      raw_label_key,
                                      "Participant_ID",
                                      random_seed=random_seed,
                                      n_splits=num_fold)

    folds = list(folds)

    # Save per-fold split columns to metadata/mvdr/mvdr.csv
    for i, (train_df, val_df) in enumerate(folds):
        df.loc[train_df.index, f"split_{i}"] = 0
        df.loc[val_df.index, f"split_{i}"] = 1
    out_dir = Path("metadata") / "mvdr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mvdr.csv"
    df.to_csv(out_path, index=False)
    print(f"Task CSV saved to {out_path} ({len(df)} rows)")

    train_dicts = []
    valid_dicts = [] # list of folds
    for train_df, val_df in folds:
        train_dicts.append(train_df.set_index('uid').to_dict(orient='index'))
        valid_dicts.append(val_df.set_index('uid').to_dict(orient='index'))

    print("Train og size:", len(train_dicts[-1]))
    print("Val og size:", len(valid_dicts[-1]))

    import json
    ensure_dir(manifest_train_path)
    with open(manifest_train_path, 'w') as f:
        json.dump(train_dicts, f, indent=5, cls=PathEncoder)

    ensure_dir(manifest_val_path)
    with open(manifest_val_path, 'w') as f:
        json.dump(valid_dicts, f, indent=5, cls=PathEncoder)

    print("Manifests created.")
    print("--- prepare_data finished ---")

