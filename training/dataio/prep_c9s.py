"""
Preparing the metadata to go into preprocessing pipeline. This is for using with default
split, label, classic model fitting
"""
from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import speaker_stratified_split, to_sb_dict_and_save
from training.dataio.utils import ensure_dir, PathEncoder


def prepare_c9s_t1(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
):
    """
    This function is task-specific.
    """
    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]

    df = df[df["split_t1"].notna()]
    df['label'] = (df['Symptoms'] != 'None') & (df['Symptoms'].notna())

    # Speaker-independent stratified split
    df_train, df_val, df_test = speaker_stratified_split(df, ratio, random_seed)

    to_sb_dict_and_save(df_train,
                        manifest_train_path,
                        df_val,
                        manifest_val_path,
                        df_test,
                        manifest_test_path)

    print("Manifests created.")
    print("--- prepare_data finished ---")



