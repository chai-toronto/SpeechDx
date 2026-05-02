from pathlib import Path
import pandas as pd
from ahb.prep.utils import save_task_csv, to_sb_dict_and_save, speaker_stratified_split_train_val


def _aphasia_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
    df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")

    age = pd.to_numeric(df.get("age_at_testing"), errors="coerce")
    df["age_bin"] = pd.qcut(age, q=5, labels=False, duplicates="drop")
    df["age_bin"] = df["age_bin"].fillna(-1).astype(int)
    return df


def _dbank_stratify_cols(df):
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower()
        df["gender"] = df["gender"].replace({"f": "female", "m": "male"})
        df["gender"] = df["gender"].where(df["gender"].isin(["female", "male"]), "unknown")
    else:
        df["gender"] = "unknown"
    return df


def _aphasia_pwa_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["label"].astype(int)
    return _aphasia_stratify_cols(df)


def _dbank_ad_df(wav_folder, metadata_path):
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["label"] = df["label"].astype(int)
    return _dbank_stratify_cols(df)


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_aphasia_dbank_pwaC_adC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_aphasia = _aphasia_pwa_df(wav_folder_train, metadata_path_train)
    df_dbank = _dbank_ad_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_aphasia, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_dbank.copy()
    _log_split_stats("Aphasia", "DBank", df_aphasia, df_dbank, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_dbank_aphasia_adC_pwaC(
    wav_folder_train, wav_folder_test, metadata_path_train, metadata_path_test,
    manifest_train_path, manifest_val_path, manifest_test_path,
    ratio, random_seed, dataset, task,
):
    df_dbank = _dbank_ad_df(wav_folder_train, metadata_path_train)
    df_aphasia = _aphasia_pwa_df(wav_folder_test, metadata_path_test)

    df_train, df_val = speaker_stratified_split_train_val(
        df_dbank, ratio, random_seed, stratify_cols=["label", "gender"]
    )
    df_test = df_aphasia.copy()
    _log_split_stats("DBank", "Aphasia", df_dbank, df_aphasia, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)
