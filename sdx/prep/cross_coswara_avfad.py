from pathlib import Path
import pandas as pd
from sdx.prep.utils import (
    save_task_csv,
    speaker_stratified_split_train_val,
    to_sb_dict_and_save,
)

COSWARA_SYMPTOM_COLS = [
    "cold",
    "cough",
    "fever",
    "diarrhoea",
    "loss_of_smell",
    "muscularpain",
    "breathing_difficulty",
    "fatigue",
    "sore_throat",
    "others_resp",
]


def _avfad_stratify_cols(df):
    df = df.copy()
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["gender"] = df["Sex"].map({"M": "male", "F": "female"}).fillna("unknown")
    age_bin = pd.cut(df["Age"], bins=[0, 30, 50, 70, 100], labels=["<30", "30-49", "50-69", "70+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df


def _coswara_stratify_cols(df):
    df = df.copy()
    df["gender"] = df["gender"].where(df["gender"].isin(["male", "female"]), "unknown")
    age_bin = pd.cut(df["age"], bins=[0, 20, 40, 60, 100], labels=["<20", "20-39", "40-59", "60+"])
    df["age_bin"] = age_bin.cat.add_categories(["unknown"]).fillna("unknown")
    return df


def _build_coswara_symp_labels(df):
    symp_matrix = df[COSWARA_SYMPTOM_COLS].fillna(False)
    for col in COSWARA_SYMPTOM_COLS:
        symp_matrix[col] = symp_matrix[col].apply(
            lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False
        )
    out = df.copy()
    out["label"] = symp_matrix.any(axis=1).astype(int)
    return out


def _build_coswara_covid_labels(df):
    positive = ["positive_asymp", "positive_mild", "positive_moderate"]
    out = df[df["covid_status"].isin(positive + ["healthy"])].copy()
    out["label"] = out["covid_status"].isin(positive).astype(int)
    return out


def _log_split_stats(train_name, test_name, df_train_src, df_test_src, df_train, df_val, df_test):
    print(f"{train_name} overall distribution:", df_train_src["label"].value_counts().to_dict())
    print(f"{test_name} overall distribution:", df_test_src["label"].value_counts().to_dict())
    print("Train distribution:", df_train["label"].value_counts().to_dict())
    print("Val distribution:", df_val["label"].value_counts().to_dict())
    print("Test distribution:", df_test["label"].value_counts().to_dict())
    print("Train speakers:", df_train["Participant_ID"].nunique())
    print("Val speakers:", df_val["Participant_ID"].nunique())
    print("Test speakers:", df_test["Participant_ID"].nunique())


def prepare_avfad_coswara_pathC_sympC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_avfad = pd.read_csv(metadata_path_train)
    df_avfad["path"] = Path(wav_folder_train).resolve() / df_avfad["path"]

    df_coswara = pd.read_csv(metadata_path_test)
    df_coswara["path"] = Path(wav_folder_test).resolve() / df_coswara["path"]
    df_coswara = _build_coswara_symp_labels(df_coswara)

    df_avfad = _avfad_stratify_cols(df_avfad)
    df_coswara = _coswara_stratify_cols(df_coswara)

    df_train, df_val = speaker_stratified_split_train_val(
        df_avfad, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_coswara.copy()
    _log_split_stats("AVFAD", "Coswara", df_avfad, df_coswara, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_coswara_avfad_sympC_pathC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_coswara = pd.read_csv(metadata_path_train)
    df_coswara["path"] = Path(wav_folder_train).resolve() / df_coswara["path"]
    df_coswara = _build_coswara_symp_labels(df_coswara)

    df_avfad = pd.read_csv(metadata_path_test)
    df_avfad["path"] = Path(wav_folder_test).resolve() / df_avfad["path"]

    df_avfad = _avfad_stratify_cols(df_avfad)
    df_coswara = _coswara_stratify_cols(df_coswara)

    df_train, df_val = speaker_stratified_split_train_val(
        df_coswara, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_avfad.copy()
    _log_split_stats("Coswara", "AVFAD", df_coswara, df_avfad, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_avfad_coswara_pathC_covidC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_avfad = pd.read_csv(metadata_path_train)
    df_avfad["path"] = Path(wav_folder_train).resolve() / df_avfad["path"]

    df_coswara = pd.read_csv(metadata_path_test)
    df_coswara["path"] = Path(wav_folder_test).resolve() / df_coswara["path"]
    df_coswara = _build_coswara_covid_labels(df_coswara)

    df_avfad = _avfad_stratify_cols(df_avfad)
    df_coswara = _coswara_stratify_cols(df_coswara)

    df_train, df_val = speaker_stratified_split_train_val(
        df_avfad, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_coswara.copy()
    _log_split_stats("AVFAD", "Coswara", df_avfad, df_coswara, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


def prepare_coswara_avfad_covidC_pathC(
    wav_folder_train,
    wav_folder_test,
    metadata_path_train,
    metadata_path_test,
    manifest_train_path,
    manifest_val_path,
    manifest_test_path,
    ratio,
    random_seed,
    dataset,
    task,
):
    df_coswara = pd.read_csv(metadata_path_train)
    df_coswara["path"] = Path(wav_folder_train).resolve() / df_coswara["path"]
    df_coswara = _build_coswara_covid_labels(df_coswara)

    df_avfad = pd.read_csv(metadata_path_test)
    df_avfad["path"] = Path(wav_folder_test).resolve() / df_avfad["path"]

    df_avfad = _avfad_stratify_cols(df_avfad)
    df_coswara = _coswara_stratify_cols(df_coswara)

    df_train, df_val = speaker_stratified_split_train_val(
        df_coswara, ratio, random_seed, stratify_cols=["label", "gender", "age_bin"]
    )
    df_test = df_avfad.copy()
    _log_split_stats("Coswara", "AVFAD", df_coswara, df_avfad, df_train, df_val, df_test)

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(df_train, manifest_train_path, df_val, manifest_val_path, df_test, manifest_test_path)


