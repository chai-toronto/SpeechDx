from pathlib import Path
import pandas as pd
from training.dataio.prep_utils import save_task_csv, speaker_stratified_split_train_val, to_sb_dict_and_save
from training.dataio.prep_aphasia_dbank import _aphasia_pwa_df, _dbank_ad_df
from training.dataio.prep_c9s import _c9s_stratify_cols
from training.dataio.prep_coswara import _coswara_stratify_cols, COSWARA_SYMPTOM_COLS
from training.dataio.prep_avfad import _avfad_stratify_cols

DATASET_TASKS = {"aphasia": ["pwaC"], "dbank": ["adC"], "c9s": ["t1", "t2"], "coswara": ["sympC", "covidC"], "avfad": ["pathC"]}
AGE_STRATIFIED_TASKS = {("aphasia", "pwaC"), ("c9s", "t1"), ("c9s", "t2"), ("coswara", "sympC"), ("coswara", "covidC"), ("avfad", "pathC")}
OR_MERGE_DATASETS = {"c9s", "coswara"}

def _paths_for_dataset(dataset, audio_roots, meta_paths):
    wav = (audio_roots or {}).get(dataset)
    meta = (meta_paths or {}).get(dataset)
    if not wav or not meta:
        raise ValueError(
            f"Missing dataset path mapping for '{dataset}'. "
            "Expected both dataset_audio_roots and dataset_metadata_paths entries."
        )
    return Path(wav), Path(meta)

def _load_task_df(dataset, task, wav_folder, metadata_path):
    if dataset == "aphasia" and task == "pwaC": return _aphasia_pwa_df(str(wav_folder), str(metadata_path))
    if dataset == "dbank" and task == "adC": return _dbank_ad_df(str(wav_folder), str(metadata_path))
    if dataset == "c9s" and task == "t1":
        df = pd.read_csv(metadata_path); df["path"] = Path(wav_folder).resolve()/df["path"]; df = df[df["split_t1"].notna()]; df["label"] = (df["Symptoms"] != "None") & (df["Symptoms"].notna()); return _c9s_stratify_cols(df)
    if dataset == "c9s" and task == "t2":
        df = pd.read_csv(metadata_path); df["path"] = Path(wav_folder).resolve()/df["path"]; df = df[df["label_t2"].notna()]; df["label"] = (df["label_t2"] == 1.0).astype(int); return _c9s_stratify_cols(df)
    if dataset == "coswara" and task == "sympC":
        df = pd.read_csv(metadata_path); df["path"] = Path(wav_folder).resolve()/df["path"]; m = df[COSWARA_SYMPTOM_COLS].fillna(False)
        for col in COSWARA_SYMPTOM_COLS: m[col] = m[col].apply(lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False)
        df["label"] = m.any(axis=1).astype(int); return _coswara_stratify_cols(df)
    if dataset == "coswara" and task == "covidC":
        df = pd.read_csv(metadata_path); df["path"] = Path(wav_folder).resolve()/df["path"]; pos = ["positive_asymp","positive_mild","positive_moderate"]; df = df[df["covid_status"].isin(pos+["healthy"])].copy(); df["label"] = df["covid_status"].isin(pos).astype(int); return _coswara_stratify_cols(df)
    if dataset == "avfad" and task == "pathC":
        df = pd.read_csv(metadata_path); df["path"] = Path(wav_folder).resolve()/df["path"]; return _avfad_stratify_cols(df)
    raise ValueError(f"Unsupported dataset/task for category prep: {dataset}/{task}")

def _merge_multitask_rows(dataset, frames):
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if dataset not in OR_MERGE_DATASETS or "uid" not in df.columns:
        return df
    if df["label"].dtype == bool:
        df["label"] = df["label"].astype(int)
    elif pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(float).round().astype(int)
    keep_cols = [c for c in df.columns if c not in {"uid", "label", "source_task"}]
    merged = (
        df.groupby("uid", as_index=False)
        .agg({"label": "max", **{c: "first" for c in keep_cols}})
    )
    merged["source_task"] = "or_union"
    return merged

def _prep_category_split(train_manifest,val_manifest,test_manifest,ratio,random_seed,dataset,task,train_datasets=None,test_datasets=None,category_settings=None,dataset_audio_roots=None,dataset_metadata_paths=None):
    train_datasets = list(train_datasets or []); test_datasets = list(test_datasets or []); settings = list(category_settings or [])
    train_frames = []
    for idx, ds in enumerate(train_datasets):
        wav, meta = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        ds_frames = []
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav, meta).copy()
            df["source_dataset"], df["source_task"], df["setting_idx"] = ds, tk, idx
            cfg = settings[idx] if idx < len(settings) else {}
            df["split_by_boundary"] = bool(cfg.get("split_by_boundary", False))
            df["num_aug_ver"] = int(cfg.get("num_aug_ver", 1))
            ds_frames.append(df)
        train_frames.append(_merge_multitask_rows(ds, ds_frames))
    test_frames = []
    for ds in test_datasets:
        wav, meta = _paths_for_dataset(ds, dataset_audio_roots, dataset_metadata_paths)
        ds_frames = []
        for tk in DATASET_TASKS[ds]:
            df = _load_task_df(ds, tk, wav, meta).copy()
            df["source_dataset"], df["source_task"] = ds, tk
            df["split_by_boundary"], df["num_aug_ver"] = False, 1
            ds_frames.append(df)
        test_frames.append(_merge_multitask_rows(ds, ds_frames))
    src, tgt = pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)
    for df in (src, tgt):
        if "gender" not in df.columns: df["gender"] = "unknown"
        if "age_bin" not in df.columns: df["age_bin"] = "unknown"
        df["gender"] = df["gender"].astype(str).fillna("unknown")
        df["age_bin"] = df["age_bin"].astype(str).fillna("unknown")
        if df["label"].dtype == bool: df["label"] = df["label"].astype(int)
        elif pd.api.types.is_numeric_dtype(df["label"]): df["label"] = df["label"].astype(float).round().astype(int)
        df["cache_uid"] = df["uid"].astype(str)
        df["uid"] = df["source_dataset"] + "_" + df["source_task"] + "::" + df["cache_uid"]
        df["Participant_ID"] = df["source_dataset"] + "_" + df["source_task"] + "::" + df["Participant_ID"].astype(str)
    strat = ["label", "gender", "source_dataset"]
    src_tasks = {(r.source_dataset, r.source_task) for r in src[["source_dataset","source_task"]].drop_duplicates().itertuples(index=False)}
    if src_tasks and all(t in AGE_STRATIFIED_TASKS for t in src_tasks): strat.append("age_bin")
    train_df, val_df = speaker_stratified_split_train_val(src, ratio, random_seed, stratify_cols=strat)
    save_task_csv(train_df, val_df, tgt.copy(), dataset, task)
    to_sb_dict_and_save(train_df, train_manifest, val_df, val_manifest, tgt.copy(), test_manifest)

def prepare_category_c2_c4(*args, **kwargs): return _prep_category_split(*args, **kwargs)
def prepare_category_c4_c2(*args, **kwargs): return _prep_category_split(*args, **kwargs)
