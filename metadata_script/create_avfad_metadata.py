import csv
import random
import shutil
from pathlib import Path

import pandas as pd

# AVFAD dataset (voice pathology)
# Source: /Users/lkieu/Downloads/avfad (or passed as argv[1])
# Metadata: AVFAD_01_00_00.xlsx
# Audio: split across 4 folders (A_to_C, D_to_L, M, N_to_Z)
# Files per participant: *004-*011 (CAPE-V sentences, reading, spontaneous speech)
# Excluded: *001-*003 (sustained vowels)
# Label: 0 = Normal, 1 = Pathological (any CMVD-I != 0)
# Splits: 60/20/20 stratified by label, grouped by participant

DATA_ROOT = Path("data/avfad")
OUT_ROOT = DATA_ROOT / "processed"

AUDIO_SUFFIXES = ["004", "005", "006", "007", "008", "009", "010", "011"]
AUDIO_FOLDERS = [
    "AVFAD_01_00_00_2_A_to_C",
    "AVFAD_01_00_00_3_D_to_L",
    "AVFAD_01_00_00_4_M",
    "AVFAD_01_00_00_5_N_to_Z",
]
RANDOM_SEED = 42


def make_stratified_splits(participants, seed):
    """60/20/20 stratified by label, grouped by participant."""
    random.seed(seed)

    normal = sorted([p for p in participants if p["label"] == 0], key=lambda x: x["pid"])
    pathol = sorted([p for p in participants if p["label"] == 1], key=lambda x: x["pid"])

    random.shuffle(normal)
    random.shuffle(pathol)

    def split_list(ids):
        n = len(ids)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2))
        test = set(p["pid"] for p in ids[:n_test])
        val = set(p["pid"] for p in ids[n_test:n_test + n_val])
        train = set(p["pid"] for p in ids[n_test + n_val:])
        return train, val, test

    n_train, n_val, n_test = split_list(normal)
    p_train, p_val, p_test = split_list(pathol)

    train = n_train | p_train
    val = n_val | p_val
    test = n_test | p_test
    return train, val, test


def main():
    import sys
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_ROOT

    # ---- Load metadata (all columns) ----
    df = pd.read_excel(src_root / "AVFAD_01_00_00.xlsx")
    df["label"] = (df["CMVD-I Dimension 1 (numeric system)"] != 0).astype(int)

    meta = {}
    extra_cols = [c for c in df.columns if c != "File ID"]
    for _, r in df.iterrows():
        pid = r["File ID"]
        meta[pid] = {"pid": pid}
        for c in extra_cols:
            meta[pid][c] = r[c]

    # ---- Build participant -> source directory mapping ----
    pid_to_src = {}
    for folder in AUDIO_FOLDERS:
        folder_path = src_root / folder
        if not folder_path.exists():
            continue
        for pid_dir in folder_path.iterdir():
            if pid_dir.is_dir():
                pid_to_src[pid_dir.name] = pid_dir

    # ---- Stratified splits ----
    participants = [meta[pid] for pid in meta if pid in pid_to_src]
    train_set, val_set, test_set = make_stratified_splits(participants, RANDOM_SEED)

    def get_split(pid):
        if pid in train_set:
            return 0
        elif pid in val_set:
            return 1
        return 2

    # ---- Build rows (one per audio file) ----
    rows = []
    for pid in sorted(pid_to_src):
        if pid not in meta:
            print(f"WARNING: {pid} not in metadata, skipping")
            continue
        m = meta[pid]
        for suffix in AUDIO_SUFFIXES:
            row = {"Participant_ID": pid}
            for c in extra_cols:
                row[c] = m[c]
            row["audio_type"] = suffix
            row["split"] = get_split(pid)
            row["path"] = f"{pid}/{pid}{suffix}.wav"
            rows.append(row)

    # ---- Copy audio to processed/audio/ ----
    dest_audio_dir = OUT_ROOT / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    copied, skipped, dropped = 0, 0, 0
    for row in rows:
        pid = row["Participant_ID"]
        src = pid_to_src[pid] / f"{pid}{row['audio_type']}.wav"
        dst = dest_audio_dir / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            skipped += 1
        elif src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"WARNING: dropping (missing audio) {src}")
            dropped += 1
            continue
        valid_rows.append(row)
    rows = valid_rows
    print(f"Audio: copied {copied}, skipped {skipped}, dropped {dropped}")

    # ---- Write CSV ----
    out_path = OUT_ROOT / "avfad.csv"

    fieldnames = ["uid", "Participant_ID"] + list(extra_cols) + ["audio_type", "split", "path"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            writer.writerow(row)

    print(f"Created {out_path} with {len(rows)} rows")
    print(f"  Train (0): {sum(1 for r in rows if r['split'] == 0)}")
    print(f"  Val (1):   {sum(1 for r in rows if r['split'] == 1)}")
    print(f"  Test (2):  {sum(1 for r in rows if r['split'] == 2)}")
    print(f"  Label 0 (normal): {sum(1 for r in rows if r['label'] == 0)}")
    print(f"  Label 1 (pathol): {sum(1 for r in rows if r['label'] == 1)}")


if __name__ == "__main__":
    main()
