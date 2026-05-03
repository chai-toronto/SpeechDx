import csv
import os
import random
import shutil
from pathlib import Path

# Coswara dataset (COVID-19 respiratory sounds)
# Source: date dirs containing participant dirs with audio files
# Audio files: counting-normal.wav and counting-fast.wav per participant
# Label: not assigned yet (covid_status kept as-is)
# Splits: 60/20/20 stratified random by covid_status, grouped by participant

DATA_ROOT = Path("data/coswara")
RAW_ROOT = DATA_ROOT / "raw"
OUT_ROOT = DATA_ROOT / "processed"
AUDIO_FILES = ["counting-normal.wav", "counting-fast.wav"]
RANDOM_SEED = 42


def build_pid_to_dir(data_root):
    """Scan date directories to map participant ID -> date_dir."""
    pid_to_dir = {}
    for entry in sorted(os.listdir(data_root)):
        d = data_root / entry
        if d.is_dir() and entry.isdigit():
            for pid_dir in os.listdir(d):
                if (d / pid_dir).is_dir():
                    pid_to_dir[pid_dir] = entry
    return pid_to_dir


def make_stratified_splits(rows, seed):
    """60/20/20 stratified by covid_status, grouped by Participant_ID."""
    random.seed(seed)

    # Group participants by covid_status
    status_to_pids = {}
    for r in rows:
        status = r["covid_status"]
        pid = r["Participant_ID"]
        status_to_pids.setdefault(status, set()).add(pid)

    train_pids, val_pids, test_pids = set(), set(), set()
    for status, pids in status_to_pids.items():
        pids = sorted(pids)
        random.shuffle(pids)
        n = len(pids)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2))
        test_pids.update(pids[:n_test])
        val_pids.update(pids[n_test:n_test + n_val])
        train_pids.update(pids[n_test + n_val:])

    for r in rows:
        pid = r["Participant_ID"]
        if pid in train_pids:
            r["split"] = 0
        elif pid in val_pids:
            r["split"] = 1
        else:
            r["split"] = 2


def main():
    # Support reading raw data from an alternate source (e.g. external drive).
    # Default layout assumes the upstream Coswara-Data repo is cloned into
    # data/coswara/raw/ and its extract_data.py has been run, producing
    # data/coswara/raw/Extracted_data/<YYYYMMDD>/<pid>/*.wav alongside
    # data/coswara/raw/combined_data.csv.
    import sys
    raw_root = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_ROOT
    audio_root = raw_root / "Extracted_data"
    if not audio_root.is_dir():
        # Backwards compat: support callers who pass the Extracted_data dir directly.
        audio_root = raw_root

    pid_to_dir = build_pid_to_dir(audio_root)
    # Also scan processed/audio/ for existing date dirs
    processed_audio = OUT_ROOT / "audio"
    if processed_audio.is_dir():
        pid_to_dir.update(build_pid_to_dir(processed_audio))

    # Try raw combined_data.csv first; fall back to existing processed CSV
    csv_path = raw_root / "combined_data.csv"
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            src_rows = list(csv.DictReader(f))
        # Column rename mapping from combined_data.csv -> output CSV
        RENAME = {
            "id": "Participant_ID",
            "a": "age",
            "g": "gender",
            "l_c": "country",
            "l_l": "locality",
            "l_s": "state",
            "rU": "returning_user",
            "ep": "english_proficient",
            "mp": "muscularpain",
            "um": "use_mask",
            "vacc": "vaccinated",
            "bd": "breathing_difficulty",
            "ftg": "fatigue",
            "st": "sore_throat",
            "ihd": "ischemic_heart_disease",
            "cld": "chronic_lung_disease",
        }
        # Columns to pass through without renaming
        PASSTHROUGH = [
            "covid_status", "record_date", "smoker", "cold", "ht",
            "diabetes", "cough", "ctDate", "ctScan", "ctScore",
            "diarrhoea", "fever", "loss_of_smell", "testType",
            "test_date", "test_status", "others_resp", "asthma",
            "others_preexist", "pneumonia",
        ]

        rows = []
        for r in src_rows:
            pid = r["id"]
            if pid not in pid_to_dir:
                continue
            date_dir = pid_to_dir[pid]
            for audio_file in AUDIO_FILES:
                rel_path = f"{date_dir}/{pid}/{audio_file}"
                row = {
                    "audio_type": audio_file.replace(".wav", ""),
                    "path": rel_path,
                }
                # Apply renames
                for src_col, dst_col in RENAME.items():
                    row[dst_col] = r.get(src_col, "")
                # Pass through remaining columns
                for col in PASSTHROUGH:
                    row[col] = r.get(col, "")
                rows.append(row)
    else:
        # Rebuild from existing processed CSV + raw audio dirs
        existing_csv = OUT_ROOT / "coswara.csv"
        print(f"No combined_data.csv found, rebuilding from {existing_csv}")
        with open(existing_csv, newline='') as f:
            existing_rows = list(csv.DictReader(f))

        # Deduplicate to one entry per participant (existing may only have counting-normal)
        seen = {}
        for r in existing_rows:
            pid = r["Participant_ID"]
            if pid not in seen:
                seen[pid] = r

        rows = []
        for pid, r in seen.items():
            if pid not in pid_to_dir:
                # Use date dir from existing path
                date_dir = r["path"].split("/")[0]
            else:
                date_dir = pid_to_dir[pid]
            for audio_file in AUDIO_FILES:
                rel_path = f"{date_dir}/{pid}/{audio_file}"
                row = {
                    "Participant_ID": pid,
                    "audio_type": audio_file.replace(".wav", ""),
                    "path": rel_path,
                }
                # Carry over all existing columns (except uid, split, path which are regenerated)
                for col in r:
                    if col not in ("uid", "split", "path", "Participant_ID", "audio_type"):
                        row[col] = r[col]
                rows.append(row)

    make_stratified_splits(rows, RANDOM_SEED)

    # ---- Copy audio to processed/audio/ ----
    dest_audio_dir = OUT_ROOT / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    copied, skipped, dropped = 0, 0, 0
    for row in rows:
        src = audio_root / row["path"]
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
    out_path = OUT_ROOT / "coswara.csv"

    fieldnames = [
        "uid", "Participant_ID", "split", "path",
        "covid_status", "age", "gender", "country", "locality", "state",
        "record_date", "english_proficient", "returning_user",
        "smoker", "cold", "ht", "diabetes", "cough",
        "ctDate", "ctScan", "ctScore",
        "diarrhoea", "fever", "loss_of_smell", "muscularpain",
        "testType", "test_date", "test_status",
        "use_mask", "vaccinated", "breathing_difficulty",
        "others_resp", "fatigue", "sore_throat",
        "ischemic_heart_disease", "asthma", "others_preexist",
        "chronic_lung_disease", "pneumonia",
        "audio_type",
    ]

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
    print(f"  Covid statuses:")
    statuses = {}
    for r in rows:
        statuses[r['covid_status']] = statuses.get(r['covid_status'], 0) + 1
    for k, v in sorted(statuses.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
