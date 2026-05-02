"""
Create unified COVID-19 Sounds (c9s) metadata CSV.

Merges the 3 platform CSVs (android/ios/web) with task 1 and task 2
split/label info. Copies voice audio from Covid19SoundFull.

Source data:
  - data/c9s/results_raw_20210426_lan_yamnet_{android,ios,web}_noloc.csv
  - data/c9s/data_0426_en_task1.csv
  - data/c9s/data_0426_en_task2.csv
  - data/Covid19SoundFull/  (read-only audio)

Output:
  - data/c9s/processed/c9s.csv
  - data/c9s/processed/audio/{Participant_ID}/{FolderName}/{voice_file}
"""

import csv
import shutil
from pathlib import Path

DATA_ROOT = Path("data")
C9S_DIR = DATA_ROOT / "c9s"
SRC_AUDIO = DATA_ROOT / "Covid19SoundFull"
FALLBACK_AUDIO = [
    DATA_ROOT / "c9s_t1" / "processed" / "audio",
    DATA_ROOT / "c9s_t2" / "processed" / "audio",
]
OUT_ROOT = C9S_DIR / "processed"

PLATFORM_CSVS = {
    "android": C9S_DIR / "results_raw_20210426_lan_yamnet_android_noloc.csv",
    "ios": C9S_DIR / "results_raw_20210426_lan_yamnet_ios_noloc.csv",
    "web": C9S_DIR / "results_raw_20210426_lan_yamnet_web_noloc.csv",
}
TASK1_CSV = C9S_DIR / "data_0426_en_task1.csv"
TASK2_CSV = C9S_DIR / "data_0426_en_task2.csv"

KEEP_COLS = [
    "Uid", "Age", "Sex", "Medhistory", "Smoking", "Language", "Date",
    "Folder Name", "Symptoms", "Covid-Tested", "Hospitalized",
    "Voice filename", "Voice check", "Sampling Rate",
]

FOLD_MAP = {"train": 0, "validation": 1, "test": 2}


def find_audio(src_path):
    """Find audio file matching the stem, regardless of extension."""
    if src_path.exists():
        return src_path
    matches = list(src_path.parent.glob(f"{src_path.stem}.*"))
    return matches[0] if matches else None


def load_csv_semicolon(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def load_csv_comma(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def step1_load_platforms():
    """Load and combine the 3 platform CSVs, deduplicating by (Uid, Folder Name, Date, Voice filename)."""
    rows = []
    seen = set()
    dupes = 0
    for platform, csv_path in PLATFORM_CSVS.items():
        raw = load_csv_semicolon(csv_path)
        for r in raw:
            uid = r.get("Uid", "")
            folder = r.get("Folder Name", "")
            date = r.get("Date", "")
            voice = r.get("Voice filename", "")
            key = (uid, folder, date, voice)
            if key in seen:
                dupes += 1
                continue
            seen.add(key)

            row = {col: r.get(col, "") for col in KEEP_COLS}
            row["platform"] = platform
            # Web entries: use Folder Name as Participant_ID
            if platform == "web":
                row["Participant_ID"] = r["Folder Name"]
                row["src_uid"] = r["Uid"]  # "form-app-users"
            else:
                row["Participant_ID"] = r["Uid"]
                row["src_uid"] = r["Uid"]
            rows.append(row)
    print(f"Step 1: Loaded {len(rows)} rows from 3 platform CSVs (dropped {dupes} duplicates)")
    return rows


def step2_merge_task1(rows):
    """Merge task 1 split/label info."""
    task1_raw = load_csv_semicolon(TASK1_CSV)

    # Build lookup keyed by (Uid, Folder Name, voice_stem) to handle
    # duplicate (Uid, Folder Name) with different recordings and labels
    t1_lookup = {}
    t1_web_lookup = {}
    for r in task1_raw:
        uid = r["Uid"]
        folder = r["Folder Name"]
        voice_stem = Path(r["Voice filename"]).stem
        info = (r["split"], r["label"])
        if uid == folder:
            t1_web_lookup[(folder, voice_stem)] = info
        else:
            t1_lookup[(uid, folder, voice_stem)] = info

    matched = 0
    for row in rows:
        pid = row["Participant_ID"]
        folder = row["Folder Name"]
        voice_stem = Path(row["Voice filename"]).stem

        info = None
        if row["platform"] == "web":
            info = t1_web_lookup.get((folder, voice_stem))
        else:
            info = t1_lookup.get((pid, folder, voice_stem))

        if info:
            row["split_t1"] = info[0]
            row["label_t1"] = info[1]
            matched += 1
        else:
            row["split_t1"] = ""
            row["label_t1"] = ""

    print(f"Step 2: Matched {matched} rows with task 1 splits/labels")


def step3_merge_task2(rows):
    """Merge task 2 split/label/categs info."""
    task2_raw = load_csv_comma(TASK2_CSV)

    # Build lookups keyed by (uid, folder, voice_stem) to handle
    # multiple recordings per (uid, folder) with different dates
    t2_lookup = {}
    t2_web_lookup = {}
    for r in task2_raw:
        parts = r["voice_path"].replace("\\", "/").split("/")
        if len(parts) >= 3:
            uid, folder, voice_fn = parts[0], parts[1], parts[2]
            voice_stem = Path(voice_fn).stem
            info = (FOLD_MAP[r["fold"]], r["label"], r["categs"])
            if uid == "form-app-users":
                t2_web_lookup[(folder, voice_stem)] = info
            else:
                t2_lookup[(uid, folder, voice_stem)] = info

    matched = 0
    for row in rows:
        pid = row["Participant_ID"]
        folder = row["Folder Name"]
        voice_stem = Path(row["Voice filename"]).stem
        if row["platform"] == "web":
            info = t2_web_lookup.get((folder, voice_stem))
        else:
            info = t2_lookup.get((pid, folder, voice_stem))

        if info:
            row["split_t2"] = info[0]
            row["label_t2"] = info[1]
            row["categs"] = info[2]
            matched += 1
        else:
            row["split_t2"] = ""
            row["label_t2"] = ""
            row["categs"] = ""

    print(f"Step 3: Matched {matched} rows with task 2 splits/labels")


def step4_copy_audio(rows):
    """Build paths, copy voice audio, drop missing."""
    dest_audio = OUT_ROOT / "audio"
    dest_audio.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    copied, skipped, missing = 0, 0, 0

    for row in rows:
        voice_fn = row["Voice filename"]
        folder = row["Folder Name"]
        src_uid = row["src_uid"]

        # Source path in Covid19SoundFull uses original Uid
        csv_src = SRC_AUDIO / src_uid / folder / voice_fn
        actual_src = find_audio(csv_src)

        # Fallback: check c9s_t1/c9s_t2 processed audio
        if actual_src is None:
            pid = row["Participant_ID"]
            for fb in FALLBACK_AUDIO:
                fb_src = fb / pid / folder / voice_fn
                actual_src = find_audio(fb_src)
                if actual_src:
                    break

        if actual_src is None:
            missing += 1
            continue

        # Drop files without a recognized audio extension
        if actual_src.suffix == "":
            missing += 1
            continue

        # Path relative to processed/audio/ uses Participant_ID
        actual_fn = actual_src.name
        rel_path = f"{row['Participant_ID']}/{folder}/{actual_fn}"
        row["path"] = rel_path

        dst = dest_audio / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(actual_src, dst)
            copied += 1

        valid_rows.append(row)

    print(f"Step 4: Audio copied={copied}, skipped={skipped}, dropped={missing}")
    return valid_rows


def step5_write_csv(rows):
    """Write the final CSV."""
    out_path = OUT_ROOT / "c9s.csv"

    fieldnames = [
        "uid", "Participant_ID", "platform", "Age", "Sex", "Medhistory",
        "Smoking", "Language", "Date", "Folder Name", "Symptoms",
        "Covid-Tested", "Hospitalized", "Voice filename", "Voice check",
        "Sampling Rate", "split_t1", "label_t1", "split_t2", "label_t2",
        "categs", "path",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            # Drop internal src_uid and original Uid columns
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    print(f"Step 5: Wrote {out_path} with {len(rows)} rows")

    # Summary
    t1_count = sum(1 for r in rows if r.get("label_t1") != "")
    t2_count = sum(1 for r in rows if r.get("label_t2") != "")
    print(f"  label_t1 populated: {t1_count}")
    print(f"  label_t2 populated: {t2_count}")


def main():
    rows = step1_load_platforms()
    step2_merge_task1(rows)
    step3_merge_task2(rows)
    rows = step4_copy_audio(rows)
    step5_write_csv(rows)


if __name__ == "__main__":
    main()
