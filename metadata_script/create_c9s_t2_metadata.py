import csv
import shutil
from pathlib import Path

# COVID-19 Sounds Task 2
# Source CSV: data/c9s/data_0426_en_task2.csv
# Audio source: data/Covid19SoundFull/{voice_path} (backslashes in CSV)
# Files may be .wav, .m4a, or .webm despite CSV listing .wav
# Label: 0 = no covid, 1 = covid
# Folds: train/validation/test -> 0/1/2

DATA_ROOT = Path("data")
SRC_AUDIO = DATA_ROOT / "Covid19SoundFull"
CSV_PATH = DATA_ROOT / "c9s" / "data_0426_en_task2.csv"
OUT_ROOT = DATA_ROOT / "c9s_t2" / "processed"

FOLD_MAP = {"train": 0, "validation": 1, "test": 2}


def find_audio(src_path):
    """Find audio file matching the stem, regardless of extension."""
    if src_path.exists():
        return src_path
    matches = list(src_path.parent.glob(f"{src_path.stem}.*"))
    if matches:
        return matches[0]
    return None


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    src_rows = load_csv(CSV_PATH)

    rows = []
    for r in src_rows:
        voice_path = r["voice_path"].replace("\\", "/")
        rows.append({
            "Participant_ID": r["uid"],
            "categs": r["categs"],
            "split": FOLD_MAP[r["fold"]],
            "label": r["label"],
            "path": voice_path,
        })

    # ---- Copy audio to processed/audio/ ----
    dest_audio_dir = OUT_ROOT / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    copied, skipped, missing = 0, 0, 0
    for row in rows:
        csv_src = SRC_AUDIO / row["path"]
        actual_src = find_audio(csv_src)

        if actual_src is None:
            print(f"WARNING: dropping (missing audio) {csv_src}")
            missing += 1
            continue

        # Update path to reflect actual extension
        row["path"] = str(actual_src.relative_to(SRC_AUDIO))

        dst = dest_audio_dir / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(actual_src, dst)
            copied += 1
        valid_rows.append(row)
    rows = valid_rows
    print(f"Audio: copied {copied}, skipped {skipped}, dropped {missing}")

    # ---- Write CSV ----
    out_path = OUT_ROOT / "c9s_t2.csv"

    fieldnames = ["uid", "Participant_ID", "categs", "split", "label", "path"]

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
    print(f"  Label 0: {sum(1 for r in rows if str(r['label']) == '0')}")
    print(f"  Label 1: {sum(1 for r in rows if str(r['label']) == '1')}")


if __name__ == "__main__":
    main()
