import csv
import shutil
from pathlib import Path

# Nemours dataset (dysarthric speech)
# Source CSV: data/nemours/Nemours-metadata.csv (semicolon-delimited)
# voice-path-new: ./data_og/Nemours/wav/{PID}/WAV/{file}.WAV
# Actual audio: data/nemours/{PID}/WAV/{file}.WAV
# Label: 0 = healthy, 1 = dysarthric
# Splits: 0/1/2 (train/val/test) already assigned

DATA_ROOT = Path("data/nemours")
RAW_ROOT = DATA_ROOT / "raw"
OUT_ROOT = DATA_ROOT / "processed"

PATH_PREFIX = "./data_og/Nemours/wav/"


def main():
    with open(RAW_ROOT / "Nemours-metadata.csv", newline='') as f:
        src_rows = list(csv.DictReader(f, delimiter=';'))

    rows = []
    for r in src_rows:
        rel_path = r["voice-path-new"].replace(PATH_PREFIX, "")
        rows.append({
            "Participant_ID": r["Uid"],
            "split": r["split"],
            "label": r["label"],
            "path": rel_path,
        })

    # ---- Copy audio to processed/audio/ ----
    dest_audio_dir = OUT_ROOT / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)

    copied, skipped, dropped = 0, 0, 0
    valid_rows = []
    for row in rows:
        src = RAW_ROOT / row["path"]
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
    out_path = OUT_ROOT / "nemours.csv"

    fieldnames = ["uid", "Participant_ID", "split", "label", "path"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            writer.writerow(row)

    print(f"Created {out_path} with {len(rows)} rows")
    print(f"  Train (0): {sum(1 for r in rows if str(r['split']) == '0')}")
    print(f"  Val (1):   {sum(1 for r in rows if str(r['split']) == '1')}")
    print(f"  Test (2):  {sum(1 for r in rows if str(r['split']) == '2')}")
    print(f"  Label 0: {sum(1 for r in rows if str(r['label']) == '0')}")
    print(f"  Label 1: {sum(1 for r in rows if str(r['label']) == '1')}")


if __name__ == "__main__":
    main()
