import argparse
import csv
import shutil
from pathlib import Path

# E-DAIC dataset
# metadata_mapped.csv contains train (training_*) and dev (development_*) participants
# labels/test_split.csv contains test participants
# Raw audio: <raw_root>/data/{Participant_ID}_P/{Participant_ID}_AUDIO.wav
# Copied to: data/edaic/processed/audio/{Participant_ID}_P/{Participant_ID}_AUDIO.wav
# Label: PHQ_Binary (0 = not depressed, 1 = depressed)

DATA_ROOT = Path("data/edaic")
RAW_ROOT = DATA_ROOT / "raw"


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                        help="Root containing metadata_mapped.csv, labels/, data/ "
                             "(default: data/edaic/raw).")
    parser.add_argument("--no-copy", action="store_true",
                        help="Skip copying audio into processed/audio/; only build the CSV.")
    args = parser.parse_args()
    raw_root = args.raw_root

    # ---- Load metadata_mapped (train + dev) ----
    meta_rows = load_csv(raw_root / "metadata_mapped.csv")

    rows = []
    for r in meta_rows:
        split = 0 if r["AVECParticipant_ID"].startswith("training_") else 1
        pid = r["Participant_ID"]
        rows.append({
            "Participant_ID": pid,
            "AVECParticipant_ID": r["AVECParticipant_ID"],
            "Gender": r["Gender"],
            "PHQ_Binary": r["PHQ_Binary"],
            "PHQ_Score": r["PHQ_Score"],
            "PCL-C (PTSD)": r["PCL-C (PTSD)"],
            "PTSD Severity": r["PTSD Severity"],
            "split": split,
            "label": r["PHQ_Binary"],
            "path": f"{pid}_P/{pid}_AUDIO.wav",
        })

    # ---- Load test split ----
    test_rows = load_csv(raw_root / "labels" / "test_split.csv")

    for r in test_rows:
        pid = r["Participant_ID"]
        rows.append({
            "Participant_ID": pid,
            "AVECParticipant_ID": "",
            "Gender": r["Gender"],
            "PHQ_Binary": r["PHQ_Binary"],
            "PHQ_Score": r["PHQ_Score"],
            "PCL-C (PTSD)": r["PCL-C (PTSD)"],
            "PTSD Severity": r["PTSD Severity"],
            "split": 2,
            "label": r["PHQ_Binary"],
            "path": f"{pid}_P/{pid}_AUDIO.wav",
        })

    # ---- Copy audio to processed/audio/ (unless --no-copy) ----
    if not args.no_copy:
        raw_audio_dir = raw_root / "data"
        dest_audio_dir = DATA_ROOT / "processed" / "audio"
        dest_audio_dir.mkdir(parents=True, exist_ok=True)

        copied, skipped = 0, 0
        for row in rows:
            src = raw_audio_dir / row["path"]
            dst = dest_audio_dir / row["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                skipped += 1
        print(f"Audio: copied {copied}, skipped {skipped} (already exist)")
    else:
        print("Audio: skipped (--no-copy)")

    # ---- Write CSV ----
    out_dir = DATA_ROOT / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edaic.csv"

    fieldnames = ["uid", "Participant_ID", "AVECParticipant_ID", "Gender",
                  "PHQ_Binary", "PHQ_Score", "PCL-C (PTSD)", "PTSD Severity",
                  "split", "label", "path"]

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
