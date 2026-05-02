import csv
import shutil
from pathlib import Path

# DAIC-WOZ dataset
# train_split_Depression_AVEC2017.csv: 107 train participants (PHQ8 columns)
# dev_split_Depression_AVEC2017.csv: 35 dev participants (PHQ8 columns)
# full_test_split.csv: 47 test participants (PHQ columns)
# Audio: {Participant_ID}_P/{Participant_ID}_AUDIO.wav
# Label: PHQ8_Binary (train/dev) / PHQ_Binary (test)

DATA_ROOT = Path("data/daic_woz")


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    # ---- Load train split ----
    train_rows = load_csv(DATA_ROOT / "train_split_Depression_AVEC2017.csv")
    # ---- Load dev split ----
    dev_rows = load_csv(DATA_ROOT / "dev_split_Depression_AVEC2017.csv")
    # ---- Load test split ----
    test_rows = load_csv(DATA_ROOT / "full_test_split.csv")

    rows = []

    # Train/dev share the same PHQ8 schema
    for split_id, split_rows in [(0, train_rows), (1, dev_rows)]:
        for r in split_rows:
            pid = r["Participant_ID"]
            rows.append({
                "Participant_ID": pid,
                "Gender": r["Gender"],
                "PHQ8_Binary": r["PHQ8_Binary"],
                "PHQ8_Score": r["PHQ8_Score"],
                "PHQ8_NoInterest": r["PHQ8_NoInterest"],
                "PHQ8_Depressed": r["PHQ8_Depressed"],
                "PHQ8_Sleep": r["PHQ8_Sleep"],
                "PHQ8_Tired": r["PHQ8_Tired"],
                "PHQ8_Appetite": r["PHQ8_Appetite"],
                "PHQ8_Failure": r["PHQ8_Failure"],
                "PHQ8_Concentrating": r["PHQ8_Concentrating"],
                "PHQ8_Moving": r["PHQ8_Moving"],
                "split": split_id,
                "label": r["PHQ8_Binary"],
                "path": f"{pid}_P/{pid}_AUDIO.wav",
            })

    # Test uses PHQ (not PHQ8) and has fewer columns
    for r in test_rows:
        pid = r["Participant_ID"]
        rows.append({
            "Participant_ID": pid,
            "Gender": r["Gender"],
            "PHQ8_Binary": r["PHQ_Binary"],
            "PHQ8_Score": r["PHQ_Score"],
            "PHQ8_NoInterest": "",
            "PHQ8_Depressed": "",
            "PHQ8_Sleep": "",
            "PHQ8_Tired": "",
            "PHQ8_Appetite": "",
            "PHQ8_Failure": "",
            "PHQ8_Concentrating": "",
            "PHQ8_Moving": "",
            "split": 2,
            "label": r["PHQ_Binary"],
            "path": f"{pid}_P/{pid}_AUDIO.wav",
        })

    # ---- Copy audio to processed/audio/ ----
    dest_audio_dir = DATA_ROOT / "processed" / "audio"
    dest_audio_dir.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for row in rows:
        src = DATA_ROOT / row["path"]
        dst = dest_audio_dir / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1
    print(f"Audio: copied {copied}, skipped {skipped} (already exist)")

    # ---- Write CSV ----
    out_dir = DATA_ROOT / "processed"
    out_path = out_dir / "daic_woz.csv"

    fieldnames = ["uid", "Participant_ID", "Gender",
                  "PHQ8_Binary", "PHQ8_Score",
                  "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep",
                  "PHQ8_Tired", "PHQ8_Appetite", "PHQ8_Failure",
                  "PHQ8_Concentrating", "PHQ8_Moving",
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
