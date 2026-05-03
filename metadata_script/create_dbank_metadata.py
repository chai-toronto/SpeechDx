import csv
import shutil
from pathlib import Path

# DementiaBank ADReSS-M (Luz et al., IEEE ICASSP-SPGC 2023):
# multilingual Alzheimer's detection. English training (adrso*.mp3) +
# Greek sample (madrs*.wav) + Greek held-out test (madrs*.wav).
# Dataset folder is named `dbank` in this repo; upstream name is ADReSS-M.
# Label: Control=0, ProbableAD=1.
# Splits: train (English) = 0, val (Greek sample-gr) = 1, test (Greek held-out) = 2.

DATA_ROOT = Path("data/dbank")
SOURCE_ROOT = DATA_ROOT / "raw"

LABEL_MAP = {"Control": 0, "ProbableAD": 1}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(r, id_field):
    return {
        "Participant_ID": r[id_field],
        "age": r.get("age", ""),
        "gender": r.get("gender", ""),
        "educ": r.get("educ", ""),
        "dx": r.get("dx", ""),
        "mmse": r.get("mmse", ""),
    }


def main():
    out_audio_dir = DATA_ROOT / "processed" / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    # --- Train: English (adrso*.mp3) ---
    train_gt = load_csv(SOURCE_ROOT / "ADReSS-M-train" / "training-groundtruth.csv")
    train_audio_dir = SOURCE_ROOT / "ADReSS-M-train" / "train"
    for r in train_gt:
        pid = r["adressfname"]
        src = train_audio_dir / f"{pid}.mp3"
        entries.append({
            **normalize_row(r, "adressfname"),
            "split": 0,
            "language": "en",
            "src": src,
            "fname": f"{pid}.mp3",
        })

    # Note: ADReSS-M-train/sample/madrs-smpl*.mp3 are Spanish samples from an
    # earlier draft release, later replaced by Greek sample-gr. Excluded.

    # --- Val: Greek "sample-gr" (madrs*.wav) ---
    sgr_gt = load_csv(SOURCE_ROOT / "ADReSS-M-sample-gr" / "sample-gr-groundtruth.csv")
    sgr_audio_dir = SOURCE_ROOT / "ADReSS-M-sample-gr" / "sample-gr"
    for r in sgr_gt:
        pid = r["addressfname"]
        src = sgr_audio_dir / f"{pid}.wav"
        entries.append({
            **normalize_row(r, "addressfname"),
            "split": 1,
            "language": "el",
            "src": src,
            "fname": f"{pid}.wav",
        })

    # --- Test: Greek held-out (madrs*.wav) ---
    test_gt = load_csv(SOURCE_ROOT / "test-gr-groundtruth.csv")
    test_audio_dir = SOURCE_ROOT / "test-gr"
    for r in test_gt:
        pid = r["addressfname"]
        src = test_audio_dir / f"{pid}.wav"
        entries.append({
            **normalize_row(r, "addressfname"),
            "split": 2,
            "language": "el",
            "src": src,
            "fname": f"{pid}.wav",
        })

    # --- Copy audio (flat layout) and build final rows ---
    rows = []
    copied = skipped = 0
    for e in entries:
        src = e["src"]
        dst = out_audio_dir / e["fname"]
        if not src.exists():
            raise FileNotFoundError(f"Missing source audio: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

        dx = e["dx"]
        if dx not in LABEL_MAP:
            raise ValueError(f"Unknown dx '{dx}' for {e['Participant_ID']}")

        rows.append({
            "Participant_ID": e["Participant_ID"],
            "age": e["age"],
            "gender": e["gender"],
            "educ": e["educ"],
            "dx": dx,
            "mmse": e["mmse"],
            "language": e["language"],
            "split": e["split"],
            "label": LABEL_MAP[dx],
            "path": e["fname"],
        })

    print(f"Audio: copied {copied}, skipped {skipped} (already exist)")

    # --- Write CSV ---
    out_csv = DATA_ROOT / "processed" / "dbank.csv"
    fieldnames = ["uid", "Participant_ID", "age", "gender", "educ",
                  "dx", "mmse", "language", "split", "label", "path"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            w.writerow(row)

    print(f"Created {out_csv} with {len(rows)} rows")
    print(f"  Train (0): {sum(1 for r in rows if r['split'] == 0)}")
    print(f"  Val   (1): {sum(1 for r in rows if r['split'] == 1)}")
    print(f"  Test  (2): {sum(1 for r in rows if r['split'] == 2)}")
    print(f"  Label 0 (Control):    {sum(1 for r in rows if r['label'] == 0)}")
    print(f"  Label 1 (ProbableAD): {sum(1 for r in rows if r['label'] == 1)}")


if __name__ == "__main__":
    main()
