"""
Create KSoF (Kassel State of Fluency) metadata and copy audio.

Source: /Users/lkieu/Downloads/KSoF_release
  - kassel-state-of-fluency-labels.csv
  - segments/<segment_id>.wav  (5597 clips, 3 seconds, 16 kHz mono)

Dest: data/ksof/processed/
  - audio/<segment_id>.wav
  - ksof.csv

Label (Unintelligible, binary): majority vote on the "Unintelligible" column
(count of 3 naive annotators who marked the clip as unintelligible).
label = 1 iff count >= 2, else 0.

Split: official KSoF partition (speaker-independent, zero overlap).
  train -> 0, dvel -> 1, test -> 2
"""
import shutil
from pathlib import Path

import pandas as pd

SRC_DIR = Path("/Users/lkieu/Downloads/KSoF_release")
DST_DIR = Path("data/ksof/processed")
AUDIO_DST = DST_DIR / "audio"
CSV_DST = DST_DIR / "ksof.csv"

SPLIT_MAP = {"train": 0, "dvel": 1, "test": 2}


def main():
    src_csv = SRC_DIR / "kassel-state-of-fluency-labels.csv"
    src_audio = SRC_DIR / "segments"

    AUDIO_DST.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv)
    # segment_id was zero-padded in filenames; pandas reads it as int
    df["segment_id_str"] = df["segment_id"].astype(str).str.zfill(4)
    df["speaker_str"] = df["speaker"].astype(str).str.zfill(3)

    # Copy audio (skip if already present)
    copied = 0
    missing = []
    for seg in df["segment_id_str"]:
        src = src_audio / f"{seg}.wav"
        dst = AUDIO_DST / f"{seg}.wav"
        if not src.exists():
            missing.append(seg)
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} files (skipped existing). Missing: {len(missing)}")
    if missing:
        print("First missing:", missing[:5])

    # Build processed CSV: required columns first, then every source column preserved.
    out = pd.DataFrame({
        "uid": range(len(df)),
        "Participant_ID": df["speaker_str"],
        "split": df["partition"].map(SPLIT_MAP).astype(int),
        "label": (df["Unintelligible"] >= 2).astype(int),
        "path": df["segment_id_str"].apply(lambda s: f"{s}.wav"),
    })
    # Append every original source column (unmodified) so no annotation is lost.
    for col in df.columns:
        if col in ("segment_id_str", "speaker_str"):
            continue
        out[col] = df[col].values

    out.to_csv(CSV_DST, index=False)
    print(f"Wrote {CSV_DST} ({len(out)} rows)")

    print("\nSplit distribution:")
    print(out["split"].value_counts().sort_index().to_dict())
    print("Label distribution:")
    print(out["label"].value_counts().sort_index().to_dict())
    print("Per-split label positives:")
    print(out.groupby("split")["label"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
