"""
Create a metadata CSV for the UASpeech dataset.

Source layout (after extracting the UASpeech archive into data/uaspeech/raw/):
    data/uaspeech/raw/audio/normalized/<speaker>/*.wav   (canonical UA-Speech tree)
    data/uaspeech/raw/<speaker>/*.wav                    (also accepted)

Staged into data/uaspeech/processed/audio/<speaker>/*.wav, parses filenames,
assigns labels and splits, and writes the CSV to
data/uaspeech/processed/uaspeech.csv.
"""

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf

# Repo root: this file lives at metadata_script/create_uaspeech_csv.py,
# so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "uaspeech" / "raw"
AUDIO_DIR = PROJECT_ROOT / "data" / "uaspeech" / "processed" / "audio"
OUTPUT_CSV = PROJECT_ROOT / "data" / "uaspeech" / "processed" / "uaspeech.csv"


def stage_audio_from_raw():
    """Copy each <speaker>/*.wav from raw/ → processed/audio/<speaker>/.

    Looks under raw/audio/normalized/ first (canonical UASpeech layout),
    then falls back to a flat raw/<speaker>/<file>.wav in case the user
    pre-flattened the tree.
    """
    if not RAW_DIR.is_dir():
        return
    candidates = [RAW_DIR / "audio" / "normalized", RAW_DIR]
    src_root = None
    matches: list[Path] = []
    for c in candidates:
        if not c.is_dir():
            continue
        found = list(c.glob("*/*.wav"))
        if found:
            src_root = c
            matches = found
            break
    if src_root is None:
        return
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in matches:
        rel = src.relative_to(src_root)  # speaker/<file>.wav
        dst = AUDIO_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Staged from {src_root}: copied {copied}, skipped {skipped}")

# Dysarthric speakers (label=1)
DYSARTHRIC = {
    "F02", "F03", "F04", "F05",
    "M01", "M04", "M05", "M07", "M08", "M09", "M10", "M11", "M12", "M14", "M16",
}
# Control speakers (label=0)
CONTROL = {
    "CF02", "CF03", "CF04", "CF05",
    "CM01", "CM04", "CM05", "CM06", "CM08", "CM09", "CM10", "CM12", "CM13",
}


def get_gender(participant_id: str) -> str:
    """Extract gender group (F, M, CF, CM) by stripping trailing digits."""
    return re.sub(r"\d+$", "", participant_id)


def assign_splits(speakers: list[str]) -> dict[str, int]:
    """
    Assign splits (0=train, 1=val, 2=test) stratified by gender group.
    ~70% train, ~10% val, ~20% test within each stratum.
    """
    strata = defaultdict(list)
    for spk in sorted(speakers):
        strata[get_gender(spk)].append(spk)

    split_map = {}
    for gender, members in sorted(strata.items()):
        n = len(members)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.1))
        # Ensure we don't exceed total
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        for i, spk in enumerate(members):
            if i < n_train:
                split_map[spk] = 0  # train
            elif i < n_train + n_val:
                split_map[spk] = 1  # val
            else:
                split_map[spk] = 2  # test

    return split_map


def main():
    stage_audio_from_raw()
    all_speakers = sorted(os.listdir(AUDIO_DIR))
    # Only keep known speakers
    all_speakers = [s for s in all_speakers if s in DYSARTHRIC | CONTROL]
    print(f"Found {len(all_speakers)} speakers: {all_speakers}")

    split_map = assign_splits(all_speakers)

    rows = []
    corrupt_files = defaultdict(list)
    uid = 0
    for speaker in all_speakers:
        speaker_dir = AUDIO_DIR / speaker
        if not speaker_dir.is_dir():
            continue

        label = 0 if speaker in CONTROL else 1
        gender = get_gender(speaker)
        split = split_map[speaker]

        wav_files = sorted(f for f in os.listdir(speaker_dir) if f.endswith(".wav"))
        for wav in wav_files:
            # Filter out M1 microphone files
            # Filename format: <Participant>_<Block>_<Word>_<Mic>.wav
            parts = wav.replace(".wav", "").split("_")
            if len(parts) >= 4:
                mic = parts[-1]
                if mic != "M6":
                    continue

            rel_path = f"{speaker}/{wav}"
            abs_path = speaker_dir / wav

            # Get duration
            try:
                info = sf.info(abs_path)
                duration = info.duration
            except Exception as e:
                corrupt_files[speaker].append(wav)
                continue

            if duration == 0:
                corrupt_files[speaker].append(wav)
                continue

            rows.append({
                "uid": uid,
                "gender": gender,
                "Participant_ID": speaker,
                "split": split,
                "label": label,
                "duration": round(duration, 4),
                "path": rel_path,
            })
            uid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

    # Verification
    print("\n--- Verification ---")
    print(f"Speakers: {df['Participant_ID'].nunique()}")
    print(f"\nSplit counts (speakers):")
    for split_val, split_name in [(0, "train"), (1, "val"), (2, "test")]:
        spks = df[df["split"] == split_val]["Participant_ID"].unique()
        print(f"  {split_name}: {len(spks)} speakers - {sorted(spks)}")

    print(f"\nSplit counts (rows):")
    total = len(df)
    for split_val, split_name in [(0, "train"), (1, "val"), (2, "test")]:
        n = len(df[df["split"] == split_val])
        print(f"  {split_name}: {n} ({n/total*100:.1f}%)")

    print(f"\nLabel counts:")
    print(df["label"].value_counts().to_string())

    print(f"\nMic check (should have no M1):")
    m1_count = df["path"].str.contains("_M1\\.").sum()
    print(f"  M1 files: {m1_count}")

    # Verify paths exist
    missing = 0
    for path in df["path"]:
        if not (AUDIO_DIR / path).exists():
            missing += 1
    print(f"  Missing files: {missing}")

    # Report corrupt files
    if corrupt_files:
        total_corrupt = sum(len(v) for v in corrupt_files.values())
        print(f"\nCorrupt/zero-duration files: {total_corrupt}")
        for spk, files in sorted(corrupt_files.items()):
            if len(files) <= 5:
                print(f"  {spk}: {files}")
            else:
                print(f"  {spk}: {len(files)} files (first 5: {files[:5]})")
    else:
        print("\nNo corrupt files found.")

    print(f"\nDuration stats (seconds):")
    print(f"  mean={df['duration'].mean():.2f}, min={df['duration'].min():.2f}, max={df['duration'].max():.2f}")


if __name__ == "__main__":
    main()
