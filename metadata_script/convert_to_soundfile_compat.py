"""
Detect audio files in a dataset CSV that are not supported by soundfile,
convert them to .wav using ffmpeg, and update the CSV paths accordingly.

Usage:
    python -m metadata_script.convert_to_soundfile_compat \
        --csv data/c9s/processed/c9s.csv \
        --audio-root data/c9s/processed/audio
"""

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf

SUPPORTED_EXTS = {f".{k.lower()}" for k in sf.available_formats().keys()}


def convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert an audio file to 16-bit WAV using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for {src}: {e.stderr.decode().strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert unsupported audio to wav and update CSV.")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--audio-root", required=True, help="Root directory for audio files (paths in CSV are relative to this)")
    parser.add_argument("--delete-original", action="store_true", help="Delete original file after successful conversion")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    audio_root = Path(args.audio_root)
    df = pd.read_csv(csv_path)

    # Drop rows with no file extension (not real audio)
    no_ext_mask = df["path"].apply(lambda p: os.path.splitext(p)[1] == "")
    n_no_ext = no_ext_mask.sum()
    if n_no_ext > 0:
        print(f"Dropping {n_no_ext} rows with no file extension")
        df = df[~no_ext_mask]

    unsupported_mask = df["path"].apply(
        lambda p: os.path.splitext(p)[1].lower() not in SUPPORTED_EXTS
    )
    unsupported = df[unsupported_mask]

    if unsupported.empty:
        print("All files are already supported by soundfile.")
        df.to_csv(csv_path, index=False)
        return

    ext_counts = unsupported["path"].apply(lambda p: os.path.splitext(p)[1].lower()).value_counts()
    print(f"Found {len(unsupported)} unsupported files:")
    for ext, count in ext_counts.items():
        print(f"  {ext}: {count}")

    converted, failed, dropped_idx = 0, 0, []
    for idx, row in unsupported.iterrows():
        src = audio_root / row["path"]
        stem, _ = os.path.splitext(row["path"])
        new_rel = stem + ".wav"
        dst = audio_root / new_rel

        if dst.exists():
            df.at[idx, "path"] = new_rel
            converted += 1
            continue

        if not src.exists():
            print(f"  Missing: {src}")
            dropped_idx.append(idx)
            failed += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if convert_to_wav(src, dst):
            df.at[idx, "path"] = new_rel
            converted += 1
            if args.delete_original:
                src.unlink()
        else:
            dropped_idx.append(idx)
            failed += 1

    if dropped_idx:
        df = df.drop(dropped_idx)

    df.to_csv(csv_path, index=False)
    print(f"\nDone: converted={converted}, failed/dropped={failed}")
    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()