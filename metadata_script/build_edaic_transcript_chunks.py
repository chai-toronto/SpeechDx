"""
Build eDAIC processed audio by cutting each participant's audio along transcript
[Start_Time:End_Time] blocks, concatenating the blocks, and recording the cut
boundaries (end times, seconds) in the new concatenated audio.

Input:
  - Source audio (local copy):  data/edaic/processed_old/audio/<PID>_P/<PID>_AUDIO.wav
  - Source CSV (metadata):      data/edaic/processed_old/edaic.csv
  - Transcript CSVs (samba):    /Volumes/desktop-estsqlc/more raw datasets/edaic/data/<PID>_P/<PID>_Transcript.csv

Output:
  - data/edaic/processed/audio/<PID>_P/<PID>_AUDIO.wav
  - data/edaic/processed/edaic.csv (same columns as input + `boundaries`)

Boundary rule (per participant, row-by-row through the transcript):
    s = max(Start_Time, last_end)
    e = End_Time
    if e > s: take audio[s:e] from source, append to output buffer
              record cumulative end-time in output as a boundary
              last_end = e
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC_AUDIO_DIR = ROOT / "data/edaic/processed_old/audio"
SRC_CSV = ROOT / "data/edaic/processed_old/edaic.csv"
TRANSCRIPT_ROOT = Path("/Volumes/desktop-estsqlc/more raw datasets/edaic/data")
DST_ROOT = ROOT / "data/edaic/processed"
DST_AUDIO_DIR = DST_ROOT / "audio"
DST_CSV = DST_ROOT / "edaic.csv"


def process_participant(src_wav: Path, transcript_csv: Path):
    audio, sr = sf.read(str(src_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = audio.shape[0]

    df = pd.read_csv(transcript_csv)
    chunks = []
    boundaries = []
    total_samples = 0
    last_end = 0.0

    for _, row in df.iterrows():
        try:
            s = float(row["Start_Time"])
            e = float(row["End_Time"])
        except (TypeError, ValueError):
            continue
        if s < last_end:
            s = last_end
        if e <= s:
            continue

        i0 = max(0, min(int(round(s * sr)), n))
        i1 = max(0, min(int(round(e * sr)), n))
        if i1 <= i0:
            continue

        chunks.append(audio[i0:i1])
        total_samples += (i1 - i0)
        boundaries.append(total_samples / sr)
        last_end = e

    if not chunks:
        return None
    return np.concatenate(chunks), sr, boundaries


def main():
    DST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC_CSV)

    new_rows = []
    n_ok = n_skip = 0
    for _, row in df.iterrows():
        pid = str(row["Participant_ID"])
        src_wav = SRC_AUDIO_DIR / f"{pid}_P" / f"{pid}_AUDIO.wav"
        transcript = TRANSCRIPT_ROOT / f"{pid}_P" / f"{pid}_Transcript.csv"

        if not src_wav.exists():
            print(f"[skip] {pid}: missing audio {src_wav}")
            n_skip += 1
            continue
        if not transcript.exists():
            print(f"[skip] {pid}: missing transcript {transcript}")
            n_skip += 1
            continue

        result = process_participant(src_wav, transcript)
        if result is None:
            print(f"[skip] {pid}: no usable transcript rows")
            n_skip += 1
            continue

        out_audio, sr, boundaries = result
        out_path = DST_AUDIO_DIR / f"{pid}_P" / f"{pid}_AUDIO.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out_audio, sr, subtype="PCM_16")

        new_row = dict(row)
        new_row["boundaries"] = json.dumps([round(b, 4) for b in boundaries])
        new_rows.append(new_row)
        n_ok += 1
        print(f"[ok]   {pid}: {len(boundaries)} chunks, {out_audio.shape[0]/sr:.1f}s")

    fieldnames = list(df.columns) + ["boundaries"]
    with open(DST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)

    print(f"\nWrote {DST_CSV}: {n_ok} rows, skipped {n_skip}")


if __name__ == "__main__":
    main()
