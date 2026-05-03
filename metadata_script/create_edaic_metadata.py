"""
Build E-DAIC metadata CSV (and optionally transcript-cut audio).

Source layout (under --raw-root, default data/edaic/raw/):
  metadata_mapped.csv                       training + dev participants
  labels/test_split.csv                     test participants
  data/<PID>_P/<PID>_AUDIO.wav              full session audio
  data/<PID>_P/<PID>_Transcript.csv         transcript with Start_Time / End_Time

Modes:
  default     : copy raw audio verbatim into processed/audio/, write base CSV.
  --no-copy   : write base CSV only; skip audio I/O. Useful when --raw-root
                points at a remote mount.
  --chunk     : for each participant, concatenate the [Start_Time:End_Time]
                segments from the transcript into one waveform, write that to
                processed/audio/, and add a `boundaries` column (cumulative
                end-times in the new audio). Subsumes the previous
                build_edaic_transcript_chunks.py.

Output:
  data/edaic/processed/edaic.csv
  data/edaic/processed/audio/<PID>_P/<PID>_AUDIO.wav

Label: PHQ_Binary (0 = not depressed, 1 = depressed).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

DATA_ROOT = Path("data/edaic")
RAW_ROOT = DATA_ROOT / "raw"
DST_ROOT = DATA_ROOT / "processed"
DST_AUDIO = DST_ROOT / "audio"
DST_CSV = DST_ROOT / "edaic.csv"

BASE_FIELDS = [
    "uid", "Participant_ID", "AVECParticipant_ID", "Gender",
    "PHQ_Binary", "PHQ_Score", "PCL-C (PTSD)", "PTSD Severity",
    "split", "label", "path",
]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_base_rows(raw_root):
    """Read metadata_mapped (train+dev) and labels/test_split — return one row per participant."""
    rows = []
    for r in load_csv(raw_root / "metadata_mapped.csv"):
        pid = r["Participant_ID"]
        rows.append({
            "Participant_ID": pid,
            "AVECParticipant_ID": r["AVECParticipant_ID"],
            "Gender": r["Gender"],
            "PHQ_Binary": r["PHQ_Binary"],
            "PHQ_Score": r["PHQ_Score"],
            "PCL-C (PTSD)": r["PCL-C (PTSD)"],
            "PTSD Severity": r["PTSD Severity"],
            "split": 0 if r["AVECParticipant_ID"].startswith("training_") else 1,
            "label": r["PHQ_Binary"],
            "path": f"{pid}_P/{pid}_AUDIO.wav",
        })
    for r in load_csv(raw_root / "labels" / "test_split.csv"):
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
    return rows


def copy_audio_verbatim(rows, raw_root, dst_audio):
    src_dir = raw_root / "data"
    dst_audio.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for row in rows:
        src = src_dir / row["path"]
        dst = dst_audio / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Audio: copied {copied}, skipped {skipped}")


def chunk_one(src_wav, transcript_csv):
    """Concat transcript segments. Returns (audio_concat, sr, boundaries) or None."""
    import numpy as np
    import pandas as pd
    import soundfile as sf

    audio, sr = sf.read(str(src_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = audio.shape[0]

    df = pd.read_csv(transcript_csv)
    chunks, boundaries = [], []
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


def chunk_audio_by_transcript(rows, raw_root, dst_audio):
    """For each row, cut audio along transcript & write a chunked wav. Adds `boundaries`.

    Drops rows whose audio or transcript is missing, or whose transcript yielded
    no usable rows.
    """
    import soundfile as sf

    src_dir = raw_root / "data"
    dst_audio.mkdir(parents=True, exist_ok=True)
    kept = []
    n_ok = n_skip = 0

    for row in rows:
        pid = row["Participant_ID"]
        src_wav = src_dir / f"{pid}_P" / f"{pid}_AUDIO.wav"
        transcript = src_dir / f"{pid}_P" / f"{pid}_Transcript.csv"

        if not src_wav.exists():
            print(f"[skip] {pid}: missing audio {src_wav}")
            n_skip += 1
            continue
        if not transcript.exists():
            print(f"[skip] {pid}: missing transcript {transcript}")
            n_skip += 1
            continue

        result = chunk_one(src_wav, transcript)
        if result is None:
            print(f"[skip] {pid}: no usable transcript rows")
            n_skip += 1
            continue

        out_audio, sr, boundaries = result
        out_path = dst_audio / f"{pid}_P" / f"{pid}_AUDIO.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out_audio, sr, subtype="PCM_16")

        row["boundaries"] = json.dumps([round(b, 4) for b in boundaries])
        kept.append(row)
        n_ok += 1
        print(f"[ok]   {pid}: {len(boundaries)} chunks, {out_audio.shape[0] / sr:.1f}s")

    print(f"Chunking: {n_ok} ok, {n_skip} skipped")
    return kept


def write_csv(rows, with_boundaries, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_FIELDS)
    if with_boundaries:
        fieldnames.append("boundaries")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for uid, row in enumerate(rows):
            row["uid"] = uid
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {out_csv}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                        help="Root containing metadata_mapped.csv, labels/, data/ "
                             "(default: data/edaic/raw).")
    parser.add_argument("--out-csv", type=Path, default=DST_CSV,
                        help=f"Output CSV path (default: {DST_CSV}).")
    parser.add_argument("--out-audio", type=Path, default=DST_AUDIO,
                        help=f"Output audio dir (default: {DST_AUDIO}).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-copy", action="store_true",
                      help="Skip copying audio; write CSV only.")
    mode.add_argument("--chunk", action="store_true",
                      help="Cut audio along each participant's transcript and add "
                           "a `boundaries` column to the CSV.")
    args = parser.parse_args()

    rows = build_base_rows(args.raw_root)

    if args.chunk:
        rows = chunk_audio_by_transcript(rows, args.raw_root, args.out_audio)
        write_csv(rows, with_boundaries=True, out_csv=args.out_csv)
    else:
        if not args.no_copy:
            copy_audio_verbatim(rows, args.raw_root, args.out_audio)
        else:
            print("Audio: skipped (--no-copy)")
        write_csv(rows, with_boundaries=False, out_csv=args.out_csv)

    print(f"  Train (0): {sum(1 for r in rows if r['split'] == 0)}")
    print(f"  Val   (1): {sum(1 for r in rows if r['split'] == 1)}")
    print(f"  Test  (2): {sum(1 for r in rows if r['split'] == 2)}")
    print(f"  Label 0:   {sum(1 for r in rows if str(r['label']) == '0')}")
    print(f"  Label 1:   {sum(1 for r in rows if str(r['label']) == '1')}")


if __name__ == "__main__":
    main()
