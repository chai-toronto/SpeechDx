"""Diarize audio via pyannoteAI API (precision-2 model).

Saves per-speaker utterances into spk_<i>/ and the unaccounted-for audio
(gaps between diarized segments) concatenated into unaccounted.wav.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannoteai.sdk import Client


def load_mono(path: Path):
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def diarize(audio_path: Path, out_dir: Path, api_key: str, exclusive: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(api_key=api_key)

    print(f"Uploading {audio_path.name}...")
    media_url = client.upload(audio_path)
    print(f"Starting diarize job (model=precision-2, exclusive={exclusive})...")
    job_id = client.diarize(media_url, exclusive=exclusive)
    print(f"Job {job_id} — polling...")
    result = client.retrieve(job_id)

    output = result.get("output", result)
    key = "exclusive_diarization" if exclusive and "exclusive_diarization" in output else "diarization"
    segments = output[key]
    print(f"Using '{key}': {len(segments)} segments")

    waveform, sr = load_mono(audio_path)
    speakers = sorted({seg["speaker"] for seg in segments})
    spk_to_idx = {lbl: i for i, lbl in enumerate(speakers)}
    print(f"Detected {len(speakers)} speakers: {speakers}")

    covered = np.zeros(len(waveform), dtype=bool)
    counts = {lbl: 0 for lbl in speakers}

    for seg in segments:
        lbl = seg["speaker"]
        s = max(0, int(seg["start"] * sr))
        e = min(len(waveform), int(seg["end"] * sr))
        if e <= s:
            continue
        covered[s:e] = True
        spk_dir = out_dir / f"spk_{spk_to_idx[lbl]}"
        spk_dir.mkdir(exist_ok=True)
        counts[lbl] += 1
        fname = f"utt_{counts[lbl]:04d}_{seg['start']:.2f}-{seg['end']:.2f}.wav"
        sf.write(str(spk_dir / fname), waveform[s:e], sr)

    unaccounted = waveform[~covered]
    if len(unaccounted) > 0:
        sf.write(str(out_dir / "unaccounted.wav"), unaccounted, sr)

    total_s = len(waveform) / sr
    uncov_s = len(unaccounted) / sr
    print(f"Duration: {total_s:.1f}s | unaccounted: {uncov_s:.1f}s ({uncov_s/total_s:.1%})")
    for lbl in speakers:
        print(f"  spk_{spk_to_idx[lbl]} ({lbl}): {counts[lbl]} utterances")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--api-key", default=os.environ.get("PYANNOTEAI_API_KEY"))
    ap.add_argument("--non-exclusive", action="store_true")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set PYANNOTEAI_API_KEY or pass --api-key")
    diarize(args.audio, args.out, args.api_key, exclusive=not args.non_exclusive)


if __name__ == "__main__":
    main()
