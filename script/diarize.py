"""Diarize an audio file with pyannote/speaker-diarization-community-1.

Saves per-speaker utterances into spk_<i>/ and the unaccounted-for audio
(gaps between diarized segments) concatenated into unaccounted.wav.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


def load_mono(path: Path):
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def diarize(audio_path: Path, out_dir: Path, hf_token: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    waveform, sr = load_mono(audio_path)
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
    out = pipeline({"waveform": wav_tensor, "sample_rate": sr})
    diar = out.exclusive_speaker_diarization

    speakers = sorted({lbl for _, _, lbl in diar.itertracks(yield_label=True)})
    spk_to_idx = {lbl: i for i, lbl in enumerate(speakers)}
    print(f"Detected {len(speakers)} speakers: {speakers}")

    covered = np.zeros(len(waveform), dtype=bool)
    counts = {lbl: 0 for lbl in speakers}

    for turn, _, lbl in diar.itertracks(yield_label=True):
        s = max(0, int(turn.start * sr))
        e = min(len(waveform), int(turn.end * sr))
        if e <= s:
            continue
        covered[s:e] = True
        spk_dir = out_dir / f"spk_{spk_to_idx[lbl]}"
        spk_dir.mkdir(exist_ok=True)
        counts[lbl] += 1
        seg = waveform[s:e]
        fname = f"utt_{counts[lbl]:04d}_{turn.start:.2f}-{turn.end:.2f}.wav"
        sf.write(str(spk_dir / fname), seg, sr)

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
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()
    diarize(args.audio, args.out, args.hf_token)


if __name__ == "__main__":
    main()