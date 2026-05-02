"""Identify-based diarization -> extract subject turns between Ellie's segments.

- Runs pyannoteAI `identify` with a saved Ellie voiceprint.
- Keeps only the top-2 speakers by segment count.
- Determines which of the two is Ellie via the identified label.
- Groups the subject's segments into "turns": all subject segments between
  two Ellie segments become one concatenated recording. Subject segments
  before the first Ellie segment are their own leading turn.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannoteai.sdk import Client


def load_mono(path: Path):
    audio, sr = sf.read(str(path), always_2d=True)
    return audio.mean(axis=1).astype(np.float32), sr


def run_identify(client: Client, audio_path: Path, voiceprints: dict, exclusive: bool,
                 cache_path: Path | None = None):
    if cache_path and cache_path.exists():
        print(f"Loading cached identify output from {cache_path}")
        return json.loads(cache_path.read_text())
    print(f"Uploading {audio_path.name}...")
    media_url = client.upload(audio_path)
    print("Starting identify job (precision-2)...")
    job_id = client.identify(media_url, voiceprints=voiceprints, exclusive=exclusive)
    print(f"Job {job_id} — polling...")
    result = client.retrieve(job_id)
    output = result.get("output", result)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(output, indent=2))
        print(f"Cached identify output -> {cache_path}")
    return output


def pick_segments(output: dict, exclusive: bool):
    for key in (["exclusiveDiarization", "exclusive_diarization"] if exclusive else []) + ["diarization"]:
        if key in output:
            return output[key], key
    raise KeyError(f"No diarization field in output: {list(output.keys())}")


def find_ellie_label(output: dict, ellie_name: str):
    """Return the raw diarization speaker label that best matches `ellie_name`.

    Prefers `voiceprints[*].match == ellie_name`; falls back to highest
    confidence[ellie_name] across speakers.
    """
    vps = output.get("voiceprints", [])
    matched = [v for v in vps if v.get("match") == ellie_name]
    if matched:
        return matched[0]["speaker"], "match"
    if vps and all(ellie_name in v.get("confidence", {}) for v in vps):
        best = max(vps, key=lambda v: v["confidence"][ellie_name])
        return best["speaker"], f"confidence={best['confidence'][ellie_name]}"
    return None, "no_voiceprint_field"


def group_turns(segments, ellie_label, subject_label):
    """Walk segments in start-time order; cut on each Ellie segment.

    Returns a list of turns, each a list of (start, end) subject segments.
    """
    segs = sorted(segments, key=lambda s: s["start"])
    turns, current = [], []
    for s in segs:
        spk = s["speaker"]
        if spk == ellie_label:
            if current:
                turns.append(current)
                current = []
        elif spk == subject_label:
            current.append((s["start"], s["end"]))
    if current:
        turns.append(current)
    return turns


def save_turns(turns, waveform, sr, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, turn in enumerate(turns, start=1):
        pieces = []
        for start, end in turn:
            a = max(0, int(start * sr))
            b = min(len(waveform), int(end * sr))
            if b > a:
                pieces.append(waveform[a:b])
        if not pieces:
            continue
        merged = np.concatenate(pieces)
        t0 = turn[0][0]
        t1 = turn[-1][1]
        fname = f"turn_{i:03d}_{t0:.2f}-{t1:.2f}_n{len(turn)}.wav"
        sf.write(str(out_dir / fname), merged, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--voiceprints", required=True, type=Path)
    ap.add_argument("--ellie-name", default="ellie")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--api-key", default=os.environ.get("PYANNOTEAI_API_KEY"))
    ap.add_argument("--non-exclusive", action="store_true")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set PYANNOTEAI_API_KEY or pass --api-key")

    voiceprints = json.loads(args.voiceprints.read_text())
    if args.ellie_name not in voiceprints:
        raise SystemExit(f"'{args.ellie_name}' not in {args.voiceprints}")

    client = Client(api_key=args.api_key)
    exclusive = not args.non_exclusive
    args.out.mkdir(parents=True, exist_ok=True)
    output = run_identify(client, args.audio, voiceprints, exclusive,
                          cache_path=args.out / "identify.json")
    segments, key = pick_segments(output, exclusive)
    print(f"Using '{key}': {len(segments)} segments")

    counts = Counter(s["speaker"] for s in segments)
    print(f"Speaker segment counts: {counts.most_common()}")
    top2 = [lbl for lbl, _ in counts.most_common(2)]
    if len(top2) < 2:
        raise SystemExit(f"Need at least 2 speakers, got: {counts}")

    ellie_label, how = find_ellie_label(output, args.ellie_name)
    print(f"Voiceprint match: Ellie -> {ellie_label} ({how})")
    if ellie_label not in top2:
        raise SystemExit(
            f"Ellie-matched speaker '{ellie_label}' not in top-2 {top2}. "
            f"Counts: {counts.most_common()}"
        )
    subject_label = [lbl for lbl in top2 if lbl != ellie_label][0]
    print(f"Ellie = '{ellie_label}' | Subject = '{subject_label}'")

    turns = group_turns(segments, ellie_label, subject_label)
    print(f"Built {len(turns)} subject turn(s)")

    waveform, sr = load_mono(args.audio)
    save_turns(turns, waveform, sr, args.out)
    total_s = sum(e - s for turn in turns for s, e in turn)
    print(f"Total subject audio across turns: {total_s:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
