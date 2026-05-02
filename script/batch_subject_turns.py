"""Batch-run subject-turn extraction across all DAIC-WOZ recordings.

For each <id>_P/<id>_AUDIO.wav:
  1. Run pyannoteAI identify with Ellie voiceprint
  2. Pick top-2 speakers; Ellie = voiceprint-matched speaker; subject = the other
  3. Save per-turn wavs and a single combined wav into one flat folder
"""

import argparse
import concurrent.futures as cf
import json
import os
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from pyannoteai.sdk import Client

from diarize_subject_turns import (
    find_ellie_label,
    group_turns,
    load_mono,
    pick_segments,
    save_turns,
)


def process_one(audio_path: Path, turns_dir: Path, combined_path: Path,
                voiceprints: dict, ellie_name: str, api_key: str) -> tuple[str, str]:
    sid = audio_path.parent.name
    if combined_path.exists():
        return sid, "skip_exists"
    try:
        turns_dir.mkdir(parents=True, exist_ok=True)
        cache_path = turns_dir / "identify.json"
        if cache_path.exists():
            output = json.loads(cache_path.read_text())
        else:
            client = Client(api_key=api_key)
            media_url = client.upload(audio_path)
            job_id = client.identify(media_url, voiceprints=voiceprints, exclusive=True)
            output = client.retrieve(job_id).get("output", {})
            cache_path.write_text(json.dumps(output, indent=2))
        segments, _ = pick_segments(output, exclusive=True)

        from collections import Counter
        counts = Counter(s["speaker"] for s in segments)
        top2 = [lbl for lbl, _ in counts.most_common(2)]
        if len(top2) < 2:
            return sid, f"fail_one_speaker:{counts}"
        ellie_label, _ = find_ellie_label(output, ellie_name)
        if ellie_label not in top2:
            return sid, f"fail_ellie_not_top2:ellie={ellie_label} top2={top2}"
        subject_label = [l for l in top2 if l != ellie_label][0]

        turns = group_turns(segments, ellie_label, subject_label)
        waveform, sr = load_mono(audio_path)
        turns_dir.mkdir(parents=True, exist_ok=True)
        save_turns(turns, waveform, sr, turns_dir)

        chunks = []
        for turn in turns:
            for s, e in turn:
                a, b = max(0, int(s * sr)), min(len(waveform), int(e * sr))
                if b > a:
                    chunks.append(waveform[a:b])
        if not chunks:
            return sid, "fail_no_subject_audio"
        merged = np.concatenate(chunks)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(combined_path), merged, sr)
        return sid, f"ok turns={len(turns)} subject_s={len(merged)/sr:.1f}"
    except Exception as e:
        return sid, f"error:{type(e).__name__}:{e}\n{traceback.format_exc()[-500:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/daic_woz/processed/audio"))
    ap.add_argument("--voiceprints", type=Path, default=Path("data/daic_woz/voiceprints/ellie.json"))
    ap.add_argument("--ellie-name", default="ellie")
    ap.add_argument("--turns-root", type=Path, default=Path("data/daic_woz/subject_turns"))
    ap.add_argument("--combined-root", type=Path, default=Path("data/daic_woz/subject_combined"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-key", default=os.environ.get("PYANNOTEAI_API_KEY"))
    ap.add_argument("--log", type=Path, default=Path("data/daic_woz/subject_combined/_batch.log"))
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set PYANNOTEAI_API_KEY")

    voiceprints = json.loads(args.voiceprints.read_text())
    subjects = sorted(p for p in args.root.iterdir() if p.is_dir())
    tasks = []
    for sdir in subjects:
        wavs = list(sdir.glob("*_AUDIO.wav"))
        if not wavs:
            continue
        sid = sdir.name
        tasks.append((wavs[0], args.turns_root / sid, args.combined_root / f"{sid}.wav"))

    args.log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(tasks)} recordings with {args.workers} workers -> {args.combined_root}")
    with args.log.open("a") as logf, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, a, t, c, voiceprints, args.ellie_name, args.api_key): a
            for a, t, c in tasks
        }
        done = 0
        for fut in cf.as_completed(futures):
            sid, status = fut.result()
            done += 1
            line = f"[{done}/{len(tasks)}] {sid}: {status}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()


if __name__ == "__main__":
    main()
