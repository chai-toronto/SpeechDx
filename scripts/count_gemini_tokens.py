"""Count Gemini API tokens for every SpeechDx task.

For each task in the registry (defined in test_qwen3omni_all.py), compute the
total input-token cost of sending every sample (audio + zero-shot prompt) to
the Gemini API.

Token model (per https://ai.google.dev/gemini-api/docs/audio):
    - audio:  32 tokens per second
    - text:   counted via client.models.count_tokens (once per task prompt)

CV tasks are de-duplicated by rowid (zero-shot predictions are fold-independent,
matching test_qwen3omni_all.py).

Usage:
    python scripts/count_gemini_tokens.py
        --> gemini_token_counts.csv  +  stdout summary
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf

# Reuse registry and sample loader from the Qwen reference script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3omni_all import TASKS, PROMPTS, load_samples  # noqa: E402

# T7/T8 manifests were moved under exps/single_task/ after the Qwen script was
# written; correct the paths in the imported registry.
TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}

AUDIO_TOKENS_PER_SECOND = 32
MODEL = "gemini-3.1-pro-preview"


def get_api_key() -> str | None:
    if k := os.environ.get("GEMINI_API_KEY"):
        return k
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "gemini-api-key", "-w"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# Some manifests/CSVs were written with legacy directory names that have since
# been renamed on disk. Rewrite the path prefix before opening the file.
_PATH_REWRITES = {
    "/data/dbank/": "/data/dementiabank/",
    "/data/c9s/": "/data/c19sounds/",
}


_TORGO_PREFIX_RE = re.compile(r"^s\d+_")


def _rewrite_path(p: str) -> str:
    for old, new in _PATH_REWRITES.items():
        if old in p:
            p = p.replace(old, new, 1)
            break
    # Torgo: manifests use "sN_NNNN.wav" but on-disk files are "NNNN.wav".
    if "/data/torgo/" in p and not Path(p).exists():
        path = Path(p)
        stripped = _TORGO_PREFIX_RE.sub("", path.name)
        if stripped != path.name:
            alt = path.with_name(stripped)
            if alt.exists():
                return str(alt)
    return p


def audio_duration_sec(path: str) -> float | None:
    try:
        info = sf.info(_rewrite_path(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return None


def per_task_stats(task: str, prompt_tokens: int) -> dict:
    samples, _ = load_samples(task)
    seen = {}
    for s in samples:
        seen[str(s["_rowid"])] = s
    unique = list(seen.values())
    paths = [s["path"] for s in unique]

    with ThreadPoolExecutor(max_workers=32) as ex:
        durations = list(ex.map(audio_duration_sec, paths))

    valid = [d for d in durations if d is not None]
    audio_tok = sum(math.ceil(d * AUDIO_TOKENS_PER_SECOND) for d in valid)
    text_tok = prompt_tokens * len(valid)
    return {
        "task": task,
        "n_unique": len(unique),
        "n_loaded": len(valid),
        "n_failed": len(unique) - len(valid),
        "total_audio_sec": sum(valid),
        "total_audio_hr": sum(valid) / 3600.0,
        "audio_tokens": audio_tok,
        "prompt_tokens_per_call": prompt_tokens,
        "text_tokens_total": text_tok,
        "total_tokens": audio_tok + text_tok,
    }


def main() -> int:
    key = get_api_key()
    if not key:
        print("ERROR: no Gemini API key found in env or macOS keychain "
              "(service 'gemini-api-key').", file=sys.stderr)
        return 1

    from google import genai
    client = genai.Client(api_key=key)

    print(f"[counter] model={MODEL}, audio rate={AUDIO_TOKENS_PER_SECOND} tok/sec")
    print("[counter] counting text prompts via Gemini count_tokens...")
    prompt_tokens: dict[str, int] = {}
    for task in sorted(PROMPTS.keys(), key=lambda t: int(t[1:])):
        try:
            resp = client.models.count_tokens(model=MODEL, contents=PROMPTS[task])
            prompt_tokens[task] = int(resp.total_tokens)
        except Exception as e:
            print(f"  [error] {task} count_tokens failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 2
        print(f"  {task}: prompt={prompt_tokens[task]} tokens")

    print("\n[counter] reading audio durations + computing per-task totals...")
    rows = []
    for task in sorted(TASKS.keys(), key=lambda t: int(t[1:])):
        r = per_task_stats(task, prompt_tokens[task])
        rows.append(r)
        print(f"  {task}: n={r['n_loaded']:>5d}  audio={r['total_audio_hr']:6.2f}h  "
              f"audio_tok={r['audio_tokens']:>11,}  total={r['total_tokens']:>11,}")
        if r["n_failed"]:
            print(f"        ({r['n_failed']} samples failed to open)")

    df = pd.DataFrame(rows)
    out_path = Path("gemini_token_counts.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 72)
    print(f"GRAND TOTAL across {len(rows)} tasks:")
    print(f"  unique samples:  {df['n_loaded'].sum():>12,}")
    print(f"  audio duration:  {df['total_audio_hr'].sum():>12.2f} hours")
    print(f"  audio tokens:    {df['audio_tokens'].sum():>12,}")
    print(f"  text tokens:     {df['text_tokens_total'].sum():>12,}")
    print(f"  TOTAL tokens:    {df['total_tokens'].sum():>12,}")
    print("=" * 72)
    print(f"\nPer-task breakdown -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
