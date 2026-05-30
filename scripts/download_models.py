#!/usr/bin/env python3
"""Pre-download all HF model weights used by run_all.py encoders.

Respects HF_HOME / HF_HUB_CACHE from the environment (set in run_all_slurm.sh:
$SCRATCH/hf and $SCRATCH/hf/hub). Also fetches the OPERA-GT checkpoint into
third_party/OPERA/cks/model/ where model/opera.py expects it.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

HF_REPOS = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "hance-ai/audiomae",
    "laion/larger_clap_general",
    "facebook/hubert-large-ls960-ft",
    "facebook/mms-300m",
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "facebook/wav2vec2-large-960h-lv60-self",
    "labhamlet/wavjepa-nat-base",
    "microsoft/wavlm-large",
    "openai/whisper-large-v3",
    # emotion2vec: FunASR primary uses modelscope, but HF mirror exists
    "emotion2vec/emotion2vec_plus_large",
]


def main():
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME", "")
    print(f"HF_HOME={os.environ.get('HF_HOME')}  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    print(f"Downloading {len(HF_REPOS)} repos to HF cache…")

    failed = []
    for repo in HF_REPOS:
        print(f"\n=== {repo} ===", flush=True)
        try:
            path = snapshot_download(repo_id=repo)
            print(f"  OK -> {path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((repo, str(e)))

    # OPERA-GT checkpoint (into third_party/OPERA/cks/model)
    repo_root = Path(__file__).resolve().parents[1]
    opera_dir = repo_root / "third_party" / "OPERA" / "cks" / "model"
    opera_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== evelyn0414/OPERA encoder-operaGT.ckpt -> {opera_dir} ===", flush=True)
    try:
        p = hf_hub_download("evelyn0414/OPERA", "encoder-operaGT.ckpt",
                            local_dir=str(opera_dir))
        print(f"  OK -> {p}")
    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append(("evelyn0414/OPERA", str(e)))

    if failed:
        print("\nSUMMARY: FAILED DOWNLOADS:")
        for r, e in failed:
            print(f"  - {r}: {e}")
        sys.exit(1)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
