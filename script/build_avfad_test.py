"""Subset 5% of AVFAD speakers into a separate test dataset.

Deterministic (random_state=42). Copies (not symlinks, per the project's
no-symlinks rule) the selected speakers' audio into
``data/avfad_test/processed/audio/`` and writes a slim
``data/avfad_test/processed/avfad_test.csv`` with the same columns.

Goal: small enough to warm caches on CPU in a few minutes; structurally
identical to the parent dataset so the same prep + warm code paths apply.
Used to byte-content-verify the ahb-rewrite cache warmer against the
legacy ``master_dataio_prep`` path without needing GPU time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO / "data" / "avfad" / "processed"
DST_ROOT = REPO / "data" / "avfad_test" / "processed"

FRACTION = 0.05
SEED = 42


def main() -> None:
    src_csv = SRC_ROOT / "avfad.csv"
    src_audio = SRC_ROOT / "audio"
    dst_csv = DST_ROOT / "avfad_test.csv"
    dst_audio = DST_ROOT / "audio"

    df = pd.read_csv(src_csv)
    speakers = sorted(df["Participant_ID"].unique())
    n = max(1, int(round(len(speakers) * FRACTION)))
    sampled = (
        pd.Series(speakers).sample(n=n, random_state=SEED).sort_values().tolist()
    )
    sub = df[df["Participant_ID"].isin(sampled)].copy()

    print(f"Sampled {len(sampled)}/{len(speakers)} speakers ({len(sub)}/{len(df)} rows).")
    print(f"  label dist: {sub['label'].value_counts().to_dict()}")

    dst_audio.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for path in sub["path"].unique():
        src = src_audio / path
        dst = dst_audio / path
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    sub.to_csv(dst_csv, index=False)
    total_mb = sum(p.stat().st_size for p in dst_audio.rglob("*") if p.is_file()) / 1e6
    print(f"  copied={copied}  skipped={skipped}  total_audio_MB={total_mb:.1f}")
    print(f"Wrote {dst_csv}")


if __name__ == "__main__":
    main()
