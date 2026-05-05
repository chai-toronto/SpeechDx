"""Drop zero-frame / unreadable audio rows from a dataset CSV, in place."""

from pathlib import Path

import pandas as pd
import soundfile as sf


def drop_invalid_audio_rows(csv_path, audio_root, path_col: str = "path") -> int:
    """Validate every audio file in `csv_path[path_col]` against `audio_root`.

    Drops rows that are unreadable or have zero frames, rewrites the CSV,
    and prints a loud banner if anything was dropped. Returns the number
    of dropped rows.
    """
    csv_path = Path(csv_path)
    audio_root = Path(audio_root)

    df = pd.read_csv(csv_path)
    if path_col not in df.columns:
        raise KeyError(f"{csv_path} has no '{path_col}' column")

    bad_idx = []
    bad_rows = []
    for i, rel in enumerate(df[path_col].astype(str)):
        fp = audio_root / rel
        try:
            info = sf.info(fp)
        except Exception as e:
            bad_idx.append(i)
            bad_rows.append((rel, f"unreadable: {type(e).__name__}: {e}"))
            continue
        if info.frames == 0 or info.samplerate == 0:
            bad_idx.append(i)
            bad_rows.append((rel, f"zero-length: frames={info.frames} sr={info.samplerate}"))

    n_dropped = len(bad_idx)
    if n_dropped == 0:
        print(f"[audio-validate] {csv_path.name}: all {len(df)} rows OK.")
        return 0

    df_clean = df.drop(index=bad_idx).reset_index(drop=True)
    df_clean.to_csv(csv_path, index=False, lineterminator="\r\n")

    bar = "!" * 78
    print()
    print(bar)
    print(f"!!! ZERO-LENGTH / UNREADABLE AUDIO DROPPED FROM {csv_path.name.upper()}: {n_dropped} ROWS")
    print(f"!!! audio root: {audio_root}")
    print(bar)
    for rel, why in bad_rows[:50]:
        print(f"  - {rel}  [{why}]")
    if len(bad_rows) > 50:
        print(f"  ... and {len(bad_rows) - 50} more")
    print(bar)
    print(f"!!! {csv_path.name}: {len(df)} -> {len(df_clean)} rows after drop.")
    print(bar)
    print()
    return n_dropped
