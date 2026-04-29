"""Invalidate stale entries in a (encoder, dataset) cache after min/max length change.

Replays the chunk_signal pad/split logic against each file's duration to flag uids
whose cached output now differs under the new (min_length, max_length). Then opens
each cache.hdf5 and deletes those uid groups (h5py removes the group plus all
v0..vN version slots in one shot). Re-running training with warm_cache=True will
recompute only the missing entries.

Usage:
    python -m training.dataio.invalidate_cache <encoder> <dataset> [--apply]
    python training/dataio/invalidate_cache.py <encoder> <dataset> [--apply]

Without --apply this is a dry-run. With --apply it deletes affected entries
(after copying cache.hdf5 -> cache.hdf5.bak on first run).
"""
import sys
import shutil
import ast
from pathlib import Path
from multiprocessing import Pool

import h5py
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
EMB_ROOT = REPO / "embeddings_avg_final"

# (old_min, new_min), (old_max, new_max) per encoder yaml change.
# Update this table whenever encoder yamls change again.
ENCODER_CHANGES = {
    "ast":          ((3, 10), (10,   10)),
    "audiomae":     ((3, 10), (10,   10)),
    "clap":         ((3, 10), (10,   10)),
    "emotion2vec":  ((3, 1),  (1e6,  250)),
    "hubert":       ((3, 1),  (1e6,  1e6)),
    "qwen3_voice":  ((1, 1),  (1000, 1e6)),
    "w2v2":         ((3, 1),  (1e6,  1e6)),
    "wavlm":        ((3, 1),  (1e6,  300)),
    "whisper":      ((3, 30), (30,   30)),
}

# Datasets that ship a duration column in their CSV — skip per-file sf.info.
DATASET_DUR_COL = {
    "aphasia":  "duration_s",
    "iemocap":  "duration",
    "ravdess":  "duration",
    "uaspeech": "duration",
}


def affected(d, mins, maxs):
    """True iff chunk_signal output differs between old and new (min, max)."""
    old_min, new_min = mins
    old_max, new_max = maxs
    if max(d, old_min) != max(d, new_min):
        return True
    if (d > old_max) != (d > new_max):
        return True
    if d > old_max and d > new_max and old_max != new_max:
        return True
    return False


def _info(p):
    try:
        i = sf.info(str(p))
        return i.frames / i.samplerate
    except Exception:
        return None


def load_durations(dataset):
    """Return (uids, durations, is_segmented).

    For edaic, durations is a list-of-lists of segment durations (split_by_boundary
    means each segment is padded/chunked independently).
    """
    csv = DATA_ROOT / dataset / "processed" / f"{dataset}.csv"
    audio = csv.parent / "audio"
    df = pd.read_csv(csv)
    uids = df["uid"].astype(str).tolist()
    if dataset == "edaic":
        seg_durs = []
        for b in df["boundaries"]:
            try:
                arr = ast.literal_eval(b)
                seg_durs.append([arr[k+1] - arr[k] for k in range(0, len(arr) - 1, 2)])
            except Exception:
                seg_durs.append([])
        return uids, seg_durs, True
    col = DATASET_DUR_COL.get(dataset)
    if col and col in df.columns:
        durs = df[col].astype(float).tolist()
    else:
        with Pool(8) as pool:
            durs = pool.map(_info, [str(audio / p) for p in df["path"]], chunksize=64)
    return uids, durs, False


def find_affected_uids(encoder, dataset):
    mins, maxs = ENCODER_CHANGES[encoder]
    uids, durs, is_segmented = load_durations(dataset)
    out = []
    for uid, d in zip(uids, durs):
        if is_segmented:
            if any(affected(x, mins, maxs) for x in d):
                out.append(uid)
        else:
            if d is None or d != d:
                continue
            if affected(d, mins, maxs):
                out.append(uid)
    return out


def cache_paths(encoder, dataset):
    """All cache.hdf5 under embeddings_avg_final/<dataset>/<encoder>/{train,val}/<mode>/."""
    base = EMB_ROOT / dataset / encoder
    if not base.exists():
        return []
    return sorted(base.glob("*/*/cache.hdf5"))


def invalidate(cache_path, uids, apply_):
    with h5py.File(cache_path, "r") as f:
        present = [u for u in uids if u in f]
        absent = [u for u in uids if u not in f]
        total = len(f)
    print(f"  {cache_path}")
    print(f"    cache entries: {total}")
    print(f"    targets present: {len(present)}  absent: {len(absent)}")
    if absent[:5]:
        print(f"    sample absent: {absent[:5]}")
    if not apply_ or not present:
        return present
    backup = cache_path.with_suffix(".hdf5.bak")
    if not backup.exists():
        print(f"    backing up -> {backup}")
        shutil.copy2(cache_path, backup)
    with h5py.File(cache_path, "a") as f:
        for u in present:
            del f[u]
    print(f"    deleted {len(present)} entries.")
    return present


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    encoder = sys.argv[1]
    dataset = sys.argv[2]
    apply_ = "--apply" in sys.argv[3:]
    if encoder not in ENCODER_CHANGES:
        sys.exit(f"Unknown encoder {encoder!r}. Known: {sorted(ENCODER_CHANGES)}")
    print(f"=== Invalidate cache: encoder={encoder} dataset={dataset} apply={apply_} ===")
    uids = find_affected_uids(encoder, dataset)
    print(f"Affected uids ({len(uids)}): {uids[:20]}{'...' if len(uids) > 20 else ''}")
    caches = cache_paths(encoder, dataset)
    if not caches:
        print(f"No cache files found at {EMB_ROOT / dataset / encoder}")
        return
    for c in caches:
        invalidate(c, uids, apply_)


if __name__ == "__main__":
    main()
