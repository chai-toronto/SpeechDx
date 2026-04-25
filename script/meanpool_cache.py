#!/usr/bin/env python3
"""Mirror tmp/ -> tmp2/, mean-pooling each cached embedding over time.

Source layout:  tmp/<dataset>/<encoder>/<split>/single/cache.hdf5
                    keys: uid/v0  with shape (T, D)

Output layout:  tmp2/<dataset>/<encoder>/<split>/single/cache.hdf5
                    keys: uid/v0  with shape (D,)   (mean over axis 0)

Usage:
    ./spa/bin/python script/meanpool_cache.py [--src tmp] [--dst tmp2]
                                              [--workers N] [--overwrite]
"""
import argparse
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def find_caches(src: Path):
    return sorted(src.rglob("cache.hdf5"))


def meanpool_one(args):
    src_path, dst_path, overwrite = args
    rel = str(src_path)
    try:
        if dst_path.exists() and not overwrite:
            return rel, "skip", 0

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = dst_path.with_suffix(dst_path.suffix + ".tmp")
        if tmp_out.exists():
            tmp_out.unlink()

        n = 0
        with h5py.File(src_path, "r", locking=False) as fin, \
             h5py.File(tmp_out, "w") as fout:
            for uid in fin.keys():
                g = fin[uid]
                if not isinstance(g, h5py.Group):
                    continue
                og = fout.create_group(uid)
                for vkey in g.keys():
                    arr = g[vkey][:]
                    if arr.ndim >= 2:
                        pooled = arr.mean(axis=0)
                    else:
                        pooled = arr
                    og.create_dataset(vkey, data=pooled.astype(arr.dtype))
                    n += 1
        os.replace(tmp_out, dst_path)
        return rel, "ok", n
    except Exception as e:
        # Clean up partial output so a rerun can try again.
        try:
            tmp_out = dst_path.with_suffix(dst_path.suffix + ".tmp")
            if tmp_out.exists():
                tmp_out.unlink()
        except OSError:
            pass
        return rel, f"err:{type(e).__name__}:{e}", 0


def verify_one(args):
    src_path, sample = args
    rel = str(src_path)
    try:
        n_keys = 0
        n_bad_shape = 0
        n_nonfinite = 0
        dims = set()
        bad_examples = []
        with h5py.File(src_path, "r", locking=False) as fin:
            uids = list(fin.keys())
            if sample and sample > 0 and len(uids) > sample:
                step = max(1, len(uids) // sample)
                uids = uids[::step][:sample]
            for uid in uids:
                g = fin[uid]
                if not isinstance(g, h5py.Group):
                    continue
                for vkey in g.keys():
                    arr = g[vkey][:]
                    n_keys += 1
                    if arr.ndim != 1:
                        n_bad_shape += 1
                        if len(bad_examples) < 3:
                            bad_examples.append(f"{uid}/{vkey} shape={arr.shape}")
                        continue
                    dims.add(int(arr.shape[0]))
                    if not np.all(np.isfinite(arr)):
                        n_nonfinite += 1
                        if len(bad_examples) < 3:
                            bad_examples.append(f"{uid}/{vkey} non-finite")
        status = "ok" if (n_bad_shape == 0 and n_nonfinite == 0 and len(dims) <= 1) else "bad"
        return rel, status, n_keys, n_bad_shape, n_nonfinite, sorted(dims), bad_examples
    except Exception as e:
        return rel, f"err:{type(e).__name__}:{e}", 0, 0, 0, [], []


def cmd_verify(args):
    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"src not found: {src}", file=sys.stderr)
        sys.exit(1)
    caches = find_caches(src)
    if not caches:
        print(f"no caches under {src}")
        return
    print(f"verifying {len(caches)} cache files under {src}")
    jobs = [(c, args.sample) for c in caches]
    results = []
    if args.workers <= 1:
        for j in tqdm(jobs, desc="verify"):
            results.append(verify_one(j))
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(verify_one, jobs),
                          total=len(jobs), desc="verify"):
                results.append(r)

    n_ok = sum(1 for r in results if r[1] == "ok")
    n_bad = sum(1 for r in results if r[1] == "bad")
    n_err = len(results) - n_ok - n_bad
    total_keys = sum(r[2] for r in results)
    print(f"\nverified: {n_ok} ok, {n_bad} bad, {n_err} errors; {total_keys} keys checked")
    for rel, status, n_keys, n_bad_shape, n_nonfinite, dims, bad_examples in results:
        if status == "ok":
            continue
        print(f"  {rel}  ->  {status}  keys={n_keys} bad_shape={n_bad_shape} non_finite={n_nonfinite} dims={dims}")
        for ex in bad_examples:
            print(f"      {ex}")
    if n_bad or n_err:
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tmp")
    ap.add_argument("--dst", default="tmp2")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel files (default 1). Each worker streams one key at a time, so RAM use is bounded.")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if destination already exists")
    ap.add_argument("--verify", action="store_true",
                    help="verify --src caches are mean-pooled (1-D, finite, consistent D); skip writing")
    ap.add_argument("--sample", type=int, default=0,
                    help="when verifying, check at most N evenly-spaced uids per file (0 = all)")
    args = ap.parse_args()

    if args.verify:
        cmd_verify(args)
        return

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        print(f"src not found: {src}", file=sys.stderr)
        sys.exit(1)
    if dst == src:
        print("dst must differ from src", file=sys.stderr)
        sys.exit(1)

    caches = find_caches(src)
    if not caches:
        print(f"no caches under {src}")
        return

    jobs = []
    for s in caches:
        rel = s.relative_to(src)
        d = dst / rel
        jobs.append((s, d, args.overwrite))

    print(f"found {len(jobs)} cache files; writing to {dst}")

    results = []
    if args.workers <= 1:
        for j in tqdm(jobs, desc="meanpool"):
            results.append(meanpool_one(j))
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(meanpool_one, jobs),
                          total=len(jobs), desc="meanpool"):
                results.append(r)

    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_skip = sum(1 for _, s, _ in results if s == "skip")
    n_err = len(results) - n_ok - n_skip
    total_items = sum(n for _, s, n in results if s == "ok")
    print(f"\ndone: {n_ok} ok, {n_skip} skipped (already exist), {n_err} errors; {total_items} keys pooled")
    if n_err:
        print("errors:")
        for rel, s, _ in results:
            if s.startswith("err:"):
                print(f"  {rel}  ->  {s}")


if __name__ == "__main__":
    main()
