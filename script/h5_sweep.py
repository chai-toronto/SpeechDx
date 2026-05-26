"""Walk all wavlm cache.hdf5 files and report which are openable.
Run from the repo root with the speech-health-ai venv active."""
import glob, os, sys, h5py

files = []
for d in sorted(glob.glob('embeddings_avg_finalv2/*/wavlm')):
    for split in ('train', 'val'):
        p = f"{d}/{split}/single/cache.hdf5"
        if os.path.exists(p):
            files.append(p)

print(f"Probing {len(files)} cache files...", flush=True)
for f in files:
    sz = os.path.getsize(f)
    try:
        with h5py.File(f, 'r') as h:
            n = len(list(h.keys()))
        print(f"OK   {sz:>14,d}b  keys={n:<8d}  {f}", flush=True)
    except Exception as e:
        msg = str(e).splitlines()[0]
        print(f"BAD  {sz:>14,d}b  {f}", flush=True)
        print(f"     {msg}", flush=True)
