#!/usr/bin/env python
"""Build one (task, encoder) frozen-prefix boundary cache.

    python scripts/lora_warm.py --task T25 --encoder wavlm --device cuda

Idempotent: already-cached (uid, version) pairs are skipped, so a requeued job
resumes. Tasks that share a dataset share the cache — run the union by passing
``--also-task`` (e.g. T25 with ``--also-task T26``) so the uid set covers both.

The cache stores every frame entering the first adapted block (fp16), so it is
as large as an ASP ``single/`` cache at that layer: plan disk accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--also-task", action="append", default=[],
                    help="extra tasks whose uids share this dataset's cache")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-versions", type=int, default=None)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                    help="encoder precision for the prefix forward: fp32 matches "
                         "the benchmark warm; fp16 autocasts on CUDA (faster; "
                         "how the leaderboard's LoRA rows were warmed)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    from sdx.config import compose_config
    from sdx.lora.targets import plan_adapters
    from sdx.lora.warm_boundary import boundary_cache_paths, warm_boundary
    from sdx.warm import _load_manifests

    hparams = compose_config(args.task, args.encoder, mode="warm")
    data_dict = _load_manifests(hparams)
    all_ids = list(data_dict["all"].keys())

    # Union in the sibling tasks' uids so one cache serves both.
    for extra in args.also_task:
        hp2 = compose_config(extra, args.encoder, mode="warm")
        if hp2["dataset"] != hparams["dataset"]:
            raise SystemExit(
                f"--also-task {extra} is dataset {hp2['dataset']}, not "
                f"{hparams['dataset']} — they cannot share a cache")
        d2 = _load_manifests(hp2)
        before = len(all_ids)
        merged = dict(data_dict["all"])
        merged.update(d2["all"])
        data_dict["all"] = merged
        all_ids = list(merged.keys())
        print(f"[lora-warm] +{len(all_ids) - before} uids from {extra} "
              f"(union {len(all_ids)})", flush=True)

    encoder = hparams["encoder"]
    plan = plan_adapters(encoder, args.encoder)
    train_path, val_path = boundary_cache_paths(hparams, plan.cut_index)

    print(f"[lora-warm] {args.task}{'+' + '+'.join(args.also_task) if args.also_task else ''}"
          f" x {args.encoder}", flush=True)
    print(f"[lora-warm] {plan.summary()}", flush=True)
    print(f"[lora-warm] uids={len(all_ids)} device={args.device}", flush=True)
    print(f"[lora-warm] train -> {train_path}", flush=True)
    print(f"[lora-warm] val   -> {val_path}", flush=True)
    if args.dry_run:
        print("[lora-warm] dry run, stopping before the encoder forward")
        return 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[lora-warm] --device cuda but no CUDA is visible")

    t0 = time.time()
    stats = warm_boundary(hparams, data_dict, all_ids, args.encoder, plan,
                          device=args.device, num_versions=args.num_versions,
                          limit=args.limit, precision=args.precision)
    elapsed = time.time() - t0
    per_chunk = elapsed / max(stats["chunks"], 1)
    print(f"[lora-warm] DONE in {elapsed:.0f}s — {stats['train']} train + "
          f"{stats['val']} val entries, {stats['chunks']} chunks "
          f"({per_chunk:.3f} s/chunk)", flush=True)
    summary = {"task": args.task, "also": args.also_task,
               "encoder": args.encoder, "cut": plan.cut_index,
               "uids": len(all_ids), "elapsed_s": round(elapsed, 1),
               "s_per_chunk": round(per_chunk, 4), **stats}
    print("[lora-warm] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
