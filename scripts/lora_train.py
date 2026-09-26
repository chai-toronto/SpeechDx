#!/usr/bin/env python
"""Train and evaluate one LoRA cell (task x encoder).

    python scripts/lora_train.py --task T25 --encoder wavlm --device cuda

Reads the boundary cache built by ``scripts/lora_warm.py``; never touches audio
and never runs the frozen prefix. Writes ``result.json`` (and, for held-out
tasks, ``best_adapter.pt`` + test predictions) to
``exps/lora/<task>/<encoder>-lr<lr>/`` unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lora-lr", type=float, default=None,
                    help="LoRA learning rate (default 1e-4, the leaderboard value)")
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap uids per split (smoke tests only)")
    ap.add_argument("--control", action="store_true",
                    help="freeze the adapters: a frozen probe under THIS "
                         "protocol, isolating adaptation from protocol drift")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the test evaluation (e.g. while choosing an LR "
                         "on validation loss)")
    ap.add_argument("--cache-root", default=None,
                    help="read the boundary cache from <ROOT>/<dataset>/<encoder>/"
                         "{train,val}/ instead of the committed cache dirs, e.g. "
                         "a node-local SSD copy (read-only)")
    args = ap.parse_args()

    import torch

    from sdx.config import compose_config
    from sdx.lora.model import LoRATailProbe
    from sdx.lora.tails import build_tail
    from sdx.lora.targets import plan_adapters
    from sdx.lora.train import (
        EPOCHS, HEAD_LR, LORA_LR, BoundaryDataset, CellConfig, collate,
        train_cell,
    )
    from sdx.lora.warm_boundary import boundary_cache_paths
    from sdx.warm import _load_manifests

    hparams = compose_config(args.task, args.encoder, mode="warm")
    encoder = hparams["encoder"]
    plan = plan_adapters(encoder, args.encoder)

    # --cache-root only moves where the cache is READ from; nothing reachable
    # from it can change a cell's result.
    cache_root = args.cache_root
    train_path, val_path = boundary_cache_paths(hparams, plan.cut_index,
                                                root=cache_root)
    if cache_root:
        print(f"[lora-train] cache root: {cache_root}")
    for p in (train_path, val_path):
        if not Path(p).exists():
            hint = (f"  --cache-root={cache_root} is set but the copy is "
                    f"missing there"
                    if cache_root else "  run scripts/lora_warm.py first")
            raise SystemExit(f"[lora-train] missing boundary cache: {p}\n{hint}")

    # CV tasks ship list-of-folds manifests and no test split; they route to
    # the per-fold trainer, which reports cross-fold validation as the board
    # does. Detect from the manifest shape rather than a hardcoded task list.
    with open(hparams["train_annotation"]) as fh:
        is_cv = isinstance(json.load(fh), list)

    data = _load_manifests(hparams)

    def split_ids(name):
        # CV manifests carry no "test" split — _load_manifests's CV branch
        # returns only train/val/all — so default to empty rather than KeyError.
        ids = list(data.get(name, {}).keys())
        return ids[:args.limit] if args.limit else ids

    def labels_for(name):  # noqa: D401 - see split_ids
        # The manifest's canonical field is "label"; ``label_encoded`` is
        # DERIVED by make_label_pipeline at dataio time and never present on a
        # raw manifest row. brain._load_train_labels reads "label" for the same
        # reason. A list label (multilabel) becomes a float tensor in collate,
        # matching the pipeline's conversion.
        return {k: v["label"] for k, v in data.get(name, {}).items()}

    n_ver = int(hparams["data_params"].get("num_aug_ver", 1))
    seed = hparams.get("random_seed", 2026)
    sets = {
        "train": BoundaryDataset(train_path, split_ids("train"),
                                 labels_for("train"), n_ver, True, seed),
        # val/test read the unaugmented val cache, as the frozen path does.
        "val": BoundaryDataset(val_path, split_ids("val"), labels_for("val")),
        # CV manifests have no test split; the dict is empty and the loader is
        # never used by the per-fold trainer.
        "test": BoundaryDataset(val_path, split_ids("test"), labels_for("test")),
    }
    loaders = {
        k: torch.utils.data.DataLoader(
            v, batch_size=args.batch_size, shuffle=(k == "train"),
            num_workers=0, collate_fn=collate)
        for k, v in sets.items()
    }

    if is_cv:
        print(f"[lora-train] {args.task} is a CV task — per-fold trainer",
              flush=True)

    tail = build_tail(encoder, args.encoder, plan.cut_index)
    num_labels = hparams["num_labels"]
    probe = LoRATailProbe(tail, plan.target_paths(), num_labels)

    # The frozen prefix must never run during training; enforce it at runtime.
    from sdx.lora.train import guard_prefix_never_runs
    release_guard = guard_prefix_never_runs(encoder, args.encoder, plan.cut_index)

    from sdx.lora.train import build_losses
    if is_cv:
        # CV losses are built PER FOLD inside the trainer — weights must come
        # from each fold's own training partition.
        loss_fn = eval_loss_fn = None
        reg_bins = None
    else:
        loss_fn, eval_loss_fn, reg_bins = build_losses(
            hparams, Path(hparams["train_annotation"]))
    lora_lr = args.lora_lr if args.lora_lr is not None else LORA_LR
    cfg = CellConfig(
        task=args.task, encoder=args.encoder, lora_lr=lora_lr,
        head_lr=args.head_lr or HEAD_LR, epochs=args.epochs or EPOCHS,
        batch_size=args.batch_size, seed=seed, device=args.device,
        control=args.control,
        task_type=hparams.get("task_type", "B"), reg_bins=reg_bins,
        out_dir=Path(args.out) if args.out else
        REPO / "exps" / "lora" / args.task / f"{args.encoder}-lr{lora_lr:g}",
    )
    print(f"[lora-train] {plan.summary()}", flush=True)
    print(f"[lora-train] n={ {k: len(v) for k, v in sets.items()} } "
          f"aug_versions={n_ver} labels={num_labels}", flush=True)

    try:
        if is_cv:
            from sdx.lora.train_cv import train_cell_cv
            result = train_cell_cv(probe, cfg, (train_path, val_path),
                                   hparams, batch_size=args.batch_size)
            # None == SDX_ONLY_FOLD run: the fold is banked, aggregation is
            # deliberately skipped, and writing result.json here would make a
            # partial cell look finished.
            if result is None:
                return 0
            if cfg.out_dir:
                Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
                (Path(cfg.out_dir) / "result.json").write_text(
                    json.dumps(result, indent=2))
            print(f"[lora-train] {args.task} x {args.encoder}: "
                  f"{result['metric_name']} {result['metric_mean']:.4f} "
                  f"+/- {result['metric_std']:.4f} over {result['n_folds']} folds",
                  flush=True)
        else:
            result = train_cell(probe, cfg, loaders["train"], loaders["val"],
                                loaders["test"], loss_fn, eval_loss_fn,
                                evaluate_test=not args.no_test)
    finally:
        release_guard()
    print("[lora-train] " + json.dumps(
        {k: v for k, v in result.items() if k != "history"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
