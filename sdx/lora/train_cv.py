"""Cross-validated LoRA cells (the CV tasks).

Matches the benchmark's CV convention (``sdx/train_cv.py``):

* **No test set.** CV tasks ship list-of-folds `train.json`/`valid.json` and no
  `test.json`; the folds partition every uid. The reported number is the
  *best-validation* metric of each fold, aggregated across folds.
* **Aggregation is a plain mean ± std over folds**, as in ``sdx/train_cv.py``.

Two consequences worth stating rather than discovering later:

1. **The reported metric is selected on the same data it is reported from.**
   Within a fold, the epoch is chosen by best validation loss and the metric is
   read off that epoch's validation predictions. That is optimistically biased
   — but it is what the published board does, so reproducing it is what makes
   the LoRA arm comparable. Do not "fix" it here; it would silently change the
   baseline.
2. **`np.mean`, not `nanmean`.** The board uses a plain mean, so a degenerate
   fold (e.g. an AUROC undefined because a validation fold is single-class)
   poisons the average. We keep the board's behaviour for comparability but
   **warn loudly** when a fold is NaN and also report ``metric_nanmean``, so
   it is visible instead of silent.

``SDX_ONLY_FOLD=<i>`` runs exactly fold *i*, banks it, and stops before
aggregation, so a 5-fold cell can run as 5 concurrent jobs instead of one
sequential process.
Each fold is banked to ``fold_<i>.json`` in the cell's output folder; a later
ordinary run finds every fold banked, skips them all, and writes ``result.json``
through the SAME aggregation code below. Nothing re-implements the arithmetic,
so the parallel and sequential paths cannot drift apart.

Throughput only: fold *i* computes identically alone or as the *i*-th step of
the loop, because the probe is reset per fold, losses are rebuilt from fold
*i*'s own partition, and the dataset seed is ``cfg.seed + i``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sdx.lora.model import GroupAwareLinearSchedule, build_param_groups
from sdx.lora.train import (
    BoundaryDataset,
    collate,
    primary_metric,
    run_epoch,
)


def load_folds(hparams) -> tuple[list[dict], list[dict]]:
    """Return (train_folds, val_folds). Raises if the manifest is not CV."""
    with open(hparams["train_annotation"]) as f:
        tr = json.load(f)
    with open(hparams["val_annotation"]) as f:
        va = json.load(f)
    if not isinstance(tr, list) or not isinstance(va, list):
        raise ValueError(
            f"{hparams['train_annotation']} is not a list-of-folds manifest — "
            f"use the single-split trainer for this task")
    if len(tr) != len(va):
        raise ValueError(f"fold count mismatch: {len(tr)} train, {len(va)} val")
    return tr, va


def train_cell_cv(probe, cfg, cache_paths, hparams, batch_size: int = 16) -> dict:
    """Train one (task, encoder) cell across all folds.

    ``probe`` is reused across folds but **reset between them**
    (``reset_parameters``), so each fold starts from the exact no-op state.
    Reloading the encoder per fold would be equivalent and much slower.

    Losses are rebuilt **per fold**: class weights and regression bins come
    from that fold's training partition only. Building them once and reusing
    would leak other folds' training labels into every fold's loss.
    """
    from sdx.lora.train import build_losses
    train_path, val_path = cache_paths
    train_folds, val_folds = load_folds(hparams)
    n_folds = len(train_folds)
    n_ver = int(hparams["data_params"].get("num_aug_ver", 1))
    device = torch.device(cfg.device)
    probe.to(device)

    only_fold = os.environ.get("SDX_ONLY_FOLD")
    only_fold = int(only_fold) if only_fold not in (None, "") else None
    if only_fold is not None and not (0 <= only_fold < n_folds):
        raise ValueError(
            f"SDX_ONLY_FOLD={only_fold} out of range for {n_folds} folds")
    bank = Path(cfg.out_dir) if cfg.out_dir else None
    if bank is not None:
        bank.mkdir(parents=True, exist_ok=True)

    per_fold, t0 = [None] * n_folds, time.time()
    for k in range(n_folds):
        if only_fold is not None and k != only_fold:
            continue
        # Already banked by an earlier job or cycle: load it rather than
        # retraining. This is what lets N one-fold jobs accumulate into a
        # complete cell that a final ordinary run can aggregate.
        fp = (bank / f"fold_{k}.json") if bank is not None else None
        if fp is not None and fp.exists():
            per_fold[k] = json.loads(fp.read_text())
            print(f"[lora-cv] fold {k}/{n_folds - 1}: already banked — skipping",
                  flush=True)
            continue
        probe.reset_parameters()          # fold independence — see module docstring
        loss_fn, eval_loss_fn, reg_bins = build_losses(
            hparams, Path(hparams["train_annotation"]), fold_idx=k)
        # The losses carry tensors (pos_weight / class weight) built on CPU;
        # move them with the model or BCEWithLogitsLoss dies on a device
        # mismatch. train_cell does this once — the CV path rebuilds per fold,
        # so it has to do it per fold too.
        loss_fn = loss_fn.to(device)
        eval_loss_fn = eval_loss_fn.to(device)
        tr, va = train_folds[k], val_folds[k]
        sets = {
            "train": BoundaryDataset(train_path, list(tr),
                                     {u: v["label"] for u, v in tr.items()},
                                     n_ver, True, cfg.seed + k),
            "val": BoundaryDataset(val_path, list(va),
                                   {u: v["label"] for u, v in va.items()}),
        }
        loaders = {n: torch.utils.data.DataLoader(
            d, batch_size=batch_size, shuffle=(n == "train"),
            num_workers=0, collate_fn=collate) for n, d in sets.items()}

        groups = build_param_groups(
            probe, lora_lr=cfg.lora_lr, head_lr=cfg.head_lr,
            head_weight_decay=cfg.head_l2, lora_weight_decay=cfg.lora_l2)
        if cfg.control:
            probe.freeze_adapters()
            groups = [g for g in groups if g["name"] != "lora"]
        opt = torch.optim.AdamW(groups)
        sched = GroupAwareLinearSchedule(opt, epoch_count=cfg.epochs)

        best = {"val_loss": float("inf"), "epoch": -1, "metric": None}
        for epoch in range(1, cfg.epochs + 1):
            sched.step(epoch)
            run_epoch(probe, loaders["train"], loss_fn, device, opt,
                      reg_bins=reg_bins, task_type=cfg.task_type)
            with torch.no_grad():
                vl, vp, vg = run_epoch(probe, loaders["val"], eval_loss_fn,
                                       device, task_type=cfg.task_type)
            # Selection on validation LOSS, as everywhere else in this track.
            if vl < best["val_loss"]:
                best = {"val_loss": vl, "epoch": epoch,
                        "metric": primary_metric(vp, vg, cfg.task_type)}
        per_fold[k] = best
        if fp is not None:
            fp.write_text(json.dumps(best, indent=2))
        print(f"[lora-cv] fold {k}/{n_folds - 1}: epoch {best['epoch']}, "
              f"val_loss {best['val_loss']:.4f}, "
              f"{cfg.metric_name} {best['metric']:.4f}", flush=True)

    if only_fold is not None:
        # A single-fold job holds only its own fold, so aggregating here would
        # be wrong — and a partial result.json would look like a finished cell.
        print(f"[lora-cv] SDX_ONLY_FOLD run complete — banked fold {only_fold}. "
              f"No aggregation; rerun without SDX_ONLY_FOLD once all {n_folds} "
              f"folds exist to write result.json.", flush=True)
        return None

    missing = [i for i, f in enumerate(per_fold) if f is None]
    if missing:
        raise RuntimeError(f"cannot aggregate: folds {missing} were never run")

    metrics = [f["metric"] for f in per_fold]
    n_nan = sum(1 for m in metrics if m is None or m != m)
    if n_nan:
        print(f"[lora-cv] ⚠ {n_nan}/{n_folds} folds produced a NaN metric. "
              f"The board aggregates with a plain mean, so the result is NaN. "
              f"Reported as-is rather than silently switching to nanmean — "
              f"see the module docstring.", flush=True)
    arr = np.array([np.nan if (m is None) else m for m in metrics], dtype=float)
    return {
        "task": cfg.task, "encoder": cfg.encoder, "cv": True,
        "n_folds": n_folds, "lora_lr": cfg.lora_lr, "head_lr": cfg.head_lr,
        "control": cfg.control, "task_type": cfg.task_type,
        "metric_name": cfg.metric_name,
        # Plain mean, matching sdx/train_cv.py. nanmean reported alongside so
        # a poisoned average is diagnosable without rerunning.
        "metric_mean": float(np.mean(arr)),
        "metric_std": float(np.std(arr)),
        "metric_nanmean": float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else None,
        "n_nan_folds": int(n_nan),
        "per_fold": [
            {"fold": i, "epoch": f["epoch"], "val_loss": f["val_loss"],
             "metric": f["metric"]} for i, f in enumerate(per_fold)],
        "val_loss_mean": float(np.mean([f["val_loss"] for f in per_fold])),
        "wall_s": round(time.time() - t0, 1),
    }
