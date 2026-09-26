"""One LoRA cell: train the adapted tail + head, evaluate once on test.

Deliberately a separate trainer from ``sdx/train.py`` rather than a flag on it.
The frozen-probe path is reader-only and asserts that no encoder module is ever
imported (``sdx/dataio/read.py:assert_no_encoder_imports``); this path *must*
instantiate encoder blocks. Keeping them apart means the published benchmark
cannot be perturbed by anything here.

Protocol: one trial per cell, 15 epochs, no early stopping, fixed learning
rates (no HP search), two optimizer groups, best-validation-loss checkpoint
evaluated once on test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sdx.lora.cache import BoundaryCache
from sdx.lora.layers import adapter_state_dict, load_adapter_state_dict
from sdx.lora.model import GroupAwareLinearSchedule, build_param_groups

# Head LR / L2: the most frequent winner of the benchmark's own 5-trial probe
# search (307 of 412 frozen-probe cells picked this configuration).
HEAD_LR = 9.476e-4
HEAD_L2 = 0.012272
# LoRA LR used for the leaderboard's LoRA rows (one value for every cell).
LORA_LR = 1e-4
EPOCHS = 15


def build_losses(hparams, train_manifest: Path, fold_idx: int | None = None):
    """Return ``(train_loss, eval_loss, reg_bins)`` matching the benchmark.

    ``reg_bins`` is ``None`` for classification, or ``(edges, weights)`` for
    regression — see the R branch below.

    ``fold_idx`` is **required for CV tasks** and must be the fold being
    trained. Class weights and regression bins are computed from the training
    partition only, so reusing one fold's weights across all folds would mix in
    information from other folds' training data. ``brain._load_train_labels``
    enforces this by refusing a list-of-folds manifest without a fold index.

    ``main.yaml`` sets ``auto_class_weights: true``, and ``brain.py`` then
    computes ``pos_weight`` from the TRAIN manifest and overrides whatever the
    task yaml set. Crucially it also keeps a separate **unweighted** loss for
    validation and test, so the reported val/test loss reflects raw
    generalization error rather than training-set class frequencies.

    That second part is load-bearing here: checkpoint selection is on
    validation loss, so using the weighted loss for validation would select a
    different epoch than the frozen board would.
    """
    import copy

    from sdx.brain import _load_train_labels, compute_auto_class_weights

    train_loss = copy.deepcopy(hparams["loss"])
    eval_loss = copy.deepcopy(hparams["loss"])
    for attr in ("pos_weight", "weight"):
        if getattr(eval_loss, attr, None) is not None:
            setattr(eval_loss, attr, None)
    eval_loss.reduction = "mean"

    if not hparams.get("auto_class_weights", False):
        return train_loss, eval_loss, None
    if not Path(train_manifest).exists():
        print(f"[auto_class_weights] manifest missing at {train_manifest}; "
              f"using the task yaml's loss as-is", flush=True)
        return train_loss, eval_loss, None

    task_type = hparams.get("task_type", "B")
    dp = hparams.get("data_params") or {}
    result = compute_auto_class_weights(
        task_type=task_type,
        labels=_load_train_labels(str(train_manifest), fold_idx=fold_idx),
        num_labels=hparams["num_labels"],
        # Clinical bin edges live in the task yaml (e.g. T8's MMSE cutoffs) and
        # take priority over quantiles, exactly as brain.py does.
        cutoffs=dp.get("clinical_cutoffs"),
    )
    if result is None:
        print("[auto_class_weights] could not compute; using loss as-is",
              flush=True)
        return train_loss, eval_loss, None
    kind, payload = result
    if kind == "regression_bins":
        edges, bin_w = payload
        # brain.py switches the TRAIN loss to elementwise and applies
        # per-sample bin weights in compute_objectives; the val/test loss stays
        # unweighted and mean-reduced.
        train_loss.reduction = "none"
        print(f"[auto_class_weights] regression_bins edges={edges.tolist()} "
              f"weights={[round(w, 3) for w in bin_w.tolist()]}", flush=True)
        return train_loss, eval_loss, (edges, bin_w)
    setattr(train_loss, "pos_weight" if kind == "pos_weight" else "weight",
            payload)
    print(f"[auto_class_weights] {kind}={payload.tolist()[:6]}"
          f"{' ...' if payload.numel() > 6 else ''}", flush=True)
    return train_loss, eval_loss, None


def guard_prefix_never_runs(encoder, model_name: str, cut: int):
    """Assert the frozen prefix is never executed during training.

    The tail only holds ``blocks[cut:]``, but the encoder object is still alive
    in the process (WavLM's tail legitimately keeps a reference to layer 0's
    attention to recompute position_bias — it calls ``compute_bias``, never
    that block's ``forward``). A structural argument is weaker than a runtime
    one, so hook every prefix block and raise if any of them is ever called.

    Returns a zero-arg function that removes the hooks.
    """
    from sdx.lora.targets import SPECS, _resolve

    blocks = _resolve(encoder, SPECS[model_name].blocks)
    handles = []

    def _boom(idx):
        def hook(*_a, **_k):
            raise AssertionError(
                f"frozen prefix block {idx} was executed "
                f"during training — the boundary cache is not being used"
            )
        return hook

    for i in range(cut):
        handles.append(blocks[i].register_forward_pre_hook(_boom(i)))
    return lambda: [h.remove() for h in handles]


@dataclass
class CellConfig:
    task: str
    encoder: str
    lora_lr: float
    head_lr: float = HEAD_LR
    head_l2: float = HEAD_L2
    lora_l2: float = 0.0
    epochs: int = EPOCHS
    batch_size: int = 16
    seed: int = 2026
    device: str = "cuda"
    control: bool = False
    task_type: str = "B"
    reg_bins: object = None

    @property
    def metric_name(self) -> str:
        return {"R": "MAE", "C": "macroAUROC_ovr"}.get(self.task_type,
                                                       "macroAUROC")
    out_dir: Path | None = None
    history: list = field(default_factory=list)


class BoundaryDataset(torch.utils.data.Dataset):
    """(uid → per-chunk boundary tensors, label), sampling one aug version."""

    def __init__(self, cache_path: Path, uids: list[str], labels: dict,
                 num_versions: int = 1, train: bool = False, seed: int = 2026):
        self.cache_path = Path(cache_path)
        self.uids = list(uids)
        self.labels = labels
        self.num_versions = num_versions
        self.train = train
        self._cache: BoundaryCache | None = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def _handle(self) -> BoundaryCache:
        if self._cache is None:  # opened lazily so workers get their own
            self._cache = BoundaryCache(self.cache_path)
        return self._cache

    def __getitem__(self, i: int):
        uid = self.uids[i]
        # Matches the frozen reader: one version drawn uniformly per read.
        v = int(self._rng.integers(self.num_versions)) if self.train else 0
        chunks = self._handle().read(uid, v)
        return uid, chunks, self.labels[uid]


def collate(batch):
    """Labels stay in their natural dtype here; ``run_epoch`` casts per task
    type. Multiclass needs int64 targets for CrossEntropyLoss, everything else
    needs float — casting to float here would silently break C."""
    uids, chunks, labels = zip(*batch)
    return list(uids), list(chunks), torch.as_tensor(np.array(labels))


def run_epoch(probe, loader, loss_fn, device, optimizer=None, reg_bins=None,
              task_type="B"):
    """One pass. Training micro-batches with gradient accumulation.

    **Why micro-batch.** The frozen probe consumed a ``(B, D)`` vector per
    recording, so a batch was trivial. Here each recording's tail forward
    builds an autograd graph over its full frame sequence, and WavLM
    materialises attention as ``(B·heads, T, T)``. mdvr's longest recordings
    are ~9,000 frames: one is ~2.6 GB of attention per layer, and holding
    sixteen graphs at once OOM'd an 80 GB H100 on the *smallest* dataset in
    the benchmark.

    Accumulating ``loss_i / N`` per recording and stepping once per batch is
    **mathematically identical** to a mean-reduced batch of N — same gradient,
    same optimizer step — while capping peak memory at one recording's graph.
    Under the benchmark-integrity rule this is a throughput/memory change, not
    a methodological one: ``batch_size`` still defines the optimizer step.
    """
    train = optimizer is not None
    probe.train(train)
    total, n = 0.0, 0
    preds, gold = [], []
    for _uids, chunk_lists, labels in loader:
        labels = labels.to(device)
        bsz = len(chunk_lists)
        if train:
            optimizer.zero_grad(set_to_none=True)

        batch_logits, batch_target = [], []
        for i, chunks in enumerate(chunk_lists):
            logit = probe([c.to(device, torch.float32) for c in chunks])
            lab = labels[i:i + 1]
            if task_type == "C":
                tgt = lab.long().reshape(-1)
                lg = logit.unsqueeze(0)
            else:
                lab = lab.float()
                if logit.shape[-1] == 1:
                    lg = logit.reshape(1)
                    tgt = lab.reshape(1)
                else:
                    lg = logit.unsqueeze(0)
                    tgt = lab
            li = loss_fn(lg, tgt)
            if reg_bins is not None and li.dim() > 0:
                edges, bin_w = reg_bins
                w = bin_w.to(device)[torch.bucketize(tgt.reshape(-1),
                                                     edges.to(device))]
                li = (li.reshape(-1) * w).mean()
            if train:
                # /bsz so the accumulated gradient equals a mean-reduced batch
                (li / bsz).backward()
            total += float(li.detach())
            batch_logits.append(lg.detach().float().cpu())
            batch_target.append(tgt.detach().float().cpu())
        if train:
            optimizer.step()
        n += bsz
        preds.append(torch.cat(batch_logits, dim=0))
        gold.append(torch.cat(batch_target, dim=0))
    return total / max(n, 1), torch.cat(preds), torch.cat(gold)


def concordance_index(target, preds) -> float:
    """Harrell's C for regression: the share of target-distinct pairs ordered
    correctly, ties in the prediction counting half. 0.5 = chance ordering,
    1.0 = perfect; equals ROC-AUC on a binary target. NaN when every target is
    tied. O(n^2), fine for the benchmark's test splits."""
    t = np.asarray(target, dtype=np.float64).ravel()
    p = np.asarray(preds, dtype=np.float64).ravel()
    if t.size < 2:
        return float("nan")
    iu, ju = np.triu_indices(t.size, k=1)
    dt, dp = t[iu] - t[ju], p[iu] - p[ju]
    comparable = dt != 0
    if not comparable.any():
        return float("nan")
    dt, dp = dt[comparable], dp[comparable]
    concordant = float((np.sign(dt) == np.sign(dp)).sum())
    tied_pred = float((dp == 0).sum())
    return (concordant + 0.5 * tied_pred) / float(comparable.sum())


def regression_metrics(preds, gold) -> dict:
    """MAE (primary, as on the benchmark board) and C-index (secondary).

    MAE is lower-is-better, so any delta built from it must be oriented before
    it is shown next to an AUROC delta — see ``oriented_delta``.
    """
    p = preds.reshape(-1).float()
    g = gold.reshape(-1).float()
    return {
        "mae": float((p - g).abs().mean()),
        "cindex": float(concordance_index(g.numpy(), p.numpy())),
    }


def oriented_delta(metric_name: str, treatment: float, baseline: float) -> float:
    """Δ with the sign fixed so **positive always means improvement**.

    AUROC/C-index are higher-is-better; MAE is lower-is-better. Without this,
    an MAE column silently reverses the meaning of every sign in a table that
    also holds AUROC deltas.
    """
    if treatment is None or baseline is None:
        return None
    return (baseline - treatment) if metric_name == "MAE" else (treatment - baseline)


def primary_metric(preds, gold, task_type: str) -> float:
    """MAE for R, one-vs-rest macro AUROC for C, macro AUROC otherwise."""
    if preds is None or gold is None:
        return None
    if task_type == "R":
        return regression_metrics(preds, gold)["mae"]
    if task_type == "C":
        # sdx/brain.py softmaxes multiclass logits before scoring; do
        # the same, then score each class one-vs-rest and average.
        import numpy as _np
        probs = torch.softmax(preds.float(), dim=-1).numpy()
        g = gold.reshape(-1).long().numpy()
        onehot = _np.zeros_like(probs)
        onehot[_np.arange(len(g)), g] = 1.0
        return macro_auroc(torch.from_numpy(probs), torch.from_numpy(onehot))
    return macro_auroc(preds, gold)


def macro_auroc(preds: torch.Tensor, gold: torch.Tensor) -> float:
    """Macro AUROC over label columns, skipping single-class ones.

    Same construction the benchmark uses in ``brain.bootstrap_macro_auc_ci``:
    per-column ``roc_auc_score``, columns with one class dropped rather than
    scored as 0.5.
    """
    from sklearn.metrics import roc_auc_score

    p = preds.numpy().reshape(len(preds), -1)
    g = gold.numpy().reshape(len(gold), -1)
    scores = []
    for j in range(p.shape[1]):
        col = g[:, j]
        if len(np.unique(col)) < 2:
            continue
        try:
            scores.append(roc_auc_score(col, p[:, j]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")


def train_cell(probe, cfg: CellConfig, train_loader, val_loader, test_loader,
               loss_fn, eval_loss_fn=None, evaluate_test: bool = True) -> dict:
    """15 epochs, no early stopping, best-val checkpoint, one test eval.

    ``evaluate_test=False`` is for choosing a LoRA learning rate: compare
    runs on validation loss only, so the test set is still touched exactly
    once, by the run you report.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    probe.to(device)

    groups = build_param_groups(
        probe, lora_lr=cfg.lora_lr, head_lr=cfg.head_lr,
        head_weight_decay=cfg.head_l2, lora_weight_decay=cfg.lora_l2)
    if cfg.control:
        # Adapters stay injected (so the tail is bit-identical to frozen) but
        # receive no gradient and no optimizer group.
        probe.freeze_adapters()
        groups = [g for g in groups if g["name"] != "lora"]
    optimizer = torch.optim.AdamW(groups)
    sched = GroupAwareLinearSchedule(optimizer, epoch_count=cfg.epochs)

    eval_loss_fn = eval_loss_fn if eval_loss_fn is not None else loss_fn
    loss_fn = loss_fn.to(device)
    eval_loss_fn = eval_loss_fn.to(device)
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        lrs = sched.step(epoch)
        if cfg.control:
            lrs = {"lora": 0.0, **lrs}
        tr_loss, _, _ = run_epoch(probe, train_loader, loss_fn, device,
                                  optimizer, reg_bins=cfg.reg_bins,
                                  task_type=cfg.task_type)
        with torch.no_grad():
            va_loss, vp, vg = run_epoch(probe, val_loader, eval_loss_fn, device,
                                        task_type=cfg.task_type)
        va_auc = primary_metric(vp, vg, cfg.task_type)
        cfg.history.append({"epoch": epoch, "train_loss": tr_loss,
                            "val_loss": va_loss, "val_auroc": va_auc,
                            "lr_lora": lrs["lora"], "lr_head": lrs["head"]})
        print(f"[lora-train] ep {epoch:>2}/{cfg.epochs} "
              f"train {tr_loss:.4f}  val {va_loss:.4f}  val_auroc {va_auc:.4f}  "
              f"lr_lora {lrs['lora']:.2e} lr_head {lrs['head']:.2e}", flush=True)
        # Selection is on validation LOSS, matching the frozen board's
        # optim_metric (hpopt.yaml) — not on AUROC.
        if va_loss < best["val_loss"]:
            best = {"val_loss": va_loss, "epoch": epoch,
                    "state": {"adapter": adapter_state_dict(probe.tail),
                              "head": {k: v.detach().clone() for k, v
                                       in probe.classifier.state_dict().items()}}}

    if best["state"] is not None:
        load_adapter_state_dict(probe.tail, best["state"]["adapter"])
        probe.classifier.load_state_dict(best["state"]["head"])
    if evaluate_test:
        with torch.no_grad():
            te_loss, tp, tg = run_epoch(probe, test_loader, eval_loss_fn, device,
                                        task_type=cfg.task_type)
    else:
        te_loss, tp, tg = float("nan"), None, None
    result = {
        "task": cfg.task, "encoder": cfg.encoder,
        "lora_lr": cfg.lora_lr, "head_lr": cfg.head_lr,
        "selected_epoch": best["epoch"], "val_loss": best["val_loss"],
        "task_type": cfg.task_type,
        "test_loss": te_loss,
        # Key stays "test_auroc" so existing readers keep working; for R it
        # holds the C-index, which is the same statistic on one footing.
        "test_auroc": primary_metric(tp, tg, cfg.task_type) if tp is not None else None,
        "metric_name": {"R": "MAE", "C": "macroAUROC_ovr"}.get(
            cfg.task_type, "macroAUROC"),
        # Computed for every regression cell but deliberately not the headline.
        "test_cindex": (regression_metrics(tp, tg)["cindex"]
                        if tp is not None and cfg.task_type == "R" else None),
        "control": cfg.control,
        "n_lora_params": 0 if cfg.control else probe.n_trainable_lora,
        "n_head_params": sum(p.numel() for p in probe.head_parameters()),
        "epochs": cfg.epochs, "wall_s": round(time.time() - t0, 1),
        "history": cfg.history,
    }
    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        torch.save(best["state"], out / "best_adapter.pt")
        if tp is not None:
            np.save(out / "test_preds.npy", tp.numpy())
            np.save(out / "test_labels.npy", tg.numpy())
    tail_msg = (f"test {result['metric_name']} {result['test_auroc']:.4f}"
                if tp is not None
                else "test NOT evaluated (--no-test)")
    print(f"[lora-train] {cfg.task} x {cfg.encoder}: "
          f"selected epoch {best['epoch']}, {tail_msg}", flush=True)
    return result
