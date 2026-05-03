import copy
import json
from pathlib import Path

import numpy as np
import torch
import speechbrain as sb
from ray import tune
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, F1Score, AUROC, Accuracy
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef, R2Score


def delong_ci(y_true, y_score, alpha=0.05):
    """Compute DeLong confidence interval for AUC."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, (np.nan, np.nan)
    
    # Placement values
    v10 = np.mean(pos[:, None] > neg[None, :], axis=1) + \
          0.5 * np.mean(pos[:, None] == neg[None, :], axis=1)
    v01 = np.mean(pos[None, :] > neg[:, None], axis=0) + \
          0.5 * np.mean(pos[None, :] == neg[:, None], axis=0)
    
    auc = np.mean(v10)
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    se = np.sqrt(var)
    
    z = 1.96  # for 95% CI
    return auc, (max(0, auc - z * se), min(1, auc + z * se))


def bootstrap_macro_auc_ci(
    y_true,
    y_score,
    subject_ids=None,
    n_boot=1000,
    ci=95,
    seed=42,
):
    """Bootstrap CI for multilabel macro-AUC.

    If subject_ids is provided, resampling is done at the subject level
    (cluster bootstrap); otherwise it is done at the sample level.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    def macro_auc(yt, ys):
        aucs = []
        for k in range(yt.shape[1]):
            if len(np.unique(yt[:, k])) < 2:
                continue
            aucs.append(roc_auc_score(yt[:, k], ys[:, k]))
        if not aucs:
            return np.nan
        return np.mean(aucs)

    point_estimate = macro_auc(y_true, y_score)

    boot_scores = []
    if subject_ids is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            boot_scores.append(macro_auc(y_true[idx], y_score[idx]))
    else:
        subject_ids = np.asarray(subject_ids)
        subjects = np.unique(subject_ids)
        subj_to_idx = {s: np.where(subject_ids == s)[0] for s in subjects}
        for _ in range(n_boot):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            idx = np.concatenate([subj_to_idx[s] for s in sampled])
            boot_scores.append(macro_auc(y_true[idx], y_score[idx]))

    boot_scores = np.asarray(boot_scores)
    boot_scores = boot_scores[~np.isnan(boot_scores)]

    if boot_scores.size == 0:
        return point_estimate, (np.nan, np.nan)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_scores, alpha)
    upper = np.percentile(boot_scores, 100 - alpha)
    return point_estimate, (lower, upper)


def bootstrap_mae_ci(y_true, y_pred, n_resamples=1000, confidence_level=0.95):
    """Compute bootstrap confidence interval for MAE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    def mae_statistic(indices):
        return np.mean(np.abs(y_true[indices] - y_pred[indices]))
    
    rng = np.random.default_rng(42)
    indices = np.arange(len(y_true))
    result = bootstrap(
        (indices,),
        statistic=mae_statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method='percentile',
        random_state=rng
    )
    mae = np.mean(np.abs(y_true - y_pred))
    return mae, (result.confidence_interval.low, result.confidence_interval.high)


def unwrap_ddp(module):
    """Unwrap a module from DistributedDataParallel if needed."""
    return getattr(module, "module", module)


def _load_train_labels(manifest_path, fold_idx=None):
    """Read training labels from the manifest's canonical ``label`` field.

    For flat dict-of-samples manifests, ``fold_idx`` is ignored. For CV
    list-of-folds manifests, ``fold_idx`` selects one fold's training set
    (each entry is already the train partition for that fold).
    """
    with open(manifest_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if fold_idx is None:
            raise ValueError(
                f"manifest {manifest_path} is a list of folds; pass fold_idx"
            )
        fold = data[fold_idx]
    else:
        fold = data
    return [sample["label"] for sample in fold.values()]


def compute_auto_class_weights(task_type, labels, num_labels, cutoffs=None):
    """Compute class weights from raw training labels.

    Returns a ``(kind, payload)`` pair, or ``None`` if it can't be computed.
    For B/L kind is ``"pos_weight"``; for C ``"weight"``; for R
    ``"regression_bins"`` with payload ``(edges_tensor, bin_weights_tensor)``.
    For R, ``cutoffs`` (e.g. clinical bin edges) takes priority. If absent, we
    use unique values when there are few (<=10), else 10 quantile bins.
    """
    if task_type == "R":
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if cutoffs is not None and len(cutoffs) > 0:
            edges = np.asarray(cutoffs, dtype=np.float64)
        else:
            unique = np.unique(y)
            if unique.size <= 1:
                return None
            if unique.size <= 10:
                edges = (unique[:-1] + unique[1:]) / 2.0
            else:
                # 10 quantile bins -> 9 internal edges (deduped for ties).
                edges = np.unique(np.quantile(y, np.linspace(0, 1, 11)[1:-1]))
                if edges.size == 0:
                    return None
        bins = np.digitize(y, bins=edges)
        n_bins = len(edges) + 1
        counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            bin_weight = np.where(counts > 0, 1.0 / counts, 0.0)
        # Normalize so the average per-sample weight is 1 (loss scale stable).
        avg = (counts * bin_weight).sum() / counts.sum()
        if avg > 0:
            bin_weight = bin_weight / avg
        return "regression_bins", (
            torch.tensor(edges, dtype=torch.float32),
            torch.tensor(bin_weight, dtype=torch.float32),
        )

    if task_type == "B":
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        pos = float((y > 0.5).sum())
        neg = float((y <= 0.5).sum())
        if pos == 0 or neg == 0:
            return None
        return "pos_weight", torch.tensor([neg / pos], dtype=torch.float32)

    if task_type == "C":
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        n = y.size
        counts = np.bincount(y, minlength=num_labels).astype(np.float64)
        # Inverse-frequency weighting: N / (K * count_k); zero-count classes
        # get weight 0 to avoid divide-by-zero (they contribute no loss anyway).
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(counts > 0, n / (num_labels * counts), 0.0)
        return "weight", torch.tensor(w, dtype=torch.float32)

    if task_type == "L":
        y = np.asarray(labels, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != num_labels:
            return None
        n = y.shape[0]
        pos = y.sum(axis=0)
        neg = n - pos
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(pos > 0, neg / pos, 1.0)
        return "pos_weight", torch.tensor(w, dtype=torch.float32)

    return None


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, fold_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        # Tune session detection moved from a constructor flag to a runtime
        # check (tune.is_session_enabled()) in on_stage_end so a single
        # brain class works for both ad-hoc training and Ray Tune trials.
        # For CV manifests (train.json is a list of fold dicts), callers that
        # know which fold this brain handles should pass fold_idx so auto class
        # weights are computed from that fold's train partition only.
        self.fold_idx = fold_idx
        
        # Storage for CI calculation during test
        self._test_preds = []
        self._test_labels = []
        self._test_pids = []

        # Re-register `counter` via add_recoverable (singular) so it lands in
        # optional_recoverables too. SB's YAML constructor uses add_recoverables
        # (plural) which only updates self.recoverables, causing KeyError in
        # _call_load_hooks when a paramfile is missing on resume.
        if "counter" in self.checkpointer.recoverables:
            self.checkpointer.add_recoverable(
                "counter",
                self.checkpointer.recoverables["counter"],
                optional_load=True,
            )
        self.checkpointer.recover_if_possible()

        # Determine task type: B (binary), C (multiclass), R (regression), L (multilabel)
        task_type = getattr(self.hparams, "task_type", None)
        num_classes = self.hparams.num_labels

        if task_type is None:
            # Backward compat: infer from num_labels
            task_type = "B" if num_classes == 1 else "C"

        self.task_type = task_type

        if task_type == "R":
            metrics = {
                "MAE": MeanAbsoluteError(),
                "MSE": MeanSquaredError(),
                "PearsonR": PearsonCorrCoef(),
                "R2": R2Score(),
            }
        else:
            task_cfg = {
                "L": {"task": "multilabel", "num_labels": num_classes, "average": "macro"},
                "C": {"task": "multiclass", "num_classes": num_classes, "average": "macro"},
                "B": {"task": "binary"},
            }
            cfg = task_cfg[task_type]
            metrics = {
                "F1": F1Score(**cfg),
                "precision": Precision(**cfg),
                "recall": Recall(**cfg),
                "accuracy": Accuracy(**{k: v for k, v in cfg.items() if k != "average"}),
                "AUROC": AUROC(**cfg),
            }
        self.error_metrics = MetricCollection(metrics).to(self.device)

        self.model = unwrap_ddp(self.modules.model)
        self.hparams.loss = self.hparams.loss.to(self.device)

        if getattr(self.hparams, "auto_class_weights", False):
            # Flat manifest: apply now. CV manifest: apply only if the caller
            # passed fold_idx (e.g. trainPerFoldCV); otherwise the CV subclasses
            # in brains.py call _apply_auto_class_weights themselves after
            # setting brain_id.
            manifest_path = getattr(self.hparams, "train_annotation", None)
            if manifest_path and Path(manifest_path).exists():
                with open(manifest_path, "r") as f:
                    is_cv = isinstance(json.load(f), list)
                if not is_cv:
                    self._apply_auto_class_weights()
                elif self.fold_idx is not None:
                    self._apply_auto_class_weights(fold_idx=self.fold_idx)

    def _apply_auto_class_weights(self, fold_idx=None):
        """Compute class weights from the train manifest and set them on the loss.

        Overrides any pos_weight/weight already set in the task YAML. For R
        tasks we bin the labels (clinical cutoffs from ``data_params`` if
        provided, else unique values or quantiles), compute per-bin weights,
        and switch the loss to ``reduction='none'`` so ``compute_objectives``
        can apply per-sample weights at training time.

        For CV manifests, pass ``fold_idx`` to compute weights from that fold's
        train partition only.
        """
        manifest_path = getattr(self.hparams, "train_annotation", None)
        if manifest_path is None or not Path(manifest_path).exists():
            print(f"[auto_class_weights] train manifest missing at {manifest_path}; skipping")
            return

        labels = _load_train_labels(manifest_path, fold_idx=fold_idx)

        # Optional clinical bin edges live in data_params (per-task yaml).
        cutoffs = None
        data_params = getattr(self.hparams, "data_params", None)
        if isinstance(data_params, dict):
            cutoffs = data_params.get("clinical_cutoffs")

        result = compute_auto_class_weights(
            task_type=self.task_type,
            labels=labels,
            num_labels=self.hparams.num_labels,
            cutoffs=cutoffs,
        )
        if result is None:
            print(f"[auto_class_weights] could not compute weights for task_type={self.task_type}; skipping")
            return

        # Snapshot an unweighted loss for val/test so the loss reported there
        # reflects raw generalization error (not training-set bin frequencies).
        # Copy first, then strip any weight attrs in case the YAML had set them.
        unweighted = copy.deepcopy(self.hparams.loss)
        for attr in ("pos_weight", "weight"):
            if hasattr(unweighted, attr) and getattr(unweighted, attr) is not None:
                setattr(unweighted, attr, None)
        unweighted.reduction = "mean"
        self._unweighted_loss = unweighted.to(self.device)

        kind, payload = result
        fold_tag = "" if fold_idx is None else f" fold={fold_idx}"
        if kind == "regression_bins":
            edges, bin_weights = payload
            self._reg_bin_edges = edges.to(self.device)
            self._reg_bin_weights = bin_weights.to(self.device)
            # Switch loss to elementwise so we can weight per-sample. The
            # reduce-by-mean happens in compute_objectives.
            self.hparams.loss.reduction = "none"
            print(
                f"[auto_class_weights]{fold_tag} task_type=R "
                f"edges={edges.tolist()} bin_weights={bin_weights.tolist()}"
            )
        else:
            tensor = payload.to(self.device)
            setattr(self.hparams.loss, kind, tensor)
            print(
                f"[auto_class_weights]{fold_tag} task_type={self.task_type} "
                f"{kind}={tensor.detach().cpu().tolist()}"
            )


    def compute_forward(self, batch, stage):
        """Read pre-cached embeddings, then run the probe.

        Cache-only path: ``ahb`` decouples encoder warming (``ahb warm``)
        from training, so the trainer never invokes the encoder. The
        legacy ``cache_encoder=False`` branch (audio → encoder → probe in
        one process) was deleted; the trainer process must already be
        looking at a warm cache.
        """
        batch = batch.to(self.device)

        if self.model.encoder.output_hidden_states:
            num_layers = self.hparams.num_layers
            emb_vars = ["emb_{}".format(i) for i in range(num_layers)]
            wavs = tuple(getattr(batch, var).data.to(self.device) for var in emb_vars)
            lens = getattr(batch, emb_vars[0]).lengths.to(self.device)
        else:
            wavs = getattr(batch, "emb_0").data.to(self.device)
            lens = getattr(batch, "emb_0").lengths.to(self.device)

        predictions = self.model.probe(wavs, lens)
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`. Usually [B, 1] for binary
            classification.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Dynamically retrieve the label using the 'label_key' from hparams
        label_key = getattr(self.hparams, "label_key", "label_encoded")
        lab = getattr(batch, label_key)
        # SpeechBrain wraps variable-length labels (e.g. multi-label lists)
        # in PaddedData; unwrap to the underlying tensor before moving device.
        if hasattr(lab, "data"):
            lab = lab.data
        lab = lab.to(predictions.device)

        if self.task_type == "C":
            lab = lab.long()
        else:
            lab = lab.float()
            if lab.dim() == 1 and self.task_type in ("R", "B"):
                lab = lab.unsqueeze(1)

        # Class weighting is a training-time gradient-shaping trick: val/test
        # should report the unweighted loss so early stopping / HP selection
        # isn't biased by training-set class frequencies. The unweighted loss
        # is snapshotted in _apply_auto_class_weights when auto weighting is on.
        if stage != sb.Stage.TRAIN and hasattr(self, "_unweighted_loss"):
            loss = self._unweighted_loss(predictions, lab)
        else:
            loss = self.hparams.loss(predictions, lab)
            # For R with auto class weights, the train loss is reduction='none';
            # apply per-sample bin weights, then reduce.
            if self.task_type == "R" and hasattr(self, "_reg_bin_weights"):
                lab_flat = lab.squeeze(-1) if lab.dim() > 1 else lab
                sample_bins = torch.bucketize(lab_flat, self._reg_bin_edges)
                sample_w = self._reg_bin_weights[sample_bins]
                loss = (loss.squeeze(-1) * sample_w).mean()

        if self.task_type == "R":
            self.error_metrics.update(predictions.squeeze(-1), lab.squeeze(-1))
        elif self.task_type == "L":
            self.error_metrics.update(predictions, lab.int())
        else:
            self.error_metrics.update(predictions, lab)

        # Store predictions and labels for CI calculation during test
        if stage == sb.Stage.TEST:
            with torch.no_grad():
                if self.task_type == "R":
                    self._test_preds.append(predictions.squeeze(-1).cpu())
                    self._test_labels.append(lab.squeeze(-1).cpu())
                elif self.task_type == "B":
                    # For binary, store sigmoid probabilities
                    self._test_preds.append(torch.sigmoid(predictions).squeeze(-1).cpu())
                    self._test_labels.append(lab.squeeze(-1).cpu())
                elif self.task_type == "C":
                    # For multiclass, store softmax probabilities
                    self._test_preds.append(torch.softmax(predictions, dim=-1).cpu())
                    self._test_labels.append(lab.cpu())
                elif self.task_type == "L":
                    # For multilabel, store sigmoid probabilities
                    self._test_preds.append(torch.sigmoid(predictions).cpu())
                    self._test_labels.append(lab.cpu())
                    # Track participant ids for subject-level cluster bootstrap.
                    self._test_pids.extend(list(batch.pid))

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        eval_stats = self.error_metrics.compute()
        self.error_metrics.reset()

        eval_stats["loss"] = stage_loss

        if stage == sb.Stage.TRAIN:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats=eval_stats,
            )

            # Replace only the previous last_train ckpt; leave val-best ckpts
            # alone so they survive across epochs for test-time loading.
            self.checkpointer.delete_checkpoints(
                num_to_keep=0,
                ckpt_predicate=lambda c: c.path.name == 'CKPT+last_train',
            )
            self.checkpointer.save_checkpoint(name='last_train')


        # At the end of validation...
        if stage == sb.Stage.VALID:
            optim_metric = getattr(self.hparams, "optim_metric", "loss")
            optim_mode = getattr(self.hparams, "optim_mode", "min")

            max_keys, min_keys = [], []
            if optim_mode == "max":
                max_keys.append(optim_metric)
            else:
                min_keys.append(optim_metric)


            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr
            )

            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                valid_stats=eval_stats,
            )

            # SB's Checkpoint set difference uses dict-eq on meta; NaN tensors
            # (e.g. PearsonR on low-variance regression) make the same ckpt
            # loaded twice compare unequal, causing double-delete.
            ckpt_meta = {}
            for k, v in eval_stats.items():
                if hasattr(v, "item") and getattr(v, "numel", lambda: 1)() == 1:
                    v = v.item()
                if isinstance(v, float) and v != v:
                    continue
                ckpt_meta[k] = v
            # Keep best-by-loss val ckpt for test loading; protect last_train
            # (which has no metric meta and would otherwise be pruned by the
            # default keep_recent=True path).
            self.checkpointer.save_and_keep_only(
                meta=ckpt_meta,
                max_keys=max_keys,
                min_keys=min_keys,
                keep_recent=False,
                ckpt_predicate=lambda c: c.path.name != 'CKPT+last_train',
            )

            # Inside a Ray Tune trial: report metrics so the searcher /
            # stopper can act. Outside (ad-hoc training, final-eval pass):
            # update the EpochCounterWithStopper for the early-stop hook.
            # ``ray.train._internal.session.get_session()`` returns the active
            # session inside a trial and ``None`` outside, with no logging.
            from ray.train._internal.session import get_session
            if get_session() is not None:
                eval_stats = detensor_dict(eval_stats)
                tune.report(eval_stats)
            else:
                epoch_counter = self.hparams.epoch_counter
                epoch_counter.update_metric(eval_stats[optim_metric])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=eval_stats,
            )
            self.test_stats = detensor_dict(eval_stats)
            
            # Compute confidence intervals
            if self._test_preds and self._test_labels:
                all_preds = torch.cat(self._test_preds, dim=0).numpy()
                all_labels = torch.cat(self._test_labels, dim=0).numpy()
                
                if self.task_type == "R":
                    # Bootstrap CI for MAE
                    mae, (mae_lo, mae_hi) = bootstrap_mae_ci(all_labels, all_preds)
                    self.test_stats["MAE_CI_low"] = mae_lo
                    self.test_stats["MAE_CI_high"] = mae_hi
                elif self.task_type == "B":
                    # DeLong CI for binary AUROC
                    auc_delong, (auc_lo, auc_hi) = delong_ci(all_labels, all_preds)
                    auc_torchmetrics = self.test_stats.get("AUROC", None)
                    # Verify DeLong AUC matches torchmetrics AUROC
                    if auc_torchmetrics is not None:
                        diff = abs(auc_delong - auc_torchmetrics)
                        print(f"[AUC CHECK] torchmetrics={auc_torchmetrics:.6f}, DeLong={auc_delong:.6f}, diff={diff:.2e}")
                        if diff > 1e-4:
                            print(f"[WARNING] AUC mismatch > 1e-4!")
                    self.test_stats["AUROC_CI_low"] = auc_lo
                    self.test_stats["AUROC_CI_high"] = auc_hi
                elif self.task_type == "C":
                    # For multiclass, compute one-vs-rest AUC CI for each class
                    # and report macro-average CI
                    n_classes = all_preds.shape[1]
                    auc_los, auc_his, auc_vals = [], [], []
                    for c in range(n_classes):
                        y_true_c = (all_labels == c).astype(int)
                        y_score_c = all_preds[:, c]
                        auc_c, (lo, hi) = delong_ci(y_true_c, y_score_c)
                        if not np.isnan(lo):
                            auc_los.append(lo)
                            auc_his.append(hi)
                            auc_vals.append(auc_c)
                    if auc_los:
                        auc_delong_macro = np.mean(auc_vals)
                        auc_torchmetrics = self.test_stats.get("AUROC", None)
                        # Note: torchmetrics uses weighted avg, DeLong uses macro avg
                        if auc_torchmetrics is not None:
                            diff = abs(auc_delong_macro - auc_torchmetrics)
                            print(f"[AUC CHECK] torchmetrics(weighted)={auc_torchmetrics:.6f}, DeLong(macro)={auc_delong_macro:.6f}, diff={diff:.2e}")
                            print(f"  (Note: weighted vs macro avg may differ)")
                        self.test_stats["AUROC_CI_low"] = np.mean(auc_los)
                        self.test_stats["AUROC_CI_high"] = np.mean(auc_his)
                elif self.task_type == "L":
                    # Multilabel: bootstrap CI for macro-AUC. Use a cluster
                    # (subject-level) bootstrap when the test set has multiple
                    # samples per subject; otherwise plain sample bootstrap.
                    subject_ids = np.asarray(self._test_pids) if self._test_pids else None
                    if subject_ids is not None and len(np.unique(subject_ids)) == len(subject_ids):
                        # Every sample is its own subject -> sample bootstrap.
                        subject_ids = None
                    auc_macro, (auc_lo, auc_hi) = bootstrap_macro_auc_ci(
                        y_true=all_labels.astype(int),
                        y_score=all_preds,
                        subject_ids=subject_ids,
                    )
                    auc_torchmetrics = self.test_stats.get("AUROC", None)
                    if auc_torchmetrics is not None and not np.isnan(auc_macro):
                        diff = abs(auc_macro - auc_torchmetrics)
                        print(f"[AUC CHECK] torchmetrics(macro)={auc_torchmetrics:.6f}, sklearn(macro)={auc_macro:.6f}, diff={diff:.2e}")
                    self.test_stats["AUROC_CI_low"] = auc_lo
                    self.test_stats["AUROC_CI_high"] = auc_hi

                # Clear storage
                self._test_preds = []
                self._test_labels = []
                self._test_pids = []

def detensor_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        v = v.item() if isinstance(v, torch.Tensor) else v
        new_d[k] = v
    return new_d