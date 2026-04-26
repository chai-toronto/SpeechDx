from pathlib import Path

import numpy as np
import torch
import speechbrain as sb
from ray import tune
from scipy.stats import bootstrap
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


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, ray_optim=False, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        # Gates `tune.report(...)` vs. stopper-counter update in on_stage_end.
        # Other ray-tune-specific config (plain EpochCounter instead of the
        # stopper variant, trial-specific save_folder) is now baked into the
        # forked main.yaml by training.config_fork.
        self.ray_optim = ray_optim
        
        # Storage for CI calculation during test
        self._test_preds = []
        self._test_labels = []

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
                "C": {"task": "multiclass", "num_classes": num_classes, "average": "weighted"},
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


    def compute_forward(self, batch, stage):
        """Runs all the computation that transforms the input into the
        output probabilities over the N classes.

        """
        batch = batch.to(self.device)

        cache_encoder = getattr(self.hparams, "cache_encoder", False)
        if cache_encoder:
            if self.model.encoder.output_hidden_states:
                num_layers = self.hparams.num_layers
                emb_vars = ["emb_{}".format(i) for i in range(num_layers)]
                wavs = tuple(getattr(batch, var).data.to(self.device) for var in emb_vars)
                lens = getattr(batch, emb_vars[0]).lengths.to(self.device)
            else:
                wavs = getattr(batch, "emb_0").data.to(self.device)
                lens = getattr(batch, "emb_0").lengths.to(self.device)
        else:
            wavs, lens = batch.signal
            # Forward pass through the model
            wavs = self.model.encoder(wavs, lens)

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

        loss = self.hparams.loss(predictions, lab)

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

            self.checkpointer.delete_checkpoints(num_to_keep=0)
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
            self.checkpointer.save_and_keep_only(meta=ckpt_meta, max_keys=max_keys, min_keys=min_keys)

            if self.ray_optim:
                eval_stats = detensor_dict(eval_stats)
                tune.report(eval_stats)
            else:
                # For Early Stopping
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
                
                # Clear storage
                self._test_preds = []
                self._test_labels = []

def detensor_dict(d: dict):
    new_d = {}
    for k, v in d.items():
        v = v.item() if isinstance(v, torch.Tensor) else v
        new_d[k] = v
    return new_d