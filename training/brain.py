from pathlib import Path

import torch
import torch.nn.functional as F
import speechbrain as sb
from training.metric import Record, roc_auc_score_rev, accuracy


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        self.checkpointer.recover_if_possible()

        self.stat = {}

    def compute_forward(self, batch, stage):
        """Runs all the computation that transforms the input into the
        output probabilities over the N classes.

        """
        # For debugging
        # label_key = getattr(self.hparams, "label_key", "label_encoded")
        # lab = getattr(batch, label_key)
        # if lab.sum() > 0:
        #     breakpoint()
        # else:
        #     return lab.unsqueeze(-1).float(), None, None
        batch = batch.to(self.device)

        cache_encoder = getattr(self.hparams, "cache_encoder", False)
        if cache_encoder:
            if self.modules.model.encoder.output_hidden_states:
                num_layers = self.hparams.num_layers
                emb_vars = ["emb_{}".format(i) for i in range(num_layers)]
                wavs = tuple(getattr(batch, var).data for var in emb_vars)
                lens = getattr(batch, emb_vars[0]).lengths
            else:
                wavs = getattr(batch, "emb_0").data
                lens = getattr(batch, "emb_0").lengths
            wavs = wavs.to(self.device)
            lens = lens.to(self.device)
        else:
            wavs, lens = batch.signal
            # Forward pass through the model
            wavs = self.modules.model.encoder(wavs, lens)
        T = wavs.size(1)
        abs_lens = (lens * T).long()
        predictions = self.modules.model.probe(wavs, lens)
        return *predictions, abs_lens

    def augment_input(self, wavs, stage):
        """Applies data augmentation based on hparams (if available)."""
        wavs, lens = wavs
        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger batch.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        return wavs, lens

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
        predictions, boundary_mask, boundary_prob, scores, abs_lens = predictions

        # Dynamically retrieve the label using the 'label_key' from hparams
        label_key = getattr(self.hparams, "label_key", "label_encoded")
        lab = getattr(batch, label_key)
        lab = lab.to(self.device) # [B]
        if lab.dim() == 1:
            lab = lab.unsqueeze(-1)  # [B, 1]
        lab = lab.to(predictions)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            lab = torch.cat([lab, lab], dim=0)

        # Compute the cost function: BCE is assumed for binary classification
        # but pos_weight is used for imbalance handling.
        weight = torch.tensor([getattr(self.hparams, "positive_class_weight", 1.0)]).to(self.device)
        loss = sb.nnet.losses.bce_loss(predictions, lab, pos_weight=weight)

        # Add load balancing loss if specified
        if boundary_mask is not None or boundary_prob is not None:
            # lb_loss = self.get_load_balancing_loss(boundary_prob, boundary_mask, N=8)
            # print("Load balancing loss: {:.4f}".format(lb_loss.item()))
            # loss = loss + lb_loss * 0.5

            boundary_prob = boundary_prob[:, :, 1].squeeze(-1)  # (B, T)
            chunk_losses = self.boundary_regularizers(
                boundary_prob,
                (boundary_prob > 0.1).float(),
                length_target=4.0  # Target ~20 frames per chunk
            )

            loss = (loss + 0.01 * chunk_losses["loss_coverage"] +
                    0.01 * chunk_losses["loss_entropy"] +
                    0.5 * chunk_losses["loss_smooth"])

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            batch.id, predictions, lab, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, lab)
            self.record.add(predictions, lab)

        if stage == sb.Stage.TEST:
            ids = batch.id
            for i in range(len(ids)):
                self.stat[ids[i]] = {
                    "duration": abs_lens[i].item(),
                    "boundary_prob": boundary_prob[i].cpu(),
                    "scores": scores[i].cpu()
                }
        return loss

    def get_load_balancing_loss(self, boundary_prob, boundary_mask, N: float = 5.0) -> torch.Tensor:
        """
        Compute load balancing loss from last forward pass.

        Encourages the model to create meaningful chunks (not too many, not too few).
        From H-Net hnet/utils/train.py lines 13-40.

        Args:
            N: Target downsampling factor (higher = more compression)

        Returns:
            loss: scalar tensor
        """
        # Extract probability of boundary class
        tokenized_prob = boundary_prob[..., 1]  # (B, L)

        # Empirical vs predicted boundary ratios
        true_ratio = boundary_mask.float().mean()  # Actual fraction of boundaries
        avg_prob = tokenized_prob.float().mean()  # Predicted fraction

        # Load balancing loss
        # Penalizes mismatch between predicted and actual boundary frequency
        lb_loss = (
            (1 - true_ratio) * (1 - avg_prob) +
            (true_ratio) * (avg_prob) * (N - 1)
        ) * N / (N - 1)

        return lb_loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.bce_loss
        )

        if stage != sb.Stage.TRAIN:
            self.record = Record()
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.checkpointer.delete_checkpoints(num_to_keep=0)
            self.checkpointer.save_checkpoint(name='last_train')
            return

        stats = self.calc_epoch_metrics(stage_loss)

        optim_metric = getattr(self.hparams, "optim_metric", "F1")
        optim_mode = getattr(self.hparams, "optim_mode", "max")
        max_keys, min_keys = [], []
        if optim_mode == "max":
            max_keys.append(optim_metric)
        else:
            min_keys.append(optim_metric)

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr
            )

            # For Early Stopping
            epoch_counter = self.hparams.epoch_counter
            epoch_counter.update_metric(stats[optim_metric])

            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.avg_train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints, based on F1
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=max_keys, min_keys=min_keys)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            torch.save(self.stat, Path(self.hparams.output_folder) / "test_diagnostics.pt")

    def calc_epoch_metrics(self, stage_loss):
        """ Call this after the epoch only"""
        # Summarize the statistics from the stage for record-keeping.
        metrics = self.error_metrics.summarize()
        preds, tgts = self.record.get_all()
        # This will dictate presentation order in logs
        stats = {
            "roc": roc_auc_score_rev(preds, tgts),
            "F1": metrics["F-score"],
            "loss": stage_loss,
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "sens": metrics['TP'] / (metrics['TP'] + metrics['FN']),
            "accuracy": accuracy(preds, tgts),

        }
        self.record.clear()
        return stats

    def boundary_regularizers(self,
                              start_prob: torch.Tensor,
                              start_hard: torch.Tensor,
                              length_target: float | None = 20.0,
                              eps: float = 1e-9):
        # Compute three auxiliary losses for stability: entropy, coverage, smoothness.
        B, T = start_prob.shape
        p = start_prob[:, 1:].clamp(eps, 1 - eps)  # avoid log(0); ignore first frame

        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))  # [B,T]
        # Binary entropy per frame; we maximize entropy - equivalent to minimizing -entropy
        loss_entropy = -ent.mean() if T >= 2 else torch.tensor(0.0, device=start_prob.device)
        num_starts = start_hard.sum(dim=1)  # [B] number of segments per sequence

        if length_target is not None and length_target > 0:
            seg_target = T / length_target  # desired number of segments based on target length
            loss_coverage = ((num_starts - seg_target) ** 2).mean() / (
                        seg_target + 1e-6)  # MSE between actual and target number of segments, normalized
        else:
            loss_coverage = torch.tensor(0.0, device=start_prob.device)

        # encourage smoothness over time and discourage rapid fluctuations
        smooth = F.mse_loss(p[:, 1:], p[:, :-1]) if T >= 3 else torch.tensor(0.0, device=start_prob.device)

        return {
            "loss_entropy": loss_entropy,
            "loss_coverage": loss_coverage,
            "loss_smooth": smooth,
        }



