from pathlib import Path

import torch
import torch.nn.functional as F
import speechbrain as sb
from torchmetrics import MetricCollection, Metric, MeanMetric
from torchmetrics.classification import Precision, Recall, F1Score, AUROC, Accuracy


def unwrap_ddp(module):
    """Unwrap a module from DistributedDataParallel if needed."""
    return getattr(module, "module", module)

LB_LOSS_WEIGHT = 0.2

class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        self.checkpointer.recover_if_possible()
        self.stat = {}

        num_classes = self.hparams.num_labels
        if num_classes == 1:
            num_classes = None
            average = None
            ext = ''
            task = 'binary'
            self.binary = True
        else:
            ext = '_weighted'
            average = 'weighted'
            task = 'multiclass'
            self.binary = False

        self.error_metrics = MetricCollection({
            f"F1{ext}": F1Score(task=task, num_classes=num_classes, average=average),
            f"precision{ext}": Precision(task=task, num_classes=num_classes, average=average),
            f"recall{ext}": Recall(task=task, num_classes=num_classes, average=average),
            f"accuracy{ext}": Accuracy(task=task, num_classes=num_classes, average=average),
            f"AUROC{ext}": AUROC(task=task, num_classes=num_classes, average=average),
        }).to(self.device)

        self.chunk_metrics = ChunkMetric().to(self.device)

        self.model = unwrap_ddp(self.modules.model)

        if self.hparams.enable_chunking:
            self.model.init_chunker(
                chunk_encoder=self.hparams.chunk_encoder,
                threshold=self.hparams.threshold,
                chunk_at=self.hparams.chunk_at,
                aggregate=self.hparams.aggregate,
            )
            self.model.chunker.to(self.device)
        self.min_chunk_size = self.hparams.min_chunk_size  # promoted to brain attribute
        self.hparams.loss = self.hparams.loss.to(self.device)

        print(self.model)

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
        lab = lab.to(predictions)
        lab = lab.float() if self.binary else lab.long()

        if self.hparams.num_labels == 1 and lab.dim() == 1:
            lab = lab.unsqueeze(1)

        loss = self.hparams.loss(predictions, lab)

        self.error_metrics.update(predictions, lab)

        last_hidden_states = getattr(self.model, "last_hidden_states", None)

        if last_hidden_states is not None:
            threshold = self.hparams.threshold
            min_chunk_size = self.min_chunk_size

            nonboundary_mask, boundary_prob, reduction, scores = last_hidden_states

            chunk_loss = self.get_load_balancing_loss(boundary_prob, nonboundary_mask, N=min_chunk_size) * LB_LOSS_WEIGHT
            print(chunk_loss)
            print(reduction)

            # boundary_prob = boundary_prob[:, :, 1].squeeze(-1)  # (B, T)
            #
            # chunk_losses = self.boundary_regularizers(
            #     boundary_prob,
            #     (boundary_prob > threshold).float(),
            #     length_target=min_chunk_size
            # )
            #
            # chunk_loss = (0.01 * chunk_losses["loss_coverage"]
            #               + 0.01 * chunk_losses["loss_entropy"]
            #               + 0.5 * chunk_losses["loss_smooth"])

            loss = loss + chunk_loss

            if stage == sb.Stage.TEST:
                ids = batch.id
                for i in range(len(ids)):
                    self.stat[ids[i]] = {
                        "boundary_prob": boundary_prob[i].cpu(),
                        # "scores": scores[i].cpu()
                    }

            self.chunk_metrics.update(chunk_loss, reduction)

        return loss

    def get_load_balancing_loss(self, boundary_prob, nonboundary_mask, N: float = 5.0) -> torch.Tensor:
        """
        Compute load balancing loss from last forward pass.

        Encourages the model to create meaningful chunks (not too many, not too few).
        From H-Net hnet/utils/train.py lines 13-40.

        Args:
            N: Target downsampling factor (higher = more compression)

        Returns:
            loss: scalar tensor
        """
        assert N > 1, "N must be greater than 1 for load balancing loss"

        # Extract probability of boundary class
        tokenized_prob = boundary_prob[..., 1]  # (B, L)

        # Empirical vs predicted boundary ratios
        true_ratio = (~nonboundary_mask).float().mean()  # Actual fraction of boundaries
        avg_prob = tokenized_prob.float().mean()  # Predicted fraction

        # Load balancing loss
        # Penalizes mismatch between predicted and actual boundary frequency
        lb_loss = (
            (1 - true_ratio) * (1 - avg_prob) +
            (true_ratio) * (avg_prob) * (N - 1)
        ) * N / (N - 1)

        return lb_loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        chunk_stats = self.chunk_metrics.compute()
        self.chunk_metrics.reset()

        eval_stats = self.error_metrics.compute()
        self.error_metrics.reset()

        eval_stats["loss"] = stage_loss
        eval_stats = eval_stats | chunk_stats

        eval_stats["clf_loss"] = stage_loss - chunk_stats["chunk_loss"]

        if stage == sb.Stage.TRAIN:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats=eval_stats,
            )

            self.checkpointer.delete_checkpoints(num_to_keep=0)
            self.checkpointer.save_checkpoint(name='last_train')


        # At the end of validation...
        if stage == sb.Stage.VALID:
            optim_metric = getattr(self.hparams, "optim_metric", "F1")
            optim_mode = getattr(self.hparams, "optim_mode", "max")
            max_keys, min_keys = [], []
            if optim_mode == "max":
                max_keys.append(optim_metric)
            else:
                min_keys.append(optim_metric)

            # For Early Stopping
            epoch_counter = self.hparams.epoch_counter
            epoch_counter.update_metric(eval_stats[optim_metric])

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr
            )

            # Sync scheduled chunk size to brain attribute
            if hasattr(epoch_counter, 'min_chunk_size'):
                self.min_chunk_size = epoch_counter.min_chunk_size

            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                valid_stats=eval_stats,
            )

            # Save the current checkpoint and delete previous checkpoints, based on F1
            self.checkpointer.save_and_keep_only(meta=eval_stats, max_keys=max_keys, min_keys=min_keys)

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=eval_stats,
            )
            torch.save(self.stat, Path(self.hparams.output_folder) / "test_diagnostics.pt")


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



class ChunkMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("chunk_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("reduction", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, chunk_loss, reduction):
        self.chunk_loss += chunk_loss.detach()
        self.reduction += reduction
        self.count += 1

    def compute(self):
        return {
            "chunk_loss": self.chunk_loss / self.count if self.count > 0 else torch.tensor(0.0),
            "reduction": self.reduction / self.count if self.count > 0 else torch.tensor(0.0)
        }

