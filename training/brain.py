from pathlib import Path

import torch
import speechbrain as sb

from training.metric import Record, roc_auc_score_rev, accuracy


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = None

    def compute_forward(self, batch, stage):
        """Runs all the computation that transforms the input into the
        output probabilities over the N classes.

        """
        batch = batch.to(self.device)
        wavs, lens = batch.signal

        # Exclusive for [B, T]  inputs
        ids = batch.id
        cache_encoder = getattr(self.hparams, "cache_encoder", False)
        if cache_encoder: # TODO: vectorize this
            cache_dir = Path(getattr(self.hparams, "cache_dir")).resolve()
            cache_done = (cache_dir / "cache_done.txt").exists()
            if not cache_done:
                encoded = self.modules.model.encoder(wavs, lens) # (B, T, D)
                t_dim = -2
                abs_lengths = (lens * encoded.size(t_dim)).long()
                # Save encoded features to disk
                if self.cache is None: # i.e. we started caching
                    self.cache = {}

                encoded = encoded.sum(dim=t_dim) / abs_lengths.unsqueeze(-1)  # (B, D)
                for i, utt_id in enumerate(ids):
                    self.cache[utt_id] = encoded[i].cpu()
            else:
                if self.cache is None: # first epoch after caching
                    self.cache = torch.load(cache_dir/'cache.pt')
                    print("Cache loaded from disk.")
                encoded = torch.stack([self.cache[utt_id] for utt_id in ids], dim=0).to(self.device)
            wavs = encoded
        else:
            # Forward pass through the model
            wavs = self.modules.model.encoder(wavs, lens)

        predictions = self.modules.model.probe(wavs, lens)
        return predictions

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

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            batch.id, predictions, lab, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, lab)
            self.record.add(predictions, lab)

        return loss

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
            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
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

        self.finalize_cache()

    def finalize_cache(self):
        # The cache is now available until the end of training
        cache_encoder = getattr(self.hparams, "cache_encoder", False)
        if cache_encoder:
            cache_dir = Path(getattr(self.hparams, "cache_dir")).resolve()
            torch.save(self.cache, cache_dir/'cache.pt')
            cache_done = cache_dir / "cache_done.txt"
            if not cache_done.exists():
                with open(cache_done, "w") as f:
                    f.write("done")
                print("Encoder cache is ready.")

    def calc_epoch_metrics(self, stage_loss):
        """ Call this after the epoch only"""
        # Summarize the statistics from the stage for record-keeping.
        metrics = self.error_metrics.summarize()
        preds, tgts = self.record.get_all()
        stats = {
            "roc": roc_auc_score_rev(preds, tgts),
            "sens": metrics['TP'] / (metrics['TP'] + metrics['FN']),
            "accuracy": accuracy(preds, tgts),
            "loss": stage_loss,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["F-score"],
        }
        self.record.clear()
        return stats



