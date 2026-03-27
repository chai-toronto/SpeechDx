from pathlib import Path

import torch
import speechbrain as sb
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, F1Score, AUROC, Accuracy


def unwrap_ddp(module):
    """Unwrap a module from DistributedDataParallel if needed."""
    return getattr(module, "module", module)


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = None
        self.checkpointer.recover_if_possible()

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

        self.model = unwrap_ddp(self.modules.model)
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
            print(eval_stats.keys())
            for score in eval_stats.values():
                if isinstance(score, torch.Tensor):
                    print(score.item())
