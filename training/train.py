#!/usr/bin/env python3
"""
Generic Recipe for training a diagnostics model using SpeechBrain.

This script is designed to be reusable across different health-related
datasets. Dataset-specific functions (like data preparation and
dataio pipeline definition) are loaded dynamically from a script
specified in the YAML file.

To run this recipe, use a specific hparams file:
> python train_generic.py hparams/respiratory.yaml

Authors
--
Yi Zhu 2025
"""

import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import LoopedLoader
from torch.utils.data import DataLoader


class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop for a generic diagnostics task."""

    def compute_forward(self, batch, stage):
        """Runs all the computation that transforms the input into the
        output probabilities over the N classes.
        """
        batch = batch.to(self.device)
        wavs, _ = self.augment_input(batch.signal, stage)
        predictions = self.modules.model(wavs)
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
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        _, lens = batch.signal
        
        # Dynamically retrieve the label using the 'label_key' from hparams
        label_key = self.hparams.get("label_key", "label_encoded")
        lab, _ = getattr(batch, label_key)
        lab = lab.to(self.device)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            lab = torch.cat([lab, lab], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function: BCE is assumed for binary classification
        # but pos_weight is used for imbalance handling.
        weight = torch.tensor([self.hparams.get("positive_class_weight", 1.0)]).to(self.device)
        loss = sb.nnet.losses.bce_loss(predictions, lab, pos_weight=weight)

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            batch.id, predictions, lab, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, lab)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.bce_loss
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        # Summarize the statistics from the stage for record-keeping.
        metrics = self.error_metrics.summarize() 
        stats = {
            "loss": stage_loss,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["F-score"],
        }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints, based on F1
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["F1"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    #Updated for CV
    def fit(
        self,
        epoch_counter,
        train_sets,
        valid_sets=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        loaders = []
        for train_set, valid_set in zip(train_sets, valid_sets):
            if not (
                isinstance(train_set, DataLoader)
                or isinstance(train_set, LoopedLoader)
            ):
                train_loader = self.make_dataloader(
                    train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
                )
            if valid_set is not None and not (
                isinstance(valid_set, DataLoader)
                or isinstance(valid_set, LoopedLoader)
            ):
                valid_loader = self.make_dataloader(
                    valid_set,
                    stage=sb.Stage.VALID,
                    ckpt_prefix=None,
                    **valid_loader_kwargs,
                )
            loaders.append((train_loader, valid_loader))

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:

            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp.
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dynamically load the data preparation module specified in the YAML
    # This module contains the 'prepare_data' and 'dataio_prep' functions.
    try:
        data_io_module = sb.import_module(hparams["data_io_script"])
    except KeyError:
        sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        prepare_data_fn = getattr(data_io_module, hparams["prepare_data_fn"])
        
        sb.utils.distributed.run_on_main(
            prepare_data_fn,
            kwargs={
                "wav_folder": hparams["wav_folder"],
                "audio_archive_path": hparams["audio_archive_path"],
                "metadata_path": hparams["metadata_path"],
                "manifest_train_path": hparams["train_annotation"],
                "manifest_val_path": hparams["val_annotation"],
                "manifest_test_path": hparams["test_annotation"],
                "ratio": hparams["ratio"],
                "random_seed": hparams["random_seed"],
                "label_key": hparams["label_key"],
                "new_test": hparams["new_test"],
                "num_fold": hparams["num_fold"],
            },
        )

    # Create dataset objects
    dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
    datasets = dataio_prep_fn(hparams)


    # Training loop
    for fold in range(hparams["num_fold"]):
        # Initialize the Brain object
        brain = DiagnosticsBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        brain.fit(
            epoch_counter=brain.hparams.epoch_counter,
            train_set=datasets[f"train_{fold}"],
            valid_set=datasets[f"valid_{fold}"],
            train_loader_kwargs=hparams["train_dataloader_options"],
            valid_loader_kwargs=hparams["val_dataloader_options"],
        )

    # Evaluation
    test_stats = brain.evaluate(
        test_set=datasets["test"],
        max_key="F1",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
