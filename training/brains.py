from pathlib import Path
from statistics import mean
from typing import Dict

import numpy as np
from ray import tune
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils import hpopt as hp
import speechbrain as sb
import torch
from torch.utils.data import DataLoader

from training.brain import DiagnosticsBrain


class Brains:
    def __init__(self, **kwargs):
        self.num_brains = kwargs.pop("num_brains", 1)
        self.brains = []
        for _ in range(self.num_brains):
            self.brains.append(DiagnosticsCVBrain(
                manager=self,
                **kwargs
            ))
        self.stats = []  # Contains one epoch stat of brains


    def __len__(self):
        return len(self.brains)

    def report(self, stat: Dict):
        """Report statistics to HP optimizer (Ray Tune or others)."""
        self.stats.append(stat)
        print(f"Received stats from {len(self.stats)}/{self.num_brains} brains.")
        if len(self.stats) == self.num_brains:
            print('All brains have reported stats for this epoch.')
            # Gather and take average the stat of all brains
            aggregated_stat = {}
            for key in stat.keys():
                aggregated_stat[key] = [stat[key] for stat in self.stats]
                aggregated_stat[key] = np.mean(aggregated_stat[key])

            print(f"Aggregated stats: {aggregated_stat}")
            # Report results to HP tuner
            # This will use Ray Tune's reporter when hpopt_mode='ray'
            tune.report(aggregated_stat)

            # Reset for next epoch
            self.stats = []

    def fit(self, **kwargs):
        train_sets = kwargs.pop("train_sets")
        valid_sets = kwargs.pop("valid_sets")
        progressbar = kwargs.pop("progressbar", None)

        if progressbar is None:
            progressbar = not self.brains[0].noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()


        loaders = []
        for brain, train_set, valid_set in zip(self.brains, train_sets, valid_sets):
            if not (
                    isinstance(train_set, DataLoader)
                    or isinstance(train_set, LoopedLoader)
            ):
                train_set = brain.make_dataloader(
                    train_set, stage=sb.Stage.TRAIN, **kwargs.get("train_loader_kwargs", {})
                )
            if valid_set is not None and not (
                    isinstance(valid_set, DataLoader)
                    or isinstance(valid_set, LoopedLoader)
            ):
                valid_set = brain.make_dataloader(
                    valid_set,
                    stage=sb.Stage.VALID,
                    ckpt_prefix=None,
                    **kwargs.get("valid_loader_kwargs", {}),
                )
            brain.on_fit_start()
            loaders.append((train_set, valid_set))

        for epoch in self.brains[0].hparams.epoch_counter:
            print(f"\nEpoch {epoch}\n-------")
            for i, brain_info in enumerate(zip(self.brains, loaders)):
                brain, (train_set, valid_set) = brain_info
                print(f"\nBrain {i}\n-------")
                brain._fit_train(train_set=train_set, epoch=epoch, enable=enable)
                brain._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)


class DiagnosticsCVBrain(DiagnosticsBrain):
    def __init__(self, manager, **kwargs):
        super().__init__(**kwargs)
        self.manager = manager

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch. Reports result to manager."""
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        stats = self.calc_epoch_metrics(stage_loss)

        # At the end of validation...
        if stage == sb.Stage.VALID:
            # Log stats
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Report to manager (which handles Ray Tune reporting)
            self.manager.report(stats)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

        # The cache is now available until the end of training
        self.finalize_cache()


