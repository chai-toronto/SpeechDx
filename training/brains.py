from statistics import mean
from typing import Dict
from speechbrain.utils import hpopt as hp
import speechbrain as sb

from training.brain import DiagnosticsBrain

class Brains:
    def __init__(self, **kwargs):
        self.num_brains = kwargs.pop("num_brains", 1)
        self.brains = []
        for _ in range(self.num_brains):
            self.brains.append(DiagnosticsCVBrain(
                manager = self,
                **kwargs
            ))
        self.stats = [] # Contains one epoch stat of brains
        self.hparams = kwargs.get("hparams")

    def __len__(self):
        return len(self.brains)

    def report(self, stat: Dict):
        self.stats.append(stat)
        if len(self.stats) == self.num_brains:
            # Gather and take average the stat of all brains
            aggregated_stat = {}
            for key in stat.keys():
                aggregated_stat[key] = [stat[key] for stat in self.stats]
                aggregated_stat[key] = mean(aggregated_stat[key])

            # Report results to HP tuner
            hp.report_result(aggregated_stat)
            # Reset for next epoch
            self.stats = []

    def fit(self, **kwargs):
        train_sets = kwargs.pop("train_sets")
        valid_sets = kwargs.pop("valid_sets")
        for brain, train_set, valid_set in zip(self.brains, train_sets, valid_sets):
            brain.fit(train_set=train_set,
                      valid_set=valid_set,
                      **kwargs)



class DiagnosticsCVBrain(DiagnosticsBrain):
    def __init__(self, manager, **kwargs):
        super(DiagnosticsCVBrain, self).__init__(**kwargs) # TODO: Logger for each fold
        self.manager = manager

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch. Reports result to manager."""
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
            # TODO: okay to update lr with CV?

            # Log stats and save checkpoint
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints, based on F1
            # self.checkpointer.save_and_keep_only(meta=stats, max_keys=["F1"])
            # TODO: add checkpoint to save multiple Brains, prolly not in this class

            self.manager.report(stats)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

