import yaml
import logging
from speechbrain.utils.epoch_loop import EpochCounterWithStopper
from speechbrain.utils.checkpoints import (
    mark_as_saver, mark_as_loader, mark_as_transfer, register_checkpoint_hooks
)

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class ChunkScheduledEpochCounter(EpochCounterWithStopper):
    def __init__(self, limit, limit_to_stop, limit_warmup, direction,
                 min_chunk_size, max_chunk_size):
        super().__init__(limit, limit_to_stop, limit_warmup, direction)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._phase_start = 0  # Epoch at which the current phase began

    def update_metric(self, current_metric):
        # Warmup relative to current phase start (true reset semantics)
        if self.current - self._phase_start <= self.limit_warmup:
            return

        # Improvement check using parent's sign/min_delta convention
        if self.sign * current_metric < self.sign * ((1 - self.min_delta) * self.best_score):
            self.best_limit = self.current
            self.best_score = current_metric

        epochs_without_improvement = self.current - self.best_limit
        if epochs_without_improvement >= self.limit_to_stop:
            if self.min_chunk_size < self.max_chunk_size:
                self.min_chunk_size = min(self.min_chunk_size * 2, self.max_chunk_size)
                self._phase_start = self.current
                self.best_limit = self.current
                self.best_score = -float("inf") if self.direction == "max" else float("inf")
                logger.info(
                    f"min_chunk_size scheduled to {self.min_chunk_size} "
                    f"(phase reset at epoch {self.current})"
                )
            else:
                self.should_stop = True
                logger.info(
                    f"Max chunk size {self.max_chunk_size} reached and patience exhausted. Stopping."
                )

    @mark_as_saver
    def _save(self, path):
        save_dict = {
            "current_epoch": self.current,
            "best_epoch": self.best_limit,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
            "min_chunk_size": self.min_chunk_size,
            "phase_start": self._phase_start,
        }
        with open(path, "w") as f:
            yaml.dump(save_dict, f)

    @mark_as_loader
    @mark_as_transfer
    def _recover(self, path, end_of_epoch=True, device=None):
        with open(path) as f:
            save_dict = yaml.safe_load(f)
        self.current = save_dict["current_epoch"]
        if not end_of_epoch:
            self.current -= 1
        self.best_limit = save_dict["best_epoch"]
        self.best_score = save_dict["best_score"]
        self.should_stop = save_dict["should_stop"]
        self.min_chunk_size = save_dict.get("min_chunk_size", self.min_chunk_size)
        self._phase_start = save_dict.get("phase_start", 0)
