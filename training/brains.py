"""
PRACTICAL FIX: brains.py with proper module handling for Ray remote actors

This version solves the PyTorch module serialization issue by ensuring
modules are instantiated locally on each remote actor rather than serialized.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import ray
from ray import tune
from speechbrain.dataio.dataloader import LoopedLoader
import speechbrain as sb
import torch
from torch.utils.data import DataLoader

from training.brain import DiagnosticsBrain


class Brains:
    """Manager for multiple diagnostic brains with concurrent GPU support."""

    def __init__(self, **kwargs):
        self.num_brains = kwargs.pop("num_brains", 1)
        self.hparams = kwargs.get("hparams")

        # Check GPU availability
        self.use_gpu = self._check_gpu_availability()

        if self.use_gpu:
            self._init_concurrent_gpu_mode(**kwargs)
        else:
            self._init_sequential_cpu_mode(**kwargs)

    def _check_gpu_availability(self):
        """Check if GPU training is possible."""
        if not self.hparams.get("use_gpu", True):
            print("GPU training disabled in configuration")
            return False

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Using sequential CPU training.")
            return False

        available_gpus = torch.cuda.device_count()
        if available_gpus < self.num_brains:
            print(f"WARNING: Need {self.num_brains} GPUs, only {available_gpus} available.")
            print(f"Falling back to sequential CPU training.")
            return False

        print(f"✓ GPU training enabled with {self.num_brains} GPUs")
        return True

    def _init_concurrent_gpu_mode(self, **kwargs):
        """Initialize brains as Ray remote actors for concurrent GPU training."""
        self.concurrent = True
        self.brains = []

        # CRITICAL: Remove 'modules' from kwargs to avoid serialization
        # Each remote actor will instantiate its own modules
        actor_kwargs = self._prepare_actor_kwargs(kwargs)

        for i in range(self.num_brains):
            brain = DiagnosticsCVBrain.remote(
                brain_id=i,
                **actor_kwargs
            )
            self.brains.append(brain)

        print(f"Initialized {self.num_brains} brains as Ray actors")

    def _prepare_actor_kwargs(self, kwargs):
        """Prepare kwargs for remote actor, removing non-serializable objects."""
        actor_kwargs = {}

        for key, value in kwargs.items():
            if key == 'modules':
                # Don't pass module instances - they'll be created remotely
                # Instead, pass the hparams which contains module specs
                continue
            else:
                actor_kwargs[key] = value

        return actor_kwargs

    def _init_sequential_cpu_mode(self, **kwargs):
        """Initialize brains as local objects for sequential CPU training."""
        self.concurrent = False
        self.brains = []

        for i in range(self.num_brains):
            brain = DiagnosticsSequentialBrain(
                brain_id=i,
                **kwargs
            )
            self.brains.append(brain)

        print(f"Initialized {self.num_brains} brains for sequential CPU training")

    def __len__(self):
        return len(self.brains)

    def fit(self, **kwargs):
        """Train all brains, using either concurrent or sequential mode."""
        if self.concurrent:
            self._fit_concurrent(**kwargs)
        else:
            self._fit_sequential(**kwargs)

    def _fit_concurrent(self, **kwargs):
        """Concurrent training for GPU mode."""
        train_sets = kwargs.pop("train_sets")
        valid_sets = kwargs.pop("valid_sets")
        train_loader_kwargs = kwargs.get("train_loader_kwargs", {})
        valid_loader_kwargs = kwargs.get("valid_loader_kwargs", {})
        progressbar = kwargs.pop("progressbar", None)

        if progressbar is None:
            progressbar_future = self.brains[0].get_progressbar_setting.remote()
            progressbar = ray.get(progressbar_future)

        enable = progressbar and sb.utils.distributed.if_main_process()

        # Setup phase
        print("Setting up data loaders for all brains...")
        setup_futures = []
        for i, brain in enumerate(self.brains):
            future = brain.setup_dataloaders.remote(
                train_set=train_sets[i],
                valid_set=valid_sets[i],
                train_loader_kwargs=train_loader_kwargs,
                valid_loader_kwargs=valid_loader_kwargs
            )
            setup_futures.append(future)
        ray.get(setup_futures)

        print("Initializing fit for all brains...")
        init_futures = [brain.on_fit_start.remote() for brain in self.brains]
        ray.get(init_futures)

        # Training loop
        for epoch in self.hparams.epoch_counter:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch} - Training {self.num_brains} brains concurrently")
            print(f"{'=' * 60}")

            # Train all brains in parallel
            train_futures = [
                brain.train_epoch.remote(epoch=epoch, enable=enable)
                for brain in self.brains
            ]
            ray.get(train_futures)
            print(f"[Training] All brains completed epoch {epoch}")

            # Validate all brains in parallel
            valid_futures = [
                brain.validate_epoch.remote(epoch=epoch, enable=enable)
                for brain in self.brains
            ]
            all_stats = ray.get(valid_futures)
            print(f"[Validation] All brains completed epoch {epoch}")

            # Aggregate and report
            self._aggregate_and_report(all_stats, epoch)

    def _fit_sequential(self, **kwargs):
        """Sequential training for CPU mode."""
        train_sets = kwargs.pop("train_sets")
        valid_sets = kwargs.pop("valid_sets")
        train_loader_kwargs = kwargs.get("train_loader_kwargs", {})
        valid_loader_kwargs = kwargs.get("valid_loader_kwargs", {})
        progressbar = kwargs.pop("progressbar", None)

        if progressbar is None:
            progressbar = not self.brains[0].noprogressbar

        enable = progressbar and sb.utils.distributed.if_main_process()

        # Setup
        for i, brain in enumerate(self.brains):
            brain.setup_dataloaders(
                train_set=train_sets[i],
                valid_set=valid_sets[i],
                train_loader_kwargs=train_loader_kwargs,
                valid_loader_kwargs=valid_loader_kwargs
            )

        for brain in self.brains:
            brain.on_fit_start()

        # Training loop
        for epoch in self.hparams.epoch_counter:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch} - Training {self.num_brains} brains sequentially")
            print(f"{'=' * 60}")

            all_stats = []
            for i, brain in enumerate(self.brains):
                print(f"\n[Brain {i + 1}/{self.num_brains}]")
                brain.train_epoch(epoch=epoch, enable=enable)
                stats = brain.validate_epoch(epoch=epoch, enable=enable)
                all_stats.append(stats)

            self._aggregate_and_report(all_stats, epoch)

    def _aggregate_and_report(self, all_stats, epoch):
        """Aggregate statistics and report to Ray Tune."""
        valid_stats = [s for s in all_stats if s is not None]

        if valid_stats:
            aggregated_stat = {}
            for key in valid_stats[0].keys():
                values = [stat[key] for stat in valid_stats]
                aggregated_stat[key] = np.mean(values)
                print(f"  {key}: {aggregated_stat[key]:.4f}")

            tune.report(aggregated_stat)
        else:
            print("[Warning] No valid stats received")


@ray.remote(num_gpus=1)
class DiagnosticsCVBrain(DiagnosticsBrain):
    """
    Remote Ray actor for concurrent GPU training.

    IMPORTANT: This class creates its own modules to avoid serialization issues.
    """

    def __init__(self, brain_id, **kwargs):
        # The key insight: SpeechBrain's Brain class can instantiate modules
        # from hparams if 'modules' is not explicitly provided
        # OR we need to create modules before calling super().__init__

        hparams = kwargs.get('hparams')

        # If modules not in kwargs, try to get from hparams
        if 'modules' not in kwargs and hparams:
            # Check if hparams has modules attribute
            if hasattr(hparams, 'modules'):
                # hparams.modules exists - use it
                # Note: This might still be an issue if hparams.modules contains
                # the parametrized modules. In that case, we need to rebuild them.
                kwargs['modules'] = hparams.modules

        # Initialize parent class
        super().__init__(**kwargs)

        self.brain_id = brain_id
        self.train_loader = None
        self.valid_loader = None
        self.last_valid_stats = None

        # Set device to allocated GPU
        if torch.cuda.is_available():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                self.device = f"cuda:{gpu_ids[0]}"
                print(f"Brain {brain_id}: Initialized on {self.device}")

                # Move all modules to the correct device
                if hasattr(self, 'modules') and self.modules:
                    for name, module in self.modules.items():
                        if hasattr(module, 'to'):
                            module.to(self.device)
                    print(f"Brain {brain_id}: Modules moved to {self.device}")
            else:
                self.device = "cpu"
                print(f"Brain {brain_id}: No GPU allocated, using CPU")
        else:
            self.device = "cpu"
            print(f"Brain {brain_id}: CUDA not available, using CPU")

    def get_progressbar_setting(self):
        """Return progressbar setting."""
        return not self.noprogressbar

    def setup_dataloaders(self, train_set, valid_set, train_loader_kwargs, valid_loader_kwargs):
        """Setup data loaders."""
        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            self.train_loader = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        else:
            self.train_loader = train_set

        if valid_set is not None:
            if not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
                self.valid_loader = self.make_dataloader(
                    valid_set, stage=sb.Stage.VALID,
                    ckpt_prefix=None, **valid_loader_kwargs
                )
            else:
                self.valid_loader = valid_set

    def train_epoch(self, epoch, enable):
        """Train for one epoch."""
        self._fit_train(train_set=self.train_loader, epoch=epoch, enable=enable)

    def validate_epoch(self, epoch, enable):
        """Validate for one epoch and return stats."""
        self._fit_valid(valid_set=self.valid_loader, epoch=epoch, enable=enable)
        return self.last_valid_stats

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Called at end of each stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        stats = self.calc_epoch_metrics(stage_loss)

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "Brain": self.brain_id},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.last_valid_stats = stats

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current, "Brain": self.brain_id},
                test_stats=stats,
            )

        self.finalize_cache()


class DiagnosticsSequentialBrain(DiagnosticsBrain):
    """Local brain for sequential CPU training."""

    def __init__(self, brain_id, **kwargs):
        super().__init__(**kwargs)
        self.brain_id = brain_id
        self.train_loader = None
        self.valid_loader = None
        self.last_valid_stats = None

    def setup_dataloaders(self, train_set, valid_set, train_loader_kwargs, valid_loader_kwargs):
        """Setup data loaders."""
        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            self.train_loader = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        else:
            self.train_loader = train_set

        if valid_set is not None:
            if not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
                self.valid_loader = self.make_dataloader(
                    valid_set, stage=sb.Stage.VALID,
                    ckpt_prefix=None, **valid_loader_kwargs
                )
            else:
                self.valid_loader = valid_set

    def train_epoch(self, epoch, enable):
        """Train for one epoch."""
        self._fit_train(train_set=self.train_loader, epoch=epoch, enable=enable)

    def validate_epoch(self, epoch, enable):
        """Validate and return stats."""
        self._fit_valid(valid_set=self.valid_loader, epoch=epoch, enable=enable)
        return self.last_valid_stats

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Called at end of each stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        stats = self.calc_epoch_metrics(stage_loss)

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "Brain": self.brain_id},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.last_valid_stats = stats

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current, "Brain": self.brain_id},
                test_stats=stats,
            )

        self.finalize_cache()