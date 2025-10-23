import numpy as np
import ray
from ray import tune
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils import hpopt as hp
import speechbrain as sb
import torch
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml

from training.brain import DiagnosticsBrain


class Brains:
    """
    Manager for multiple diagnostic brains with support for both concurrent
    and sequential training.

     Concurrent Mode:
        - Each brain gets its own GPU via Ray remote actors
        - All brains train/validate in parallel
        - Requires num_fold GPUs available

    Sequential Mode:
        - Falls back to sequential training when GPUs unavailable
    """

    def __init__(self, hparams_file, overrides, run_opts, **kwargs):
        """
        Initialize Brains manager.

        Args:
            hparams_file: Path to YAML hyperparameters file
            overrides: Dictionary of parameter overrides
            run_opts: SpeechBrain run options
            **kwargs: Additional keyword arguments (for backward compatibility)
        """
        # Load hparams to determine configuration
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        self.num_brains = hparams.get("num_fold", kwargs.get("num_brains", 1))
        self.hparams = hparams
        self.hparams_file = hparams_file
        self.overrides = overrides
        self.run_opts = run_opts

        # Determine execution mode based on GPU availability and configuration
        self.sequential = hparams.get('sequential', True)

        if not self.sequential:
            self._init_concurrent_gpu_mode()
        else:
            self._init_sequential_mode()

    def _check_gpu_availability(self):
        """Check if GPU training is possible and advisable."""
        # Check if user explicitly disabled GPU
        if not self.hparams.get("use_gpu", True):
            print("GPU training disabled in configuration (use_gpu=False)")
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Falling back to CPU training.")
            print("This will be significantly slower than GPU training.")
            return False

        # Check if enough GPUs available
        available_gpus = torch.cuda.device_count()
        if available_gpus < self.num_brains:
            print(f"WARNING: Not enough GPUs for concurrent training!")
            print(f"  Required: {self.num_brains} GPUs (num_fold={self.num_brains})")
            print(f"  Available: {available_gpus} GPUs")
            print(f"  Falling back to sequential CPU training.")
            print(f"\nTo enable GPU training:")
            print(f"  1. Reduce num_fold to {available_gpus} or less, OR")
            print(f"  2. Add more GPUs to your system")
            return False

        print(f"✓ GPU training enabled with {self.num_brains} GPUs")
        return True

    def _init_concurrent_gpu_mode(self):
        """Initialize brains as Ray remote actors for concurrent GPU training."""
        self.concurrent = True
        self.brains = []

        for i in range(self.num_brains):
            # Pass hparams_file and overrides instead of loaded hparams
            brain = DiagnosticsCVBrain.options(
                num_gpus=1,
                num_cpus=self.hparams.get('num_workers', 4)
            ).remote(
                brain_id=i,
                hparams_file=str(self.hparams_file),  # Ensure it's a string path
                overrides=self.overrides,
                run_opts=self.run_opts,
            )
            self.brains.append(brain)

        print(f"Initialized {self.num_brains} brains as Ray actors (concurrent GPU mode)")

    def _init_sequential_mode(self):
        """Initialize brains as local objects for sequential training."""
        self.concurrent = False
        self.brains = []

        for i in range(self.num_brains):
            # For local brains, we can use the loaded hparams
            brain = DiagnosticsSequentialBrain(
                brain_id=i,
                hparams=self.hparams,
                run_opts=self.run_opts,
            )
            self.brains.append(brain)

        print(f"Initialized {self.num_brains} brains for sequential training")

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
            # Get from first brain
            progressbar_future = self.brains[0].get_progressbar_setting.remote()
            progressbar = ray.get(progressbar_future)

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Setup data loaders for all brains in parallel
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

        # Initialize fit for all brains
        print("Initializing fit for all brains...")
        init_futures = [brain.on_fit_start.remote() for brain in self.brains]
        ray.get(init_futures)

        # Training loop - iterate through epochs
        for epoch in self.hparams.get('epoch_counter'):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} - Training all {self.num_brains} brains concurrently")
            print(f"{'='*60}")

            # Phase 1: Train all brains concurrently
            print(f"\n[Training Phase] Launching {self.num_brains} concurrent training jobs...")
            train_futures = []
            for i, brain in enumerate(self.brains):
                future = brain.train_epoch.remote(epoch=epoch, enable=enable)
                train_futures.append(future)

            # Wait for all training to complete
            ray.get(train_futures)
            print(f"[Training Phase] All brains completed training for epoch {epoch}")

            # Phase 2: Validate all brains concurrently
            print(f"\n[Validation Phase] Launching {self.num_brains} concurrent validation jobs...")
            valid_futures = []
            for i, brain in enumerate(self.brains):
                future = brain.validate_epoch.remote(epoch=epoch, enable=enable)
                valid_futures.append(future)

            # Wait for all validation to complete and collect stats
            all_stats = ray.get(valid_futures)
            print(f"[Validation Phase] All brains completed validation for epoch {epoch}")

            # Phase 3: Aggregate statistics across all brains
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

        # Setup data loaders for all brains
        print("Setting up data loaders for all brains...")
        for i, brain in enumerate(self.brains):
            brain.setup_dataloaders(
                train_set=train_sets[i],
                valid_set=valid_sets[i],
                train_loader_kwargs=train_loader_kwargs,
                valid_loader_kwargs=valid_loader_kwargs
            )

        # Initialize fit for all brains
        print("Initializing fit for all brains...")
        for brain in self.brains:
            brain.on_fit_start()

        # Training loop - iterate through epochs
        for epoch in self.hparams.get('epoch_counter', 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} - Training {self.num_brains} brains sequentially (CPU)")
            print(f"{'='*60}")

            all_stats = []

            # Train and validate each brain sequentially
            for i, brain in enumerate(self.brains):
                print(f"\n[Brain {i}/{self.num_brains}]")

                # Training phase
                brain.train_epoch(epoch=epoch, enable=enable)

                # Validation phase
                stats = brain.validate_epoch(epoch=epoch, enable=enable)
                all_stats.append(stats)

            # Aggregate statistics across all brains
            self._aggregate_and_report(all_stats, epoch)

    def _aggregate_and_report(self, all_stats, epoch):
        """Aggregate statistics from all brains and report to Ray Tune."""
        print(f"\n[Aggregation Phase] Collecting stats from {len(all_stats)} brains...")

        # Filter out None values
        valid_stats = [s for s in all_stats if s is not None]

        if valid_stats:
            # Aggregate all metrics
            aggregated_stat = {}
            for key in valid_stats[0].keys():
                values = [stat[key] for stat in valid_stats]
                aggregated_stat[key] = np.mean(values)
                print(f"  {key}: {aggregated_stat[key]:.4f} (mean of {len(values)} brains)")

            print(f"\n[Reporting Phase] Reporting aggregated stats to Ray Tune")
            print(f"Aggregated stats: {aggregated_stat}")

            # Report aggregated results to Ray Tune
            tune.report(aggregated_stat)
        else:
            print("[Warning] No valid stats received from brains")

        print(f"{'='*60}\n")


@ray.remote
class DiagnosticsCVBrain(DiagnosticsBrain):
    """
    Remote Ray actor version of DiagnosticsBrain for concurrent GPU training.
    Each instance gets its own GPU (num_gpus=1).

    This class reconstructs hparams from the hparams_file in each actor to avoid
    serialization issues with PyTorch modules.
    """

    def __init__(self, brain_id, hparams_file, overrides, run_opts):
        """
        Initialize brain by loading hparams from file.

        Args:
            brain_id: Unique identifier for this brain
            hparams_file: Path to YAML hyperparameters file
            overrides: Dictionary of parameter overrides
            run_opts: SpeechBrain run options
        """
        # Load hparams in this actor's process
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Initialize parent class with loaded hparams
        super().__init__(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"]
        )

        self.brain_id = brain_id
        self.train_loader = None
        self.valid_loader = None
        self.last_valid_stats = None

        # Set device to the GPU allocated by Ray
        if torch.cuda.is_available():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                # self.device = f"cuda:{gpu_ids[0]}"
                self.device = torch.device(f"cuda:0")
                print(f"Brain {brain_id} initialized on device {self.device}")
            else:
                self.device = "cpu"
                print(f"Brain {brain_id} initialized on CPU (no GPU allocated)")
        else:
            self.device = "cpu"
            print(f"Brain {brain_id} initialized on CPU (CUDA not available)")

    def get_progressbar_setting(self):
        """Return whether progressbar should be shown."""
        return not self.noprogressbar

    def setup_dataloaders(self, train_set, valid_set, train_loader_kwargs, valid_loader_kwargs):
        """Setup data loaders for this brain."""
        # Create train loader if needed
        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            self.train_loader = self.make_dataloader(
                train_set,
                stage=sb.Stage.TRAIN,
                **train_loader_kwargs
            )
        else:
            self.train_loader = train_set

        # Create validation loader if needed
        if valid_set is not None:
            if not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
                self.valid_loader = self.make_dataloader(
                    valid_set,
                    stage=sb.Stage.VALID,
                    ckpt_prefix=None,
                    **valid_loader_kwargs
                )
            else:
                self.valid_loader = valid_set

        print(f"Brain {self.brain_id}: Data loaders setup complete")

    def train_epoch(self, epoch, enable):
        """Train for one epoch."""
        print(f"Brain {self.brain_id}: Starting training for epoch {epoch}")
        with torch.detect_anomaly():
            self._fit_train(train_set=self.train_loader, epoch=epoch, enable=enable)
        print(f"Brain {self.brain_id}: Completed training for epoch {epoch}")

    def validate_epoch(self, epoch, enable):
        """Validate for one epoch and return statistics."""
        print(f"Brain {self.brain_id}: Starting validation for epoch {epoch}")
        self._fit_valid(valid_set=self.valid_loader, epoch=epoch, enable=enable)
        print(f"Brain {self.brain_id}: Completed validation for epoch {epoch}")
        return self.last_valid_stats

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Store the train loss until the validation stage
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        # Calculate metrics
        stats = self.calc_epoch_metrics(stage_loss)

        # At the end of validation, store stats for retrieval
        if stage == sb.Stage.VALID:
            # Log stats locally
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "Brain": self.brain_id},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Store stats to be returned by validate_epoch
            self.last_valid_stats = stats
            print(f"Brain {self.brain_id}: Validation stats - {stats}")

        # Handle test stage
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current, "Brain": self.brain_id},
                test_stats=stats,
            )

        # Finalize cache if enabled
        self.finalize_cache()


class DiagnosticsSequentialBrain(DiagnosticsBrain):
    """
    Local (non-Ray) version of DiagnosticsBrain for sequential CPU training.
    Used as fallback when GPUs are not available.
    """

    def __init__(self, brain_id, hparams, run_opts):
        """
        Initialize brain with already-loaded hparams.

        Args:
            brain_id: Unique identifier for this brain
            hparams: Already-loaded hyperparameters dictionary
            run_opts: SpeechBrain run options
        """
        super().__init__(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"]
        )

        self.brain_id = brain_id
        self.train_loader = None
        self.valid_loader = None
        self.last_valid_stats = None
        print(f"Brain {brain_id} initialized for sequential training")

    def setup_dataloaders(self, train_set, valid_set, train_loader_kwargs, valid_loader_kwargs):
        """Setup data loaders for this brain."""
        # Create train loader if needed
        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            self.train_loader = self.make_dataloader(
                train_set,
                stage=sb.Stage.TRAIN,
                **train_loader_kwargs
            )
        else:
            self.train_loader = train_set

        # Create validation loader if needed
        if valid_set is not None:
            if not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
                self.valid_loader = self.make_dataloader(
                    valid_set,
                    stage=sb.Stage.VALID,
                    ckpt_prefix=None,
                    **valid_loader_kwargs
                )
            else:
                self.valid_loader = valid_set

        print(f"Brain {self.brain_id}: Data loaders setup complete")

    def train_epoch(self, epoch, enable):
        """Train for one epoch."""
        print(f"Brain {self.brain_id}: Starting training for epoch {epoch}")
        self._fit_train(train_set=self.train_loader, epoch=epoch, enable=enable)
        print(f"Brain {self.brain_id}: Completed training for epoch {epoch}")

    def validate_epoch(self, epoch, enable):
        """Validate for one epoch and return statistics."""
        print(f"Brain {self.brain_id}: Starting validation for epoch {epoch}")
        self._fit_valid(valid_set=self.valid_loader, epoch=epoch, enable=enable)
        print(f"Brain {self.brain_id}: Completed validation for epoch {epoch}")
        return self.last_valid_stats

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch."""
        # Store the train loss until the validation stage
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        # Calculate metrics
        stats = self.calc_epoch_metrics(stage_loss)

        # At the end of validation, store stats for retrieval
        if stage == sb.Stage.VALID:
            # Log stats locally
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "Brain": self.brain_id},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Store stats to be returned by validate_epoch
            self.last_valid_stats = stats
            print(f"Brain {self.brain_id}: Validation stats - {stats}")

        # Handle test stage
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current, "Brain": self.brain_id},
                test_stats=stats,
            )

        # Finalize cache if enabled
        self.finalize_cache()