#!/usr/bin/env python3
"""
Generic Recipe for training a diagnostics model using SpeechBrain with Ray Tune HP optimization.

This script is designed to be reusable across different health-related
datasets. Dataset-specific functions (like data preparation and
dataio pipeline definition) are loaded dynamically from a script
specified in the YAML file.

To run this recipe, use a specific hparams file:
> python train_generic.py hparams/respiratory.yaml

To run with hyperparameter optimization:
> python train_generic.py hparams/respiratory.yaml --hpopt='ray'

Authors
--
Yi Zhu 2025
"""
import importlib
import shutil
import sys
import os
from functools import partial
from pathlib import Path

from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.searcher import ConcurrencyLimiter
from ray.tune.stopper import TrialPlateauStopper

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
import ray
from ray import tune
from ray.tune import CLIReporter

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from training.brain import DiagnosticsBrain
from training.brains import Brains, DiagnosticsCVBrain

def train_with_ray(config, hparams_file, run_opts, overrides):
    """Ray Tune trainable function that wraps the SpeechBrain training loop.

    Args:
        config: Dict containing hyperparameters from Ray Tune
        hparams_file: Path to the yaml hyperparameter file
        run_opts: SpeechBrain run options
        overrides: Command line overrides (dict or string)
    """
    # Update overrides with Ray Tune config
    ray_overrides = overrides.copy() if overrides else {}
    for key, value in config.items():
        print(f"Ray Tune override: {key} = {value}")
        ray_overrides[key] = value

    # Load hyperparameters with Ray Tune config overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, ray_overrides)

    # Create experiment directory with trial-specific folder
    trial_id = tune.get_context().get_trial_id() or "default"
    hparams["output_folder"] = os.path.join(hparams["output_folder"], trial_id)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=ray_overrides,
    )

    # Dynamically load the data preparation module
    try:
        data_io_module = importlib.import_module(hparams["data_io_script"])
    except KeyError:
        sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

    # Create dataset objects
    dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
    datasets = dataio_prep_fn(hparams)

    # ============== KEY CHANGE: New Brains API ==============
    # Initialize the Brains object by passing hparams_file and overrides
    # instead of the loaded hparams (which contains unpickleable modules)
    brains = Brains(
        hparams_file=hparams_file,      # Pass file path
        overrides=ray_overrides,         # Pass overrides dict
        run_opts=run_opts,               # Pass run_opts
    )
    # Each Ray actor will independently load hparams from the file
    # ========================================================

    train_sets = [datasets[f"train_{i}"] for i in range(hparams["num_fold"])]
    valid_sets = [datasets[f"val_{i}"] for i in range(hparams["num_fold"])]

    brains.fit(
        train_sets=train_sets,
        valid_sets=valid_sets,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
        progressbar=hparams["progressbar"]
    )


def parse_hp_search_space(hparams):
    """Parse hyperparameter search space from hparams dict.

    Looks for parameters with special Ray Tune sampling functions in the yaml.
    Example yaml format:
        lr_start: !tune.loguniform [0.0001, 0.01]
        batch_size: !tune.choice [8, 16, 32]
        dp: !tune.uniform [0.1, 0.5]

    Args:
        hparams: Hyperparameter dict from yaml

    Returns:
        Dict containing Ray Tune search space
    """
    search_space = {}

    # Map common parameter patterns to Ray Tune functions
    tune_mapping = {
        "!tune.choice": tune.choice,
        "!tune.uniform": tune.uniform,
        "!tune.loguniform": tune.loguniform,
        "!tune.randint": tune.randint,
        "!tune.quniform": tune.quniform,
    }

    # Simple search space extraction - can be extended
    # For now, define common hyperparameters to tune
    if hparams.get("hpopt_config"):
        for param, config in hparams["hpopt_config"].items():
            if isinstance(config, dict) and "type" in config:
                tune_fn = tune_mapping.get(config["type"])
                if tune_fn == tune.choice:
                    search_space[param] = tune_fn(config["values"])
                else:
                    search_space[param] = tune_fn(*config["values"])
    else:
        # Default search space if not specified
        search_space = {
            "lr_start": tune.loguniform(1e-5, 1e-2),
            "dp": tune.uniform(0.1, 0.5),
            "num_fc_neurons": tune.choice([512, 768, 1024]),
        }

    return search_space


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    if os.environ.get("WORLD_SIZE"):
        sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    hparams_file = Path(hparams_file).resolve()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dynamically load the data preparation module
    try:
        data_io_module = importlib.import_module(hparams["data_io_script"])
    except KeyError:
        sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

    # Warm cache early so Ray workers can open it read-only;
    # must run before Ray spawns parallel trials
    dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
    datasets = dataio_prep_fn(hparams)

    # Data preparation, to be run on only one process
    if not hparams["skip_prep"]:
        prepare_data_fn = getattr(data_io_module, hparams["prepare_data_fn"])

        sb.utils.distributed.run_on_main(
            prepare_data_fn,
            kwargs={
                "wav_folder": hparams["wav_folder"],
                "metadata_path": hparams["metadata_path"],
                "manifest_train_path": hparams["train_annotation"],
                "manifest_val_path": hparams["val_annotation"],
                "manifest_test_path": hparams["test_annotation"],
                "ratio": hparams.get("ratio", None),
                "random_seed": hparams["random_seed"],
                "raw_label_key": hparams["raw_label_key"],
                "new_test": hparams["new_test"],
                "num_fold": hparams["num_fold"],
            },
        )

    optim_metric = hparams.get("optim_metric", "F1")
    optim_mode = hparams.get("optim_mode", "max")
    best_config = None
    # Check if hyperparameter optimization is enabled
    if hparams.get("hpopt_mode") == "ray":
        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Parse search space
        search_space = parse_hp_search_space(hparams)

        trainable = tune.with_parameters(
            train_with_ray,
            hparams_file=hparams_file,
            run_opts=run_opts,
            overrides=overrides,
        )

        # Configure Ray Tune
        tune_config = hparams.get("ray_tune_config", {})

        # Set up reporter
        reporter = CLIReporter(
            metric_columns=["F1", "loss", "precision", "recall", "roc", "sens", "accuracy"],
            max_report_frequency=30,
        )

        optuna_search = OptunaSearch(
            metric = optim_metric,
            mode = optim_mode,
        )

        search_alg = ConcurrencyLimiter(
            optuna_search,
            max_concurrent=hparams.get("max_concurrent_trials", 4)
        )

        scheduler = ASHAScheduler(
            metric=optim_metric,
            mode=optim_mode,
            grace_period=hparams.get("grace_period", 10),
            reduction_factor=hparams.get("reduction_factor", 2),
        )


        resources_per_trial = tune_config.get("resources_per_trial", {"cpu": 1, "gpu": 0})
        if not hparams.get('sequential', True):
            num_workers = hparams.get("num_workers", 1)
            num_folds = hparams.get("num_fold", 1)

            resources_split = []
            for _ in range(num_folds):
                resources_split.append({
                    "CPU": num_workers,
                    "GPU": 1
                })
                resources_per_trial["cpu"] -= num_workers
                resources_per_trial["gpu"] -= 1

            resources_split.insert(0, {'CPU': resources_per_trial["cpu"]})
            resources_per_trial = tune.PlacementGroupFactory(resources_split)

        storage_path = (Path(hparams["output_folder"]) / "ray_results").resolve()
        if hparams["continue_exp"]:
            print(f"Continuing hyperparameter optimization from {storage_path}")
            resume="AUTO"
        else:
            resume=False
            if storage_path.exists():
                shutil.rmtree(storage_path)

        # Run hyperparameter optimization
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=tune_config.get("num_samples", 10),
            resume=resume,
            progress_reporter=reporter,
            storage_path=storage_path.as_posix(),
            name="hp_optimization",
            search_alg=search_alg,
            scheduler=scheduler,
            resources_per_trial=resources_per_trial,
        )

        # Print best hyperparameters
        best_config = analysis.get_best_config(metric=optim_metric, mode=optim_mode, scope="all")
        print(f"\nBest hyperparameters found: {best_config}")

        # Save best config
        best_config_path = os.path.join(hparams["output_folder"], "best_hparams.txt")
        with open(best_config_path, "w") as f:
            for key, value in best_config.items():
                f.write(f"{key}: {value}\n")

        ray.shutdown()
        # Final Training with best HP

    # Train final model
    if best_config is not None:
        # Update overrides with best config
        ray_overrides = overrides.copy() if overrides else {}
        for key, value in best_config.items():
            ray_overrides[key] = value
        overrides = ray_overrides

    hparams["output_folder"] = os.path.join(hparams["output_folder"], 'final_model')

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Seed for consistent final result
    sb.utils.seed.seed_everything(hparams["random_seed"])

    hparams_file = Path(hparams_file).resolve()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    brain = DiagnosticsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )

    if not hparams.get("test_only", False):
        # Not the usual CV case anymore as train and val are merged
        try:
            brain.fit(
                epoch_counter=hparams["epoch_counter"],
                train_set=datasets["train_og"],
                valid_set=datasets["val_og"],
                train_loader_kwargs=hparams["train_dataloader_options"],
                valid_loader_kwargs=hparams["val_dataloader_options"],
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)
    brain.evaluate(
        test_set=datasets["test_val"],
        test_loader_kwargs=hparams["test_dataloader_options"]
    )



