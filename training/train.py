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
import sys
import os
from functools import partial

import ray
from ray import tune
from ray.tune import CLIReporter

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils import hpopt as hp

from training.brain import DiagnosticsBrain
from training.brains import Brains


def train_with_ray(config, *, hparams_file, run_opts, overrides):
    """Ray Tune trainable function that wraps the SpeechBrain training loop.

    Args:
        config: Dict containing hyperparameters from Ray Tune
        hparams_file: Path to the yaml hyperparameter file
        run_opts: SpeechBrain run options
        overrides: Command line overrides
    """
    # Update overrides with Ray Tune config
    ray_overrides = overrides.copy() if overrides else {}
    for key, value in config.items():
        ray_overrides[key] = value

    # Load hyperparameters with Ray Tune config overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, ray_overrides)

    # Create experiment directory with trial-specific folder
    trial_id = tune.get_trial_id() or "default"
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

    # Initialize the Brains object with Ray Tune reporter
    brains = Brains(
        num_brains=hparams["num_fold"],
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Replace the report function to use Ray Tune
    original_report = hp.report_result
    hp.report_result = lambda results: tune.report(**results)

    try:
        train_sets = [datasets[f"train_{i}"] for i in range(hparams["num_fold"])]
        valid_sets = [datasets[f"valid_{i}"] for i in range(hparams["num_fold"])]

        brains.fit(
            epoch_counter=brains.hparams.epoch_counter,
            train_sets=train_sets,
            valid_sets=valid_sets,
            train_loader_kwargs=hparams["train_dataloader_options"],
            valid_loader_kwargs=hparams["val_dataloader_options"],
        )
    finally:
        # Restore original report function
        hp.report_result = original_report


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

    # Initialize ddp
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dynamically load the data preparation module
    try:
        data_io_module = importlib.import_module(hparams["data_io_script"])
    except KeyError:
        sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

    # Data preparation, to be run on only one process
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

    # Check if hyperparameter optimization is enabled
    if hparams.get("hpopt_mode") == "ray":
        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Parse search space
        search_space = parse_hp_search_space(hparams)


        def trainable(config):
            return train_with_ray(config=config, hparams_file=hparams_file, run_opts=run_opts, overrides=overrides)


        # Configure Ray Tune
        tune_config = hparams.get("ray_tune_config", {})

        # Set up reporter
        reporter = CLIReporter(
            metric_columns=["F1", "loss", "precision", "recall"],
            max_report_frequency=30,
        )

        # Run hyperparameter optimization
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=tune_config.get("num_samples", 10),
            metric="F1",
            mode="max",
            progress_reporter=reporter,
            storage_path=os.path.join(hparams["output_folder"], "ray_results"),
            name="hp_optimization",
            stop={"training_iteration": hparams["number_of_epochs"]},
            resources_per_trial=tune_config.get("resources_per_trial", {"cpu": 1, "gpu": 1}),
        )

        # Print best hyperparameters
        best_config = analysis.get_best_config(metric="F1", mode="max")
        print(f"\nBest hyperparameters found: {best_config}")

        # Save best config
        best_config_path = os.path.join(hparams["output_folder"], "best_hparams.txt")
        with open(best_config_path, "w") as f:
            for key, value in best_config.items():
                f.write(f"{key}: {value}\n")

        ray.shutdown()

    else:
        # Standard training without HP optimization
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Create dataset objects
        dataio_prep_fn = getattr(data_io_module, hparams["dataio_prep_fn"])
        datasets = dataio_prep_fn(hparams)

        # Initialize the Brains object
        brains = Brains(
            num_brains=hparams["num_fold"],
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        train_sets = [datasets[f"train_{i}"] for i in range(hparams["num_fold"])]
        valid_sets = [datasets[f"valid_{i}"] for i in range(hparams["num_fold"])]

        brains.fit(
            epoch_counter=brains.hparams.epoch_counter,
            train_sets=train_sets,
            valid_sets=valid_sets,
            train_loader_kwargs=hparams["train_dataloader_options"],
            valid_loader_kwargs=hparams["val_dataloader_options"],
        )