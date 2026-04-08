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
Larry Kieu 2026
"""
import importlib
import json
import shutil
import sys
import os
from pathlib import Path

import yaml
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.searcher import ConcurrencyLimiter

from training.dataio.preprocessing import master_dataio_prep

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
import ray
from ray import tune
from ray.tune import CLIReporter

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from training.brain import DiagnosticsBrain
from training.brains import Brains, DiagnosticsCVBrain
from training.hp_utils import parse_hp_search_space

def dataio_prep(hparams):
    """
    This function is dataset-specific.
    It defines the data processing pipelines and creates the DynamicItemDatasets.

    For a new task, modify the label_pipeline and the output_keys.
    """
    # Retrieve the data
    with open(hparams["train_annotation"], "r") as f:
        train = json.load(f)

    with open(hparams["val_annotation"], "r") as f:
        val = json.load(f)

    with open(hparams['test_annotation'], "r") as f:
        test = json.load(f)

    data_dict = {
        "train": train,
        "val": val,
        "test": test,
        "all": train | val | test, # for cache warming
    }

    datasets = master_dataio_prep(data_dict, hparams)
    return datasets

def train_with_ray(config, hparams_file, run_opts, overrides, project_root):
    """Ray Tune trainable function that wraps the SpeechBrain training loop.

    Args:
        config: Dict containing hyperparameters from Ray Tune
        hparams_file: Path to the yaml hyperparameter file
        run_opts: SpeechBrain run options
        overrides: Command line overrides (dict or string)
        project_root: Absolute path to the project root directory
    """
    # Update overrides with Ray Tune config
    ray_overrides = overrides.copy() if overrides else {}
    for key, value in config.items():
        print(f"Ray Tune override: {key} = {value}")
        ray_overrides[key] = value

    # Load hyperparameters with Ray Tune config overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, ray_overrides)

    # Resolve relative paths against project root for Ray workers
    root = Path(project_root)
    for key, val in hparams.items():
        if isinstance(val, str) and not Path(val).is_absolute() and (
                key.endswith("_dir") or key.endswith("_folder")
                or key.endswith("_path") or key.endswith("_annotation")
        ):
            hparams[key] = str(root / val)

    # Create experiment directory with trial-specific folder
    trial_id = tune.get_context().get_trial_id() or "default"
    hparams["output_folder"] = os.path.join(hparams["output_folder"], trial_id)
    hparams["save_folder"] = os.path.join(hparams["save_folder"], trial_id)

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

    # Cache should already be warmed by the main process; open read-only here
    hparams["warm_cache"] = False

    # Create dataset objects

    datasets = dataio_prep(hparams)

    # Rebuild checkpointer with trial-specific save_folder
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables=hparams["checkpointer"].recoverables,
    )

    brain = DiagnosticsBrain(
        ray_optim=True,
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["val"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )


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


    # Data preparation, to be run on only one process
    if not hparams["skip_prep"]:
        # Dynamically load the data preparation module
        try:
            data_io_module = importlib.import_module(hparams["data_io_script"])
        except KeyError:
            sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

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
                "dataset": hparams["dataset"],
                "task": hparams["task"],
            },
        )

    # Warm cache (if True) early so Ray workers can open it read-only;
    # must run before Ray spawns parallel trials
    datasets = dataio_prep(hparams)


    _project_root = str(Path.cwd().resolve())

    optim_metric = hparams.get("optim_metric", "F1")
    optim_mode = hparams.get("optim_mode", "max")
    best_config = None
    # Check if hyperparameter optimization is enabled
    if not hparams.get("test_only"):
        # Initialize Ray
        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "excludes": [
                    "data/", "exps/", "tmp/", ".idea/",
                    "*.DS_Store", "uv.lock", "pyproject.toml",
                    ".python-version", "exps_old/", "data_zip", "data_extra"
                ],
            },
        )

        # Parse search space
        search_space = parse_hp_search_space(hparams)

        trainable = tune.with_parameters(
            train_with_ray,
            hparams_file=hparams_file,
            run_opts=run_opts,
            overrides=overrides,
            project_root=_project_root,
        )

        # Configure Ray Tune
        tune_config = hparams.get("ray_tune_config", {})

        # Set up reporter
        task_type = hparams.get("task_type", "B")
        if task_type == "R":
            metric_cols = ["loss", "MAE", "MSE", "R2", "PearsonR"]
        else:
            metric_cols = ["F1", "loss", "precision", "recall", "AUROC", "accuracy"]
        reporter = CLIReporter(
            metric_columns=metric_cols,
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

        stopper = tune.stopper.TrialPlateauStopper(
            metric=optim_metric,
            mode=optim_mode,
            grace_period=hparams['hpopt_params']['limit_warmup'],
            num_results=hparams['grace_period'] # correct order, semantics from SB
        )

        resources_per_trial = tune_config.get("resources_per_trial", {"cpu": 1, "gpu": 0})

        storage_path = (Path(hparams["output_folder"]) / "results").resolve()

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
            stop=stopper,
            progress_reporter=reporter,
            storage_path=storage_path.as_posix(),
            name="hp_optimization",
            search_alg=search_alg,
            resources_per_trial=resources_per_trial,
        )

        trial_id = analysis.get_best_trial(metric=optim_metric, mode=optim_mode, scope="all").trial_id

        # Print best hyperparameters
        best_config = analysis.get_best_config(metric=optim_metric, mode=optim_mode, scope="all")
        print(f"\nBest hyperparameters found: {best_config}")
        best_config["trial_id"] = trial_id

        # Save best config
        best_config_path = os.path.join(hparams["output_folder"], "best_hparams.yaml")
        with open(best_config_path, "w") as f:
            yaml.dump(best_config, f)

        ray.shutdown()

    if best_config is None:
        # Try to load best config
        print("Loading best configs")
        best_config_path = os.path.join(hparams["output_folder"], "best_hparams.yaml")
        assert os.path.exists(best_config_path), "Cant find best config"

        with open(best_config_path, "r") as f:
            best_config = yaml.safe_load(f)

    # Point dir to best trial
    exp_root = hparams["output_folder"]
    hparams["output_folder"] = os.path.join(exp_root, best_config['trial_id'])
    hparams["save_folder"] = os.path.join(hparams["save_folder"], best_config['trial_id'])

    # Rebuild checkpointer with trial-specific save_folder
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables=hparams["checkpointer"].recoverables,
    )

    brain = DiagnosticsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    brain.evaluate(
        test_set=datasets["test"],
        test_loader_kwargs=hparams["test_dataloader_options"]
    )

    # Write test results to file
    results_path = os.path.join(exp_root, "test_results.txt")
    with open(results_path, "w") as f:
        for name, score in brain.test_stats.items():
            f.write(f"{name}: {score}\n")



