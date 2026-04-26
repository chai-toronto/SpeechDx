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
from training.dataio.preprocessing import master_dataio_prep_cross

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
import ray
from ray import tune
from ray.tune import CLIReporter

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from training.brain import DiagnosticsBrain
from training.brains import Brains, DiagnosticsCVBrain
from training.config_fork import fork_trial_config
from training.hp_utils import parse_hp_search_space


def _as_override_dict(overrides):
    """Normalize SB's overrides (YAML string or dict or None) to a dict."""
    if overrides is None or overrides == "":
        return {}
    if isinstance(overrides, dict):
        return dict(overrides)
    return yaml.safe_load(overrides) or {}


#: Top-level source paths in ``main.yaml`` whose absolute values get stamped
#: into each trial's forked config. Derived paths (``save_folder``,
#: ``wav_folder``, ``train_cache_dir``, ...) are deliberately excluded —
#: they use ``!ref <source_key>/...`` in the yaml, so stamping the source is
#: enough for the chain to resolve to an absolute value automatically. In
#: particular, stamping ``save_folder`` here would shadow the
#: ``!ref <output_folder>/save`` chain and break per-trial nesting.
_PATH_STAMP_KEYS = (
    "data_folder",
    "slurm_tmpdir",
    "output_folder",
    "train_annotation",
    "val_annotation",
    "test_annotation",
)


def _collect_resolved_paths(hparams, project_root):
    """Pre-resolve top-level source paths to absolute strings.

    These get stamped into the forked main.yaml so Ray workers never see a
    relative path (they may run from a different CWD). Only SOURCE keys
    (:data:`_PATH_STAMP_KEYS`) are resolved — derived paths reach the worker
    via the yaml's ``!ref`` chains.
    """
    resolved = {}
    for key in _PATH_STAMP_KEYS:
        val = hparams.get(key)
        if not isinstance(val, str):
            continue
        p = Path(val)
        resolved[key] = str(p if p.is_absolute() else (project_root / p).resolve())
    return resolved

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
    if hparams.get("cross_eval", False):
        data_dict = {
            "train": train,
            "val": val,
            "test": test,
            "all_train": train | val, # for cache warming
            "all_test": test, # for cache warming
        }

        # datasets, cache_handles = master_dataio_prep_cross(data_dict, hparams)
        datasets = master_dataio_prep_cross(data_dict, hparams)
    else:
        data_dict = {
            "train": train,
            "val": val,
            "test": test,
            "all": train | val | test, # for cache warming
        }

        # datasets, cache_handles = master_dataio_prep(data_dict, hparams)
        datasets = master_dataio_prep(data_dict, hparams)
    # return datasets, cache_handles
    return datasets

def train_with_ray(config, hparams_file, run_opts, overrides, resolved_paths):
    """Ray Tune trainable function that wraps the SpeechBrain training loop.

    Each trial forks the config tree into ``<trial_dir>/config/`` on first
    invocation and then loads hparams from the fork with no overrides. On
    resume the existing fork is reused verbatim, so edits to the base
    ``training/config/*.yaml`` between runs never leak into a resumed trial.

    Args:
        config: Hyperparameter sample from Ray Tune (lr_s, l2, ...).
        hparams_file: Path to the source main yaml (only read on first call).
        run_opts: SpeechBrain run options.
        overrides: Command-line overrides (YAML string or dict).
        resolved_paths: Dict of top-level path-like hparams pre-resolved to
            absolute strings by the main process.
    """
    trial_id = tune.get_context().get_trial_id() or "default"
    trial_dir = Path(resolved_paths["output_folder"]) / trial_id

    # Assemble the full stamp set for the forked main.yaml:
    # absolute paths → Ray sample → CLI overrides → trial-specific keys.
    fork_overrides = {}
    fork_overrides.update(resolved_paths)
    fork_overrides.update(_as_override_dict(overrides))
    for key, value in config.items():
        print(f"Ray Tune override: {key} = {value}")
        fork_overrides[key] = value
    fork_overrides["output_folder"] = str(trial_dir)
    fork_overrides["warm_cache"] = False

    forked_yaml = fork_trial_config(
        base_yaml=Path(hparams_file),
        trial_dir=trial_dir,
        overrides=fork_overrides,
    )
    with open(forked_yaml) as fin:
        hparams = load_hyperpyyaml(fin)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=str(forked_yaml),
    )

    datasets = dataio_prep(hparams)  # Ray trials keep their cache handles open

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
    # Convert SLURM SIGTERM into SystemExit so try/finally cleanup runs
    # (lets HDF5 cache files close cleanly instead of leaving corrupt headers).
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    if os.environ.get("WORLD_SIZE"):
        sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    hparams_file = Path(hparams_file).resolve()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Log key config values for verification
    print(f"[CONFIG] test_only={hparams.get('test_only', False)}, warm_cache={hparams.get('warm_cache', True)}")

    # Wipe output_folder unless continuing a previous experiment or test_only
    if not hparams.get("continue_exp", False) and not hparams.get("test_only", False):
        output_folder = Path(hparams["output_folder"])
        if output_folder.exists():
            print(f"Wiping output_folder: {output_folder}")
            shutil.rmtree(output_folder)

    # Data preparation, to be run on only one process.
    # Also skip if all manifests already exist — avoids concurrent jobs
    # racing on the same manifest writes.
    manifests_exist = all(
        Path(hparams[k]).exists()
        for k in ("train_annotation", "val_annotation", "test_annotation")
    )
    if not hparams["skip_prep"] and not manifests_exist:
        # Dynamically load the data preparation module
        try:
            data_io_module = importlib.import_module(hparams["data_io_script"])
        except KeyError:
            sys.exit("Error: 'data_io_script' path must be defined in the YAML file.")

        prepare_data_fn = getattr(data_io_module, hparams["prepare_data_fn"])
        if hparams.get("cross_eval", False):
            print("Running cross-dataset evaluation data prep...")
            sb.utils.distributed.run_on_main(
                prepare_data_fn,
                kwargs={
                    "wav_folder_train": hparams["wav_folder_train"],
                    "metadata_path_train": hparams["metadata_path_train"],
                    "wav_folder_test": hparams["wav_folder_test"],
                    "metadata_path_test": hparams["metadata_path_test"],
                    "manifest_train_path": hparams["train_annotation"],
                    "manifest_val_path": hparams["val_annotation"],
                    "manifest_test_path": hparams["test_annotation"],
                    "ratio": hparams.get("ratio", None),
                    "random_seed": hparams["random_seed"],
                    "dataset": hparams["dataset"],
                    "task": hparams["task"],
                },
            )
        else:
            print("Running standard data prep...")
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
    print("Cache warm complete.")

    # Cache-only mode: stop here so the cache can be reused by later runs.
    if hparams.get("cache_only", False):
        print("[CACHE_ONLY] Cache generation complete; exiting before training.")
        sys.exit(0)

    _project_root = Path.cwd().resolve()
    resolved_paths = _collect_resolved_paths(hparams, _project_root)
    base_output_folder = resolved_paths["output_folder"]

    optim_metric = hparams.get("optim_metric", "F1")
    optim_mode = hparams.get("optim_mode", "max")
    best_config = None
    # Check if hyperparameter optimization is enabled
    if not hparams.get("test_only"):
        # Initialize Ray
        tune_config = hparams.get("ray_tune_config", {})
        resources_per_trial = tune_config.get("resources_per_trial", {"cpu": 1, "gpu": 0})
        ray.init(ignore_reinit_error=True)

        # Parse search space
        search_space = parse_hp_search_space(hparams)

        trainable = tune.with_parameters(
            train_with_ray,
            hparams_file=str(hparams_file),
            run_opts=run_opts,
            overrides=overrides,
            resolved_paths=resolved_paths,
        )

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
            max_concurrent=hparams.get("max_concurrent_trials", 1)
        )

        stopper = tune.stopper.TrialPlateauStopper(
            metric=optim_metric,
            mode=optim_mode,
            grace_period=hparams['hpopt_params']['limit_warmup'],
            num_results=hparams['grace_period'] # correct order, semantics from SB
        )

        storage_path = Path(base_output_folder) / "results"

        if hparams["continue_exp"]:
            print(f"Continuing hyperparameter optimization from {storage_path}")
            resume="AUTO+RESTART_ERRORED"
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
        best_config_path = os.path.join(base_output_folder, "best_hparams.yaml")
        with open(best_config_path, "w") as f:
            yaml.dump(best_config, f)

        ray.shutdown()

    if best_config is None:
        # Try to load best config
        print("Loading best configs")
        best_config_path = os.path.join(base_output_folder, "best_hparams.yaml")
        assert os.path.exists(best_config_path), "Cant find best config"

        with open(best_config_path, "r") as f:
            best_config = yaml.safe_load(f)

    # Re-load the best trial's forked config. This gives the final-test brain
    # the exact same hparams the winning trial was trained under — no drift
    # from edits to the base training/config/*.yaml between HP opt and eval.
    best_trial_dir = Path(base_output_folder) / best_config["trial_id"]
    best_forked_yaml = best_trial_dir / "config" / "main.yaml"
    if best_forked_yaml.exists():
        with open(best_forked_yaml) as fin:
            hparams = load_hyperpyyaml(fin)
    else:
        # Legacy trial dirs predate training.config_fork — no per-trial
        # config/main.yaml exists. Reload base hparams and apply best_config
        # values as overrides. save_folder resolves to the top-level ./save
        # layout these runs used, where the final checkpoint lives.
        print(f"Forked config missing at {best_forked_yaml}; "
              f"falling back to base hparams with best_config overrides")
        legacy_overrides = yaml.dump(
            {k: v for k, v in best_config.items() if k != "trial_id"}
        )
        merged = (overrides or "") + "\n" + legacy_overrides
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, merged)

    # Reuse main-process ``datasets`` — manifests and dataloader options are
    # identical across trials; only lr_s/l2 changed, so the cached dataset
    # pipeline is still valid for test evaluation.

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

    hparams["warm_cache"] = False
    datasets = dataio_prep(hparams)
    brain.evaluate(
        test_set=datasets["test"],
        test_loader_kwargs=hparams["test_dataloader_options"]
    )

    # Write test results to file
    results_path = os.path.join(base_output_folder, "test_results.txt")
    with open(results_path, "w") as f:
        for name, score in brain.test_stats.items():
            f.write(f"{name}: {score}\n")

    # Drop Ray Tune storage now that test eval is on disk.
    ray_storage = Path(base_output_folder) / "results"
    if ray_storage.exists():
        shutil.rmtree(ray_storage, ignore_errors=True)
        print(f"Cleaned {ray_storage}")



