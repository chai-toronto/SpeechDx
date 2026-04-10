#!/usr/bin/env python3
"""
Per-fold cross-validation with independent HP optimization per fold.

For each fold, runs a separate Ray Tune / Optuna HP search, then
aggregates the best validation metrics across folds (mean +/- std).
No test-set evaluation.

Usage:
    python -m training.trainPerFoldCV training/config/main.yaml

Authors
--
Larry Kieu 2026
"""
import importlib
import json
import shutil
import sys
import os
from pathlib import Path

import numpy as np
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
from training.hp_utils import parse_hp_search_space

def dataio_prep(hparams):
    # Retrieve the data
    with open(hparams["train_annotation"], "r") as f:
        train_folds = json.load(f)

    with open(hparams["val_annotation"], "r") as f:
        val_folds = json.load(f)

    data_dict = {}
    for i in range(hparams['data_params']['num_fold']):
        data_dict[f'train_{i}'] = train_folds[i]
        data_dict[f'val_{i}'] = val_folds[i]

    data_dict['all'] = train_folds[0] | val_folds[0] # For cache purpose

    datasets = master_dataio_prep(data_dict, hparams)

    return datasets


def train_fold_with_ray(config, hparams_file, run_opts, overrides, project_root, fold_idx):
    """Ray Tune trainable for a single fold.

    Same as train.py's train_with_ray but uses fold-specific datasets.
    """
    ray_overrides = overrides.copy() if overrides else {}
    for key, value in config.items():
        ray_overrides[key] = value

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, ray_overrides)

    # Resolve relative paths for Ray workers
    root = Path(project_root)
    for key, val in hparams.items():
        if isinstance(val, str) and not Path(val).is_absolute() and (
            key.endswith("_dir") or key.endswith("_folder")
            or key.endswith("_path") or key.endswith("_annotation")
        ):
            hparams[key] = str(root / val)

    trial_id = tune.get_context().get_trial_id() or "default"
    hparams["output_folder"] = os.path.join(hparams["output_folder"], f"fold_{fold_idx}", trial_id)
    hparams["save_folder"] = os.path.join(hparams["save_folder"], f"fold_{fold_idx}", trial_id)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=ray_overrides,
    )

    hparams["warm_cache"] = False

    datasets = dataio_prep(hparams)

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
        train_set=datasets[f"train_{fold_idx}"],
        valid_set=datasets[f"val_{fold_idx}"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )


def run_fold_hp_optimization(fold_idx, hparams, hparams_file, run_opts, overrides,
                             project_root, search_space):
    """Run a complete HP optimization for one fold. Returns (best_config, best_metrics)."""

    optim_metric = hparams.get("optim_metric", "F1")
    optim_mode = hparams.get("optim_mode", "max")

    tune_config = hparams.get("ray_tune_config", {})
    resources_per_trial = tune_config.get("resources_per_trial", {"cpu": 1, "gpu": 0})

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [
                "data/", "exps/", "tmp/", ".idea/",
                "*.DS_Store", "uv.lock", "pyproject.toml",
                ".python-version", "exps_old/", "data_extra/", "data_zip/", "metadata/"
            ],
        },
    )

    trainable = tune.with_parameters(
        train_fold_with_ray,
        hparams_file=hparams_file,
        run_opts=run_opts,
        overrides=overrides,
        project_root=project_root,
        fold_idx=fold_idx,
    )

    reporter = CLIReporter(
        metric_columns=["F1", "loss", "precision", "recall", "AUROC", "accuracy"],
        max_report_frequency=30,
    )

    optuna_search = OptunaSearch(
        metric=optim_metric,
        mode=optim_mode,
    )

    search_alg = ConcurrencyLimiter(
        optuna_search,
        max_concurrent=hparams.get("max_concurrent_trials", 4),
    )

    stopper = tune.stopper.TrialPlateauStopper(
        metric=optim_metric,
        mode=optim_mode,
        grace_period=hparams.get("limit_warmup", 2),
        num_results=hparams.get("grace_period", 5),
    )

    storage_path = (Path(hparams["output_folder"]) / "ray_results" / f"fold_{fold_idx}").resolve()

    if hparams.get("continue_exp", False):
        print(f"Fold {fold_idx}: Continuing HP optimization from {storage_path}")
        resume = "AUTO"
    else:
        resume = False
        if storage_path.exists():
            shutil.rmtree(storage_path)

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

    best_trial = analysis.get_best_trial(metric=optim_metric, mode=optim_mode, scope="all")
    best_config = analysis.get_best_config(metric=optim_metric, mode=optim_mode, scope="all")

    # Get the metrics from the best-performing epoch, not just the last one
    df = best_trial.dataframe(metric=optim_metric, mode=optim_mode)
    if optim_mode == "min":
        best_row = df.loc[df[optim_metric].idxmin()]
    else:
        best_row = df.loc[df[optim_metric].idxmax()]
    best_metrics = best_row.to_dict()

    print(f"\nFold {fold_idx} best config: {best_config}")
    print(f"Fold {fold_idx} best {optim_metric}: {best_metrics.get(optim_metric, 'N/A')}")

    # Save per-fold best config
    fold_config_path = os.path.join(hparams["output_folder"], f"best_hparams_fold_{fold_idx}.yaml")
    with open(fold_config_path, "w") as f:
        yaml.dump(best_config, f)

    ray.shutdown()
    return best_config, best_metrics


# Recipe begins!
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    if os.environ.get("WORLD_SIZE"):
        sb.utils.distributed.ddp_init_group(run_opts)

    hparams_file = Path(hparams_file).resolve()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    # Data preparation (manifest generation) — must run before dataio_prep
    if not hparams["skip_prep"]:
        # Dynamically load data preparation module
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
                "random_seed": hparams["random_seed"],
                "raw_label_key": hparams['data_params']["raw_label_key"],
                "num_fold": hparams['data_params']["num_fold"],
            },
        )

    # Warm cache early so Ray workers can open read-only
    datasets = dataio_prep(hparams)

    project_root = str(Path.cwd().resolve())
    search_space = parse_hp_search_space(hparams)
    num_folds = hparams['data_params']["num_fold"]

    optim_metric = hparams.get("optim_metric", "F1")

    all_best_configs = {}
    all_best_metrics = {}

    print(f"\n{'='*60}")
    print(f"Per-Fold HP Optimization: {num_folds} folds")
    print(f"{'='*60}\n")

    for fold_idx in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Starting HP optimization for fold {fold_idx}/{num_folds - 1}")
        print(f"{'='*60}\n")

        best_config, best_metrics = run_fold_hp_optimization(
            fold_idx=fold_idx,
            hparams=hparams,
            hparams_file=hparams_file,
            run_opts=run_opts,
            overrides=overrides,
            project_root=project_root,
            search_space=search_space,
        )

        all_best_configs[fold_idx] = best_config
        all_best_metrics[fold_idx] = best_metrics

    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("Per-Fold HP Optimization Complete — Aggregated Results")
    print(f"{'='*60}\n")

    # Collect all metric keys from the first fold
    metric_keys = [k for k in all_best_metrics[0].keys()
                   if isinstance(all_best_metrics[0][k], (int, float))]

    summary = {}
    for key in metric_keys:
        values = [all_best_metrics[i][key] for i in range(num_folds)
                  if key in all_best_metrics[i]]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[key] = {"mean": float(mean_val), "std": float(std_val)}
            print(f"  {key}: {mean_val:.4f} +/- {std_val:.4f}")

    # Per-fold detail
    print(f"\nPer-fold {optim_metric}:")
    for i in range(num_folds):
        val = all_best_metrics[i].get(optim_metric, "N/A")
        print(f"  Fold {i}: {val}")

    # Save summary
    summary_path = os.path.join(hparams["output_folder"], "test_results.yaml")
    fold_detail = {
        f"fold_{i}": {
            "best_config": all_best_configs[i],
            "best_metrics": {k: float(v) for k, v in all_best_metrics[i].items()
                            if isinstance(v, (int, float))},
        }
        for i in range(num_folds)
    }
    with open(summary_path, "w") as f:
        yaml.dump({"summary": summary, "folds": fold_detail}, f, default_flow_style=False)

    print(f"\nSummary saved to {summary_path}")
