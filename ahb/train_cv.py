"""Per-fold CV trainer with independent HP optimization per fold.

For each fold runs a separate Ray Tune / Optuna HP search, then aggregates
the best validation metrics across folds (mean ± std) into
``test_results.yaml``. No test-set evaluation (CV reports cross-fold
uncertainty as the equivalent of a held-out test).

Reader-only: cache must already be warm (use ``ahb warm``).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

import numpy as np  # noqa: E402
import ray  # noqa: E402
import speechbrain as sb  # noqa: E402
import yaml  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune import CLIReporter  # noqa: E402
from ray.tune.search.optuna import OptunaSearch  # noqa: E402
from ray.tune.search.searcher import ConcurrencyLimiter  # noqa: E402

from ahb.brain import DiagnosticsBrain  # noqa: E402
from ahb.config import CONFIG_DIR, compose_yaml_text  # noqa: E402
from ahb.config_fork import fork_trial_config  # noqa: E402
from ahb.dataio.read import (  # noqa: E402
    assert_no_encoder_imports,
    build_read_datasets_cv,
    load_manifest_data_cv,
)
from ahb.prep.dispatch import ensure_manifest  # noqa: E402
from ahb.ray_search import (  # noqa: E402
    as_override_dict,
    collect_resolved_paths,
    parse_hp_search_space,
)
from ahb.registry import encoders as registry_encoders  # noqa: E402


def _all_trials_errored(output_folder: Path) -> bool:
    """True iff every Ray Tune trial dir under ``output_folder`` has an
    ``error.txt`` (and there is at least one trial). Used to skip auto-resume
    when every trial in the partial state failed — resuming would error again,
    so wipe-and-restart is the right move.
    """
    trial_dirs = [p for p in output_folder.glob("**/hp_optimization/*")
                  if p.is_dir()]
    if not trial_dirs:
        return False
    return all((t / "error.txt").exists() for t in trial_dirs)


def _train_fold_trial(config: dict, hparams_file: str, run_opts: dict,
                      overrides: str, resolved_paths: dict, fold_idx: int) -> None:
    """Ray Tune trainable for a single fold.

    Same fork-based flow as ``ahb/train.py:_train_one_trial`` but the trial
    dir nests under ``fold_<idx>/`` so each fold's HP search is isolated.
    """
    trial_id = tune.get_context().get_trial_id() or "default"
    trial_dir = Path(resolved_paths["output_folder"]) / f"fold_{fold_idx}" / trial_id

    fork_overrides = {}
    fork_overrides.update(resolved_paths)
    fork_overrides.update(as_override_dict(overrides))
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

    train_folds, val_folds = load_manifest_data_cv(hparams)
    datasets = build_read_datasets_cv(
        train_folds[fold_idx], val_folds[fold_idx], hparams,
    )

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables=hparams["checkpointer"].recoverables,
    )

    brain = DiagnosticsBrain(
        fold_idx=fold_idx,
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


def _run_fold_hp_optimization(fold_idx, hparams, hparams_file, run_opts,
                              overrides, resolved_paths, search_space):
    """One full HP search for one fold. Returns (best_config, best_metrics)."""
    optim_metric = hparams.get("optim_metric", "F1")
    optim_mode = hparams.get("optim_mode", "max")

    tune_config = hparams.get("ray_tune_config", {})
    resources_per_trial = tune_config.get(
        "resources_per_trial", {"cpu": 1, "gpu": 0},
    )

    ray.init(ignore_reinit_error=True)
    try:
        trainable = tune.with_parameters(
            _train_fold_trial,
            hparams_file=str(hparams_file),
            run_opts=run_opts,
            overrides=overrides,
            resolved_paths=resolved_paths,
            fold_idx=fold_idx,
        )

        reporter = CLIReporter(
            metric_columns=["F1", "loss", "precision", "recall", "AUROC", "accuracy"],
            max_report_frequency=30,
        )

        optuna_search = OptunaSearch(
            metric=optim_metric, mode=optim_mode,
            seed=hparams.get("hp_search_seed"),
        )
        search_alg = ConcurrencyLimiter(
            optuna_search,
            max_concurrent=hparams.get("max_concurrent_trials", 1),
        )
        stopper = tune.stopper.TrialPlateauStopper(
            metric=optim_metric, mode=optim_mode,
            grace_period=hparams.get("limit_warmup", 2),
            num_results=hparams.get("grace_period", 5),
        )

        storage_path = (
            Path(resolved_paths["output_folder"]) / "ray_results" / f"fold_{fold_idx}"
        )
        if hparams.get("continue_exp", False):
            print(f"Fold {fold_idx}: continuing HP search from {storage_path}")
            resume = "AUTO+RESTART_ERRORED"
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

        best_trial = analysis.get_best_trial(
            metric=optim_metric, mode=optim_mode, scope="all",
        )
        best_config = analysis.get_best_config(
            metric=optim_metric, mode=optim_mode, scope="all",
        )

        # Match by trial_id because best_trial.local_path may point at a Ray
        # session artifacts dir, while trial_dataframes is keyed by storage logdir.
        df = None
        for logdir, trial_df in analysis.trial_dataframes.items():
            if best_trial.trial_id in logdir:
                df = trial_df
                break
        if df is None:
            raise KeyError(
                f"Could not find trial dataframe for best trial {best_trial.trial_id}"
            )
        if optim_mode == "min":
            best_row = df.loc[df[optim_metric].idxmin()]
        else:
            best_row = df.loc[df[optim_metric].idxmax()]
        best_metrics = best_row.to_dict()

        print(f"\nFold {fold_idx} best config: {best_config}")
        print(f"Fold {fold_idx} best {optim_metric}: {best_metrics.get(optim_metric, 'N/A')}")

        # Persist trial_id so test_only can locate the winning trial's
        # forked yaml + checkpointer save_folder for re-evaluation.
        best_config["trial_id"] = best_trial.trial_id

        fold_config_path = (
            Path(resolved_paths["output_folder"]) / f"best_hparams_fold_{fold_idx}.yaml"
        )
        fold_config_path.parent.mkdir(parents=True, exist_ok=True)
        fold_config_path.write_text(yaml.safe_dump(best_config))
    finally:
        ray.shutdown()
    return best_config, best_metrics


def _evaluate_fold_test_only(*, fold_idx: int, output_folder: Path,
                             run_opts: dict, optim_metric: str,
                             optim_mode: str) -> tuple[dict, dict]:
    """Reload one fold's best trial and re-run val-set evaluation.

    Mirrors train.py's test_only path: loads ``fold_<i>/<trial_id>/config/main.yaml``
    (the forked yaml the winning trial trained under), instantiates the brain
    with the trial's checkpointer, and runs ``brain.evaluate`` against the val
    set (CV's "test" — there is no held-out test split). Returns the best_config
    plus brain.test_stats so the caller can re-aggregate test_results.yaml.

    Errors out for legacy CV runs whose ``best_hparams_fold_<i>.yaml`` lacks
    ``trial_id``: those need to be re-trained once before test-only works.
    """
    fold_config_path = output_folder / f"best_hparams_fold_{fold_idx}.yaml"
    if not fold_config_path.exists():
        raise FileNotFoundError(
            f"Fold {fold_idx}: missing {fold_config_path}; re-train this run before --test-only."
        )
    best_config = yaml.safe_load(fold_config_path.read_text()) or {}
    trial_id = best_config.get("trial_id")
    if not trial_id:
        raise KeyError(
            f"Fold {fold_idx}: {fold_config_path} has no trial_id (legacy run); "
            f"re-train this CV experiment to add trial_id, then retry --test-only."
        )

    trial_dir = output_folder / f"fold_{fold_idx}" / trial_id
    forked_yaml = trial_dir / "config" / "main.yaml"
    if not forked_yaml.exists():
        raise FileNotFoundError(
            f"Fold {fold_idx}: forked config missing at {forked_yaml}; re-train required."
        )

    with open(forked_yaml) as fin:
        hparams = load_hyperpyyaml(fin)

    train_folds, val_folds = load_manifest_data_cv(hparams)
    datasets = build_read_datasets_cv(
        train_folds[fold_idx], val_folds[fold_idx], hparams,
    )

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables=hparams["checkpointer"].recoverables,
    )

    brain = DiagnosticsBrain(
        fold_idx=fold_idx,
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    evaluate_kwargs = {
        "test_set": datasets["val"],
        "test_loader_kwargs": hparams["val_dataloader_options"],
    }
    if optim_mode == "min":
        evaluate_kwargs["min_key"] = optim_metric
    else:
        evaluate_kwargs["max_key"] = optim_metric
    brain.evaluate(**evaluate_kwargs)

    return best_config, dict(brain.test_stats)


def cmd_train_cv(task: str, encoder: str, *,
                 probe: str = "AvgTProbe", probe_yaml: str = "Probe.yaml",
                 tag: str = "run1", overrides: str = "",
                 level_dir: str | None = None) -> None:
    """Run per-fold HP optimization for one (task, encoder, probe).

    Cache must be warm; ``build_read_datasets_cv``'s pre-flight raises
    ``RuntimeError`` with an actionable message otherwise.

    See ``ahb.train.cmd_train`` for ``level_dir``.
    """
    ensure_manifest(task)

    encoder_yaml = registry_encoders().get(encoder)
    if encoder_yaml is None:
        raise KeyError(f"encoder {encoder!r} not in ahb/configs/registry.yaml")

    text = compose_yaml_text(
        model_name=encoder, encoder_yaml=encoder_yaml,
        task_yaml=f"{task}.yaml", probe_yaml=probe_yaml, probe_name=probe,
        experiment_tag=tag,
        warm_cache_override=False, test_only=False, cache_only=False,
        level_dir=level_dir,
    )
    suffix = f"_de{level_dir}" if level_dir else ""
    base_yaml = CONFIG_DIR / f"_tmp_run_ahb_cv_{os.getpid()}{suffix}.yaml"
    base_yaml.write_text(text)

    run_opts: dict = {}
    try:
        with open(base_yaml) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        assert_no_encoder_imports()

        # ``continue_exp`` is auto-set from disk state — see train.py for
        # the full rationale. CV variant: partial state means at least one
        # fold has a Tune storage; resume rather than wipe.
        output_folder = Path(hparams["output_folder"])
        if hparams.get("test_only", False):
            hparams["continue_exp"] = True
        elif not hparams.get("continue_exp", False) and output_folder.exists():
            has_partial = (any(output_folder.glob("fold_*/results"))
                           or any(output_folder.glob("fold_*/best_hparams.yaml")))
            has_complete = (output_folder / "test_results.yaml").exists()
            if has_partial and not has_complete:
                if _all_trials_errored(output_folder):
                    print(f"All previous Tune trials errored at "
                          f"{output_folder}; wiping for a fresh run.")
                else:
                    print(f"Auto-resuming partial Tune state at {output_folder}")
                    hparams["continue_exp"] = True

        if not hparams.get("continue_exp", False):
            if output_folder.exists():
                print(f"Wiping output_folder: {output_folder}")
                shutil.rmtree(output_folder)

        project_root = Path.cwd().resolve()
        resolved_paths = collect_resolved_paths(hparams, project_root)
        num_folds = hparams["data_params"]["num_fold"]
        optim_metric = hparams.get("optim_metric", "F1")
        optim_mode = hparams.get("optim_mode", "max")

        all_best_configs: dict = {}
        all_best_metrics: dict = {}

        if hparams.get("test_only", False):
            print(f"\n{'='*60}\nPer-fold test-only re-evaluation: {num_folds} folds\n{'='*60}\n")
            for fold_idx in range(num_folds):
                print(f"\n{'='*60}\nFold {fold_idx}/{num_folds - 1}\n{'='*60}\n")
                best_config, best_metrics = _evaluate_fold_test_only(
                    fold_idx=fold_idx,
                    output_folder=Path(resolved_paths["output_folder"]),
                    run_opts=run_opts,
                    optim_metric=optim_metric, optim_mode=optim_mode,
                )
                all_best_configs[fold_idx] = best_config
                all_best_metrics[fold_idx] = best_metrics
        else:
            search_space = parse_hp_search_space(hparams)
            print(f"\n{'='*60}\nPer-fold HP search: {num_folds} folds\n{'='*60}\n")
            for fold_idx in range(num_folds):
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx}/{num_folds - 1}")
                print(f"{'='*60}\n")

                best_config, best_metrics = _run_fold_hp_optimization(
                    fold_idx=fold_idx, hparams=hparams,
                    hparams_file=base_yaml, run_opts=run_opts,
                    overrides=overrides, resolved_paths=resolved_paths,
                    search_space=search_space,
                )
                all_best_configs[fold_idx] = best_config
                all_best_metrics[fold_idx] = best_metrics

        print(f"\n{'='*60}\nAggregated results\n{'='*60}\n")

        metric_keys = [k for k in all_best_metrics[0].keys()
                       if isinstance(all_best_metrics[0][k], (int, float))]
        summary: dict = {}
        for key in metric_keys:
            values = [all_best_metrics[i][key] for i in range(num_folds)
                      if key in all_best_metrics[i]]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                print(f"  {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")

        print(f"\nPer-fold {optim_metric}:")
        for i in range(num_folds):
            print(f"  Fold {i}: {all_best_metrics[i].get(optim_metric, 'N/A')}")

        summary_path = Path(resolved_paths["output_folder"]) / "test_results.yaml"
        fold_detail = {
            f"fold_{i}": {
                "best_config": all_best_configs[i],
                "best_metrics": {
                    k: float(v) for k, v in all_best_metrics[i].items()
                    if isinstance(v, (int, float))
                },
            }
            for i in range(num_folds)
        }
        summary_path.write_text(
            yaml.safe_dump({"summary": summary, "folds": fold_detail},
                           default_flow_style=False)
        )
        print(f"\nSummary saved to {summary_path}")

        ray_results_dir = Path(resolved_paths["output_folder"]) / "ray_results"
        if ray_results_dir.exists():
            shutil.rmtree(ray_results_dir, ignore_errors=True)
            print(f"Cleaned {ray_results_dir}")
    finally:
        base_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit("Use `python -m ahb single train -t <task> -e <encoder>` instead "
                     "(auto-routes to per-fold CV when the task yaml sets `num_fold`).")
