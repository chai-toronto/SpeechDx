"""Single-split trainer with Ray Tune HP search.

Reader-only: this process never imports an encoder module. Cache must be
warm before invocation (use ``sdx warm`` or ``sdx run``); the read-path
pre-flight raises a clear error otherwise.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Ray Tune sets cwd to a per-trial directory by default; that breaks every
# relative path in our hparams. Set this BEFORE importing ray.
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

import ray  # noqa: E402
import speechbrain as sb  # noqa: E402
import yaml  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune import CLIReporter  # noqa: E402
from ray.tune.search.optuna import OptunaSearch  # noqa: E402
from ray.tune.search.searcher import ConcurrencyLimiter  # noqa: E402

from sdx.brain import DiagnosticsBrain  # noqa: E402
from sdx.config import CONFIG_DIR, compose_yaml_text  # noqa: E402
from sdx.config_fork import fork_trial_config  # noqa: E402
from sdx.dataio.read import (  # noqa: E402
    assert_no_encoder_imports,
    build_read_datasets_standard,
    load_manifest_data_standard,
)
from sdx.prep.dispatch import ensure_manifest  # noqa: E402
from sdx.ray_search import (  # noqa: E402
    as_override_dict,
    collect_resolved_paths,
    parse_hp_search_space,
    ray_init_kwargs,
)
from sdx.registry import encoders as registry_encoders  # noqa: E402


def _all_trials_errored(output_folder: Path) -> bool:
    """True iff every Ray Tune trial dir under ``output_folder`` has an
    ``error.txt`` (and there is at least one trial). Used to skip auto-resume
    when a prior run had every trial fail (e.g. systemic cache miss) —
    resuming would just error again, so wipe-and-restart is the right move.
    """
    trial_dirs = [p for p in output_folder.glob("**/hp_optimization/*")
                  if p.is_dir()]
    if not trial_dirs:
        return False
    return all((t / "error.txt").exists() for t in trial_dirs)


def _train_one_trial(config: dict, hparams_file: str, run_opts: dict,
                     overrides: str, resolved_paths: dict) -> None:
    """Ray Tune trainable: fork the base yaml, load it, fit the brain.

    The fork lives at ``<trial_dir>/config/main.yaml`` so a SLURM resume
    sees exactly the config the trial originally launched under (no drift
    from edits to ``sdx/configs/*.yaml`` between runs).
    """
    trial_id = tune.get_context().get_trial_id() or "default"
    trial_dir = Path(resolved_paths["output_folder"]) / trial_id

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

    data_dict = load_manifest_data_standard(hparams)
    datasets = build_read_datasets_standard(data_dict, hparams)

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

    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["val"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )


def cmd_train(task: str, encoder: str, *,
              probe: str = "wavrx", probe_yaml: str = "wavrx.yaml",
              tag: str = "run1", overrides: str = "",
              level_dir: str | None = None) -> None:
    """Run HP search + final eval for one (task, encoder, probe) triple.

    Cache must be warm; ``build_read_datasets_standard``'s pre-flight
    raises ``RuntimeError`` with an actionable message otherwise.

    If ``level_dir`` is given, output paths and manifest paths are
    rerouted from ``./exps/single_task/`` to ``./exps/data_eff/<level_dir>/`` —
    used by ``sdx run-data-eff`` so subsampled-manifest results land
    in the data-efficiency tree (cache paths are unchanged; data-eff
    reuses the full-benchmark encoder cache).
    """
    ensure_manifest(task)

    encoder_yaml = registry_encoders().get(encoder)
    if encoder_yaml is None:
        raise KeyError(
            f"encoder {encoder!r} not in sdx/configs/registry.yaml"
        )

    text = compose_yaml_text(
        model_name=encoder, encoder_yaml=encoder_yaml,
        task_yaml=f"{task}.yaml", probe_yaml=probe_yaml, probe_name=probe,
        experiment_tag=tag,
        warm_cache_override=False, test_only=False, cache_only=False,
        level_dir=level_dir,
    )
    suffix = f"_de{level_dir}" if level_dir else ""
    base_yaml = CONFIG_DIR / f"_tmp_run_sdx_{os.getpid()}{suffix}.yaml"
    base_yaml.write_text(text)

    run_opts: dict = {}
    try:
        with open(base_yaml) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Encoder isolation invariant: trainer process must not have any
        # encoder modules loaded — only stub_encoder, probe, pool.
        assert_no_encoder_imports()

        # ``continue_exp`` is now an internal hparam — auto-set from disk
        # state rather than exposed as a user-facing flag. Three triggers:
        #   1. test_only re-evaluates the saved best trial; never wipe.
        #   2. Partial Tune state on disk (storage/ or best_hparams.yaml
        #      present, but no test_results.{txt,yaml}) → resume, don't
        #      wipe. Reaches here outside test_only when cli.py didn't
        #      already skip-or-overwrite this pair.
        # Otherwise: fresh train, wipe any stale folder.
        output_folder = Path(hparams["output_folder"])
        if hparams.get("test_only", False):
            hparams["continue_exp"] = True
        elif not hparams.get("continue_exp", False) and output_folder.exists():
            storage_path = output_folder / "results"
            has_partial = (storage_path.exists()
                           or (output_folder / "best_hparams.yaml").exists())
            has_complete = ((output_folder / "test_results.txt").exists()
                            or (output_folder / "test_results.yaml").exists())
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
        base_output_folder = resolved_paths["output_folder"]

        optim_metric = hparams.get("optim_metric", "F1")
        optim_mode = hparams.get("optim_mode", "max")
        best_config = None

        if not hparams.get("test_only"):
            tune_config = hparams.get("ray_tune_config", {})
            resources_per_trial = tune_config.get(
                "resources_per_trial", {"cpu": 1, "gpu": 0},
            )
            # Hardened ray.init: CPU cap + per-pid temp dir + no dashboard.
            # See sdx.ray_search.ray_init_kwargs for the rationale (prevents
            # the GCS startup timeout under --workers N and CV per-fold churn).
            ray.init(**ray_init_kwargs())
            try:
                search_space = parse_hp_search_space(hparams)

                trainable = tune.with_parameters(
                    _train_one_trial,
                    hparams_file=str(base_yaml),
                    run_opts=run_opts,
                    overrides=overrides,
                    resolved_paths=resolved_paths,
                )

                task_type = hparams.get("task_type", "B")
                if task_type == "R":
                    metric_cols = ["loss", "MAE", "MSE", "R2", "PearsonR"]
                else:
                    metric_cols = ["F1", "loss", "precision", "recall", "AUROC", "accuracy"]
                reporter = CLIReporter(
                    metric_columns=metric_cols, max_report_frequency=30,
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
                    grace_period=hparams["hpopt_params"]["limit_warmup"],
                    num_results=hparams["grace_period"],
                )

                storage_path = Path(base_output_folder) / "results"
                if hparams["continue_exp"]:
                    print(f"Continuing HP optimization from {storage_path}")
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

                trial_id = analysis.get_best_trial(
                    metric=optim_metric, mode=optim_mode, scope="all",
                ).trial_id
                best_config = analysis.get_best_config(
                    metric=optim_metric, mode=optim_mode, scope="all",
                )
                print(f"\nBest hyperparameters found: {best_config}")
                best_config["trial_id"] = trial_id

                best_config_path = Path(base_output_folder) / "best_hparams.yaml"
                best_config_path.parent.mkdir(parents=True, exist_ok=True)
                best_config_path.write_text(yaml.safe_dump(best_config))
            finally:
                ray.shutdown()

        if best_config is None:
            print("Loading best configs")
            best_config_path = Path(base_output_folder) / "best_hparams.yaml"
            if not best_config_path.exists():
                raise FileNotFoundError(f"can't find best config at {best_config_path}")
            best_config = yaml.safe_load(best_config_path.read_text())

        # Final eval: load the best trial's forked yaml so the brain sees
        # exactly the config the winning trial trained under.
        best_trial_dir = Path(base_output_folder) / best_config["trial_id"]
        best_forked_yaml = best_trial_dir / "config" / "main.yaml"
        if best_forked_yaml.exists():
            with open(best_forked_yaml) as fin:
                hparams = load_hyperpyyaml(fin)
        else:
            print(f"Forked config missing at {best_forked_yaml}; "
                  f"falling back to base hparams + best_config overrides")
            legacy_overrides = yaml.safe_dump(
                {k: v for k, v in best_config.items() if k != "trial_id"}
            )
            merged = (overrides or "") + "\n" + legacy_overrides
            with open(base_yaml) as fin:
                hparams = load_hyperpyyaml(fin, merged)

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

        data_dict = load_manifest_data_standard(hparams)
        datasets = build_read_datasets_standard(data_dict, hparams)
        evaluate_kwargs = {
            "test_set": datasets["test"],
            "test_loader_kwargs": hparams["test_dataloader_options"],
        }
        if optim_mode == "min":
            evaluate_kwargs["min_key"] = optim_metric
        else:
            evaluate_kwargs["max_key"] = optim_metric
        brain.evaluate(**evaluate_kwargs)

        results_path = Path(base_output_folder) / "test_results.txt"
        with results_path.open("w") as f:
            for name, score in brain.test_stats.items():
                f.write(f"{name}: {score}\n")

        ray_storage = Path(base_output_folder) / "results"
        if ray_storage.exists():
            shutil.rmtree(ray_storage, ignore_errors=True)
            print(f"Cleaned {ray_storage}")
    finally:
        base_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    # Convert SLURM SIGTERM to SystemExit so try/finally cleanup runs.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit("Use `python -m sdx train <task> <encoder>` instead.")
