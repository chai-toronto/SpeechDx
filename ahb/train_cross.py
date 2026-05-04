"""Cross-task trainer with Ray Tune HP search.

Same shape as ``ahb.train.cmd_train`` but uses ``build_read_datasets_cross``
(three caches) and the ``main_cross.yaml`` base. Default probe is ``Probe``
(not ``AvgTProbe``) — matches the legacy ``run_all_cross.py`` behavior.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

import ray  # noqa: E402
import speechbrain as sb  # noqa: E402
import yaml  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune import CLIReporter  # noqa: E402
from ray.tune.search.optuna import OptunaSearch  # noqa: E402
from ray.tune.search.searcher import ConcurrencyLimiter  # noqa: E402

from ahb.brain import DiagnosticsBrain  # noqa: E402
from ahb.config import (  # noqa: E402
    CONFIG_DIR,
    MAIN_CROSS,
    MAIN_CROSS_CATEGORY,
    compose_yaml_text,
)
from ahb.config_fork import fork_trial_config  # noqa: E402
from ahb.dataio.read import (  # noqa: E402
    assert_no_encoder_imports,
    build_read_datasets_category,
    build_read_datasets_cross,
    load_manifest_data_cross,
)
from ahb.prep.dispatch import ensure_manifest  # noqa: E402
from ahb.ray_search import (  # noqa: E402
    as_override_dict,
    collect_resolved_paths,
    parse_hp_search_space,
)
from ahb.registry import encoders as registry_encoders  # noqa: E402


def _train_one_trial_cross(config: dict, hparams_file: str, run_opts: dict,
                           overrides: str, resolved_paths: dict) -> None:
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

    data_dict = load_manifest_data_cross(hparams)
    if hparams.get("cross_eval_category"):
        datasets = build_read_datasets_category(data_dict, hparams)
    else:
        datasets = build_read_datasets_cross(data_dict, hparams)

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


def cmd_train_cross(task: str, encoder: str, *,
                    probe: str = "Probe", probe_yaml: str = "Probe.yaml",
                    tag: str = "run1", overrides: str = "") -> None:
    """Cross-task HP search + final eval for one (task, encoder, probe)."""
    ensure_manifest(task)

    encoder_yaml = registry_encoders().get(encoder)
    if encoder_yaml is None:
        raise KeyError(f"encoder {encoder!r} not in ahb/configs/registry.yaml")

    task_yaml_path = CONFIG_DIR / "cross_tasks" / f"{task}.yaml"
    is_category = "\ntrain_datasets:" in task_yaml_path.read_text()
    base_main_yaml = MAIN_CROSS_CATEGORY if is_category else MAIN_CROSS
    text = compose_yaml_text(
        model_name=encoder, encoder_yaml=encoder_yaml,
        task_yaml=f"{task}.yaml", probe_yaml=probe_yaml, probe_name=probe,
        experiment_tag=tag,
        warm_cache_override=False, test_only=False, cache_only=False,
        base_main_yaml=base_main_yaml,
        task_include_prefix="cross_tasks",
    )
    base_yaml = CONFIG_DIR / f"_tmp_run_ahb_cross_{os.getpid()}.yaml"
    base_yaml.write_text(text)

    run_opts: dict = {}
    try:
        with open(base_yaml) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        assert_no_encoder_imports()

        # test_only re-evaluates the saved best trial; never wipe under it.
        if hparams.get("test_only", False):
            hparams["continue_exp"] = True

        if not hparams.get("continue_exp", False):
            output_folder = Path(hparams["output_folder"])
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
            ray.init(ignore_reinit_error=True)
            try:
                search_space = parse_hp_search_space(hparams)

                trainable = tune.with_parameters(
                    _train_one_trial_cross,
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
                    trainable, config=search_space,
                    num_samples=tune_config.get("num_samples", 10),
                    resume=resume, stop=stopper, progress_reporter=reporter,
                    storage_path=storage_path.as_posix(),
                    name="hp_optimization",
                    search_alg=search_alg, resources_per_trial=resources_per_trial,
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
            modules=hparams["modules"], opt_class=hparams["opt_class"],
            hparams=hparams, run_opts=run_opts, checkpointer=checkpointer,
        )

        data_dict = load_manifest_data_cross(hparams)
        if hparams.get("cross_eval_category"):
            datasets = build_read_datasets_category(data_dict, hparams)
        else:
            datasets = build_read_datasets_cross(data_dict, hparams)
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
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit("Use `python -m ahb cross train <task> <encoder>` instead.")
