"""Shared hyperparameter optimization utilities."""

from ray import tune


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
