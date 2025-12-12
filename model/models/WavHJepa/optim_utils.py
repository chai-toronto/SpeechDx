"""
Optimizer utilities for per-parameter learning rate modulation.

Adapted from H-Net's parameter annotation pattern for hierarchical models.
"""

import torch


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """
    Annotates a parameter with optimization parameters.
    Updates the parameter's _optim attribute with the given kwargs.

    Args:
        param: Parameter tensor to annotate
        **kwargs: Optimization params (e.g., lr_multiplier=0.5, weight_decay=0.01)

    Example:
        >>> apply_optimization_params(param, lr_multiplier=0.5)
        >>> apply_optimization_params(param, weight_decay=0.0)
    """
    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs


def group_params_by_optim(model: torch.nn.Module, base_lr: float, base_wd: float) -> list[dict]:
    """
    Creates parameter groups for the optimizer based on _optim annotations.

    This function groups parameters with identical optimization settings together,
    automatically applying special rules (e.g., zero weight decay for biases/norms).

    Args:
        model: Model with annotated parameters
        base_lr: Base learning rate
        base_wd: Base weight decay

    Returns:
        List of parameter group dicts with format:
        [{"params": [...], "lr": float, "weight_decay": float}, ...]

    Example:
        >>> from models.WavHJepa.optim_utils import apply_optimization_params, group_params_by_optim
        >>> # Annotate some parameters
        >>> for param in model.encoder.parameters():
        ...     apply_optimization_params(param, lr_multiplier=1.0)
        >>> for param in model.decoder.parameters():
        ...     apply_optimization_params(param, lr_multiplier=0.5)
        >>> # Create parameter groups
        >>> param_groups = group_params_by_optim(model, base_lr=2e-4, base_wd=0.01)
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    param_groups = []
    all_keys = set()

    # First pass: annotate special cases and collect all keys
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases and layer norms get zero weight decay by default
        if name.endswith(".bias") or ".norm." in name or "layer_norm" in name:
            apply_optimization_params(param, weight_decay=0.0)

        # Collect all _optim keys
        if hasattr(param, "_optim"):
            all_keys.update(param._optim.keys())

    all_keys = list(all_keys)
    all_tuples = []

    # Second pass: group parameters with identical settings
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Get current parameter's optimization settings
        if not hasattr(param, "_optim"):
            param._optim = {}

        # Create tuple of settings for grouping
        current_tuple = tuple(param._optim.get(key, None) for key in all_keys)

        # Compute effective lr and weight decay
        lr_multiplier = param._optim.get("lr_multiplier", 1.0)
        weight_decay = param._optim.get("weight_decay", base_wd)
        effective_lr = base_lr * lr_multiplier

        # Find or create group
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            param_groups.append({
                "params": [param],
                "lr": effective_lr,
                "weight_decay": weight_decay,
            })
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)

    return param_groups
