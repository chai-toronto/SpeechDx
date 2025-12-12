"""
Hydra-compatible structured configs for JEPA transformer components.

These dataclasses define the configuration schema for transformer encoder/decoder
layers and can be instantiated directly via Hydra from YAML configs.
"""

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn


class ForwardReturn(TypedDict):
    """Return type for JEPA forward pass. Kept as TypedDict for type checking."""
    local_features: torch.Tensor
    contextual_features: torch.Tensor
    reconstruction_loss: float
    codebook_entropy_loss: float
    loss: float
    preds: torch.Tensor
    targets: torch.Tensor
    idxs_context: torch.Tensor
    target_masks: torch.BoolTensor


@dataclass
class TransformerLayerCFG:
    """
    Configuration for a single transformer layer (encoder or decoder).

    Compatible with nn.TransformerEncoderLayer and nn.TransformerDecoderLayer.
    The activation parameter accepts string names (e.g., "gelu", "relu") which
    will be converted to nn.Module instances by the JEPA class.
    """

    d_model: int = 768
    """Embedding dimension."""

    nhead: int = 12
    """Number of attention heads."""

    batch_first: bool = True
    """If True, input shape is (batch, seq, feature)."""

    norm_first: bool = False
    """If True, layer norm is done prior to attention and feedforward operations (pre-norm)."""

    bias: bool = True
    """Whether to use bias in linear layers."""

    mlp_ratio: float = 4.0
    """Ratio of feedforward dimension to d_model (dim_feedforward = d_model * mlp_ratio)."""

    dropout: float = 0.0
    """Dropout probability."""

    activation: str = "gelu"
    """Activation function name (e.g., 'gelu', 'relu', 'silu')."""

    layer_norm_eps: float = 1e-6
    """Epsilon for layer normalization."""

    @property
    def dim_feedforward(self) -> int:
        """Computed feedforward dimension based on mlp_ratio."""
        return int(self.d_model * self.mlp_ratio)

    def to_layer_kwargs(self) -> dict:
        """
        Convert to kwargs dict for nn.TransformerEncoderLayer/DecoderLayer.

        Handles activation string -> nn.Module conversion.
        """
        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }

        activation_fn = activation_map.get(self.activation.lower())
        if activation_fn is None:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Supported: {list(activation_map.keys())}"
            )

        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "batch_first": self.batch_first,
            "norm_first": self.norm_first,
            "bias": self.bias,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": activation_fn,
            "layer_norm_eps": self.layer_norm_eps,
        }


@dataclass
class TransformerEncoderCFG:
    """
    Configuration for nn.TransformerEncoder.

    Note: The norm parameter must be set separately when instantiating the encoder.
    """

    num_layers: int = 12
    """Number of transformer layers."""

    enable_nested_tensor: bool = False
    """Whether to use nested tensor optimization."""

    mask_check: bool = True
    """Whether to check mask validity."""

    def to_encoder_kwargs(self) -> dict:
        """Convert to kwargs dict for nn.TransformerEncoder."""
        return {
            "num_layers": self.num_layers,
            "enable_nested_tensor": self.enable_nested_tensor,
            "mask_check": self.mask_check,
        }
