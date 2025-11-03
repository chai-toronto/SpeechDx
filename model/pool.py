from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentiveTemporalPoolLite(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn  = nn.Linear(input_dim, 1)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        scores = torch.tanh(self.attn(x)).squeeze(-1)               # (B, T_max)

        return attn_pool(lengths, scores, x)

def attn_pool(lengths, scores, x):
    """
    scores: (B, T_max) unnormalized attention scores
    x: (B, T_max, D) input features
    lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
    Returns:
        pooled:  (B, D)
    """
    T_max = scores.size(-1)
    if lengths is None:
        # No padding case
        weights = F.softmax(scores, dim=1)  # (B, T_max)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled

    lengths = (lengths * T_max).long()  # Convert into absolute lengths
    lengths = lengths.clamp(min=1, max=T_max)      # avoid zero-lengths

    # Build mask: True for real tokens
    t = torch.arange(T_max, device=x.device).unsqueeze(0)  # (1, T_max)
    mask = t < lengths.unsqueeze(1)  # (B, T_max)

    # Mask BEFORE softmax so pads get zero probability
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_scores, dim=1)  # (B, T_max)

    pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
    return pooled


class AttentiveTemporalPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn  = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        h = torch.tanh(self.attn(x))           # (B, T_max, H)
        scores = self.score(h).squeeze(-1)     # (B, T_max)

        return attn_pool(lengths, scores, x)

class AvgTPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        if lengths is None:
            return x.mean(dim=-2)
        else:
            B, T_max, D = x.shape
            lengths = (lengths * T_max).long()  # Convert into absolute lengths
            lengths = lengths.clamp(min=1)      # avoid div by zero
            sum_x = x.sum(dim=-2)         # (B, D)
            pooled = sum_x / lengths.unsqueeze(1)  # (B, D)
            return pooled


class LayerWeightedAvgPool(nn.Module):
    """
    Learnable softmax weights over layers to pool across the layer dimension.

    Args:
        num_layers (int): number of layers (L).
        init (str): 'uniform' (default) or 'last' to bias toward higher layers.
        temperature (float): softmax temperature; <1.0 makes weights peakier.
        learnable (bool): if False, keeps uniform fixed weights.

    Forward:
        x: Tuples of each layer (B, T_max, D) tensor of batch x time x features.
        layer_mask: optional boolean mask of shape (L,) where False drops a layer
        return_weights: if True, also returns the normalized weights (L,)

    Returns:
        pooled: x with the layer dimension removed (weighted average over L)
        (optionally) weights: the softmax weights over layers (L,)
    """
    def __init__(self, num_layers, layer_dim = 1, init="uniform", temperature=1.0, learnable=True):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = float(temperature)

        # logits -> softmax -> weights
        logits = torch.rand(num_layers)
        if init == "last":
            # bias toward deeper layers (monotonic increasing logits)
            logits = torch.linspace(-1.0, 1.0, steps=num_layers)

        self.logits = nn.Parameter(logits, requires_grad=learnable)

    def forward(self, x: Tuple[Tensor], layer_mask=None, return_weights=False):
        # Get the number of layers present in x along the chosen dimension
        num_layer = len(x)
        assert num_layer == self.num_layers, f"Expected L={self.num_layers}, got L={num_layer}"

        logits = self.logits / self.temperature

        if layer_mask is not None:
            # layer_mask: bool or {0,1} of shape (L,)
            mask = layer_mask.to(dtype=torch.bool)
            if mask.shape != (num_layer,):
                raise ValueError(f"layer_mask must have shape (L,), got {mask.shape}")
            # Exclude masked layers by setting their logit to -inf before softmax
            logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        w = F.softmax(logits, dim=0)  # (L,)

        pooled = torch.zeros_like(x[0])  # (B, T_max, D)
        for wi, xi in zip(w, x):
            pooled.add_(xi, alpha=wi.item())

        return (pooled, w) if return_weights else pooled


