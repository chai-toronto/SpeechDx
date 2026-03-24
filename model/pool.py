from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from torch import Tensor


class AttentiveTemporalPoolLite(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn  = nn.Linear(input_dim, 1)
        self.last_scores = None
        self.input_dim = self.output_dim = input_dim

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        scores = torch.tanh(self.attn(x)).squeeze(-1)               # (B, T_max)
        self.last_scores = scores.clone().detach()
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
    weights = F.softmax(masked_scores, dim=-1)  # (B, T_max)

    pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
    return pooled


class AttentiveTemporalPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn  = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.input_dim = self.output_dim = input_dim

        self.last_scores = None

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        h = F.gelu(self.attn(x))           # (B, T_max, H)
        scores = self.score(h).squeeze(-1)     # (B, T_max)

        self.last_scores = scores.clone().detach()
        return attn_pool(lengths, scores, x)

class AvgTPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = self.output_dim = input_dim

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

class ASP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        if isinstance(input_dim, float):
            input_dim_int = int(input_dim)
            assert input_dim - input_dim_int == 0, "Float input_dim can't be cast to int"
            input_dim = input_dim_int

        self.pool = AttentiveStatisticsPooling(input_dim, attention_channels = input_dim, global_context=True)
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, 2*D)
        """
        x = x.transpose(1,2)  # (B, D, T_max)
        x = self.pool(x, lengths).transpose(1, 2)  # (B, 1, 2*D)
        x = x.squeeze(1)  # (B, 2*D)
        return x


class LayerWeightedAvgPool(nn.Module):
    """
    Learnable softmax weights over layers to pool across the layer dimension.
    Now with numerical stability improvements.

    Args:
        num_layers (int): number of layers (L).
        init (str): 'uniform' (default) or 'last' to bias toward higher layers.
        temperature (float): softmax temperature; <1.0 makes weights peakier.
        learnable (bool): if False, keeps uniform fixed weights.
        eps (float): small constant for numerical stability

    Forward:
        x: Tuples of each layer (B, T_max, D) tensor of batch x time x features.
        layer_mask: optional boolean mask of shape (L,) where False drops a layer
        return_weights: if True, also returns the normalized weights (L,)

    Returns:
        pooled: x with the layer dimension removed (weighted average over L)
        (optionally) weights: the softmax weights over layers (L,)
    """

    def __init__(self, num_layers, layer_dim=1, init="uniform", temperature=1.0, learnable=True, eps=1e-8):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = float(temperature)
        self.eps = eps

        # Initialize logits with proper scaling
        if init == "uniform":
            logits = torch.zeros(num_layers)  # Start with zeros for uniform after softmax
        elif init == "last":
            # Bias toward deeper layers with reasonable range
            logits = torch.linspace(-0.5, 0.5, steps=num_layers)
        else:
            logits = torch.zeros(num_layers)

        self.logits = nn.Parameter(logits, requires_grad=learnable)

    def forward(self, x: Tuple[Tensor], lengths, layer_mask=None, return_weights=False):
        # Get the number of layers present in x
        num_layer = len(x)
        assert num_layer == self.num_layers, f"Expected L={self.num_layers}, got L={num_layer}"

        # Apply temperature scaling
        logits = self.logits / self.temperature

        if layer_mask is not None:
            # layer_mask: bool or {0,1} of shape (L,)
            mask = layer_mask.to(dtype=torch.bool)
            if mask.shape != (num_layer,):
                raise ValueError(f"layer_mask must have shape (L,), got {mask.shape}")
            # Use large negative value instead of -inf for stability
            logits = torch.where(mask, logits, torch.full_like(logits, -1e9))

        w = F.softmax(logits, dim=0)  # (L,)

        # Initialize pooled tensor
        pooled = torch.zeros_like(x[0])  # (B, T_max, D)

        T = pooled.shape[1]
        lengths = (lengths * T).long().clamp(max=T)  # Convert to absolute lengths
        mask = length_to_mask(lengths, max_len=T) if lengths is not None else None # (B, T_max)

        for wi, xi in zip(w, x):
            xi = xi * mask.unsqueeze(-1) if mask is not None else xi
            # Turn all nan/inf into zeros before pooling
            xi = torch.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
            pooled = pooled + xi * wi

        return (pooled, w) if return_weights else pooled











