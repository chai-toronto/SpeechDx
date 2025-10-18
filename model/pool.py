import torch
import torch.nn as nn
import torch.nn.functional as F

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
          weights: (B, T_max) softmax over real tokens only (zeros on pads)
        """
        B, T_max, D = x.shape
        scores = torch.tanh(self.attn(x)).squeeze(-1)               # (B, T_max)

        if lengths is None:
            # No padding case
            weights = F.softmax(scores, dim=1)                       # (B, T_max)
            pooled  = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
            return pooled, weights

        lengths = (lengths * T_max).long()  # Convert into absolute lengths
        # Build mask: True for real tokens

        t = torch.arange(T_max, device=x.device).unsqueeze(0)  # (1, T_max)
        mask = t < lengths.unsqueeze(1)                        # (B, T_max)

        # Handle any zero-length sequences explicitly to avoid softmax(all -inf)
        empty = lengths == 0
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0.0  # harmless placeholder

        # Mask BEFORE softmax so pads get zero probability
        masked_scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(masked_scores, dim=1)              # (B, T_max)
        weights = torch.where(mask, weights, torch.zeros_like(weights))  # clean pads

        # For truly empty rows, force weights=0 to avoid NaNs in grads
        if empty.any():
            weights[empty] = 0.0

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # (B, D)
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
          weights: (B, T_max) softmax over real tokens only (zeros on pads)
        """
        B, T_max, D = x.shape
        h = torch.tanh(self.attn(x))           # (B, T_max, H)
        scores = self.score(h).squeeze(-1)     # (B, T_max)

        if lengths is None:
            # No padding case
            weights = F.softmax(scores, dim=1)                       # (B, T_max)
            pooled  = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
            return pooled, weights

        lengths = (lengths * T_max).long()  # Convert into absolute lengths

        # Build mask: True for real tokens
        t = torch.arange(T_max, device=x.device).unsqueeze(0)                   # (1, T_max)
        mask = t < lengths.unsqueeze(1)                        # (B, T_max)

        # Handle any zero-length sequences explicitly to avoid softmax(all -inf)
        empty = lengths == 0
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0.0  # harmless placeholder

        # Mask BEFORE softmax so pads get zero probability
        masked_scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(masked_scores, dim=1)              # (B, T_max)
        weights = torch.where(mask, weights, torch.zeros_like(weights))  # clean pads

        # For truly empty rows, force weights=0 to avoid NaNs in grads
        if empty.any():
            weights[empty] = 0.0

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # (B, D)
        return pooled

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
        x: tensor with a layer dimension (e.g., (B, L, T, D) or (L, B, T, D))
        layer_dim: which dimension is the layer axis (default: 1)
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
        logits = torch.zeros(num_layers)
        if init == "last":
            # bias toward deeper layers (monotonic increasing logits)
            logits = torch.linspace(-1.0, 1.0, steps=num_layers)

        self.logits = nn.Parameter(logits, requires_grad=learnable)
        self.layer_dim = layer_dim

    def forward(self, x, layer_mask=None, return_weights=False):
        # Get the number of layers present in x along the chosen dimension
        L = x.size(self.layer_dim)
        assert L == self.num_layers, f"Expected L={self.num_layers}, got L={L}"

        logits = self.logits / self.temperature

        if layer_mask is not None:
            # layer_mask: bool or {0,1} of shape (L,)
            mask = layer_mask.to(dtype=torch.bool)
            if mask.shape != (L,):
                raise ValueError(f"layer_mask must have shape (L,), got {mask.shape}")
            # Exclude masked layers by setting their logit to -inf before softmax
            logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        w = F.softmax(logits, dim=0)  # (L,)

        # Weighted sum over the layer dimension
        # torch.tensordot removes the specified axes:
        pooled = torch.tensordot(x, w, dims=([self.layer_dim], [0]))
        # tensordot moves dimensions; keep original order except the removed axis
        # (PyTorch keeps remaining x-dims in order)
        # print(f'pooled shape: {pooled.shape}')

        return (pooled, w) if return_weights else pooled


