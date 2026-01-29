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

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        scores = torch.tanh(self.attn(x)).squeeze(-1)               # (B, T_max)

        return attn_pool(lengths, scores, x), scores

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

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          pooled:  (B, D)
        """
        h = F.gelu(self.attn(x))           # (B, T_max, H)
        scores = self.score(h).squeeze(-1)     # (B, T_max)

        return attn_pool(lengths, scores, x), scores

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
            return pooled, None

class ASP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pool = AttentiveStatisticsPooling(input_dim, attention_channels = input_dim, global_context=True)

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
        return x, None


# class LayerWeightedAvgPool(nn.Module):
#     """
#     Learnable softmax weights over layers to pool across the layer dimension.
#
#     Args:
#         num_layers (int): number of layers (L).
#         init (str): 'uniform' (default) or 'last' to bias toward higher layers.
#         temperature (float): softmax temperature; <1.0 makes weights peakier.
#         learnable (bool): if False, keeps uniform fixed weights.
#
#     Forward:
#         x: Tuples of each layer (B, T_max, D) tensor of batch x time x features.
#         layer_mask: optional boolean mask of shape (L,) where False drops a layer
#         return_weights: if True, also returns the normalized weights (L,)
#
#     Returns:
#         pooled: x with the layer dimension removed (weighted average over L)
#         (optionally) weights: the softmax weights over layers (L,)
#     """
#     def __init__(self, num_layers, layer_dim = 1, init="uniform", temperature=1.0, learnable=True):
#         super().__init__()
#         self.num_layers = num_layers
#         self.temperature = float(temperature)
#
#         # logits -> softmax -> weights
#         logits = torch.ones(num_layers)
#         if init == "last":
#             # bias toward deeper layers (monotonic increasing logits)
#             logits = torch.linspace(-1.0, 1.0, steps=num_layers)
#
#         self.logits = nn.Parameter(logits, requires_grad=learnable)
#
#     def forward(self, x: Tuple[Tensor], layer_mask=None, return_weights=False):
#         # Get the number of layers present in x along the chosen dimension
#         num_layer = len(x)
#         assert num_layer == self.num_layers, f"Expected L={self.num_layers}, got L={num_layer}"
#
#         logits = self.logits / self.temperature
#
#         if layer_mask is not None:
#             # layer_mask: bool or {0,1} of shape (L,)
#             mask = layer_mask.to(dtype=torch.bool)
#             if mask.shape != (num_layer,):
#                 raise ValueError(f"layer_mask must have shape (L,), got {mask.shape}")
#             # Exclude masked layers by setting their logit to -inf before softmax
#             logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
#
#         w = F.softmax(logits, dim=0)  # (L,)
#
#         pooled = torch.zeros_like(x[0])  # (B, T_max, D)
#         for wi, xi in zip(w, x):
#             pooled.add_(xi, alpha=wi.item())
#
#         return (pooled, w) if return_weights else pooled


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
        lengths = (lengths * T).long()  # Convert to absolute lengths
        mask = length_to_mask(lengths) if lengths is not None else None # (B, T_max)

        # Weighted sum with numerical checks
        for wi, xi in zip(w, x):
            xi = xi * mask.unsqueeze(-1) if mask is not None else xi
            # Turn all nan/inf into zeros before pooling
            xi = torch.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize each [1, D] vector to unit norm to prevent large values
            xi = xi / torch.linalg.vector_norm(xi, dim=-1, keepdim=True).clamp(min=self.eps)
            if not torch.isnan(xi).any() and not torch.isinf(xi).any():
                pooled = pooled + xi * wi.item()
            else:
                print(f"Warning: Skipping layer due to NaN/Inf values")

        # Final safety check
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

        return (pooled, w) if return_weights else pooled


class ChunkPool(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True
        self.up_proj = nn.Linear(d_model, d_out)


    def forward(self, x, lengths = None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.

        Returns:
            sorted_hidden: (B, T_chunk, D) reordered hidden states with boundaries first
            chunk_mask: (B, T_chunk) boolean mask where True indicates non-boundary tokens
            boundary_mask: (B, T_max) boolean mask where True indicates non-boundary tokens
            boundary_prob: (B, T_max, 2) probabilities for non-boundary and boundary classes, respectively
        """
        B, T, D = x.shape
        mask = None
        if lengths is not None:
            T = x.shape[1]
            lengths = (lengths * T).long()  # Convert to absolute lengths
            mask = ~length_to_mask(lengths).bool() # (B, T), True for pads

        # Cosine similarity between consecutive tokens
        q = F.normalize(self.q_proj(x[:, :-1]), dim=-1)  # (B, L-1, D)
        k = F.normalize(self.k_proj(x[:, 1:]), dim=-1)  # (B, L-1, D)
        cos_sim = (q * k).sum(dim=-1)  # (B, L-1)

        # Convert to boundary score: high cosine sim � low boundary prob
        boundary_score = (1 - cos_sim) / 2  # (B, L-1), range [0, 1]


        # Using Sigmoid instead
        # q = self.q_proj(x[:, :-1])  # (B, L-1, D)
        # k = self.k_proj(x[:, 1:])  # (B, L-1, D)
        # sim = (q * k).sum(dim=-1)  # (B, L-1)
        # sim = F.sigmoid(sim)
        #
        # boundary_score = 1 - sim  # (B, L-1), range [0, 1]

        boundary_score = F.pad(boundary_score, (1, 0), value=1.0)  # First token always boundary

        boundary_score[:, 1::2] = 0.0  # Force even positions to be non-boundaries (for stability)

        boundary_prob = torch.stack(((1 - boundary_score), boundary_score), dim=-1)

        selected_idx = boundary_prob[:, :, 1] > 0.1  # (B, L)

        boundary_mask = selected_idx != 1  # (shape hidden_states.shape[:-1])

        # Handle padding: force padded positions to NOT be boundaries
        if mask is not None:
            boundary_mask = boundary_mask | mask

        # Reorder so boundaries come first
        # Strategy: assign large indices to non-boundaries, small to boundaries
        token_idx = torch.arange(T, device=x.device)[None, :]  # (1, L)
        token_idx = token_idx + boundary_mask.long() * T  # Non-boundaries get +L
        sorted_idx = torch.argsort(token_idx, dim=1)  # (B, L)

        # Count boundaries per batch
        num_boundaries = (~boundary_mask).sum(dim=1)  # (B,)
        max_boundaries = num_boundaries.max().item()

        if max_boundaries == 0:
            raise ValueError("No boundaries detected in any sequence.")

        # Gather reordered tokens (only first max_boundaries)
        sorted_hidden = torch.gather(
            x, dim=1,
            index=sorted_idx[:, :max_boundaries].unsqueeze(-1).expand(-1, -1, D)
        )

        # True = Non-boundary tokens (to be masked out)
        chunk_mask = torch.arange(max_boundaries, device=x.device)[None, :] >= num_boundaries[:, None]

        sorted_hidden = self.up_proj(sorted_hidden * (~chunk_mask).unsqueeze(-1).float())

        return sorted_hidden, chunk_mask, boundary_mask, boundary_prob



if __name__ == "__main__":
    # Simple test
    # B, T, D = 2, 10, 4
    # x = torch.randn(B, T, D)
    x = torch.Tensor([[1, 0],
                      [-1, 0]])
    x = x.unsqueeze(0) # (1, 2, 2)

    # lengths = torch.tensor([1.0, 0.8])
    lengths = None

    chunk_pool = ChunkPool(d_model=2)
    sorted_hidden, chunk_mask, boundary_mask, boundary_prob = chunk_pool(x, lengths)

    print("Input shape:", x.shape)
    print("Sorted hidden shape:", sorted_hidden.shape)
    print("Chunk mask shape:", chunk_mask.shape)
    print("Boundary mask shape:", boundary_mask.shape)
    print("Boundary prob shape:", boundary_prob.shape)






