"""
Hierarchical Chunking Encoder for JEPA.

Inspired by H-Net (https://arxiv.org/pdf/2507.07955) with PyTorch-native implementation.
Implements two-level hierarchical chunking with learned boundary prediction.

Key components:
- RoutingModule: Predicts chunk boundaries using cosine similarity + learned classifier
- ChunkLayer: Extracts boundary tokens
- DeChunkLayer: Reconstructs full sequence using EMA-based interpolation
- ChunkingEncoderStage: Single hierarchical stage (encoder -> route -> chunk -> process -> dechunk -> decoder)
- ChunkingEncoder: Main interface, drop-in replacement for nn.TransformerEncoder
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChunkingEncoderConfig:
    """Configuration for ChunkingEncoder."""

    # Model dimensions per stage [outer, inner]
    d_model: List[int] = field(default_factory=lambda: [768, 1152])

    # Number of attention heads per stage
    num_heads: List[int] = field(default_factory=lambda: [12, 12])

    # Number of layers per component per stage
    encoder_layers: List[int] = field(default_factory=lambda: [2, 2])
    decoder_layers: List[int] = field(default_factory=lambda: [2, 2])
    main_layers: List[int] = field(default_factory=lambda: [0, 4])  # 0 for outer (has recursion), 8 for inner

    # MLP ratio
    mlp_ratio: float = 4.0

    # Transformer settings
    norm_first: bool = False
    dropout: float = 0.0
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6

    # Learning rate multipliers per stage (optional)
    lr_multipliers: Optional[List[float]] = None


class RoutingModule(nn.Module):
    """
    Predicts chunk boundaries using cosine similarity between consecutive tokens.

    Combines two signals:
    1. Cosine similarity (frozen projections)
    2. Learnable boundary classifier

    From H-Net dc.py lines 47-164.
    """

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict chunk boundaries.

        Args:
            hidden_states: (B, L, D) - input sequence
            mask: (B, L) - True for valid tokens (optional)

        Returns:
            boundary_prob: (B, L, 2) - probabilities [no_boundary, boundary]
            boundary_mask: (B, L) - boolean mask of predicted boundaries
        """
        B, L, D = hidden_states.shape

        # Cosine similarity between consecutive tokens
        q = F.normalize(self.q_proj(hidden_states[:, :-1]), dim=-1)  # (B, L-1, D)
        k = F.normalize(self.k_proj(hidden_states[:, 1:]), dim=-1)  # (B, L-1, D)
        cos_sim = (q * k).sum(dim=-1)  # (B, L-1)

        # Convert to boundary score: high cosine sim � low boundary prob
        boundary_score = (1 - cos_sim) / 2  # (B, L-1), range [0, 1]
        boundary_score = F.pad(boundary_score, (1, 0), value=1.0)  # First token always boundary

        boundary_prob = torch.stack(((1 - boundary_score), boundary_score), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])

        # Handle padding: force padded positions to NOT be boundaries
        if mask is not None:
            boundary_mask = boundary_mask & mask

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return boundary_prob, boundary_mask, selected_probs


class ChunkLayer(nn.Module):
    """
    Extracts tokens at boundary positions.

    Reorders sequence so boundary tokens come first, then truncates to max boundaries.
    From H-Net dc.py lines 167-210.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract boundary tokens.

        Args:
            hidden_states: (B, L, D) - input sequence
            boundary_mask: (B, L) - boolean mask of boundaries
            mask: (B, L) - valid token mask (optional)

        Returns:
            chunked: (B, M, D) - only boundary tokens
            chunk_mask: (B, M) - valid chunk mask
        """
        B, L, D = hidden_states.shape

        # Reorder so boundaries come first
        # Strategy: assign large indices to non-boundaries, small to boundaries
        token_idx = torch.arange(L, device=hidden_states.device)[None, :]  # (1, L)
        token_idx = token_idx + (~boundary_mask).long() * L  # Non-boundaries get +L
        sorted_idx = torch.argsort(token_idx, dim=1)  # (B, L)

        # Count boundaries per batch
        num_boundaries = boundary_mask.sum(dim=1)  # (B,)
        max_boundaries = num_boundaries.max().item()

        if max_boundaries == 0:
            # Fallback: no boundaries predicted, use all tokens
            max_boundaries = L
            chunk_mask = mask if mask is not None else torch.ones(B, L, dtype=torch.bool, device=hidden_states.device)
            return hidden_states, chunk_mask

        # Gather reordered tokens (only first max_boundaries)
        sorted_hidden = torch.gather(
            hidden_states, dim=1,
            index=sorted_idx[:, :max_boundaries].unsqueeze(-1).expand(-1, -1, D)
        )

        # Create mask for valid chunks
        chunk_mask = torch.arange(max_boundaries, device=hidden_states.device)[None, :] < num_boundaries[:, None]

        return sorted_hidden, chunk_mask


class DeChunkLayer(nn.Module):
    """
    Reconstructs full sequence from boundary tokens using EMA-based interpolation.

    Implements the algorithm from H-Net dc.py lines 213-336, specifically the
    step() function which shows the EMA pattern:
        result = p * current_hidden_states + (1-p) * previous_value

    The forward pass applies this EMA scan across the chunked sequence, then
    maps results back to full sequence positions.
    """

    def forward(
        self,
        chunked_hidden: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct full sequence from boundary tokens.

        Args:
            chunked_hidden: (B, M, D) - hidden states at boundaries only
            boundary_mask: (B, L) - original boundary mask
            boundary_prob: (B, L, 2) - boundary probabilities
            chunk_mask: (B, M) - valid chunk mask (optional)

        Returns:
            reconstructed: (B, L, D) - full sequence
        """
        B, L = boundary_mask.shape
        M, D = chunked_hidden.shape[1], chunked_hidden.shape[2]

        # Extract boundary probabilities (probability of being a boundary)
        p = boundary_prob[..., 1].clamp(1e-4, 1-1e-4)  # (B, L)

        # Get probabilities for boundary positions only
        # Need to handle case where boundary_mask might be all False
        num_boundaries = boundary_mask.sum(dim=1)  # (B,)
        if num_boundaries.max() == 0:
            # No boundaries, just return chunked_hidden as is (shouldn't happen with routing)
            return chunked_hidden[:, :L, :] if M >= L else F.pad(chunked_hidden, (0, 0, 0, L - M))

        # Vectorized: Gather probabilities at boundary positions
        # Use advanced indexing to extract p values at boundary locations
        # Create indices for gathering: for each batch, get positions where boundary_mask is True
        boundary_indices = boundary_mask.nonzero(as_tuple=False)  # (N, 2) where N = total boundaries
        batch_idx = boundary_indices[:, 0]  # Batch indices
        seq_idx = boundary_indices[:, 1]  # Sequence positions

        # Get p values at these positions
        p_gathered = p[batch_idx, seq_idx]  # (N,)

        # Scatter into (B, M) tensor
        p_at_boundaries = torch.zeros(B, M, device=chunked_hidden.device, dtype=chunked_hidden.dtype)

        # Count boundaries per batch to know where to place values
        boundary_counts = boundary_mask.sum(dim=1)  # (B,)

        # Create target indices for scattering
        # For each boundary, compute which position it should go to in the M dimension
        cumsum = torch.cat([torch.zeros(1, dtype=torch.long, device=boundary_mask.device),
                           boundary_counts.cumsum(dim=0)[:-1]])
        target_positions = torch.arange(len(p_gathered), device=boundary_mask.device) - cumsum[batch_idx]

        # Only keep positions < M (in case there are more boundaries than M)
        valid_mask = target_positions < M
        # Ensure dtype matches for mixed precision training
        p_at_boundaries[batch_idx[valid_mask], target_positions[valid_mask]] = p_gathered[valid_mask].to(p_at_boundaries.dtype)

        # Fully vectorized EMA scan
        # The recurrence is: h[t] = p[t] * x[t] + (1-p[t]) * h[t-1], with h[-1] = 0
        #
        # Expanding: h[t] = sum_{i=0}^{t} [p[i] * x[i] * prod_{j=i+1}^{t} (1-p[j])]
        #
        # We can compute this using matrix operations:
        # Create a lower triangular matrix where entry [t, i] contains prod_{j=i+1}^{t} (1-p[j])

        # Handle padding by masking
        if chunk_mask is None:
            chunk_mask = torch.ones(B, M, dtype=torch.bool, device=chunked_hidden.device)

        # Compute (1-p) for each position
        one_minus_p = 1 - p_at_boundaries  # (B, M)
        # Mask invalid positions - they should not affect the EMA
        one_minus_p = torch.where(chunk_mask, one_minus_p, torch.ones_like(one_minus_p))

        # Compute log for numerical stability
        log_one_minus_p = torch.log(one_minus_p.clamp(min=1e-10))  # (B, M)

        # Cumulative sum of log(1-p)
        log_cumsum = torch.cumsum(log_one_minus_p, dim=1)  # (B, M)

        # Create matrices for broadcasting
        # We need: decay[t, i] = prod_{j=i+1}^{t} (1-p[j])
        # This equals: exp(sum_{j=i+1}^{t} log(1-p[j]))
        #            = exp(log_cumsum[t] - log_cumsum[i])
        # For diagonal (i=t): prod is empty, so decay[t,t] = 1

        # log_cumsum_i[t, i] should have log_cumsum[i] for all i
        log_cumsum_i = log_cumsum.unsqueeze(1).expand(B, M, M)  # (B, M, M)

        # log_cumsum_t[t, i] should have log_cumsum[t] for all i (but only use i <= t)
        log_cumsum_t = log_cumsum.unsqueeze(2).expand(B, M, M)  # (B, M, M)

        # Compute difference: log_cumsum[t] - log_cumsum[i]
        # For diagonal: log_cumsum[t] - log_cumsum[t] = 0 ✓
        # For i < t: log_cumsum[t] - log_cumsum[i] = sum_{j=i+1}^{t} log(1-p[j]) ✓
        log_decay = log_cumsum_t - log_cumsum_i  # (B, M, M)

        # Create lower triangular mask
        tril_mask = torch.tril(torch.ones(M, M, device=chunked_hidden.device, dtype=torch.bool))  # (M, M)

        # Apply mask - set upper triangle to large negative value
        # Use dtype-specific min value to avoid overflow in fp16
        min_val = torch.finfo(log_decay.dtype).min / 2  # Divide by 2 for safety
        log_decay = torch.where(tril_mask, log_decay, torch.tensor(min_val, dtype=log_decay.dtype, device=log_decay.device))

        # Compute decay factors
        decay_factors = torch.exp(log_decay)  # (B, M, M)

        # Compute p[i] * x[i] for each position
        weighted_input = p_at_boundaries.unsqueeze(-1) * chunked_hidden  # (B, M, D)

        # Mask invalid chunks - convert mask to same dtype as weighted_input
        weighted_input = weighted_input * chunk_mask.unsqueeze(-1).to(weighted_input.dtype)  # (B, M, D)

        # Apply decay factors and sum
        # ema_output[b, t, d] = sum_{i=0}^{t} decay_factors[b, t, i] * weighted_input[b, i, d]
        # This is a batch matrix multiplication: (B, M, M) @ (B, M, D) -> (B, M, D)
        ema_output = torch.bmm(decay_factors, weighted_input)  # (B, M, D)

        # Map boundary tokens back to full sequence positions
        # Each position gets the value from its corresponding boundary token
        plug_back_idx = boundary_mask.cumsum(dim=1) - 1  # (B, L) - which boundary owns each position
        plug_back_idx = plug_back_idx.clamp(min=0, max=M-1)  # Clamp to valid range

        reconstructed = torch.gather(
            ema_output, dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        return reconstructed


class ChunkingEncoderStage(nn.Module):
    """
    Single stage of hierarchical chunking encoder.

    Structure:
    - Innermost stage: encoder � main_network
    - Non-innermost: encoder � route � chunk � main_network (recursive) � dechunk � residual � decoder
    """

    def __init__(self, config: ChunkingEncoderConfig, stage_idx: int = 0, is_innermost: bool = False):
        super().__init__()
        self.stage_idx = stage_idx
        self.is_innermost = is_innermost

        d_model = config.d_model[stage_idx]

        # Encoder: process input
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads[stage_idx],
            dim_feedforward=int(d_model * config.mlp_ratio),
            batch_first=True,
            norm_first=config.norm_first,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers[stage_idx],
            norm=nn.LayerNorm(d_model, eps=config.layer_norm_eps),
        )

        if not is_innermost:
            # Routing and chunking modules
            self.routing_module = RoutingModule(d_model)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer()

            # Projection to next stage dimension (if different)
            next_d_model = config.d_model[stage_idx + 1]
            if d_model != next_d_model:
                self.proj_to_next = nn.Linear(d_model, next_d_model)
                self.proj_from_next = nn.Linear(next_d_model, d_model)
            else:
                self.proj_to_next = None
                self.proj_from_next = None

            # Recursive main network (next stage)
            self.main_network = ChunkingEncoderStage(
                config, stage_idx=stage_idx + 1, is_innermost=True
            )

            # Residual projection (initialized to zero for stable training)
            self.residual_proj = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

            # Decoder: process after dechunking
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads[stage_idx],
                dim_feedforward=int(d_model * config.mlp_ratio),
                batch_first=True,
                norm_first=config.norm_first,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
            )
            self.decoder = nn.TransformerEncoder(
                decoder_layer,
                num_layers=config.decoder_layers[stage_idx],
                norm=nn.LayerNorm(d_model, eps=config.layer_norm_eps),
            )
        else:
            # Innermost: just main network (no recursion)
            main_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads[stage_idx],
                dim_feedforward=int(d_model * config.mlp_ratio),
                batch_first=True,
                norm_first=config.norm_first,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
            )
            self.main_network = nn.TransformerEncoder(
                main_layer,
                num_layers=config.main_layers[stage_idx],
                norm=nn.LayerNorm(d_model, eps=config.layer_norm_eps),
            )

    def forward(
        self, hidden_states: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through one stage.

        Args:
            hidden_states: (B, L, D) - input sequence
            src_key_padding_mask: (B, L) - True for positions to ignore

        Returns:
            output: (B, L, D) - processed sequence
            boundary_outputs: list of (boundary_prob, boundary_mask) tuples for load balancing
        """
        # Encode input
        encoded = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)

        if self.is_innermost:
            # Base case: just process through main network
            output = self.main_network(encoded, src_key_padding_mask=src_key_padding_mask)
            return output, []

        # Non-innermost: hierarchical chunking
        # Convert padding mask to valid token mask for routing
        valid_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None

        # Route: predict chunk boundaries
        boundary_prob, boundary_mask, _ = self.routing_module(encoded, mask=valid_mask)

        # Chunk: extract boundary tokens
        chunked, chunk_mask = self.chunk_layer(encoded, boundary_mask, mask=valid_mask)

        # Project to next stage dimension if needed
        if self.proj_to_next is not None:
            chunked = self.proj_to_next(chunked)

        # Process chunks through main network (recursive)
        chunked_out, inner_boundary_outputs = self.main_network(
            chunked, src_key_padding_mask=~chunk_mask
        )

        # Project back from next stage dimension if needed
        if self.proj_from_next is not None:
            chunked_out = self.proj_from_next(chunked_out)

        # Dechunk: reconstruct full sequence
        dechunked = self.dechunk_layer(chunked_out, boundary_mask, boundary_prob, chunk_mask)

        # Residual connection
        residual = self.residual_proj(encoded)
        combined = dechunked + residual

        # Decode
        output = self.decoder(combined, src_key_padding_mask=src_key_padding_mask)

        # Collect boundary outputs for load balancing loss
        boundary_outputs = [(boundary_prob, boundary_mask)] + inner_boundary_outputs

        return output, boundary_outputs


class ChunkingEncoder(nn.Module):
    """
    Two-level hierarchical chunking encoder for JEPA.

    Drop-in replacement for nn.TransformerEncoder with hierarchical processing:
    1. Outer stage: routes and chunks input sequence
    2. Inner stage: processes chunked sequence
    3. Reconstruction and decoding

    Supports:
    - Padding via src_key_padding_mask
    - Learning rate modulation per stage
    - Load balancing loss computation
    """

    def __init__(self, config: ChunkingEncoderConfig):
        super().__init__()
        self.config = config
        self.root_stage = ChunkingEncoderStage(config, stage_idx=0, is_innermost=False)

        # Store for lr modulation
        self._lr_multipliers = config.lr_multipliers if hasattr(config, 'lr_multipliers') else None

        # Apply lr multipliers if provided
        if self._lr_multipliers is not None:
            self.apply_lr_multipliers(self._lr_multipliers)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through chunking encoder.

        Args:
            src: (B, L, D) - input sequence
            src_key_padding_mask: (B, L) - True for positions to mask

        Returns:
            output: (B, L, D) - processed sequence

        Side effect: stores boundary_outputs in self._last_boundary_outputs for loss computation
        """
        output, boundary_outputs = self.root_stage(src, src_key_padding_mask, output_hidden_states)
        self._last_boundary_outputs = boundary_outputs
        return output

    def get_load_balancing_loss(self, N: float = 5.0) -> torch.Tensor:
        """
        Compute load balancing loss from last forward pass.

        Encourages the model to create meaningful chunks (not too many, not too few).
        From H-Net hnet/utils/train.py lines 13-40.

        Args:
            N: Target downsampling factor (higher = more compression)

        Returns:
            loss: scalar tensor
        """
        if not hasattr(self, '_last_boundary_outputs') or not self._last_boundary_outputs:
            return torch.tensor(0.0)

        total_loss = 0.0
        for boundary_prob, boundary_mask in self._last_boundary_outputs:
            # Extract probability of boundary class
            tokenized_prob = boundary_prob[..., 1]  # (B, L)

            # Empirical vs predicted boundary ratios
            true_ratio = boundary_mask.float().mean()  # Actual fraction of boundaries
            avg_prob = tokenized_prob.float().mean()  # Predicted fraction

            # Load balancing loss
            # Penalizes mismatch between predicted and actual boundary frequency
            lb_loss = (
                (1 - true_ratio) * (1 - avg_prob) +
                (true_ratio) * (avg_prob) * (N - 1)
            ) * N / (N - 1)

            total_loss = total_loss + lb_loss

        return total_loss / len(self._last_boundary_outputs)

    def apply_lr_multipliers(self, lr_multipliers: List[float]):
        """
        Apply learning rate multipliers to different stages.

        Uses parameter annotation pattern from H-Net to mark parameters
        with different learning rates for hierarchical optimization.

        Args:
            lr_multipliers: list of floats, one per stage (e.g., [1.0, 0.5])
                           Higher values = higher learning rate
        """
        from model.models.WavHJepa.optim_utils import apply_optimization_params

        # Apply to outer stage (stage 0)
        if len(lr_multipliers) > 0:
            for param in self.root_stage.encoder.parameters():
                apply_optimization_params(param, lr_multiplier=lr_multipliers[0])
            if hasattr(self.root_stage, 'decoder'):
                for param in self.root_stage.decoder.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[0])
            if hasattr(self.root_stage, 'routing_module'):
                for param in self.root_stage.routing_module.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[0])
            if hasattr(self.root_stage, 'residual_proj'):
                for param in self.root_stage.residual_proj.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[0])
            # Apply to projection layers (if they exist)
            if hasattr(self.root_stage, 'proj_to_next') and self.root_stage.proj_to_next is not None:
                for param in self.root_stage.proj_to_next.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[0])
            if hasattr(self.root_stage, 'proj_from_next') and self.root_stage.proj_from_next is not None:
                for param in self.root_stage.proj_from_next.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[0])

        # Apply to inner stage (stage 1)
        if len(lr_multipliers) > 1 and hasattr(self.root_stage, 'main_network'):
            if isinstance(self.root_stage.main_network, ChunkingEncoderStage):
                for param in self.root_stage.main_network.parameters():
                    apply_optimization_params(param, lr_multiplier=lr_multipliers[1])
