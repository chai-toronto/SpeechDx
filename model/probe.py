import time

import torch
import torch.nn as nn

from model.pool import ASP
from model.chunker import ChunkPool


class Model(nn.Module):
    """
    A wrapper class for various probes.
    """
    def __init__(self, probe, encoder):
        super().__init__()
        self.probe = probe
        self.encoder = encoder

        self.chunker = None
        self.last_hidden_states = None

    def init_chunker(self, chunk_encoder=False, chunk_at=2, threshold=0.5, aggregate="mean"):
        assert self.chunker is None, "Chunker already initialized"

        if not chunk_encoder:
            # Chunk before temp pool
            layer = getattr(self.probe, "tpooler", None)
            assert layer is not None, "Probe must have a temporal pooler for chunking"
            d_in, d_out = layer.input_dim, layer.output_dim
            self.chunker = ChunkPool(d_in, d_out, threshold, aggregate)
            layer.register_forward_pre_hook(self.chunker_forward_hook)

        else:
            d_in = d_out = self.encoder.d_transformer
            self.chunker = ChunkPool(d_in, d_out, threshold, aggregate)
            self.encoder.register_transformer_pre_hook(self.chunker_forward_hook, layer_idx=chunk_at)

    def forward(self, x, lengths=None):
        """
        x: (B, T) matrix of batch x time (raw waveform)
        lengths: (B,) relative lengths (to T_max) per sequence
        Returns:
          logits: (B, num_labels)
        """

        x = self.encoder(x, lengths=lengths)  # (B, T_max, D) or (B, L, T_max, D)
        return self.probe(x, lengths)

    def chunker_forward_hook(self, module, inputs):
        encoder_style = len(inputs) == 1  # Mimi/Qwen3Voice style: only hidden_states
        if encoder_style:
            x = inputs[0]
            lengths = None
        else:
            x, lengths = inputs
        x_chunk, nonchunk_mask, boundary_mask, boundary_prob = self.chunker(x, lengths)

        # Stat gathering
        scores = getattr(self.probe.tpooler, 'last_scores', None)
        reduction = chunk_stat(x, x_chunk, nonchunk_mask, lengths)

        # reconstruct lengths from mask
        lengths = (~nonchunk_mask).sum(dim=1).float() / nonchunk_mask.size(1)
        self.last_hidden_states = (boundary_mask,
                                   boundary_prob,
                                   reduction,
                                   scores)

        if encoder_style:
            return (x_chunk,)
        return (x_chunk, lengths)



class LinearProbe(nn.Module):
    """
    A simple probe with only a linear layer.
    """
    def __init__(self, input_dim, num_labels, bias=True):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)

    def forward(self, x, lengths=None):
        """
        x: (B, D) matrix of batch x features
        Returns:
          logits: (B, num_labels)
        """
        return self.classifier(x)

class TemporalProbe(nn.Module):
    """
    A probe with a temporal pooling layer followed by a linear layer.
    """
    def __init__(self, input_dim, num_labels, temp_pooler, bias=True):
        super().__init__()
        self.tpooler = temp_pooler
        if isinstance(self.tpooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        pooled = self.tpooler(x, lengths)
        output = self.classifier(pooled)

        return output


class LayerTemporalProbe(nn.Module):
    """
    A probe that performs layer pool -> temporal pool -> linear layer.
    """

    def __init__(self, input_dim, num_labels, layer_pooler, temp_pooler, bias=True):
        super().__init__()
        self.tpooler = temp_pooler
        if isinstance(self.tpooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)
        self.layer_pooler = layer_pooler

    def forward(self, x, lengths=None):
        """
        x: Tuples of each layer (B, T_max, D) tensor of batch x time x features.
        Note: L can be at any position (e.g., B, T_max, L, D), specifiable in layer_pooler.
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        layer_pooled = self.layer_pooler(x, lengths)  # (B, T_max, D)
        pooled = self.tpooler(layer_pooled, lengths)  # (B, D)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.classifier(pooled)

class ChunkTProbe(nn.Module):
    def __init__(self, input_dim, num_labels, temp_pooler, bias=True):
        super().__init__()
        d_out = int(input_dim * 1.5)

        self.chunker = ChunkPool(input_dim, d_out=d_out) # For the upsampler

        input_dim = d_out

        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

        self.pooler = temp_pooler

        if isinstance(self.pooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension

        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)

        self.stat = []

        self.last_hidden_states = None

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_chunk, nonchunk_mask, boundary_mask, boundary_prob = self.chunker(x, lengths)

        chunk_stat(x, x_chunk, nonchunk_mask, lengths)

        x_chunk = x_chunk + self.ffn(x_chunk)

        # reconstruct lengths from mask
        lengths = (~nonchunk_mask).sum(dim=1).float() / nonchunk_mask.size(1)

        pooled = self.pooler(x_chunk, lengths)

        scores = self.pooler.last_scores if hasattr(self.pooler, 'last_scores') else None

        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

        output = self.classifier(pooled)

        self.last_hidden_states = (boundary_mask,
                                    boundary_prob,
                                    scores)

        return output

class LayerChunkTProbe(nn.Module):
    """
    A probe that performs layer pool -> chunking -> temporal pool -> linear layer.
    """

    def __init__(self,
                 input_dim,
                 num_labels,
                 layer_pooler,
                 temp_pooler,
                 threshold=0.5,
                 min_chunk_size=2,):
        super().__init__()
        d_out = int(input_dim * 1.5)

        self.chunker = ChunkPool(input_dim,
                                 d_out=d_out,
                                 threshold=threshold,
                                 min_chunk_size=min_chunk_size)  # For the upsampler

        input_dim = d_out

        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

        self.pooler = temp_pooler

        if isinstance(self.pooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension

        self.classifier = nn.Linear(input_dim, num_labels)

        self.layer_pooler = layer_pooler

        self.stat = []

        self.last_hidden_states = None

    def forward(self, x, lengths=None):
        """
        x: Tuples of each layer (B, T_max, D) tensor of batch x time x features.
        Note: L can be at any position (e.g., B, T_max, L, D), specifiable in layer_pooler.
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        layer_pooled = self.layer_pooler(x, lengths)  # (B, T_max, D)

        x_chunk, nonchunk_mask, boundary_mask, boundary_prob = self.chunker(layer_pooled, lengths)

        reduction = chunk_stat(layer_pooled, x_chunk, nonchunk_mask, lengths)

        x_chunk = x_chunk + self.ffn(x_chunk)

        # reconstruct lengths from mask
        lengths = (~nonchunk_mask).sum(dim=1).float() / nonchunk_mask.size(1)

        pooled = self.pooler(x_chunk, lengths)  # (B, D)

        scores = self.pooler.last_scores if hasattr(self.pooler, 'last_scores') else None
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        output = self.classifier(pooled)

        self.last_hidden_states = (boundary_mask,
                                    boundary_prob,
                                    reduction,
                                    scores)

        return output

def chunk_stat(x, x_chunk, nonchunk_mask, lengths=None):
    T_orig, T_chunk = x.shape[1], x_chunk.shape[1]

    if lengths is None:
        lengths = torch.ones(1, device=x.device)

    orig_lens = (lengths * T_orig).float()
    chunk_lens = (~nonchunk_mask).sum(dim=1).float()
    reduction = (orig_lens - chunk_lens) / orig_lens

    return reduction.mean().item() * 100


class XTTSProbe(nn.Module):
    def __init__(self, input_dim, num_labels, temp_pooler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = temp_pooler
        if isinstance(self.pooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension
        self.classifier = nn.Linear(input_dim + 512, num_labels)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features, where D includes both GPT cond latent and speaker embedding.
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        spk_emb = x[:, 0, :512]  # Assuming speaker embedding is at the first time step and is 512-dim
        gpt_cond_latent = x[:, 1:, :]  # The rest is GPT cond latent

        pooled_lat = self.pooler(gpt_cond_latent) # (B, D)
        both = torch.cat([pooled_lat, spk_emb], dim=-1) # (B, D + 512)
        return self.classifier(both)

