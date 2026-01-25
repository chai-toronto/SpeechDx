import time

import torch
import torch.nn as nn

from model.pool import ASP, ChunkPool


class Model(nn.Module):
    """
    A wrapper class for various probes.
    """
    def __init__(self, probe, encoder):
        super().__init__()
        self.probe = probe
        self.encoder = encoder

    def forward(self, x, lengths=None):
        """
        x: (B, T) matrix of batch x time (raw waveform)
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        x = self.encoder(x, lengths=lengths)  # (B, T_max, D) or (B, L, T_max, D)
        return self.probe(x, lengths)

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
        self.pooler = temp_pooler
        if isinstance(self.pooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pooled, scores = self.pooler(x, lengths)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        output = self.classifier(pooled)

        return output

class LayerTemporalProbe(nn.Module):
    """
    A probe that performs layer pool -> temporal pool -> linear layer.
    """

    def __init__(self, input_dim, num_labels, layer_pooler, temp_pooler, bias=True):
        super().__init__()
        self.pooler = temp_pooler
        if isinstance(self.pooler, ASP):
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
        pooled, scores = self.pooler(layer_pooled, lengths)  # (B, D)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.classifier(pooled)

class ChunkTProbe(nn.Module):
    def __init__(self, input_dim, num_labels, temp_pooler, bias=True):
        super().__init__()
        self.chunker = ChunkPool(input_dim, d_out = 1536) # For the upsampler
        input_dim = 1536
        self.pooler = temp_pooler
        if isinstance(self.pooler, ASP):
            input_dim = input_dim * 2  # ASP doubles the dimension
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)
        self.stat = []
        self.scores = None

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_chunk, mask, boundary_mask, boundary_prob = self.chunker(x, lengths)
        self.chunk_stat(x, x_chunk, lengths, mask)
        # reconstruct lengths from mask
        lengths = (~mask).sum(dim=1).float() / mask.size(1)

        pooled, scores = self.pooler(x_chunk, lengths)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        output = self.classifier(pooled)

        return (output,
                boundary_mask.clone().detach(),
                boundary_prob.clone().detach(),
                scores.clone().detach())

    def chunk_stat(self, x, x_chunk, lengths, chunk_mask):
        B, T_max, D = x.size()
        _, T_chunk, _ = x_chunk.size()

        orig_lens = (lengths * T_max).long()
        chunk_lens = (~chunk_mask).sum(dim=1).long()
        reduction = (orig_lens - chunk_lens).float() / orig_lens.float()
        print("Average batch reduction: {:.2f}%".format(reduction.mean().item() * 100))
        self.stat.append(reduction.mean().item())
        print("Overall average reduction: {:.2f}%".format(sum(self.stat) / len(self.stat) * 100))


