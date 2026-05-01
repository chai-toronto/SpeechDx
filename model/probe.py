import torch
import torch.nn as nn

from model.pool import ASP


class Model(nn.Module):
    """
    A wrapper class for various probes.
    """
    def __init__(self, probe, encoder, cache_pool=None):
        super().__init__()
        self.probe = probe
        self.encoder = encoder
        self.cache_pool = cache_pool

    def forward(self, x, lengths=None):
        """
        x: (B, T) matrix of batch x time (raw waveform)
        lengths: (B,) relative lengths (to T_max) per sequence
        Returns:
          logits: (B, num_labels)
        """

        x = self.encoder(x, lengths=lengths)  # (B, T_max, D) or (B, L, T_max, D)
        logits = self.probe(x, lengths)
        # Keep probe contract stable only for mean-cache mode, where cached
        # embeddings can carry singleton middle dimensions (e.g., Bx1xC).
        if (
            self.cache_pool == "mean"
            and logits.dim() > 2
            and all(dim == 1 for dim in logits.shape[1:-1])
        ):
            logits = logits.reshape(logits.shape[0], logits.shape[-1])
        return logits


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
        pooled = self.pooler(x, lengths)
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
        pooled = self.pooler(layer_pooled, lengths)  # (B, D)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.classifier(pooled)


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

class EyeProbe(nn.Module):
    def forward(self, x, lengths=None):
        return x
