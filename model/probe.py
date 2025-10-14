import torch
import torch.nn as nn

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
        x = self.encoder(x)  # (B, T_max, D) or (B, L, T_max, D)
        return self.probe(x, lengths)

class LinearProbe(nn.Module):
    """
    A simple probe with only a linear layer.
    """
    def __init__(self, input_dim, num_labels, bias=True):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels, bias=bias)

    def forward(self, x):
        """
        x: (B, D) matrix of batch x features
        Returns:
          logits: (B, num_labels)
        """
        return self.classifier(x)


class TemporalProbe(LinearProbe):
    """
    A probe with a temporal pooling layer followed by a linear layer.
    """
    def __init__(self, input_dim, num_labels, pooler, bias=True):
        super().__init__(input_dim, num_labels, bias=bias)
        self.pooler = pooler

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) matrix of batch x time x features
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        pooled = self.pooler(x, lengths)
        return self.classifier(pooled)

class LayerTemporalProbe(TemporalProbe):
    """
    A probe that performs layer pool -> temporal pool -> linear layer.
    """

    def __init__(self, input_dim, num_labels, layer_pooler, temp_pooler, bias=True):
        super().__init__(input_dim, num_labels, temp_pooler, bias=bias)
        self.layer_pooler = layer_pooler

    def forward(self, x, lengths=None):
        """
        x: (B, L, T_max, D) matrix of batch x layers x time x features.
        Note: L can be at any position (e.g., B, T_max, L, D), specifiable in layer_pooler.
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Returns:
          logits: (B, num_labels)
        """
        layer_pooled = self.layer_pooler(x)  # (B, T_max, D)
        pooled = self.pooler(layer_pooled, lengths)  # (B, D)
        return self.classifier(pooled)