import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class Encoder(nn.Module):
    def __init__(self, feature_dim: int = 128, freeze_encoder: bool = False, output_hidden_states: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = 2
        self.output_hidden_states = output_hidden_states

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        B, T = x.shape
        if self.output_hidden_states:
            return torch.randn(B, self.num_layers, 1000, self.feature_dim)
        else:
            return torch.randn(B, 1000, self.feature_dim)


# Example usage
if __name__ == "__main__":
    B, T, D = 4, 100, 64
    x = torch.randn(B, T)
    enc = Encoder(feature_dim=D, freeze_encoder=False, output_hidden_states=True)
    out = enc(x)
    print(out.shape)  # (4, 100, 64)
