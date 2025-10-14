import torch
import torch.nn as nn
class DummyWaveModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = nn.Linear(1, 1)

    def forward(self, x, lengths=None):
        """ X is tensor of B x T"""
        return self.model(x.mean(dim = 1, keepdim=True))