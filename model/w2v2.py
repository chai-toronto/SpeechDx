import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model


class Wav2Vec2(nn.Module):
    def __init__(self, ssl_encoder_source, num_labels, freeze_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(ssl_encoder_source)
        self.model = Wav2Vec2Model.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False
        self.freeze_encoder = freeze_encoder

        self.temp_pool = AttentionPool(self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, x):
        # inputs = self.processor(x)
        inputs = x
        inputs = inputs.to(device = x.device)

        if self.freeze_encoder:
            with torch.no_grad():
                output = self.model(inputs).last_hidden_state
        else:
            output = self.model(inputs).last_hidden_state

        pooled = self.temp_pool(output)
        return self.classifier(pooled)

class AttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn  = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) actual lengths per sequence. If None, we assume no padding.
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
            return pooled

        # Build mask: True for real tokens

        t = torch.arange(T_max).unsqueeze(0)                   # (1, T_max)
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