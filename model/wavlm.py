import torch
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn

class WavLM(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder
        if freeze_encoder:
            self.feature_extractor.requires_grad = False
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        B, T = x.shape
        if lengths is not None:
            assert x.size(0) == lengths.size(0)
            # Convert lengths to absolute
            lengths = (lengths * T).long()

            # Build mask: 1 for valid, 0 for padded
            t = torch.arange(T, device=x.device).unsqueeze(0) # (1, T)
            mask = (t < lengths.unsqueeze(1)) * 1 # (B, T)
        input_values = self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)
        if self.output_hidden_states:
            features = self.feature_extractor(input_values, output_hidden_states=True, attention_mask=mask)
            features = torch.stack(features.hidden_states[1:], dim=1) # (B, L, T, D), excluding the input embeddings
        else:
            features = self.feature_extractor(input_values, attention_mask=mask).last_hidden_state # (B, T, D)
        return features