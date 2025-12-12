import torch
from speechbrain.dataio.dataio import length_to_mask
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn

class WavLM(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        input_values = self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        mask = None
        if lengths is not None:
            T = input_values.shape[1]
            lengths = (lengths * T).long() # Convert to absolute lengths
            mask = length_to_mask(lengths)

        if self.output_hidden_states:
            features = self.feature_extractor(input_values,
                                              output_hidden_states=True,
                                              attention_mask=mask
                                              ).hidden_states[1:]
        else:
            features = self.feature_extractor(input_values,
                                              attention_mask=mask
                                              ).last_hidden_state # (B, T, D)
        return features