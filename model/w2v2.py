import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from model.pool import AttentiveTemporalPool


class Wav2Vec2(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(ssl_encoder_source, sampling_rate=sample_rate)
        self.model = Wav2Vec2Model.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False
        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states

        self.sample_rate = sample_rate

    def forward(self, x):
        inputs = self.processor(x.cpu().numpy().tolist(),
                                return_tensors="pt",
                                sample_rate=self.sample_rate)

        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with (torch.enable_grad() if not self.freeze_encoder else torch.no_grad()):
            output = self.model(**inputs)

        if self.output_hidden_states:
            hidden_states = output.hidden_states  # tuple of (B, T, D), including input embeddings
            output = torch.stack(hidden_states, dim=1)  # (B, L, T, D)
        else:
            output = output.last_hidden_state  # (B, T, D)

        return output

