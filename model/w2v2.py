import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class Wav2Vec2(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(ssl_encoder_source, sampling_rate=sample_rate)
        self.model = Wav2Vec2Model.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states

        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        x = self.processor(x.cpu().numpy().tolist(),
                                return_tensors="pt",
                                sampling_rate=self.sample_rate)

        x = {k: v.to(x.device) for k, v in x.items() if k != 'attention_mask'}

        B, T = x.shape
        mask = None
        if lengths is not None:
            assert x.size(0) == lengths.size(0)
            # Convert lengths to absolute
            lengths = (lengths * T).long()

            # Build mask: 1 for valid, 0 for padded
            t = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

            mask = (t < lengths.unsqueeze(1)) * 1  # (B, T)

        with (torch.enable_grad() if not self.freeze_encoder else torch.no_grad()):
            output = self.model(**x, attention_mask=mask)

        if self.output_hidden_states:
            hidden_states = output.hidden_states[1:]  # tuple of (B, T, D), including input embeddings
            output = torch.stack(hidden_states, dim=1)  # (B, L, T, D)
        else:
            output = output.last_hidden_state  # (B, T, D)

        return output

