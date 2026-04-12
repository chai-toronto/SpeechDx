import torch
import torch.nn as nn
from speechbrain.dataio.dataio import length_to_mask
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class HuBERT(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # HuBERT uses the same feature extractor as Wav2Vec2
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_encoder_source)
        self.model = HubertModel.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states

        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        """
        If output_hidden_states is True, returns a tuple of hidden state tensors from all transformer layers.
        Else, returns the last hidden state as a tensor.

        Note: processor already normalizes the input waveform.
        """
        # Convert to list of numpy arrays for proper batch handling
        if isinstance(x, torch.Tensor):
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
        feature = self.processor(x_list,
                                 return_tensors="pt",
                                 sampling_rate=self.sample_rate,
                                 padding=True).input_values
        x = feature.to(x.device, dtype=x.dtype)

        mask = None
        if lengths is not None:
            T = x.shape[1]
            lengths = (lengths * T).long()  # Convert to absolute lengths
            mask = length_to_mask(lengths)  # (B, T)

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            if self.output_hidden_states:
                return self.model(x, attention_mask=mask).hidden_states[1:]  # tuple of layers (B, T, D)
            else:
                return self.model(x, attention_mask=mask).last_hidden_state  # (B, T, D) matrix
