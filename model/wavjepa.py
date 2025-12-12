from transformers import AutoModel, BatchFeature
import torch.nn as nn

import torch

class WavJEPA(nn.Module):
    def __init__(self, ssl_encoder_source="labhamlet/wavjepa-nat-base", freeze_encoder=True, output_hidden_states=False, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoModel.from_pretrained(ssl_encoder_source, trust_remote_code=True)

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder
        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        if output_hidden_states:
            raise NotImplementedError("WavJEPA does not support output_hidden_states=True")

        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        input_values = self.extract_features(x, lengths)['input_values']
        input_values = input_values.to(device=x.device, dtype=x.dtype)
        features, _ = self.model(input_values)
        # features: (B, 2, T, D), timestamps: (B, T)
        # timestamp is the time index of each frame along T dimension
        features = features.mean(dim=1, keepdims=False)  # average the 2 channels to get (B, T, D)
        return features

    def extract_features(self, x, lengths=None):
        # x is Tensor
        # x: [B, T], lengths: [B], relative
        if lengths is not None:
            x = [
                xi[:li]
                for xi, li in zip(x.unbind(0), lengths)
            ]

        feats = []
        for wav in x:
            wav = self._normalize_audio(wav.unsqueeze(0), -14.0)
            feats.append(torch.cat((wav, wav), dim=0).T) # make it 2 channels

        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)  # shape: [B, T, 2]
        inputs = BatchFeature({"input_values": feats.permute(0, 2, 1)})  # shape: [B, 2, T]
        return inputs

    def _normalize_audio(self, audio_data, target_dBFS=-14.0):
        rms = torch.sqrt(torch.mean(audio_data ** 2))  # Calculate the RMS of the audio
        if rms == 0:  # Avoid division by zero in case of a completely silent audio
            return audio_data
        current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
        gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
        gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
        normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
        return normalized_audio

