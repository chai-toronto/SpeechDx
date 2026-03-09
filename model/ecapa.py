from contextlib import nullcontext

import torch
import torch.nn as nn
from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


class ECAPA_TDNN(nn.Module):
    def __init__(self, freeze_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            local_strategy=LocalStrategy.COPY,
        )

        self.output_hidden_states = False
        self.freeze_encoder = freeze_encoder

    def forward(self, x, lengths=None):
        context = torch.no_grad if self.freeze_encoder else nullcontext
        with context():
            return self.model.encode_batch(x).squeeze(1)  # (B, D)

if __name__ == "__main__":
    model = ECAPA_TDNN()
    dummy_wav = torch.randn(2, 16000 * 20)  # batch of 2, 5 seconds each
    dummy_lengths = torch.tensor([1.0, 0.8])  # first is full length, second is 80%
    features = model(dummy_wav, dummy_lengths)
    print(features.shape)
