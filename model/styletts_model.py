import os
import tempfile

import torch
import torch.nn as nn
import soundfile as sf

import sys
sys.path.append()

# A pseudo module, infer only
class StyleTTS2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = tts.StyleTTS2()
        self.freeze_encoder = True
        self.output_hidden_states = False

    def forward(self, x, lengths=None):
        batch_size = x.shape[0]
        embeddings = []

        for i in range(batch_size):
            wav_i = x[i]  # (T,)

            # Trim to actual length if lengths provided
            if lengths is not None:
                actual_len = int(lengths[i].item() * wav_i.shape[0])
                wav_i = wav_i[:actual_len]

            # StyleTTS requires file paths — write a temporary WAV
            with tempfile.TemporaryDirectory() as td:
                tmp_path = os.path.join(td, "ref.wav")
                sf.write(tmp_path, wav_i.cpu().numpy(), self.sample_rate, subtype='PCM_16')

                emb = compute_style({"ref_audio": tmp_path})

            embeddings.append(emb.squeeze())  # (1, D) or (T', D)



if __name__ == "__main__":
    wav = torch.randn(2, 22000 * 5)  # batch of 2, 5 seconds each
    enc = StyleTTS2_LJ()
    emb = enc(wav)