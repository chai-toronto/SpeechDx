"""
VoiceCraft encoder — hook-based feature extraction via EnCodec + VoiceCraft decoder.

Encodes audio with EnCodec at 24kHz, then feeds codes through VoiceCraft's
decoder transformer and captures hidden states from a specified layer via a
forward hook.  Returns frame-level features of shape (B, T', D).
"""

import os
import sys
import librosa
import torch
import torch.nn as nn
from encodec import EncodecModel
from encodec.utils import convert_audio

# VoiceCraft is not pip-installable; add cloned repo to path
_vc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VoiceCraft")
if os.path.isdir(_vc_dir) and _vc_dir not in sys.path:
    sys.path.insert(0, _vc_dir)


class VoiceCraftEncoder(nn.Module):
    """
    VoiceCraft hook-based feature extractor.

    Parameters
    ----------
    voicecraft_source : str
        HuggingFace model id, e.g. ``"pyp1/VoiceCraft"``.
    freeze_encoder : bool
        If True, all parameters are frozen.
    sample_rate : int
        Expected input waveform sample rate (should be 24000).
    hook_layer : int
        Decoder transformer layer index to hook (default 12, mid-point of 24 layers).
    """

    def __init__(self, voicecraft_source, freeze_encoder, sample_rate,
                 hook_layer=12, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy import — VoiceCraft must be on PYTHONPATH or installed
        from VCmodels.voicecraft import VoiceCraft as _VoiceCraft

        self.model = _VoiceCraft.from_pretrained(voicecraft_source)
        self.model.eval()

        # EnCodec for audio tokenisation (24kHz)
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.eval()
        for p in self.codec.parameters():
            p.requires_grad = False

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.model.parameters():
                p.requires_grad = False

        self.sample_rate = sample_rate
        self.output_hidden_states = False

        # Register hook on the specified decoder layer
        self._hooked_output = None
        target_layer = self.model.decoder.layers[hook_layer]
        target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # output may be a tensor or a tuple (hidden, …)
        if isinstance(output, (tuple, list)):
            self._hooked_output = output[0]
        else:
            self._hooked_output = output

    @torch.no_grad()
    def forward(self, x, lengths=None):
        """
        Parameters
        ----------
        x : Tensor (B, T)
            Waveforms at ``self.sample_rate``.
        lengths : Tensor (B,), optional
            Relative lengths in [0, 1].  Currently unused.

        Returns
        -------
        Tensor (B, T', D)
            Hidden states from the hooked decoder layer.
        """
        device = x.device

        # Resample to codec rate if needed
        wav = x

        # EnCodec expects (B, channels, T)
        wav_enc = wav.unsqueeze(1).to(device)

        # Move codec to same device
        self.codec = self.codec.to(device)

        encoded = self.codec.encode(wav_enc)
        # encoded is a list of (codes, scale) tuples per frame
        codes = torch.cat([fr[0] for fr in encoded], dim=-1)  # (B, n_q, T_codes)

        # Move model to same device
        self.model = self.model.to(device)

        self._hooked_output = None

        # VoiceCraft.forward() expects a complex batch dict — bypass it and
        # manually embed codes then feed through the decoder transformer.
        # codes shape: (B, n_q, T_codes) — use only the codebooks the model knows
        n_q = min(codes.shape[1], self.model.args.n_codebooks)
        embedded = torch.stack(
            [self.model.audio_embedding[k](codes[:, k, :]) for k in range(n_q)],
            dim=0,
        )  # (K, B, T_codes, D)
        embedded = embedded.sum(dim=0)  # (B, T_codes, D)
        embedded = self.model.audio_positional_embedding(embedded)  # (B, T_codes, D)

        # Feed through the decoder (TransformerEncoder); triggers the hook
        _ = self.model.decoder((embedded, None))

        h = self._hooked_output
        if h is None:
            raise RuntimeError(
                "VoiceCraft hook did not fire. Check hook_layer index and forward signature."
            )

        return h


if __name__ == "__main__":
    model = VoiceCraftEncoder(
        voicecraft_source="pyp1/VoiceCraft",
        freeze_encoder=True,
        sample_rate=24000,
        hook_layer=12,
    )

    dummy_wav = torch.randn(1, 24000 * 2)  # batch of 2, 5 seconds at 24kHz
    dummy_lengths = torch.tensor([1.0, 0.8])

    features = model(dummy_wav, dummy_lengths)
    print(features.shape)  # expected: (2, T', 1024)
