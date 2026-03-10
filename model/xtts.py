"""
XTTS v2 speaker encoder — wraps Coqui TTS ``get_conditioning_latents``.

Returns an utterance-level embedding of shape (B, 1, D).
Because the Coqui API requires file paths, each sample in the batch is
written to a temporary WAV file before extraction.
"""

import os
import tempfile

import torch
import torch.nn as nn
import soundfile as sf

# TTS 0.22 checkpoints use pickle-based torch.save; PyTorch 2.6+ defaults to
# weights_only=True which rejects them.  Allow unsafe loading before importing.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS


class XTTS(nn.Module):
    """
    XTTS v2 speaker-embedding extractor.

    Parameters
    ----------
    model_name : str
        Coqui TTS model identifier, e.g. ``tts_models/multilingual/multi-dataset/xtts_v2``.
    freeze_encoder : bool
        If True, all parameters are frozen (typical for probing).
    sample_rate : int
        Expected input waveform sample rate (should be 22050 for XTTS native rate).
    which : str
        Which embedding to extract: ``"speaker_embedding"`` (512-d) or
        ``"gpt_cond_latent"`` (variable length).
    """

    def __init__(self, model_name, freeze_encoder, sample_rate,
                 which="speaker_embedding", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tts = TTS(model_name)
        self.tts_model = self.tts.synthesizer.tts_model
        self.tts_model.eval()

        self.which = which
        self.freeze_encoder = freeze_encoder
        self.sample_rate = sample_rate
        self.output_hidden_states = False

        if freeze_encoder:
            for param in self.tts_model.parameters():
                param.requires_grad = False

    def _extract_from_path(self, wav_path):
        """Extract both conditioning latents from a wav file path.

        Returns (gpt_cond_latent, speaker_embedding) as raw tensors.
        """
        return self.tts_model.get_conditioning_latents(audio_path=[wav_path])

    def _extract_from_tensor(self, wav_tensor):
        """Extract both conditioning latents from a waveform tensor.

        Writes to a temp file since the Coqui API requires file paths.
        Returns (gpt_cond_latent, speaker_embedding) as raw tensors.
        """
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "ref.wav")
            sf.write(tmp_path, wav_tensor.cpu().numpy(), self.sample_rate, subtype='PCM_16')
            return self._extract_from_path(tmp_path)

    def encode_path(self, wav_path):
        """Encode a single wav file path, returning the selected embedding.

        Returns
        -------
        Tensor
            ``(D,)`` for speaker_embedding, ``(T', D)`` for gpt_cond_latent.
        """
        gpt_cond_latent, speaker_embedding = self._extract_from_path(wav_path)
        return self._concat_embedding(gpt_cond_latent, speaker_embedding)

    def encode_paths(self, wav_paths):
        """Encode a list of wav file paths, returning both embeddings per path.

        Returns
        -------
        list[dict]
            Each dict has keys ``"gpt_cond_latent"`` and ``"speaker_embedding"``.
        """
        results = []
        for path in wav_paths:
            gpt_cond_latent, speaker_embedding = self._extract_from_path(path)
            results.append({
                "gpt_cond_latent": gpt_cond_latent.squeeze(0).cpu(),
                "speaker_embedding": speaker_embedding.squeeze().cpu(),
            })
        return results

    def _concat_embedding(self, gpt_cond_latent, speaker_embedding):
        """spk emb padded over dim and insert @ pos 0 """
        speaker_embedding = speaker_embedding.squeeze(-1)  # (1, D)
        expanded_spk_emb = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)],
                                                 dim = 1) # [1, D]

        return torch.cat([expanded_spk_emb, gpt_cond_latent.squeeze(0), ], dim=0)  # (T', D)

    @torch.no_grad()
    def forward(self, x, lengths=None):
        """
        Parameters
        ----------
        x : Tensor (B, T)
            Waveforms at ``self.sample_rate``.
        lengths : Tensor (B,), optional
            Relative lengths in [0, 1]. Used to trim padding before saving.

        Returns
        -------
        Tensor (B, 1, D)
            Speaker embeddings.
        """
        batch_size = x.shape[0]
        embeddings = []

        for i in range(batch_size):
            wav_i = x[i]  # (T,)

            # Trim to actual length if lengths provided
            if lengths is not None:
                actual_len = int(lengths[i].item() * wav_i.shape[0])
                wav_i = wav_i[:actual_len]

            gpt_cond_latent, speaker_embedding = self._extract_from_tensor(wav_i)
            emb = self._concat_embedding(gpt_cond_latent, speaker_embedding)
            embeddings.append(emb.squeeze())  # (D,) or (T', D)

        max_len = max(emb.shape[0] for emb in embeddings)

        padded_embeddings = []

        for emb in embeddings:
            if emb.shape[0] < max_len:
                pad = torch.zeros(max_len - emb.shape[0], emb.shape[1], device=emb.device)
                emb = torch.cat([emb, pad], dim=0)

            padded_embeddings.append(emb)

        embeddings = padded_embeddings

        out = torch.stack(embeddings, dim=0)  # (B, D) or (B, T', D)
        return out.to(x.device)


if __name__ == "__main__":
    model = XTTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        freeze_encoder=True,
        sample_rate=22050,
        which="speaker_embedding",
    )

    dummy_wav = torch.randn(2, 22050 * 5)  # batch of 2, 5 seconds at 22050 Hz
    dummy_lengths = torch.tensor([1.0, 0.8])

    features = model(dummy_wav, dummy_lengths)
    print(features.shape)  # expected: (2, 1, 512)
