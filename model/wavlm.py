import torch
from speechbrain.dataio.dataio import length_to_mask
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn

from model.wavlm_sdpa_patch import disable_wavlm_sdpa, enable_wavlm_sdpa

_ATTN_IMPLS = ("sdpa", "original")


class WavLM(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate,
                 *args, attn_impl: str = "sdpa", **kwargs):
        super().__init__(*args, **kwargs)
        # attn_impl selects WavLMAttention's implementation (class-wide, so it
        # applies to every WavLM in the process):
        #   "sdpa"     — model/wavlm_sdpa_patch.py: calls q/k/v_proj as modules
        #                so wrappers such as LoRA adapters take effect. Bit-
        #                identical to "original" (same op sequence).
        #   "original" — transformers' stock kernel, which reads the raw
        #                projection weights.
        if attn_impl not in _ATTN_IMPLS:
            raise ValueError(f"WavLM attn_impl: expected one of {_ATTN_IMPLS}, "
                             f"got {attn_impl!r}")
        (enable_wavlm_sdpa if attn_impl == "sdpa" else disable_wavlm_sdpa)()
        self.attn_impl = attn_impl
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        """Encode a batch of waveforms.

        x       : ``(B, T)`` tensor (or ``(T,)``). For a ragged batch the
                  caller right-zero-pads every row to a common ``T``.
        lengths : ``(B,)`` relative true lengths in ``(0, 1]`` — required to
                  interpret a padded ``x``. ``None`` means every row is full.

        With ``lengths`` each row is sliced back to its true length so the
        processor normalizes it per utterance (not over the zero padding),
        and the transformer is masked so padded frames don't leak in.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape
        if lengths is None:
            abs_len = None
            x_list = [x[i].detach().cpu().numpy() for i in range(B)]
        else:
            abs_len = (lengths.to(x.device).float() * T).round().long().clamp(1, T)
            x_list = [x[i, :int(abs_len[i])].detach().cpu().numpy() for i in range(B)]

        input_values = self.processor(
            x_list, sampling_rate=self.sample_rate,
            return_tensors="pt", padding=True,
        ).input_values.to(device=x.device, dtype=x.dtype)

        mask = None
        if abs_len is not None:
            mask = length_to_mask(abs_len, max_len=input_values.shape[1]).to(x.device)

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

    def feature_lengths(self, sample_lengths):
        """Encoder output frame count for each input sample length.

        Lets the batched cache warmer crop padded frames off a ``(B, T', D)``
        output back to each chunk's true frame count.
        """
        return self.feature_extractor._get_feat_extract_output_lengths(
            torch.as_tensor(sample_lengths)).long()


if __name__ == "__main__":
    # simple test
    model = WavLM(
        ssl_encoder_source="microsoft/wavlm-base-plus",
        freeze_encoder=True,
        output_hidden_states=False,
        sample_rate=16000
    )

    dummy_wav = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds each
    dummy_lengths = torch.tensor([1.0, 0.8])  # first is full length, second is 80%

    features = model(dummy_wav, dummy_lengths)
    print(features.shape)  # should print (2, T', D) where T' depends on the model's downsampling