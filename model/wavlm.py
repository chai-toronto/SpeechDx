import torch
from speechbrain.dataio.dataio import length_to_mask
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn

# transformers' WavLMAttention runs F.multi_head_attention_forward, which
# materializes (B, H, T, T) score AND (B, H, T, T) gated-pos-bias tensors —
# both T²-bound. Two patches reduce that:
#   - SDPA   : kills the score + softmax matrices (bias still materialized).
#   - flex   : score_mod recreates the gated bias inside the kernel, so
#              nothing (B, H, T, T) is ever allocated.
# Patches are applied lazily by the WavLM class init based on attn_impl.
from model.wavlm_sdpa_patch import enable_wavlm_sdpa, disable_wavlm_sdpa
from model.wavlm_flex_patch import enable_wavlm_flex_attention, disable_wavlm_flex_attention


def _apply_attn_impl(impl: str) -> str:
    """Install the requested WavLM attention patch (or none). Returns the
    impl that ended up active (may differ from requested if flex falls back
    to sdpa on a non-CUDA build or import failure)."""
    impl = (impl or "original").lower()
    if impl not in {"flex", "sdpa", "original"}:
        raise ValueError(
            f"WavLM attn_impl: expected one of flex / sdpa / original, "
            f"got {impl!r}."
        )
    if impl == "flex":
        try:
            disable_wavlm_sdpa()
            enable_wavlm_flex_attention()
            return "flex"
        except Exception as e:  # noqa: BLE001 — fall back, don't crash
            print(f"[wavlm] flex attention setup failed ({type(e).__name__}: "
                  f"{e}); falling back to sdpa.")
            disable_wavlm_flex_attention()
            enable_wavlm_sdpa()
            return "sdpa"
    if impl == "sdpa":
        disable_wavlm_flex_attention()
        enable_wavlm_sdpa()
        return "sdpa"
    # original
    disable_wavlm_flex_attention()
    disable_wavlm_sdpa()
    return "original"

# Map YAML strings to torch dtypes for the optional autocast knob below.
_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16":  torch.float16,  "fp16": torch.float16,  "half": torch.float16,
    "float32":  torch.float32,  "fp32": torch.float32,
}


def _resolve_dtype(spec):
    """Accept None, a torch.dtype, or a string alias from wavlm.yaml."""
    if spec is None or isinstance(spec, torch.dtype):
        return spec
    if isinstance(spec, str):
        try:
            return _DTYPE_ALIASES[spec.lower()]
        except KeyError as e:
            raise ValueError(
                f"WavLM compute_dtype: unknown alias {spec!r}; "
                f"expected one of {sorted(_DTYPE_ALIASES)}."
            ) from e
    raise TypeError(f"WavLM compute_dtype: unexpected type {type(spec).__name__}")


class WavLM(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states,
                 sample_rate, *args, compute_dtype=None,
                 attn_impl: str = "sdpa", **kwargs):
        super().__init__(*args, **kwargs)
        # Install the requested attention patch BEFORE WavLMModel is built so
        # the class state is consistent across all instances in this process.
        # The active impl may differ from the requested one if flex falls back.
        self.attn_impl = _apply_attn_impl(attn_impl)

        # ``compute_dtype`` controls both the loaded weight dtype and the
        # autocast context wrapping forward. Loading the weights at the
        # target dtype is what halves the materialized position bias (the
        # warm-batch bottleneck): the bias comes out of self.rel_attn_embed
        # which is a Parameter — and Parameter dtype follows the model dtype,
        # not autocast. Autocast on top remains as a safety net for any op
        # that allocates an fp32 intermediate.
        self.compute_dtype = _resolve_dtype(compute_dtype)

        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        # ``from_pretrained`` accepts ``dtype`` (the new spelling — older
        # versions accepted ``torch_dtype`` with a deprecation warning).
        # In some transformers builds ``torch_dtype`` is silently dropped,
        # leaving the model in fp32; defensively cast with ``.to(...)``
        # after loading so the param/buffer dtype is pinned regardless.
        load_kwargs = {}
        if self.compute_dtype is not None:
            load_kwargs["dtype"] = self.compute_dtype
        try:
            self.feature_extractor = WavLMModel.from_pretrained(
                ssl_encoder_source, **load_kwargs,
            )
        except TypeError:
            # Fallback for older transformers that only know torch_dtype.
            load_kwargs = ({"torch_dtype": self.compute_dtype}
                           if self.compute_dtype is not None else {})
            self.feature_extractor = WavLMModel.from_pretrained(
                ssl_encoder_source, **load_kwargs,
            )
        if self.compute_dtype is not None:
            self.feature_extractor.to(self.compute_dtype)

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

        # Cast the model input to compute_dtype before forward so the
        # whole encoder path (input → weights → output) is one dtype,
        # matching what the on-disk cache will store. If compute_dtype
        # is None we fall back to the input wav's dtype (fp32) as before.
        input_dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        input_values = self.processor(
            x_list, sampling_rate=self.sample_rate,
            return_tensors="pt", padding=True,
        ).input_values.to(device=x.device, dtype=input_dtype)

        mask = None
        if abs_len is not None:
            mask = length_to_mask(abs_len, max_len=input_values.shape[1]).to(x.device)

        # torch.autocast is a no-op when the device backend doesn't support
        # the requested dtype, so guarding by device.type keeps CPU smoke
        # tests honest (CPU autocast bf16 works on AVX512 hosts but mixing
        # paths is fragile; we only autocast on GPU here).
        if self.compute_dtype is not None and input_values.device.type == "cuda":
            ac_ctx = torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        else:
            from contextlib import nullcontext
            ac_ctx = nullcontext()

        with ac_ctx:
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