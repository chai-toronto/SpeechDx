"""
Bark encoder — hook-based feature extraction from Bark's semantic (text) model.

Encodes audio with EnCodec at 24kHz, takes the first codebook, and feeds
tokens through Bark's semantic transformer.  Hidden states are captured from
a specified transformer layer via a forward hook.  Returns frame-level
features of shape (B, T', D).
"""

import torch
import torch.nn as nn
import torchaudio

# Bark checkpoints use pickle-based torch.save; PyTorch 2.6+ defaults to
# weights_only=True which rejects them.  Allow unsafe loading before importing.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from encodec import EncodecModel


class BarkEncoder(nn.Module):
    """
    Bark semantic-model hook-based feature extractor.

    Parameters
    ----------
    freeze_encoder : bool
        If True, all parameters are frozen.
    sample_rate : int
        Expected input waveform sample rate (should be 24000).
    model_type : str
        Bark model bundle to load (default ``"text"`` = semantic model).
    hook_layer : int
        Transformer layer index to hook (default 12, mid-point of 24 layers).
    """

    def __init__(self, freeze_encoder, sample_rate, model_type="text",
                 hook_layer=12, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from bark.generation import load_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = load_model(model_type=model_type, use_gpu=torch.cuda.is_available())
        self.semantic_model = bundle["model"]
        self.semantic_model.eval()

        # EnCodec for audio tokenisation (24kHz)
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.eval()
        for p in self.codec.parameters():
            p.requires_grad = False

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.semantic_model.parameters():
                p.requires_grad = False

        self.sample_rate = sample_rate
        self.output_hidden_states = False

        # Register hook — find transformer blocks
        self._hooked_output = None
        blocks = self._find_transformer_blocks()
        layer_idx = min(hook_layer, len(blocks) - 1)
        blocks[layer_idx].register_forward_hook(self._hook_fn)

    def _find_transformer_blocks(self):
        """Locate the main transformer block list in the semantic model."""
        # Common attribute names in GPT-style models
        for attr in ["h", "blocks", "layers", "transformer.h", "transformer.blocks"]:
            parts = attr.split(".")
            obj = self.semantic_model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if isinstance(obj, nn.ModuleList) and len(obj) >= 4:
                    return obj
            except AttributeError:
                continue

        # Fallback: search all ModuleLists
        candidates = []
        for name, module in self.semantic_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 4:
                candidates.append((name, module))
        if candidates:
            _, blocks = max(candidates, key=lambda x: len(x[1]))
            return blocks

        raise RuntimeError("Cannot find transformer blocks in Bark semantic model.")

    def _hook_fn(self, module, input, output):
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
            Hidden states from the hooked transformer layer.
        """
        device = x.device
        wav = x
        # EnCodec: (B, 1, T) -> codes (B, n_q, T_codes)
        wav_enc = wav.unsqueeze(1).to(device)
        self.codec = self.codec.to(device)

        encoded = self.codec.encode(wav_enc)
        codes = torch.cat([fr[0] for fr in encoded], dim=-1)  # (B, n_q, T_codes)

        # Use first codebook only (semantic tokens)
        tokens = codes[:, 0, :]  # (B, T_codes)

        # Move semantic model to device
        self.semantic_model = self.semantic_model.to(device)
        self._hooked_output = None

        # Bark's GPT-style model expects token ids
        # Feed through the model — try common signatures
        try:
            _ = self.semantic_model(tokens)
        except TypeError:
            try:
                _ = self.semantic_model(input_ids=tokens)
            except TypeError:
                _ = self.semantic_model(tokens, merge_context=False)

        h = self._hooked_output
        if h is None:
            raise RuntimeError(
                "Bark hook did not fire. Check hook_layer index and forward signature."
            )

        # Normalise to (B, T', D)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if h.dim() == 3:
            # If shape is (B, D, T') rather than (B, T', D), transpose
            if h.shape[1] > h.shape[2]:
                h = h.transpose(1, 2)

        return h


if __name__ == "__main__":
    model = BarkEncoder(
        freeze_encoder=True,
        sample_rate=24000,
        model_type="text",
        hook_layer=12,
    )

    dummy_wav = torch.randn(2, 24000 * 1)  # batch of 2, 5 seconds at 24kHz
    dummy_lengths = torch.tensor([1.0, 0.8])

    features = model(dummy_wav, dummy_lengths)
    print(features.shape)  # expected: (2, T', 1024)
