"""Route ``WavLMAttention`` through its ``q_proj`` / ``k_proj`` / ``v_proj`` modules.

transformers' WavLM attention (4.57.3) hands the raw projection weights to
``F.multi_head_attention_forward``::

    F.multi_head_attention_forward(..., torch.cat((q_proj.bias, k_proj.bias, v_proj.bias)),
                                   ..., q_proj_weight=q_proj.weight, ...)

so the projection modules are never *called*. Anything that wraps them — a
LoRA adapter on ``q_proj`` / ``v_proj`` (``sdx/lora``), a forward hook — is
bypassed, or crashes because the wrapper has no ``.weight`` / ``.bias``.

This patch replaces ``WavLMAttention.torch_multi_head_self_attention`` with a
version that calls the modules but otherwise replays, op for op, what
``F.multi_head_attention_forward`` does on this path (``need_weights=False``):
project in ``(T, B, E)`` layout, merge the gated relative-position bias with
the key-padding mask as an additive float mask, run
``F.scaled_dot_product_attention`` with the same tensor views, then
``out_proj``. Keeping the layouts identical is what makes it bit-exact: an
earlier variant that projected in ``(B, T, E)`` differed by ~1e-4 on CPU for
padded batches. Verified bit-identical to the stock kernel on WavLM-Large, fp32,
CPU and MPS, for single chunks and padded batches (``tests/test_wavlm_patch.py``
covers a small random WavLM).

Memory and speed are unchanged: with ``need_weights=False`` torch already runs
SDPA inside ``F.multi_head_attention_forward``, and the ``(B*H, T, T)`` gated
position bias is materialized either way.

``output_attentions=True`` falls through to the stock implementation (which
needs unwrapped projections). Apply with ``enable_wavlm_sdpa()``; it patches
the class, so it affects every WavLM in the process, and is idempotent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.wavlm import modeling_wavlm as _mw

_PATCH_FLAG = "_sdx_sdpa_patched"


def _torch_multi_head_self_attention_sdpa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gated_position_bias: torch.Tensor,
    output_attentions: bool,
):
    """Drop-in replacement for ``WavLMAttention.torch_multi_head_self_attention``."""
    if output_attentions:
        return type(self)._sdx_orig_torch_multi_head_self_attention(
            self, hidden_states, attention_mask, gated_position_bias,
            output_attentions)

    bsz, tgt_len, embed_dim = hidden_states.shape
    heads, head_dim = self.num_heads, self.head_dim
    # Same (T, B, E) layout as the stock path: query = key = value.
    x = hidden_states.transpose(0, 1)

    def heads_view(t):
        # (T, B, E) -> (B*H, T, d) -> (B, H, T, d), exactly as torch does.
        t = t.view(tgt_len, bsz * heads, head_dim).transpose(0, 1)
        return t.view(bsz, heads, tgt_len, head_dim)

    q = heads_view(self.q_proj(x))
    k = heads_view(self.k_proj(x))
    v = heads_view(self.v_proj(x))

    attn_mask = gated_position_bias
    if attention_mask is not None:
        # WavLM convention: 1 = keep. torch turns the bool padding mask into
        # an additive float mask and adds it to the (B*H, T, T) bias.
        key_padding = torch.zeros(
            attention_mask.shape, dtype=attn_mask.dtype, device=attn_mask.device,
        ).masked_fill_(attention_mask.ne(1), float("-inf"))
        key_padding = (key_padding.view(bsz, 1, 1, tgt_len)
                       .expand(-1, heads, -1, -1)
                       .reshape(bsz * heads, 1, tgt_len))
        attn_mask = attn_mask + key_padding
    attn_mask = attn_mask.view(bsz, heads, tgt_len, tgt_len)

    dropout_p = self.dropout if self.training else 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, False)
    out = out.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
    out = self.out_proj(out).view(tgt_len, bsz, -1)
    # [T, B, E] -> [B, T, E], as the stock wrapper does.
    return out.transpose(0, 1), None


def enable_wavlm_sdpa() -> None:
    """Patch transformers' ``WavLMAttention`` to call its projections. Idempotent."""
    cls = _mw.WavLMAttention
    if getattr(cls, _PATCH_FLAG, False):
        return
    cls._sdx_orig_torch_multi_head_self_attention = cls.torch_multi_head_self_attention
    cls.torch_multi_head_self_attention = _torch_multi_head_self_attention_sdpa
    setattr(cls, _PATCH_FLAG, True)


def disable_wavlm_sdpa() -> None:
    """Restore the stock ``torch_multi_head_self_attention``. Idempotent."""
    cls = _mw.WavLMAttention
    if not getattr(cls, _PATCH_FLAG, False):
        return
    cls.torch_multi_head_self_attention = cls._sdx_orig_torch_multi_head_self_attention
    delattr(cls, "_sdx_orig_torch_multi_head_self_attention")
    setattr(cls, _PATCH_FLAG, False)


def wavlm_sdpa_enabled() -> bool:
    return bool(getattr(_mw.WavLMAttention, _PATCH_FLAG, False))
