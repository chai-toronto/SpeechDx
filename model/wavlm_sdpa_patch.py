"""Route ``WavLMAttention`` through ``F.scaled_dot_product_attention``.

The transformers WavLM implementation (4.57.3 — still true on main as of
this writing) computes attention via ``F.multi_head_attention_forward``,
which materializes the full ``(B, H, T, T)`` score matrix. For 300 s audio
chunks (T=15000 frames) that's ~14 GB fp32 per chunk per layer, which
caps the warm batch size hard on a 40 GB MIG slice.

PyTorch's SDPA accepts an additive float ``attn_mask`` and on GPU routes
to the memory-efficient backend (or FlashAttention-2 on supported masks),
turning attention memory from O(B·H·T²) into O(B·H·T·d_k). WavLM's
*gated* relative position bias is just such an additive bias on the
attention scores — so the only thing we have to do is reshape it from
``(B*H, T, T)`` to ``(B, H, T, T)`` and hand it to SDPA.

Behavioural notes:
- ``output_attentions=True`` would require materializing weights anyway,
  so this patch refuses that path (raises) — the warm path never asks
  for it (``output_hidden_states: False`` in wavlm.yaml; the model returns
  ``last_hidden_state`` only).
- Key padding (``attention_mask`` in the WavLM forward) is folded into the
  same additive mask by adding ``-inf`` where the key is a pad token.
- Dropout matches the original (``self.dropout`` is the float prob).
- ``self.training`` is False in our use (encoder is ``.eval()``-ed and
  parameters are frozen), so dropout is a no-op either way; we still pass
  it through for completeness.

Apply once via ``enable_wavlm_sdpa()``; subsequent calls are no-ops.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.wavlm import modeling_wavlm as _mw

_PATCH_FLAG = "_sdx_sdpa_patched"


def _torch_multi_head_self_attention_sdpa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    gated_position_bias: torch.Tensor,
    output_attentions: bool,
):
    """Drop-in replacement for ``WavLMAttention.torch_multi_head_self_attention``."""
    if output_attentions:
        raise NotImplementedError(
            "wavlm_sdpa_patch: output_attentions=True is unsupported; SDPA "
            "does not materialize the per-head score matrix. Set "
            "output_attentions=False (the warm path already does)."
        )

    B, T, _ = hidden_states.shape
    H = self.num_heads
    d_k = self.head_dim

    q = self.q_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)

    # gated_position_bias arrives as (B*H, T, T); the upstream code flattens
    # via .view(B, H, T, T).flatten(0, 1), so .view back is exact.
    attn_bias = gated_position_bias.view(B, H, T, T)

    if attention_mask is not None:
        # WavLM convention: 1 = keep, 0 = pad (vs HF "padding mask" elsewhere
        # which is True=pad). The original code does `attention_mask.ne(1)`
        # to derive the bool padding mask — match that.
        key_padding = attention_mask.ne(1)  # (B, T) — True at pad positions
        # Promote to additive float mask and broadcast over (H, T_q).
        pad_bias = torch.zeros(
            (B, 1, 1, T), dtype=attn_bias.dtype, device=attn_bias.device,
        ).masked_fill(key_padding[:, None, None, :], float("-inf"))
        attn_bias = attn_bias + pad_bias

    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    # (B, H, T, d_k) -> (B, T, H*d_k)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * d_k)
    attn_output = self.out_proj(attn_output)
    return attn_output, None


def enable_wavlm_sdpa() -> None:
    """Monkey-patch transformers' ``WavLMAttention`` to use SDPA. Idempotent."""
    cls = _mw.WavLMAttention
    if getattr(cls, _PATCH_FLAG, False):
        return
    cls._sdx_orig_torch_multi_head_self_attention = cls.torch_multi_head_self_attention
    cls.torch_multi_head_self_attention = _torch_multi_head_self_attention_sdpa
    setattr(cls, _PATCH_FLAG, True)


def disable_wavlm_sdpa() -> None:
    """Restore the original ``torch_multi_head_self_attention``. For tests."""
    cls = _mw.WavLMAttention
    if not getattr(cls, _PATCH_FLAG, False):
        return
    cls.torch_multi_head_self_attention = cls._sdx_orig_torch_multi_head_self_attention
    delattr(cls, "_sdx_orig_torch_multi_head_self_attention")
    setattr(cls, _PATCH_FLAG, False)
