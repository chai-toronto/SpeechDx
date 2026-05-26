"""Compute WavLM's gated relative position bias *inside* a flex_attention
kernel, so the ``(B, H, T, T)`` bias tensor never materializes.

The SDPA patch ([[wavlm_sdpa_patch.py]]) eliminated the score / softmax
matrices, but `WavLMAttention.forward` still allocates the gated relative
position bias as a `(B, H, T, T)` float tensor before handing it to
attention. For T=15000 (300 s chunks), that's ~14 GB fp32 / ~7 GB bf16 per
forward — the binding constraint on the warm batch size.

flex_attention (PyTorch ≥ 2.5) takes a ``score_mod`` callable that runs
inside the Triton kernel and reads its inputs by indexed lookup. We can
recreate WavLM's gated bias entirely there:

    gated_bias[b, h, q, k] = gate[b, h, q] * rel_attn_embed[bucket(k - q), h]

with the bucket function from transformers' ``_relative_positions_bucket``
inlined. The captured tensors are:

  - ``gate``           shape (B, H, T)         — recomputed per layer.
  - ``rel_attn_embed`` shape (num_buckets, H)  — tiny, constant per layer.

…neither of which is O(T²). Per-layer peak attention memory drops to
O(B·H·T·d_k) — the dotted-Q/K/V tensors themselves.

Patches `WavLMAttention.forward` (not just the inner kernel — we need to
skip the position-bias build). Idempotent. Mutually exclusive with the SDPA
patch (which would otherwise reroute the inner kernel underneath us).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers.models.wavlm import modeling_wavlm as _mw

_PATCH_FLAG = "_sdx_flex_patched"

# Compile with dynamic=True so the per-batch T (padded to max chunk in
# batch) can vary without thrashing the inductor cache. fullgraph=False
# keeps Python escape hatches available for the score_mod's int math.
_flex_attention = torch.compile(flex_attention, dynamic=True, fullgraph=False)


def _build_score_mod(gate: torch.Tensor, rel_embed: torch.Tensor,
                     num_buckets: int, max_distance: int):
    """Closure that recreates WavLM's gated rel-pos bias inside the kernel.

    `gate` (B, H, T) and `rel_embed` (num_buckets, H) are captured as
    flex_attention "buffer" tensors — looked up element-wise per (b, h, q, k)
    rather than read in bulk. The bucket function inlines
    ``WavLMAttention._relative_positions_bucket``.
    """
    num_buckets_half = num_buckets // 2
    max_exact = num_buckets_half // 2
    # Constants outside the closure so they're folded into the compiled
    # graph rather than re-read per element.
    inv_log_scale = 1.0 / math.log(max_distance / max_exact)
    range_factor = float(num_buckets_half - max_exact)
    last_bucket = num_buckets_half - 1
    inv_max_exact = 1.0 / max_exact

    def score_mod(score, b, h, q_idx, k_idx):
        # rp = k - q ; signed bucket offset for positive rp.
        rp = k_idx - q_idx
        sign_off = (rp > 0).to(torch.long) * num_buckets_half
        abs_rp = rp.abs()
        is_small = abs_rp < max_exact
        # Large-bucket branch: log-spaced. Clamp to ≥1 so log() is finite
        # in the small/zero branch (the result is discarded by `where`).
        safe = torch.clamp(abs_rp, min=1).to(score.dtype)
        large_f = max_exact + (torch.log(safe * inv_max_exact)
                               * inv_log_scale
                               * range_factor)
        large = large_f.to(torch.long)
        large = torch.clamp(large, max=last_bucket)
        bucket = sign_off + torch.where(is_small, abs_rp, large)
        pos_bias = rel_embed[bucket, h]
        return score + gate[b, h, q_idx] * pos_bias

    return score_mod


def _build_key_padding_block_mask(attention_mask: torch.Tensor,
                                   B: int, T: int, device) -> "torch.Tensor":
    """create_block_mask for key padding (mask out padded keys)."""
    am = attention_mask  # (B, T) — 1 = keep, 0 = pad

    def mask_mod(b, h, q_idx, k_idx):
        return am[b, k_idx] == 1

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T,
                             device=device, _compile=True)


def _wavlm_attention_forward_flex(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_bias: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    index: int = 0,
):
    """Drop-in replacement for ``WavLMAttention.forward`` using flex_attention.

    Upstream contract: only the first attention layer has ``rel_attn_embed``;
    it computes the rel-pos bias and returns it as ``position_bias``, which
    every subsequent layer receives and reuses. We keep the same handoff
    but ship the *embedding weight* (small: ``(num_buckets, H)``) instead
    of the materialized ``(H, T, T)`` tensor. The score_mod looks bucket
    values up element-wise inside the kernel.
    """
    if output_attentions:
        raise NotImplementedError(
            "wavlm_flex_patch: output_attentions=True is unsupported."
        )
    B, T, _ = hidden_states.shape
    H = self.num_heads
    d_k = self.head_dim

    # Layer 0 carries the embedding; subsequent layers receive it via
    # the position_bias arg from the layer above.
    if hasattr(self, "rel_attn_embed"):
        rel_embed_weight = self.rel_attn_embed.weight
    elif position_bias is not None:
        rel_embed_weight = position_bias
    else:
        raise RuntimeError(
            "wavlm_flex_patch: this attention layer has no rel_attn_embed "
            "and no position_bias was passed in. Expected layer 0 to "
            "populate position_bias and downstream layers to forward it."
        )

    # Gate computation — same expression as upstream, kept as (B, H, T).
    gated_hs = hidden_states.view(B, T, H, d_k).permute(0, 2, 1, 3)
    rel_proj = self.gru_rel_pos_linear(gated_hs)             # (B, H, T, 8)
    rel_proj = rel_proj.view(B, H, T, 2, 4).sum(-1)          # (B, H, T, 2)
    gate_a, gate_b = torch.sigmoid(rel_proj).chunk(2, dim=-1) # each (B, H, T, 1)
    # gru_rel_pos_const has shape (1, H, 1, 1) — broadcasts cleanly against
    # gate_b (B, H, T, 1). DO NOT squeeze it; squeezing the trailing 1
    # gives (1, H, 1) which broadcasts as (1, 1, H, 1) and collides H↔T.
    gate = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
    gate = gate.squeeze(-1)                                  # (B, H, T)

    # Q/K/V projections.
    q = self.q_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, T, H, d_k).transpose(1, 2)

    score_mod = _build_score_mod(
        gate, rel_embed_weight, self.num_buckets, self.max_distance,
    )

    if attention_mask is not None:
        block_mask = _build_key_padding_block_mask(
            attention_mask, B, T, hidden_states.device,
        )
    else:
        block_mask = None

    attn_output = _flex_attention(q, k, v, score_mod=score_mod,
                                  block_mask=block_mask)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H * d_k)
    attn_output = self.out_proj(attn_output)
    # Forward the embedding weight so downstream layers can reuse it.
    return attn_output, None, rel_embed_weight


def enable_wavlm_flex_attention() -> None:
    """Patch ``WavLMAttention.forward`` to use flex_attention. Idempotent."""
    cls = _mw.WavLMAttention
    if getattr(cls, _PATCH_FLAG, False):
        return
    # Make sure the SDPA patch isn't also active — they would conflict
    # (SDPA patches the inner kernel; flex replaces the outer forward).
    if getattr(cls, "_sdx_sdpa_patched", False):
        from model.wavlm_sdpa_patch import disable_wavlm_sdpa
        disable_wavlm_sdpa()
    cls._sdx_orig_forward = cls.forward
    cls.forward = _wavlm_attention_forward_flex
    setattr(cls, _PATCH_FLAG, True)


def disable_wavlm_flex_attention() -> None:
    cls = _mw.WavLMAttention
    if not getattr(cls, _PATCH_FLAG, False):
        return
    cls.forward = cls._sdx_orig_forward
    delattr(cls, "_sdx_orig_forward")
    setattr(cls, _PATCH_FLAG, False)
