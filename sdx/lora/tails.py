"""Prefix tap and adapted-tail execution.

Two halves of the same cut:

* :func:`capture_boundary` runs an encoder forward and grabs the tensor
  entering block ``cut`` — the only thing the boundary cache stores. A
  pre-hook short-circuits the forward so the frozen suffix is never computed
  during cache construction.
* :class:`EncoderTail` replays ``blocks[cut:]`` over that tensor, plus whatever
  the encoder applies between its last block and the benchmark's tap point, so
  that ``tail(capture(x)) == encoder(x)``. ``scripts/lora_parity.py`` asserts
  this against the real weights.

There is no generic "run the last N blocks" in transformers, but the variation
is narrow enough to be declared rather than coded: each encoder's
:class:`~sdx.lora.targets.EncoderTargetSpec` names a *runner* (how a block is
called), static per-block kwargs, and the *post* modules up to the tap. Only
blocks that thread state their parent builds need a runner of their own:

* **sequential** (Whisper, and most HF / timm encoders): call each block with
  the spec's ``block_kwargs``, keep output ``[0]`` of a tuple, then apply
  ``post`` (the final norm, plus any CLS-drop / pooling the wrapper does
  before the tap).
* **wavlm**: WavLM threads a ``position_bias`` through its layers, created by
  layer 0 (the only one with ``rel_attn_embed``). It is a pure function of
  sequence length, so the tail recomputes it rather than caching an O(T^2)
  tensor. The Large config is ``do_stable_layer_norm=True``, so
  ``encoder.layer_norm`` is applied AFTER the loop and belongs to the tail.
* **delegate** (Qwen3-TTS / Mimi): the parent transformer computes a causal
  mask and rotary embeddings inside its ``forward``, which accepts
  ``hidden_states`` directly and does no pre-loop work — so the tail truncates
  ``layers`` and delegates, letting the parent rebuild the mask and rotary for
  the same sequence length.
"""

from __future__ import annotations

import contextlib
import types
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdx.lora.targets import SPECS, EncoderTargetSpec, _resolve


class _StopAtBoundary(Exception):
    """Raised by the tap hook to abandon the rest of the frozen forward."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__("boundary reached")
        self.tensor = tensor


@contextlib.contextmanager
def _boundary_hook(block: nn.Module, out: dict):
    def _pre(_mod, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        out["hidden"] = hidden.detach()
        raise _StopAtBoundary(hidden)

    handle = block.register_forward_pre_hook(_pre, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def capture_boundary(encoder: nn.Module, model_name: str, cut: int,
                     forward: Callable[[], object],
                     spec: EncoderTargetSpec | None = None) -> torch.Tensor:
    """Run ``forward()`` and return the activation entering ``blocks[cut]``.

    ``forward`` is a zero-arg thunk that invokes the encoder however the
    caller normally would (the wrappers each preprocess differently), e.g.
    ``lambda: encoder(wav)``. The frozen suffix is never executed.
    """
    spec = spec or SPECS[model_name]
    blocks = _resolve(encoder, spec.blocks)
    if not 0 <= cut < len(blocks):
        raise IndexError(f"cut {cut} out of range for {len(blocks)} blocks")
    out: dict = {}
    with _boundary_hook(blocks[cut], out):
        try:
            forward()
        except _StopAtBoundary:
            pass
    if "hidden" not in out:
        raise RuntimeError(
            f"{model_name}: block {cut} was never reached — the forward did "
            f"not traverse {spec.blocks}"
        )
    return out["hidden"]


class EncoderTail(nn.Module):
    """``blocks[cut:]`` and the ``post`` modules up to the tap, per the spec.

    The base class is the ``"sequential"`` runner. Subclasses override
    :meth:`run_blocks` only; ``post`` is always applied here.
    """

    def __init__(self, encoder: nn.Module, model_name: str, cut: int,
                 spec: EncoderTargetSpec | None = None):
        super().__init__()
        self.model_name = model_name
        self.cut = cut
        self.spec = spec or SPECS[model_name]
        blocks = _resolve(encoder, self.spec.blocks)
        self.num_blocks = len(blocks)
        self.blocks = nn.ModuleList(list(blocks)[cut:])
        self.block_kwargs = dict(self.spec.block_kwargs)
        apply_post = (self.spec.post_if is None
                      or bool(_resolve(encoder, self.spec.post_if)))
        self.post = nn.ModuleList(
            [p if isinstance(p, nn.Module) else _resolve(encoder, p)
             for p in self.spec.post] if apply_post else [])

    def _call_kwargs(self, attention_mask) -> dict:
        kwargs = dict(self.block_kwargs)
        if attention_mask is not None:
            if "attention_mask" not in kwargs:
                raise ValueError(
                    f"{self.model_name}: its blocks take no attention_mask")
            kwargs["attention_mask"] = attention_mask
        return kwargs

    def run_blocks(self, hidden: torch.Tensor, attention_mask=None) -> torch.Tensor:
        kwargs = self._call_kwargs(attention_mask)
        for layer in self.blocks:
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def forward(self, hidden: torch.Tensor, attention_mask=None) -> torch.Tensor:
        hidden = self.run_blocks(hidden, attention_mask)
        for op in self.post:
            hidden = op(hidden)
        return hidden


def _wavlm_attention_via_modules(self, hidden_states, attention_mask,
                                 gated_position_bias, output_attentions):
    """``WavLMAttention.torch_multi_head_self_attention`` that CALLS q/k/v_proj.

    transformers' version hands ``q_proj.weight`` / ``.bias`` straight to
    ``F.multi_head_attention_forward``, bypassing the projection modules, so a
    LoRA adapter wrapped around ``q_proj`` / ``v_proj`` would crash (it has no
    ``.bias``) or, worse, be silently ignored. Same math otherwise: separate
    q/k/v projections, the gated relative position bias as an additive mask,
    padded keys at -inf, scaled dot-product attention, then ``out_proj``.
    """
    if output_attentions:
        raise NotImplementedError(
            "LoRA WavLM tail: output_attentions=True is not supported")
    bsz, seq_len, _ = hidden_states.shape
    heads, head_dim = self.num_heads, self.head_dim

    def split(x):
        return x.view(bsz, seq_len, heads, head_dim).transpose(1, 2)

    q = split(self.q_proj(hidden_states))
    k = split(self.k_proj(hidden_states))
    v = split(self.v_proj(hidden_states))
    bias = gated_position_bias.view(bsz, heads, seq_len, seq_len)
    if attention_mask is not None:
        # WavLM convention: 1 = keep; HF derives the padding mask with ne(1).
        bias = bias.masked_fill(attention_mask.ne(1)[:, None, None, :],
                                float("-inf"))
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias.to(q.dtype),
        dropout_p=self.dropout if self.training else 0.0)
    out = out.transpose(1, 2).reshape(bsz, seq_len, heads * head_dim)
    return self.out_proj(out), None


class WavLMTail(EncoderTail):
    """WavLM suffix: recompute position_bias and thread it through the blocks.

    Also rebinds each tail block's attention so it calls ``q_proj`` / ``v_proj``
    as modules (see :func:`_wavlm_attention_via_modules`); without that, HF's
    WavLM attention reads the raw weights and LoRA never takes effect. Only
    the tail's own blocks are rebound (per instance), so the frozen benchmark
    warm keeps transformers' original kernel. ``scripts/lora_parity.py``
    checks the rebound tail against the unmodified encoder.
    """

    def __init__(self, encoder, model_name, cut, spec=None):
        super().__init__(encoder, model_name, cut, spec)
        # Layer 0 owns rel_attn_embed, and the bias must be rebuilt from
        # sequence length on every forward. Register it as a REAL submodule so
        # ``.to(device)`` moves it with the rest of the tail: compute_bias
        # builds its output on rel_attn_embed's device, so a copy held outside
        # the module tree stays on CPU while the tail runs on CUDA.
        #
        # Registering it is safe: freeze_all_but_adapters() freezes every
        # parameter and unfreezes only lora_A/lora_B, and the checkpoint is
        # adapter-only, so layer 0's weights are never trained or saved.
        all_blocks = list(_resolve(encoder, self.spec.blocks))
        # When cut == 0 the tail already owns block 0; reuse it rather than
        # registering the same module twice.
        src_block = self.blocks[0] if cut == 0 else all_blocks[0]
        self.bias_source = _resolve(src_block, self.spec.attn)
        import inspect as _inspect
        self._takes_index = "index" in _inspect.signature(
            type(self.blocks[0]).forward).parameters
        for block in self.blocks:
            attn = _resolve(block, self.spec.attn)
            attn.torch_multi_head_self_attention = types.MethodType(
                _wavlm_attention_via_modules, attn)

    def _position_bias(self, hidden: torch.Tensor) -> torch.Tensor:
        attn = self.bias_source
        bsz, seq_len, _ = hidden.size()
        bias = attn.compute_bias(seq_len, seq_len)
        return (bias.unsqueeze(0).repeat(bsz, 1, 1, 1)
                .view(bsz * attn.num_heads, seq_len, seq_len)).to(hidden.dtype)

    def run_blocks(self, hidden, attention_mask=None):
        position_bias = self._position_bias(hidden)
        for i, layer in enumerate(self.blocks):
            # The post-LN layer takes `index`; the stable-LN (pre-LN) layer that
            # Large uses does not. Pass it only when the signature has it.
            kwargs = {"attention_mask": attention_mask,
                      "position_bias": position_bias}
            if self._takes_index:
                kwargs["index"] = self.cut + i
            hidden, position_bias = layer(hidden, **kwargs)[:2]
        # The final norm is the spec's ``post``, gated on do_stable_layer_norm:
        # post-LN variants normalize before the loop, inside the cached prefix.
        return hidden


class DelegatingTail(EncoderTail):
    """For parents that accept ``hidden_states`` and do no pre-loop work.

    Truncates the parent's ``layers`` to the suffix and calls it, so the
    parent rebuilds masks / rotary embeddings itself. Used for Qwen's
    Mimi-style ``encoder_transformer``.
    """

    def __init__(self, encoder, model_name, cut, spec=None):
        super().__init__(encoder, model_name, cut, spec)
        self._parent = [_resolve(encoder, self.spec.blocks.rsplit(".", 1)[0])]
        self._attr = self.spec.blocks.rsplit(".", 1)[1]

    def run_blocks(self, hidden, attention_mask=None):
        if attention_mask is not None:
            raise ValueError(
                f"{self.model_name}: the delegating tail takes no attention_mask")
        parent, full = self._parent[0], None
        try:
            full = getattr(parent, self._attr)
            setattr(parent, self._attr, self.blocks)
            out = parent(hidden)
        finally:
            if full is not None:
                setattr(parent, self._attr, full)
        return getattr(out, "last_hidden_state", out)


RUNNERS: dict[str, type[EncoderTail]] = {
    "sequential": EncoderTail,
    "wavlm": WavLMTail,
    "delegate": DelegatingTail,
}


def build_tail(encoder: nn.Module, model_name: str, cut: int,
               spec: EncoderTargetSpec | None = None) -> EncoderTail:
    spec = spec or SPECS.get(model_name)
    if spec is None:
        raise KeyError(
            f"no LoRA target spec for {model_name!r}. Known: {sorted(SPECS)}")
    if spec.runner not in RUNNERS:
        raise KeyError(
            f"{model_name}: unknown runner {spec.runner!r}. Known: {sorted(RUNNERS)}")
    return RUNNERS[spec.runner](encoder, model_name, cut, spec)
