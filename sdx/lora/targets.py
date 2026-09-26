"""Adapter target selection, the parameter audit, and the per-encoder specs.

Given a live encoder wrapper, resolve its transformer blocks, locate the
query/value projections, and walk backward from the final block, adding
blocks while the LoRA parameter total stays within the ceiling.

The cut layer this returns decides what the boundary cache stores, so the
plan must be computed (and its audit must pass) before any cache is built.

Parameter count per adapted projection is ``r * (in_features + out_features)``
(``A`` is ``r x in``, ``B`` is ``out x r``). For a square projection of width
``d`` that is ``2rd``, so query+value over one block costs ``4rd``. At the
defaults (r=8, 100k ceiling) that gives:

    encoder      blocks  d     adapted  cut  adapter params
    whisper      32      1280  2        30   81,920
    wavlm        24      1024  3        21   98,304
    qwen3voice   8       512   6        2    98,304
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

DEFAULT_RANK = 8
DEFAULT_BUDGET = 100_000


@dataclass(frozen=True)
class EncoderTargetSpec:
    """Where the adaptable pieces live inside one encoder wrapper, and how to
    replay them up to the benchmark's tap point.

    Paths are dotted attribute chains. ``blocks``, ``post`` and ``post_if`` are
    resolved against the wrapper module (the ``!new:model.*`` object the yaml
    builds); ``attn`` is resolved against a single block, and ``q``/``v``
    against that block's attention.

    Projections are declared one of two ways, never both: separate ``q`` and
    ``v`` linears, or one fused ``qkv`` linear laid out ``[q; k; v]`` on its
    output (timm ViT, data2vec 2.0), which gets a slice-aware adapter that
    leaves K untouched.

    The last four fields drive :class:`sdx.lora.tails.EncoderTail`, so adding
    an encoder is usually a spec, not a new tail class:

    * ``runner`` — how ``blocks[cut:]`` are executed (a key of
      ``sdx.lora.tails.RUNNERS``). ``"sequential"`` calls each block with
      ``block_kwargs`` and keeps output ``[0]``; the others exist for blocks
      that thread state their parent builds (WavLM's position bias, Mimi's
      rotary/causal mask).
    * ``block_kwargs`` — static keyword arguments for every block call.
    * ``post`` — what the encoder applies between its last block and the tap,
      in order: a module path on the wrapper (typically the final layer norm),
      or a stateless op instance for work the wrapper does inline
      (:class:`DropTokens`, :class:`MeanTokens`).
    * ``post_if`` — optional boolean attribute gating ``post``. For WavLM and
      the wav2vec2 family the final ``layer_norm`` is applied AFTER the loop
      only in the stable-layer-norm (pre-LN) variant; the post-LN variant
      applies it before the loop, where it belongs to the cached prefix.
    """

    blocks: str
    attn: str
    q: str = ""
    v: str = ""
    note: str = ""
    runner: str = "sequential"
    block_kwargs: dict[str, Any] = field(default_factory=dict)
    post: tuple[str | nn.Module, ...] = ()
    post_if: str | None = None
    qkv: str = ""

    def __post_init__(self):
        separate = self.q and self.v and not self.qkv
        fused = self.qkv and not self.q and not self.v
        if not (separate or fused):
            raise ValueError(
                "declare either separate q= and v= projections or one fused "
                f"qkv= projection (got q={self.q!r} v={self.v!r} qkv={self.qkv!r})")


class DropTokens(nn.Module):
    """Drop the first ``n`` tokens (a CLS token, data2vec's extra tokens)."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, self.n:]

    def extra_repr(self) -> str:
        return f"n={self.n}"


class MeanTokens(nn.Module):
    """Mean over the token axis, keeping it, so the tap stays ``(B, 1, D)``."""

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


# Model-specific specs for the three best encoders on the main board. Keys
# are the --encoder names in sdx/configs/registry.yaml. Verified against
# transformers 4.57.3 (the pinned version). Whether a projection is separate
# or fused is declared here and asserted at runtime by ``_audit_projection`` /
# ``_audit_fused`` rather than assumed, and each spec must pass the
# split-parity check (``scripts/lora_parity.py``) before its caches are built.
# sdx/lora/README.md explains how to add another encoder.
SPECS: dict[str, EncoderTargetSpec] = {
    "whisper": EncoderTargetSpec(
        blocks="encoder.layers",
        attn="self_attn",
        q="q_proj",
        v="v_proj",
        block_kwargs={"attention_mask": None, "layer_head_mask": None},
        post=("encoder.layer_norm",),
        note="Whisper-large-v3 encoder; the wrapper keeps only "
             "WhisperModel.encoder.",
    ),
    "wavlm": EncoderTargetSpec(
        # The wrapper holds WavLMModel as ``feature_extractor``
        # (model/wavlm.py), NOT ``model`` — the name collides with HF's own
        # conv feature extractor, so do not "correct" it.
        blocks="feature_extractor.encoder.layers",
        attn="attention",
        q="q_proj",
        v="v_proj",
        runner="wavlm",
        post=("feature_extractor.encoder.layer_norm",),
        post_if="feature_extractor.config.do_stable_layer_norm",
        note="WavLM-Large (microsoft/wavlm-large), stable layer norm.",
    ),
    "qwen3voice": EncoderTargetSpec(
        blocks="mimi_encoder.encoder_transformer.layers",
        attn="self_attn",
        q="q_proj",
        v="v_proj",
        runner="delegate",
        note="Qwen3-TTS-Tokenizer-12Hz, Mimi-style encoder_transformer only; "
             "the quantizer and decoder are never adapted.",
    ),
}

# Encoders whose q/v projections are square by construction. A projection whose
# out_features differs from in_features is not necessarily fused (Whisper's
# attention is square too), but a 3x ratio almost certainly is, so the audit
# treats that as fatal.
_FUSED_RATIO_TOL = 2.5


def _resolve(root, path: str):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            # ``post_if`` paths end in a config attribute, not a module.
            available = ([n for n, _ in obj.named_children()][:12]
                         if isinstance(obj, nn.Module) else "<not a module>")
            raise AttributeError(
                f"cannot resolve {path!r}: {type(obj).__name__} has no {part!r}. "
                f"Available children: {available}"
            )
        obj = getattr(obj, part)
    return obj


@dataclass
class BlockTarget:
    """One adapted block and its two projections.

    For a fused spec ``q_path == v_path`` (the one ``qkv`` linear) and the
    shapes are those of its q and v thirds.
    """

    index: int
    q_path: str
    v_path: str
    q_shape: tuple[int, int]
    v_shape: tuple[int, int]
    params: int


@dataclass
class AdapterPlan:
    """Result of the suffix walk. ``cut_index`` is what the cache is keyed on."""

    model_name: str
    rank: int
    budget: int
    num_blocks: int
    width: int
    cut_index: int
    targets: list[BlockTarget] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def adapted_blocks(self) -> int:
        return len(self.targets)

    @property
    def total_params(self) -> int:
        return sum(t.params for t in self.targets)

    @property
    def unused_capacity(self) -> int:
        return self.budget - self.total_params

    def target_paths(self) -> list[str]:
        out = []
        for t in self.targets:
            out.extend(dict.fromkeys([t.q_path, t.v_path]))  # fused: one path
        return out

    def summary(self) -> str:
        return (
            f"{self.model_name}: L={self.num_blocks} d={self.width} "
            f"N={self.adapted_blocks} cut={self.cut_index} "
            f"params={self.total_params:,} "
            f"(unused {self.unused_capacity:,} of {self.budget:,})"
        )


def _audit_projection(mod, path: str, kind: str, warnings: list[str]) -> tuple[int, int]:
    """Validate one projection and return (in_features, out_features)."""
    if not isinstance(mod, nn.Linear):
        raise TypeError(
            f"{path} is {type(mod).__name__}, not nn.Linear — LoRA target "
            f"resolution is wrong for this encoder."
        )
    fan_in, fan_out = mod.in_features, mod.out_features
    # A fused QKV projection wrapped as one linear would silently adapt K as
    # well and use a different factorization from the separate-projection
    # encoders.
    if fan_out >= _FUSED_RATIO_TOL * fan_in:
        raise ValueError(
            f"{path} looks FUSED: in={fan_in} out={fan_out} "
            f"(ratio {fan_out / fan_in:.1f}). Wrapping it would also adapt K. "
            f"Declare it as qkv= in the spec for a slice-aware {kind} "
            f"adapter — refusing to proceed."
        )
    if fan_out != fan_in:
        warnings.append(
            f"{path}: non-square projection in={fan_in} out={fan_out} "
            f"(not fused, but the 4rd parameter formula does not apply)"
        )
    return fan_in, fan_out


def _audit_fused(mod, path: str, warnings: list[str]) -> tuple[int, int]:
    """Validate a fused ``[q; k; v]`` projection; return (in, per-slice out)."""
    if not isinstance(mod, nn.Linear):
        raise TypeError(
            f"{path} is {type(mod).__name__}, not nn.Linear — LoRA target "
            f"resolution is wrong for this encoder."
        )
    fan_in, fan_out = mod.in_features, mod.out_features
    # Anything but equal thirds (e.g. grouped-query K/V) would mis-slice.
    if fan_out % 3:
        raise ValueError(
            f"{path} is declared fused but out={fan_out} does not split into "
            f"equal q/k/v thirds — refusing to proceed."
        )
    d = fan_out // 3
    if d != fan_in:
        warnings.append(
            f"{path}: non-square fused slices in={fan_in} out={d} "
            f"(the 4rd parameter formula does not apply)"
        )
    return fan_in, d


def plan_adapters(
    encoder: nn.Module,
    model_name: str,
    *,
    rank: int = DEFAULT_RANK,
    budget: int = DEFAULT_BUDGET,
    spec: EncoderTargetSpec | None = None,
) -> AdapterPlan:
    """Select the adapted suffix for ``encoder`` under the parameter ceiling.

    Walks backward from the final block, adding a block only if the running
    total stays within ``budget``. Stops at the first block that would exceed
    it — the suffix is contiguous by construction, so a later smaller block is
    never "skipped into".
    """
    spec = spec or SPECS.get(model_name)
    if spec is None:
        raise KeyError(
            f"no LoRA target spec for {model_name!r}. Known: {sorted(SPECS)}"
        )

    blocks = _resolve(encoder, spec.blocks)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            f"{spec.blocks} resolved to {type(blocks).__name__}, expected a "
            f"ModuleList of transformer blocks"
        )

    num_blocks = len(blocks)
    warnings: list[str] = []
    targets: list[BlockTarget] = []
    total = 0
    width = 0

    # Backward walk: final block first.
    for i in range(num_blocks - 1, -1, -1):
        attn = _resolve(blocks[i], spec.attn)
        if spec.qkv:
            q_path = v_path = f"{spec.blocks}.{i}.{spec.attn}.{spec.qkv}"
            q_in, q_out = v_in, v_out = _audit_fused(
                _resolve(attn, spec.qkv), q_path, warnings)
        else:
            q_path = f"{spec.blocks}.{i}.{spec.attn}.{spec.q}"
            v_path = f"{spec.blocks}.{i}.{spec.attn}.{spec.v}"
            q_in, q_out = _audit_projection(
                _resolve(attn, spec.q), q_path, "query", warnings)
            v_in, v_out = _audit_projection(
                _resolve(attn, spec.v), v_path, "value", warnings)
        width = width or q_in

        cost = rank * (q_in + q_out) + rank * (v_in + v_out)
        if total + cost > budget:
            break
        total += cost
        targets.append(BlockTarget(
            index=i, q_path=q_path, v_path=v_path,
            q_shape=(q_in, q_out), v_shape=(v_in, v_out), params=cost,
        ))

    if not targets:
        raise ValueError(
            f"{model_name}: even one block ({cost:,} params) exceeds the "
            f"{budget:,} ceiling at rank {rank}"
        )

    targets.reverse()  # ascending block order reads better in reports
    cut_index = targets[0].index

    if cut_index == 0:
        warnings.append(
            "the whole encoder is adapted (cut at block 0) — the boundary "
            "cache degenerates to the encoder's own input features"
        )

    return AdapterPlan(
        model_name=model_name, rank=rank, budget=budget,
        num_blocks=num_blocks, width=width, cut_index=cut_index,
        targets=targets, warnings=warnings,
    )
