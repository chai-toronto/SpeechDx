"""Spec-driven tail tests: runners, block_kwargs, post / post_if.

Synthetic module trees, no checkpoints. The real-encoder split-parity gate
(``tail(capture(x)) == encoder(x)`` on every registered spec) lives in
``scripts/lora_parity.py``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sdx.lora.targets import EncoderTargetSpec
from sdx.lora.tails import (
    DelegatingTail,
    EncoderTail,
    WavLMTail,
    build_tail,
    capture_boundary,
)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)


class _Block(nn.Module):
    """HF-shaped block: returns a tuple and records the kwargs it was given."""

    def __init__(self, d):
        super().__init__()
        self.attention = _Attn(d)
        self.seen: list[dict] = []

    def forward(self, x, attention_mask=None, head_mask=None):
        self.seen.append({"attention_mask": attention_mask, "head_mask": head_mask})
        a = self.attention
        return (x + a.v_proj(torch.tanh(a.q_proj(x))),)


class _Encoder(nn.Module):
    """Stable-LN-shaped: loop, then an optional post-loop norm."""

    def __init__(self, n=6, d=8, stable=True):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n)])
        self.layer_norm = nn.LayerNorm(d)
        self.config = SimpleNamespace(do_stable_layer_norm=stable)

    def forward(self, x):
        if not self.config.do_stable_layer_norm:
            x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x, attention_mask=None, head_mask=None)[0]
        if self.config.do_stable_layer_norm:
            x = self.layer_norm(x)
        return x


class _Wrap(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = _Encoder(**kw)

    def forward(self, x):
        return self.model(x)


SPEC = EncoderTargetSpec(
    blocks="model.layers", attn="attention", q="q_proj", v="v_proj",
    block_kwargs={"attention_mask": None, "head_mask": None},
    post=("model.layer_norm",),
    post_if="model.config.do_stable_layer_norm",
)


@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("cut", [0, 3, 5])
def test_split_parity(stable, cut):
    """tail(capture(x)) == encoder(x), for both layer-norm placements."""
    torch.manual_seed(0)
    wrap = _Wrap(stable=stable).eval()
    x = torch.randn(1, 11, 8)
    with torch.no_grad():
        ref = wrap(x)
        boundary = capture_boundary(wrap, "fake", cut, lambda: wrap(x), spec=SPEC)
        got = build_tail(wrap, "fake", cut, spec=SPEC)(boundary)
    assert torch.equal(ref, got)


def test_post_if_false_leaves_the_norm_in_the_prefix():
    tail = build_tail(_Wrap(stable=False), "fake", 3, spec=SPEC)
    assert len(tail.post) == 0
    tail = build_tail(_Wrap(stable=True), "fake", 3, spec=SPEC)
    assert len(tail.post) == 1


class _AddOne(nn.Module):
    def forward(self, h):
        return h + 1


class _Double(nn.Module):
    def forward(self, h):
        return h * 2


def test_post_ops_apply_in_declared_order():
    wrap = _Wrap()
    wrap.model.add_one = _AddOne()
    wrap.model.times_two = _Double()
    spec = EncoderTargetSpec(blocks="model.layers", attn="attention",
                             q="q_proj", v="v_proj",
                             post=("model.add_one", "model.times_two"))
    tail = build_tail(wrap, "fake", 5, spec=spec)
    x = torch.randn(1, 2, 8)
    with torch.no_grad():
        assert torch.equal(tail(x), (tail.run_blocks(x) + 1) * 2)


def test_block_kwargs_reach_every_block():
    wrap = _Wrap()
    tail = build_tail(wrap, "fake", 3, spec=SPEC)
    tail(torch.randn(1, 4, 8))
    for block in tail.blocks:
        assert block.seen == [{"attention_mask": None, "head_mask": None}]


def test_attention_mask_is_forwarded_only_where_blocks_take_one():
    wrap = _Wrap()
    mask = torch.ones(1, 4, dtype=torch.bool)
    tail = build_tail(wrap, "fake", 5, spec=SPEC)
    tail(torch.randn(1, 4, 8), attention_mask=mask)
    assert tail.blocks[0].seen[-1]["attention_mask"] is mask

    no_mask = EncoderTargetSpec(blocks="model.layers", attn="attention",
                                q="q_proj", v="v_proj", block_kwargs={"head_mask": None})
    tail = build_tail(wrap, "fake", 5, spec=no_mask)
    with pytest.raises(ValueError, match="no attention_mask"):
        tail(torch.randn(1, 4, 8), attention_mask=mask)


def test_runner_selects_the_tail_class():
    wrap = _Wrap()
    base = dict(blocks="model.layers", attn="attention", q="q_proj", v="v_proj")
    assert type(build_tail(wrap, "fake", 3, spec=EncoderTargetSpec(**base))) is EncoderTail
    assert isinstance(build_tail(wrap, "fake", 3,
                                 spec=EncoderTargetSpec(**base, runner="delegate")),
                      DelegatingTail)
    with pytest.raises(KeyError, match="unknown runner"):
        build_tail(wrap, "fake", 3, spec=EncoderTargetSpec(**base, runner="nope"))


def test_unknown_encoder_without_spec_raises():
    with pytest.raises(KeyError, match="no LoRA target spec"):
        build_tail(_Wrap(), "not_an_encoder", 3)


def test_wavlm_runner_is_registered_for_wavlm():
    from sdx.lora.tails import RUNNERS
    from sdx.lora.targets import SPECS

    assert RUNNERS[SPECS["wavlm"].runner] is WavLMTail


# -- post ops and the fused-qkv encoder families ------------------------------


class _VitBlock(nn.Module):
    """timm-shaped: fused qkv, returns a bare tensor."""

    def __init__(self, d):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, 3 * d)

    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        return x + torch.tanh(q) * k + v


class _OperaLike(nn.Module):
    """forward_feature shape: blocks -> drop CLS -> mean -> norm -> unsqueeze."""

    def __init__(self, n=6, d=8):
        super().__init__()
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([_VitBlock(d) for _ in range(n)])
        self.model.norm = nn.LayerNorm(d)

    def forward(self, x):
        for blk in self.model.blocks:
            x = blk(x)
        return self.model.norm(x[:, 1:, :].mean(dim=1)).unsqueeze(1)


def test_pool_before_norm_tap_is_reproduced_by_post_ops():
    from sdx.lora.targets import DropTokens, MeanTokens

    spec = EncoderTargetSpec(blocks="model.blocks", attn="attn", qkv="qkv",
                             post=(DropTokens(1), MeanTokens(), "model.norm"))
    torch.manual_seed(0)
    enc = _OperaLike().eval()
    x = torch.randn(2, 13, 8)
    with torch.no_grad():
        ref = enc(x)
        boundary = capture_boundary(enc, "fake", 2, lambda: enc(x), spec=spec)
        got = build_tail(enc, "fake", 2, spec=spec)(boundary)
    assert got.shape == ref.shape == (2, 1, 8)
    assert torch.allclose(ref, got, atol=1e-6)


# -- real transformers WavLM: LoRA must go through q_proj / v_proj -------------

def _tiny_wavlm_wrapper():
    from transformers import WavLMConfig, WavLMModel

    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=64, do_stable_layer_norm=True,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        feat_extract_norm="layer", num_buckets=32, max_bucket_distance=80)

    class _Wrap(nn.Module):
        # Same attribute name as model/wavlm.py's wrapper.
        def __init__(self):
            super().__init__()
            self.feature_extractor = WavLMModel(cfg)

        def forward(self, x):
            return self.feature_extractor(x).last_hidden_state

    torch.manual_seed(0)
    return _Wrap().eval()


def test_wavlm_tail_matches_hf_and_lora_actually_takes_effect():
    """HF's WavLM attention reads q_proj.weight/.bias directly, so a wrapped
    projection used to crash (LoRALinear has no .bias) — and would otherwise
    be ignored. Building the tail enables model/wavlm_sdpa_patch.py, which
    calls the modules; the tail must still match the stock kernel."""
    from model.wavlm_sdpa_patch import (
        disable_wavlm_sdpa,
        enable_wavlm_sdpa,
        wavlm_sdpa_enabled,
    )

    was = wavlm_sdpa_enabled()
    try:
        disable_wavlm_sdpa()  # reference = transformers' stock kernel
        _check_wavlm_tail()
    finally:
        (enable_wavlm_sdpa if was else disable_wavlm_sdpa)()


def _check_wavlm_tail():
    from sdx.lora.layers import lora_branches
    from sdx.lora.model import LoRATailProbe
    from sdx.lora.targets import SPECS, plan_adapters

    enc = _tiny_wavlm_wrapper()
    wav = torch.randn(1, 4000)
    cut = 2
    with torch.no_grad():
        ref = enc(wav)
        boundary = capture_boundary(enc, "wavlm", cut, lambda: enc(wav))
        tail = build_tail(enc, "wavlm", cut).eval()
        assert torch.allclose(tail(boundary), ref, atol=1e-5)

    plan = plan_adapters(enc, "wavlm", budget=2 * 4 * 8 * 32)  # two blocks
    assert plan.cut_index == cut
    probe = LoRATailProbe(tail, plan.target_paths(), num_labels=1).eval()
    with torch.no_grad():
        assert torch.equal(probe.tail(boundary), tail(boundary))  # B = 0
        for b in lora_branches(probe.tail):
            b.lora_B.normal_(std=0.5)
        assert not torch.allclose(probe.tail(boundary), ref, atol=1e-3)
    assert SPECS["wavlm"].runner == "wavlm"
