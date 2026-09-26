"""Unit tests for the LoRA target selection / parameter audit.

These use synthetic module trees so they run without downloading any
checkpoint. The real-encoder check lives in ``scripts/lora_parity.py``.
"""

import pytest
import torch.nn as nn

from sdx.lora.targets import EncoderTargetSpec, SPECS, plan_adapters


class _Attn(nn.Module):
    def __init__(self, d, q_out=None, v_out=None):
        super().__init__()
        self.q_proj = nn.Linear(d, q_out or d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, v_out or d)


class _Block(nn.Module):
    def __init__(self, d, **kw):
        super().__init__()
        self.self_attn = _Attn(d, **kw)


class _Enc(nn.Module):
    """Minimal stand-in shaped like the wrappers: <root>.layers[i].self_attn."""

    def __init__(self, n_blocks, d, **kw):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d, **kw) for _ in range(n_blocks)])


SPEC = EncoderTargetSpec(blocks="layers", attn="self_attn", q="q_proj", v="v_proj")


def _plan(n_blocks, d, *, rank=8, budget=100_000, **kw):
    return plan_adapters(_Enc(n_blocks, d, **kw), "fake",
                         rank=rank, budget=budget, spec=SPEC)


@pytest.mark.parametrize("n_blocks,d,exp_n,exp_params,exp_cut", [
    (24, 1024, 3, 98_304, 21),   # WavLM-Large
    (32, 1280, 2, 81_920, 30),   # Whisper-Large-v3
    (8, 512, 6, 98_304, 2),      # Qwen3-TTS-Tokenizer-12Hz
    (12, 768, 4, 98_304, 8),     # a 768-d, 12-block encoder (e.g. AST)
    (48, 1280, 2, 81_920, 46),   # a 1280-d, 48-block encoder (e.g. MMS-1B)
])
def test_matches_plan_table(n_blocks, d, exp_n, exp_params, exp_cut):
    """Block geometries reproduce the table in sdx/lora/targets.py."""
    p = _plan(n_blocks, d)
    assert (p.adapted_blocks, p.total_params, p.cut_index) == (
        exp_n, exp_params, exp_cut)
    assert p.total_params <= 100_000
    assert p.width == d


def test_suffix_is_contiguous_and_ends_at_final_block():
    p = _plan(24, 1024)
    idx = [t.index for t in p.targets]
    assert idx == list(range(p.cut_index, 24))


def test_targets_are_only_q_and_v():
    """no key projection is ever matched."""
    p = _plan(12, 768)
    paths = p.target_paths()
    assert len(paths) == 2 * p.adapted_blocks
    assert all(x.endswith(("q_proj", "v_proj")) for x in paths)
    assert not any("k_proj" in x for x in paths)


def test_budget_is_a_ceiling_not_a_target():
    """A budget between 1x and 2x one block's cost adapts exactly one block."""
    p = _plan(24, 1024, budget=40_000)
    assert p.adapted_blocks == 1
    assert p.total_params == 32_768
    assert p.unused_capacity == 7_232


def test_rank_scales_parameters_linearly():
    assert _plan(12, 768, rank=4).targets[0].params == 12_288
    assert _plan(12, 768, rank=8).targets[0].params == 24_576


def test_fused_qkv_is_fatal():
    """a fused projection must fail loudly, never be wrapped."""
    with pytest.raises(ValueError, match="FUSED"):
        _plan(12, 768, q_out=768 * 3)


def test_non_square_projection_warns_but_proceeds():
    p = _plan(12, 768, v_out=1024)
    assert any("non-square" in w for w in p.warnings)
    # r*(768+768) + r*(768+1024)
    assert p.targets[0].params == 8 * 1536 + 8 * 1792


def test_budget_smaller_than_one_block_raises():
    with pytest.raises(ValueError, match="exceeds"):
        _plan(24, 1024, budget=1_000)


def test_unknown_encoder_name_raises():
    with pytest.raises(KeyError):
        plan_adapters(_Enc(4, 256), "not_an_encoder")


def test_top3_specs_are_registered_under_registry_names():
    from sdx.registry import encoders as registry_encoders

    assert set(SPECS) == {"whisper", "wavlm", "qwen3voice"}
    assert set(SPECS) <= set(registry_encoders())
    for name, spec in SPECS.items():
        assert spec.blocks and spec.attn, name
        assert (spec.q and spec.v) or spec.qkv, name


def test_every_spec_names_a_known_runner():
    from sdx.lora.tails import RUNNERS

    for name, spec in SPECS.items():
        assert spec.runner in RUNNERS, (name, spec.runner)


def test_bad_path_error_names_available_children():
    bad = EncoderTargetSpec(blocks="nope", attn="self_attn", q="q_proj", v="v_proj")
    with pytest.raises(AttributeError, match="Available children"):
        plan_adapters(_Enc(4, 256), "fake", spec=bad)


# -- fused [q; k; v] projections ----------------------------------------------


class _FusedAttn(nn.Module):
    def __init__(self, d, out=None):
        super().__init__()
        self.qkv = nn.Linear(d, out or 3 * d)


class _FusedEnc(nn.Module):
    def __init__(self, n_blocks, d, **kw):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Module() for _ in range(n_blocks)])
        for b in self.blocks:
            b.attn = _FusedAttn(d, **kw)


FUSED = EncoderTargetSpec(blocks="blocks", attn="attn", qkv="qkv")


@pytest.mark.parametrize("n_blocks,d,exp_n,exp_cut", [
    (12, 768, 4, 8),    # AudioMAE (ViT-B)
    (12, 384, 8, 4),    # OPERA-GT (ViT-S)
])
def test_fused_plan_costs_the_same_as_separate_q_and_v(n_blocks, d, exp_n, exp_cut):
    fused = plan_adapters(_FusedEnc(n_blocks, d), "fake", spec=FUSED)
    separate = _plan(n_blocks, d)
    assert fused.total_params == separate.total_params
    assert (fused.adapted_blocks, fused.cut_index) == (exp_n, exp_cut)
    assert fused.width == d


def test_fused_plan_has_one_target_path_per_block():
    plan = plan_adapters(_FusedEnc(12, 768), "fake", spec=FUSED)
    assert plan.target_paths() == [f"blocks.{i}.attn.qkv" for i in range(8, 12)]


def test_fused_projection_that_is_not_thirds_raises():
    with pytest.raises(ValueError, match="equal q/k/v thirds"):
        plan_adapters(_FusedEnc(4, 64, out=64 * 3 + 2), "fake", spec=FUSED)


@pytest.mark.parametrize("kw", [
    dict(q="q_proj", v="v_proj", qkv="qkv"),   # both
    dict(q="q_proj"),                          # half a separate pair
    dict(),                                    # neither
])
def test_spec_must_declare_exactly_one_projection_kind(kw):
    with pytest.raises(ValueError, match="either separate"):
        EncoderTargetSpec(blocks="layers", attn="self_attn", **kw)
