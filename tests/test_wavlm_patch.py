"""model/wavlm_sdpa_patch.py: WavLM attention that calls q/k/v_proj as modules.

A small random WavLM, no checkpoints. The patch must be bit-identical to
transformers' stock kernel, since it is on by default in the benchmark warm.
"""

import pytest
import torch

from model.wavlm_sdpa_patch import (
    disable_wavlm_sdpa,
    enable_wavlm_sdpa,
    wavlm_sdpa_enabled,
)


@pytest.fixture(autouse=True)
def _restore_patch_state():
    was = wavlm_sdpa_enabled()
    yield
    (enable_wavlm_sdpa if was else disable_wavlm_sdpa)()


def _tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel

    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=64, do_stable_layer_norm=True,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        feat_extract_norm="layer", num_buckets=32, max_bucket_distance=80)
    torch.manual_seed(0)
    return WavLMModel(cfg).eval()


def _padded_batch():
    torch.manual_seed(1)
    lengths = [4000, 2500, 3100]
    x = torch.zeros(len(lengths), max(lengths))
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for i, n in enumerate(lengths):
        x[i, :n] = torch.randn(n)
        mask[i, :n] = 1
    return x, mask


def _run(model, x, mask=None, **kw):
    with torch.no_grad():
        out = model(x, attention_mask=mask, output_hidden_states=True, **kw)
    return out


@pytest.mark.parametrize("batch", ["serial", "padded"])
def test_patch_is_bit_identical_to_stock_kernel(batch):
    model = _tiny_wavlm()
    if batch == "serial":
        x, mask = torch.randn(1, 4000), None
    else:
        x, mask = _padded_batch()

    disable_wavlm_sdpa()
    ref = _run(model, x, mask)
    enable_wavlm_sdpa()
    got = _run(model, x, mask)

    assert torch.equal(got.last_hidden_state, ref.last_hidden_state)
    assert len(got.hidden_states) == len(ref.hidden_states)
    for g, r in zip(got.hidden_states, ref.hidden_states):
        assert torch.equal(g, r)


def test_patch_calls_the_projection_modules():
    """The point of the patch: hooks / LoRA wrappers on q_proj actually run."""
    model = _tiny_wavlm()
    calls = []
    for layer in model.encoder.layers:
        layer.attention.q_proj.register_forward_hook(
            lambda *_: calls.append(1))
    x = torch.randn(1, 4000)

    disable_wavlm_sdpa()
    _run(model, x)
    assert calls == []  # stock kernel reads q_proj.weight, never calls it

    enable_wavlm_sdpa()
    _run(model, x)
    assert len(calls) == len(model.encoder.layers)


def test_output_attentions_falls_back_to_stock_kernel():
    model = _tiny_wavlm()
    x, mask = _padded_batch()

    disable_wavlm_sdpa()
    ref = _run(model, x, mask, output_attentions=True)
    enable_wavlm_sdpa()
    got = _run(model, x, mask, output_attentions=True)

    assert got.attentions is not None and got.attentions[0] is not None
    assert torch.equal(got.last_hidden_state, ref.last_hidden_state)
    for g, r in zip(got.attentions, ref.attentions):
        assert torch.equal(g, r)


def test_enable_and_disable_are_idempotent():
    from transformers.models.wavlm import modeling_wavlm as mw

    disable_wavlm_sdpa()
    stock = mw.WavLMAttention.torch_multi_head_self_attention

    enable_wavlm_sdpa()
    enable_wavlm_sdpa()  # must not save the patched method as the "original"
    assert wavlm_sdpa_enabled()
    assert mw.WavLMAttention.torch_multi_head_self_attention is not stock
    assert mw.WavLMAttention._sdx_orig_torch_multi_head_self_attention is stock

    disable_wavlm_sdpa()
    disable_wavlm_sdpa()
    assert not wavlm_sdpa_enabled()
    assert mw.WavLMAttention.torch_multi_head_self_attention is stock


def _fake_pretrained(monkeypatch):
    from transformers import Wav2Vec2FeatureExtractor

    import model.wavlm as mw

    monkeypatch.setattr(mw.WavLMModel, "from_pretrained",
                        staticmethod(lambda *_a, **_k: _tiny_wavlm()))
    monkeypatch.setattr(mw.AutoFeatureExtractor, "from_pretrained",
                        staticmethod(lambda *_a, **_k: Wav2Vec2FeatureExtractor()))
    return mw.WavLM


def test_wrapper_attn_impl_selects_the_kernel(monkeypatch):
    WavLM = _fake_pretrained(monkeypatch)

    WavLM("tiny", True, False, 16000, attn_impl="original")
    assert not wavlm_sdpa_enabled()
    enc = WavLM("tiny", True, False, 16000)  # default
    assert enc.attn_impl == "sdpa" and wavlm_sdpa_enabled()


def test_wrapper_rejects_unknown_attn_impl_before_loading(monkeypatch):
    import model.wavlm as mw

    def _no_load(*_a, **_k):
        raise AssertionError("weights loaded before attn_impl was validated")

    monkeypatch.setattr(mw.WavLMModel, "from_pretrained", staticmethod(_no_load))
    monkeypatch.setattr(mw.AutoFeatureExtractor, "from_pretrained",
                        staticmethod(_no_load))
    with pytest.raises(ValueError, match="attn_impl"):
        mw.WavLM("tiny", True, False, 16000, attn_impl="bogus")
