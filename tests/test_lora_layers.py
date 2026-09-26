"""LoRA layer tests — gradient isolation, checkpoint round trip and
merged-weight equivalence, plus the zero-init property that makes a fresh
adapter an exact no-op.
"""

import pytest
import torch
import torch.nn as nn

from sdx.lora.layers import (
    LoRAFusedQKV,
    LoRALinear,
    adapter_state_dict,
    freeze_all_but_adapters,
    inject_adapters,
    lora_branches,
    load_adapter_state_dict,
)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)


class _Net(nn.Module):
    def __init__(self, d=32, n=3):
        super().__init__()
        self.layers = nn.ModuleList([_Attn(d) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            x = l.v_proj(l.q_proj(x))
        return x


def _paths(n=3):
    return [f"layers.{i}.{p}" for i in range(n) for p in ("q_proj", "v_proj")]


def test_fresh_adapter_is_exactly_a_noop():
    """B=0 => BA=0, so injection cannot perturb the frozen forward at all."""
    torch.manual_seed(0)
    net, x = _Net(), torch.randn(4, 32)
    before = net(x)
    inject_adapters(net, _paths())
    net.eval()  # disable LoRA dropout
    after = net(x)
    assert torch.equal(before, after), (before - after).abs().max()


def test_adapter_changes_output_once_b_is_nonzero():
    torch.manual_seed(0)
    net, x = _Net(), torch.randn(4, 32)
    before = net(x)
    adapters = inject_adapters(net, _paths())
    for a in adapters.values():
        nn.init.normal_(a.lora_B, std=0.02)
    net.eval()
    assert not torch.allclose(before, net(x))


def test_scaling_is_alpha_over_rank():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    assert lora.scaling == 2.0
    nn.init.constant_(lora.lora_A, 1.0)
    nn.init.constant_(lora.lora_B, 1.0)
    x = torch.randn(2, 16)
    expected = base(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * 2.0
    assert torch.allclose(lora(x), expected, atol=1e-6)


def test_gradient_isolation():
    """only lora_A/lora_B get gradients; no base weight moves."""
    torch.manual_seed(0)
    net = _Net()
    inject_adapters(net, _paths())
    trainable, frozen = freeze_all_but_adapters(net)
    assert trainable == 6 * 8 * (32 + 32)  # 6 projections, r=8, square d=32
    assert frozen > 0

    base_before = {n: p.detach().clone() for n, p in net.named_parameters()
                   if "lora_" not in n}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.1)
    net(torch.randn(4, 32)).pow(2).mean().backward()

    for n, p in net.named_parameters():
        if "lora_" in n:
            assert p.grad is not None, n
        else:
            assert p.grad is None, f"frozen param {n} received a gradient"
    opt.step()
    for n, p in net.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base_before[n]), f"frozen param {n} moved"


def test_only_q_and_v_are_wrapped():
    net = _Net()
    inject_adapters(net, _paths())
    for layer in net.layers:
        assert isinstance(layer.q_proj, LoRALinear)
        assert isinstance(layer.v_proj, LoRALinear)
        assert isinstance(layer.k_proj, nn.Linear)
        assert not isinstance(layer.k_proj, LoRALinear)


def test_checkpoint_round_trip_reproduces_predictions():
    """Only the frozen prefix is cached; the tail never re-runs it."""
    torch.manual_seed(0)
    net = _Net()
    adapters = inject_adapters(net, _paths())
    for a in adapters.values():
        nn.init.normal_(a.lora_B, std=0.05)
    net.eval()
    x = torch.randn(4, 32)
    expected = net(x)
    state = adapter_state_dict(net)
    assert len(state) == 12  # 6 projections x {A, B}

    fresh = _Net()
    fresh.load_state_dict({k: v for k, v in net.state_dict().items()
                           if "lora_" not in k and "base." not in k}, strict=False)
    # rebuild with the same frozen weights, then load only the adapter
    fresh = _Net()
    for i, layer in enumerate(net.layers):
        fresh.layers[i].q_proj.load_state_dict(layer.q_proj.base.state_dict())
        fresh.layers[i].v_proj.load_state_dict(layer.v_proj.base.state_dict())
        fresh.layers[i].k_proj.load_state_dict(layer.k_proj.state_dict())
    inject_adapters(fresh, _paths())
    fresh.eval()
    assert not torch.allclose(fresh(x), expected)  # zero-init, not yet loaded
    load_adapter_state_dict(fresh, state)
    assert torch.allclose(fresh(x), expected, atol=1e-6)


def test_loading_wrong_adapter_state_raises():
    net = _Net()
    inject_adapters(net, _paths())
    state = adapter_state_dict(net)
    state.pop(next(iter(state)))
    with pytest.raises(KeyError):
        load_adapter_state_dict(net, state)


def test_merged_weight_equivalence():
    """W + (alpha/r)BA reproduces the unmerged adapter output."""
    torch.manual_seed(0)
    base = nn.Linear(24, 24)
    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    nn.init.normal_(lora.lora_B, std=0.05)
    lora.eval()
    x = torch.randn(5, 24)
    merged = nn.Linear(24, 24)
    with torch.no_grad():
        merged.weight.copy_(lora.merged_weight())
        merged.bias.copy_(base.bias)
    assert torch.allclose(lora(x), merged(x), atol=1e-5)


def test_double_injection_raises():
    net = _Net()
    inject_adapters(net, _paths())
    with pytest.raises(ValueError, match="already has an adapter"):
        inject_adapters(net, _paths())


def test_injecting_non_linear_raises():
    net = _Net()
    with pytest.raises(TypeError):
        inject_adapters(net, ["layers.0"])


def test_dropout_is_active_in_train_mode_only():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(64, 64), rank=8, dropout=0.9)
    nn.init.normal_(lora.lora_B, std=1.0)
    x = torch.randn(8, 64)
    lora.train()
    assert not torch.allclose(lora(x), lora(x))
    lora.eval()
    assert torch.allclose(lora(x), lora(x))


# -- fused [q; k; v] projections (timm ViT, data2vec 2.0) ---------------------


class _FusedAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)


class _FusedNet(nn.Module):
    def __init__(self, d=32, n=3):
        super().__init__()
        self.layers = nn.ModuleList([_FusedAttn(d) for _ in range(n)])

    def forward(self, x):
        for l in self.layers:
            q, k, v = l.qkv(x).chunk(3, dim=-1)
            x = torch.tanh(q) * k + v
        return x


def _fused_paths(n=3):
    return [f"layers.{i}.qkv" for i in range(n)]


def _slice(base: nn.Linear, i: int, d: int) -> nn.Linear:
    out = nn.Linear(base.in_features, d)
    with torch.no_grad():
        out.weight.copy_(base.weight[i * d:(i + 1) * d])
        out.bias.copy_(base.bias[i * d:(i + 1) * d])
    return out


def test_fused_fresh_adapter_is_exactly_a_noop():
    torch.manual_seed(0)
    net, x = _FusedNet(), torch.randn(4, 7, 32)
    before = net(x)
    inject_adapters(net, _fused_paths(), fused=True)
    net.eval()
    assert torch.equal(before, net(x))


def test_fused_equals_two_separate_adapters_on_the_slices():
    """The fused adapter IS separate q/v LoRA: same A, B, scaling, per slice."""
    torch.manual_seed(0)
    d, x = 24, torch.randn(5, 24)
    base = nn.Linear(d, 3 * d)
    fused = LoRAFusedQKV(base, rank=8, alpha=16.0, dropout=0.0)
    nn.init.normal_(fused.q.lora_B, std=0.1)
    nn.init.normal_(fused.v.lora_B, std=0.1)

    q = LoRALinear(_slice(base, 0, d), rank=8, alpha=16.0, dropout=0.0)
    v = LoRALinear(_slice(base, 2, d), rank=8, alpha=16.0, dropout=0.0)
    with torch.no_grad():
        for sep, br in ((q, fused.q), (v, fused.v)):
            sep.lora_A.copy_(br.lora_A)
            sep.lora_B.copy_(br.lora_B)
    expected = torch.cat([q(x), _slice(base, 1, d)(x), v(x)], dim=-1)
    assert torch.allclose(fused(x), expected, atol=1e-6)


def test_fused_never_touches_k():
    torch.manual_seed(0)
    base = nn.Linear(16, 48)
    fused = LoRAFusedQKV(base, rank=4, dropout=0.0)
    nn.init.normal_(fused.q.lora_B, std=1.0)
    nn.init.normal_(fused.v.lora_B, std=1.0)
    x = torch.randn(3, 16)
    got, ref = fused(x).chunk(3, -1), base(x).chunk(3, -1)
    assert torch.equal(got[1], ref[1])
    assert not torch.allclose(got[0], ref[0]) and not torch.allclose(got[2], ref[2])


def test_fused_parameter_count_matches_separate_q_and_v():
    """r*(in+d) per slice — the budget is comparable across encoder families."""
    net = _FusedNet(d=32, n=3)
    inject_adapters(net, _fused_paths(), fused=True)
    trainable, _ = freeze_all_but_adapters(net)
    assert trainable == 6 * 8 * (32 + 32)  # same as test_gradient_isolation
    assert len(list(lora_branches(net))) == 6


def test_fused_gradient_isolation():
    torch.manual_seed(0)
    net = _FusedNet()
    inject_adapters(net, _fused_paths(), fused=True)
    freeze_all_but_adapters(net)
    net(torch.randn(4, 32)).pow(2).mean().backward()
    for n, p in net.named_parameters():
        if "lora_" in n:
            assert p.grad is not None, n
        else:
            assert p.grad is None, f"frozen param {n} received a gradient"


def test_fused_checkpoint_round_trip():
    torch.manual_seed(0)
    net = _FusedNet()
    inject_adapters(net, _fused_paths(), fused=True)
    for b in lora_branches(net):
        nn.init.normal_(b.lora_B, std=0.05)
    net.eval()
    x = torch.randn(4, 32)
    expected, state = net(x), adapter_state_dict(net)
    assert len(state) == 12  # 3 blocks x {q, v} x {A, B}
    assert all(k.endswith(("q.lora_A", "q.lora_B", "v.lora_A", "v.lora_B"))
               for k in state)

    fresh = _FusedNet()
    for i, layer in enumerate(net.layers):
        fresh.layers[i].qkv.load_state_dict(layer.qkv.base.state_dict())
    inject_adapters(fresh, _fused_paths(), fused=True)
    fresh.eval()
    load_adapter_state_dict(fresh, state)
    assert torch.equal(fresh(x), expected)


def test_fused_merged_weight_equivalence():
    torch.manual_seed(0)
    base = nn.Linear(24, 72)
    fused = LoRAFusedQKV(base, rank=8, alpha=16.0, dropout=0.0)
    nn.init.normal_(fused.q.lora_B, std=0.05)
    nn.init.normal_(fused.v.lora_B, std=0.05)
    merged = nn.Linear(24, 72)
    with torch.no_grad():
        merged.weight.copy_(fused.merged_weight())
        merged.bias.copy_(base.bias)
    x = torch.randn(5, 24)
    assert torch.allclose(fused(x), merged(x), atol=1e-5)


def test_fused_q_and_v_draw_independent_dropout_masks():
    """Separate q/v adapters each drop their own input; so must the slices."""
    torch.manual_seed(0)
    fused = LoRAFusedQKV(nn.Linear(64, 192), rank=8, dropout=0.5)
    with torch.no_grad():
        fused.v.lora_A.copy_(fused.q.lora_A)
        fused.v.lora_B.fill_(1.0)
        fused.q.lora_B.fill_(1.0)
    fused.train()
    x = torch.randn(4, 64)
    q, _, v = (fused(x) - fused.base(x)).chunk(3, -1)
    assert not torch.allclose(q, v)


def test_fused_requires_equal_thirds():
    with pytest.raises(ValueError, match="thirds"):
        LoRAFusedQKV(nn.Linear(16, 40))


def test_fused_double_injection_raises():
    net = _FusedNet()
    inject_adapters(net, _fused_paths(), fused=True)
    with pytest.raises(ValueError, match="already has an adapter"):
        inject_adapters(net, _fused_paths(), fused=True)
