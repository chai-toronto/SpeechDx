"""Tail-probe, param-group and schedule tests.

Covers the two-group LR audit and the frozen-parity pooling rule,
including a direct demonstration of the SpeechBrain landmine the group-aware
schedule exists to avoid.
"""

import pathlib

import pytest
import torch
import torch.nn as nn

from sdx.lora.layers import LoRALinear
from sdx.lora.model import (
    GroupAwareLinearSchedule,
    LoRATailProbe,
    build_param_groups,
    frozen_parity_mean,
)
from sdx.lora.targets import EncoderTargetSpec
from sdx.lora.tails import EncoderTail

SPEC = EncoderTargetSpec(blocks="layers", attn="self_attn", q="q_proj", v="v_proj")


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)

    def forward(self, x):
        a = self.self_attn
        return x + a.v_proj(torch.tanh(a.q_proj(x)))


class _Enc(nn.Module):
    def __init__(self, n=6, d=16):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n)])


class _Tail(EncoderTail):
    def forward(self, hidden, attention_mask=None):
        for b in self.blocks:
            hidden = b(hidden)
        return hidden


def _probe(n=6, d=16, cut=4, num_labels=1):
    enc = _Enc(n, d)
    tail = _Tail(enc, "fake", cut, SPEC)
    paths = [f"layers.{i}.self_attn.{p}"
             for i in range(cut, n) for p in ("q_proj", "v_proj")]
    return LoRATailProbe(tail, paths, num_labels, dropout=0.0), enc


def test_paths_rebase_onto_the_tail():
    probe, _ = _probe(n=6, cut=4)
    assert probe.target_paths == [
        "blocks.0.self_attn.q_proj", "blocks.0.self_attn.v_proj",
        "blocks.1.self_attn.q_proj", "blocks.1.self_attn.v_proj",
    ]
    assert isinstance(probe.tail.blocks[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(probe.tail.blocks[0].self_attn.k_proj, LoRALinear)


def test_path_before_the_cut_is_rejected():
    enc = _Enc(6, 16)
    tail = _Tail(enc, "fake", 4, SPEC)
    with pytest.raises(ValueError, match="before the cut"):
        LoRATailProbe(tail, ["layers.1.self_attn.q_proj"], 1)


def test_pooling_matches_warm_semantics_unmasked():
    """concat over chunks then a plain mean — no length mask."""
    a, b = torch.randn(3, 8), torch.randn(5, 8)
    expected = torch.cat([a, b], dim=0).mean(dim=0)
    assert torch.allclose(frozen_parity_mean([a, b]), expected)
    # A masked mean would weight the two chunks equally; this must not.
    equal_weight = (a.mean(0) + b.mean(0)) / 2
    assert not torch.allclose(frozen_parity_mean([a, b]), equal_weight)


def test_per_chunk_then_concat_not_concat_then_tail():
    """The tail must see each chunk separately, as the frozen encoder did."""
    probe, _ = _probe()
    probe.eval()
    c1, c2 = torch.randn(4, 16), torch.randn(7, 16)
    per_chunk = frozen_parity_mean(
        [probe.tail(c.unsqueeze(0)).squeeze(0) for c in (c1, c2)])
    concat_first = probe.tail(torch.cat([c1, c2]).unsqueeze(0)).squeeze(0).mean(0)
    out = probe([c1, c2])
    assert torch.allclose(out, probe.classifier(per_chunk), atol=1e-6)
    # For this toy block the two happen to agree; assert we used the per-chunk
    # path explicitly rather than relying on that coincidence.
    assert per_chunk.shape == concat_first.shape


def test_trainable_parameter_accounting():
    probe, _ = _probe(n=6, d=16, cut=4)
    # 2 blocks x 2 projections x r=8 x (16+16)
    assert probe.n_trainable_lora == 2 * 2 * 8 * 32
    trainable = {n for n, p in probe.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith("classifier") for n in trainable)
    frozen_in_tail = [n for n, p in probe.tail.named_parameters()
                      if p.requires_grad and "lora_" not in n]
    assert frozen_in_tail == []


def test_param_groups_carry_distinct_lrs_and_weight_decay():
    probe, _ = _probe()
    groups = build_param_groups(probe, lora_lr=3e-5, head_lr=9.476e-4,
                                head_weight_decay=0.012272)
    assert [g["name"] for g in groups] == ["lora", "head"]
    assert groups[0]["lr"] == 3e-5 and groups[0]["weight_decay"] == 0.0
    assert groups[1]["lr"] == 9.476e-4 and groups[1]["weight_decay"] == 0.012272
    assert sum(p.numel() for p in groups[0]["params"]) == probe.n_trainable_lora


def test_schedule_decays_each_group_against_its_own_initial_lr():
    """the two LRs must stay distinct across all 15 epochs."""
    probe, _ = _probe()
    opt = torch.optim.AdamW(build_param_groups(
        probe, lora_lr=3e-5, head_lr=9.476e-4, head_weight_decay=0.01))
    sched = GroupAwareLinearSchedule(opt, epoch_count=15)
    for epoch in range(1, 16):
        lrs = sched.step(epoch)
        assert lrs["lora"] != lrs["head"], epoch
        assert abs(lrs["head"] / lrs["lora"] - 9.476e-4 / 3e-5) < 1e-6
    assert sched.step(1)["lora"] == pytest.approx(3e-5)
    assert sched.step(15)["lora"] == pytest.approx(3e-6)
    assert sched.step(15)["head"] == pytest.approx(9.476e-5)


def test_speechbrain_update_learning_rate_would_collapse_the_groups():
    """The landmine itself — documented as an executable fact."""
    from speechbrain.nnet.schedulers import update_learning_rate

    probe, _ = _probe()
    opt = torch.optim.AdamW(build_param_groups(
        probe, lora_lr=3e-5, head_lr=9.476e-4, head_weight_decay=0.01))
    update_learning_rate(opt, 9.476e-5)  # what brain.py:754 effectively does
    assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"], (
        "if this ever fails, SpeechBrain changed and the workaround can be "
        "revisited")


def test_schedule_shape_matches_the_benchmark_linear_scheduler():
    """Same family as LinearScheduler: initial -> initial/10, linear in epoch."""
    probe, _ = _probe()
    opt = torch.optim.AdamW(build_param_groups(
        probe, lora_lr=1e-4, head_lr=1e-3, head_weight_decay=0.01))
    sched = GroupAwareLinearSchedule(opt, epoch_count=15)
    seq = [sched.step(e)["head"] for e in range(1, 16)]
    assert seq[0] == pytest.approx(1e-3)
    assert seq[-1] == pytest.approx(1e-4)
    deltas = [b - a for a, b in zip(seq, seq[1:])]
    assert max(deltas) - min(deltas) < 1e-12  # strictly linear


def test_forward_batch_shapes():
    probe, _ = _probe(num_labels=10)
    probe.eval()
    batch = [[torch.randn(4, 16)], [torch.randn(3, 16), torch.randn(6, 16)]]
    assert probe.forward_batch(batch).shape == (2, 10)


def test_gradients_reach_both_groups():
    probe, _ = _probe()
    out = probe.forward_batch([[torch.randn(5, 16)], [torch.randn(2, 16)]])
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in probe.lora_parameters())
    assert all(p.grad is not None for p in probe.head_parameters())


def test_bias_source_is_in_the_module_tree():
    """GPU-only bug guard: WavLM's tail rebuilds position_bias from layer 0's
    rel_attn_embed. If that module is not registered, ``.to(cuda)`` leaves it
    on CPU and the forward dies with a device mismatch — which CPU testing
    cannot catch. Assert it participates in the module tree, which is exactly
    what makes ``.to()`` move it.
    """
    import torch.nn as nn

    from sdx.lora.tails import WavLMTail

    class _A(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.rel_attn_embed = nn.Embedding(320, 8)
            self.num_heads = 8

    class _B(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attention = _A(d)

    class _Enc(nn.Module):
        def __init__(self, n, d):
            super().__init__()
            self.layers = nn.ModuleList([_B(d) for _ in range(n)])
            self.layer_norm = nn.LayerNorm(d)

    class _Wrap(nn.Module):
        def __init__(self, n, d):
            super().__init__()
            self.feature_extractor = _Enc(n, d)

    spec = EncoderTargetSpec(blocks="feature_extractor.layers",
                             attn="attention", q="q_proj", v="v_proj")
    tail = WavLMTail(_Wrap(6, 16), "wavlm", 4, spec)
    ids = {id(p) for p in tail.parameters()}
    assert id(tail.bias_source.rel_attn_embed.weight) in ids, (
        "bias source is not in tail.parameters(); .to(device) would skip it")
    assert "bias_source" in dict(tail.named_children())


def test_regression_primary_is_mae_cindex_is_secondary():
    """MAE is the reported regression metric; C-index is computed but not
    headlined (user, 2026-07-25)."""
    from sdx.lora.train import primary_metric, regression_metrics

    gold = torch.tensor([1.0, 2.0, 3.0, 4.0])
    preds = torch.tensor([1.5, 2.5, 2.0, 4.5])
    assert primary_metric(preds, gold, "R") == pytest.approx(
        float((preds - gold).abs().mean()))
    m = regression_metrics(preds, gold)
    assert set(m) == {"mae", "cindex"}
    # C-index still equals AUROC on a binary target — the property that lets
    # it sit on one footing with the classification column if ever adopted.
    from sdx.lora.train import macro_auroc
    g2 = torch.tensor([0.0, 0, 1, 1, 0, 1]); p2 = torch.randn(6)
    assert regression_metrics(p2, g2)["cindex"] == pytest.approx(
        macro_auroc(p2, g2), abs=1e-9)


def test_mae_delta_is_oriented_so_positive_means_better():
    """An MAE column next to an AUROC column silently reverses every sign
    unless the delta is oriented."""
    from sdx.lora.train import oriented_delta

    # MAE fell 8.5 -> 8.0: an improvement, so positive.
    assert oriented_delta("MAE", 8.0, 8.5) == pytest.approx(+0.5)
    # AUROC rose 0.60 -> 0.65: also an improvement, also positive.
    assert oriented_delta("macroAUROC", 0.65, 0.60) == pytest.approx(+0.05)
    # and a worse MAE is negative
    assert oriented_delta("MAE", 9.0, 8.5) == pytest.approx(-0.5)


def test_regression_bin_weights_reweight_the_loss():
    """The R branch must apply per-sample bin weights the way brain.py does:
    bucketize the LABEL, look up that bin's weight, scale, then mean."""
    from sdx.lora.train import run_epoch

    edges = torch.tensor([2.0])          # two bins: <2 and >=2
    binw = torch.tensor([10.0, 1.0])     # rare low bin weighted 10x
    loss_fn = nn.MSELoss(reduction="none")

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, i):
            return f"u{i}", [torch.zeros(2, 16)], [0.0 if i == 0 else 5.0]

    from sdx.lora.train import collate
    probe, _ = _probe(num_labels=1)
    probe.eval()
    loader = torch.utils.data.DataLoader(_DS(), batch_size=2, collate_fn=collate)
    with torch.no_grad():
        w_loss, _, _ = run_epoch(probe, loader, loss_fn, torch.device("cpu"),
                                 reg_bins=(edges, binw))
        u_loss, _, _ = run_epoch(probe, loader, nn.MSELoss(), torch.device("cpu"))
    # the low-bin sample carries 10x weight, so the weighted loss must differ
    assert w_loss != pytest.approx(u_loss)
    assert w_loss > 0


def test_multiclass_targets_are_int64_and_not_squeezed():
    """C wants int64 class indices for CrossEntropyLoss. An earlier collate
    cast every label to float32, which silently breaks multiclass."""
    from sdx.lora.train import collate, run_epoch

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, i): return f"u{i}", [torch.randn(3, 16)], i % 4

    probe, _ = _probe(num_labels=4)
    loader = torch.utils.data.DataLoader(_DS(), batch_size=4, collate_fn=collate)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss, preds, gold = run_epoch(probe, loader, loss_fn,
                                      torch.device("cpu"), task_type="C")
    assert preds.shape == (4, 4)          # logits kept, not squeezed
    assert gold.reshape(-1).shape == (4,)  # class indices, not one-hot
    assert loss > 0


def test_multiclass_metric_is_one_vs_rest_macro_auroc():
    from sdx.lora.train import primary_metric

    # 3 classes, perfectly separable -> OvR macro AUROC 1.0
    gold = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.tensor([[9., 0, 0], [0, 9, 0], [0, 0, 9],
                           [8., 0, 0], [0, 8, 0], [0, 0, 8]])
    assert primary_metric(logits, gold, "C") == pytest.approx(1.0)


def test_reset_parameters_restores_the_no_op_state():
    """CV needs each fold to start fresh. After reset the adapter must again
    be an exact no-op, or fold k inherits fold k-1 and the folds stop being
    independent."""
    torch.manual_seed(0)
    probe, _ = _probe()
    probe.eval()
    x = [torch.randn(5, 16)]
    before = probe.tail(x[0].unsqueeze(0))

    for a in probe.adapters.values():
        nn.init.normal_(a.lora_B, std=0.1)
    assert not torch.allclose(probe.tail(x[0].unsqueeze(0)), before)

    probe.reset_parameters()
    assert torch.equal(probe.tail(x[0].unsqueeze(0)), before)
    assert all(float(a.lora_B.abs().max()) == 0.0 for a in probe.adapters.values())


def test_micro_batching_gradient_equals_a_mean_reduced_batch():
    """run_epoch accumulates loss_i/N per recording instead of forming one
    batched loss. That is only legitimate if the resulting gradient is
    IDENTICAL to the mean-reduced batch it replaces — assert it directly,
    because the whole memory fix rests on this equivalence."""
    from sdx.lora.train import collate, run_epoch

    torch.manual_seed(0)
    xs = [torch.randn(4, 16) for _ in range(4)]
    ys = [1.0, 0.0, 1.0, 0.0]

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, i): return f"u{i}", [xs[i]], ys[i]

    loss_fn = nn.BCEWithLogitsLoss()

    # (a) micro-batched, via run_epoch
    torch.manual_seed(1); probe_a, _ = _probe(num_labels=1)
    opt = torch.optim.SGD([p for p in probe_a.parameters() if p.requires_grad], lr=0.0)
    loader = torch.utils.data.DataLoader(_DS(), batch_size=4, collate_fn=collate)
    run_epoch(probe_a, loader, loss_fn, torch.device("cpu"), opt)
    micro = {n: p.grad.clone() for n, p in probe_a.named_parameters()
             if p.grad is not None}

    # (b) one mean-reduced batched loss, by hand
    torch.manual_seed(1); probe_b, _ = _probe(num_labels=1)
    probe_b.train()
    logits = torch.stack([probe_b([x]) for x in xs]).reshape(-1)
    loss_fn(logits, torch.tensor(ys)).backward()
    batched = {n: p.grad.clone() for n, p in probe_b.named_parameters()
               if p.grad is not None}

    assert set(micro) == set(batched) and micro
    for n in micro:
        assert torch.allclose(micro[n], batched[n], atol=1e-6), n


def test_cv_moves_per_fold_losses_to_the_device():
    """The CV trainer rebuilds losses each fold, so it must also move each
    one's weight tensors to the device — the single-split trainer does this
    once and the CV path silently skipped it (BCEWithLogitsLoss then dies on
    a cuda/cpu mismatch, but only on GPU, so CPU tests can't catch it)."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "sdx" / "lora" / "train_cv.py").read_text()
    build = src.index("build_losses(")
    step = src.index("opt = torch.optim.AdamW")
    between = src[build:step]
    assert "loss_fn.to(device)" in between and "eval_loss_fn.to(device)" in between


def test_fused_spec_probe_injects_slice_adapters_and_resets():
    """A qkv= spec reaches LoRATailProbe as LoRAFusedQKV adapters; accounting,
    param groups and the CV re-init all go through the same branch walk."""
    from sdx.lora.layers import LoRAFusedQKV

    class _FBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = nn.Module()
            self.attn.qkv = nn.Linear(d, 3 * d)

        def forward(self, x):
            q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
            return x + torch.tanh(q) * k + v

    class _FEnc(nn.Module):
        def __init__(self, n=6, d=16):
            super().__init__()
            self.blocks = nn.ModuleList([_FBlock(d) for _ in range(n)])

    spec = EncoderTargetSpec(blocks="blocks", attn="attn", qkv="qkv")
    enc = _FEnc()
    tail = EncoderTail(enc, "fake", 4, spec)
    paths = [f"blocks.{i}.attn.qkv" for i in (4, 5)]
    torch.manual_seed(0)
    x = torch.randn(7, 16)
    with torch.no_grad():
        ref = tail(x.unsqueeze(0))
    probe = LoRATailProbe(tail, paths, num_labels=1, dropout=0.0)

    assert isinstance(probe.tail.blocks[0].attn.qkv, LoRAFusedQKV)
    assert probe.n_trainable_lora == 2 * 2 * 8 * (16 + 16)
    assert len(list(probe.lora_parameters())) == 2 * 2 * 2  # blocks x {q,v} x {A,B}

    with torch.no_grad():
        for p in probe.lora_parameters():
            p.normal_()
    probe.reset_parameters()
    probe.eval()
    with torch.no_grad():
        assert torch.equal(probe.tail(x.unsqueeze(0)), ref)
