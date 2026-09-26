#!/usr/bin/env python
"""Split-parity check for a LoRA spec, run against the real encoder weights.

Asserts that splitting an encoder at its cut layer is lossless:

    tail(capture_boundary(x))  ==  encoder(x)

within tolerance, for random audio at realistic chunk lengths. If this fails,
the tail is missing something the frozen forward does (a position bias, a final
layer norm, a mask) and every downstream number would be quietly wrong.

CPU/fp32, no training, no data: runs on a laptop. An encoder is supported
only once this passes for it.

    python scripts/lora_parity.py
    python scripts/lora_parity.py --encoder whisper --seconds 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sdx.lora.tails import build_tail, capture_boundary  # noqa: E402
from sdx.lora.targets import SPECS, plan_adapters  # noqa: E402

# Default: every registered spec. A spec is only usable once it passes here.
PANEL = list(SPECS)
# Realistic chunk lengths (Coswara mean 11 s); fixed-input encoders at their
# native chunk (encoder yaml max_length). Unlisted encoders default to 11 s.
DEFAULT_SECONDS = {"wavlm": 11.0, "whisper": 30.0, "qwen3voice": 11.0}
# fp32 CPU: accumulated differences over a 30-block prefix land ~1e-5.
ATOL, RTOL = 2e-4, 1e-3


def load_encoder(model_name: str):
    from hyperpyyaml import load_hyperpyyaml

    from sdx.registry import encoders as registry_encoders

    path = REPO / "sdx" / "configs" / "encoders" / registry_encoders()[model_name]
    with open(path) as fin:
        hp = load_hyperpyyaml(fin)
    return hp["encoder"], hp


def check(model_name: str, seconds: float | None) -> bool:
    encoder, hp = load_encoder(model_name)
    encoder.eval()
    sr = hp["sample_rate"]
    secs = seconds or DEFAULT_SECONDS.get(model_name, 11.0)
    torch.manual_seed(0)
    wav = torch.randn(1, int(sr * secs)) * 0.1

    plan = plan_adapters(encoder, model_name)
    cut = plan.cut_index

    with torch.no_grad():
        reference = encoder(wav)
        if isinstance(reference, tuple):
            reference = reference[-1]
        boundary = capture_boundary(encoder, model_name, cut, lambda: encoder(wav))
        tail = build_tail(encoder, model_name, cut).eval()
        got = tail(boundary)

    ok_shape = reference.shape == got.shape
    if not ok_shape:
        print(f"  ✗ SHAPE mismatch: reference {tuple(reference.shape)} "
              f"vs tail {tuple(got.shape)}")
        return False

    diff = (reference - got).abs()
    max_abs = diff.max().item()
    scale = reference.abs().max().item()
    ok = torch.allclose(reference, got, atol=ATOL, rtol=RTOL)
    mark = "✓" if ok else "✗"
    print(f"  {mark} cut={cut}/{plan.num_blocks}  boundary={tuple(boundary.shape)}  "
          f"out={tuple(got.shape)}")
    print(f"    max|diff|={max_abs:.3e}  (max|ref|={scale:.3f}, "
          f"rel={max_abs / max(scale, 1e-9):.2e}, tol atol={ATOL})")
    if not ok:
        print(f"    mean|diff|={diff.mean().item():.3e}  "
              f"frames differing >{ATOL}: "
              f"{(diff.max(dim=-1).values > ATOL).sum().item()}/{diff.shape[-2]}")
    return ok


def check_adapted(model_name: str, seconds: float | None) -> bool:
    """The same check end-to-end, with adapters injected.

    A freshly injected adapter has ``B = 0``, so the adapted tail must still
    reproduce the frozen encoder exactly. Then check the parameter budget,
    that only adapters get gradients, and that a save/load round trip is
    faithful.
    """
    from sdx.lora.layers import (
        adapter_state_dict,
        load_adapter_state_dict,
        lora_branches,
    )
    from sdx.lora.model import LoRATailProbe

    encoder, hp = load_encoder(model_name)
    encoder.eval()
    sr = hp["sample_rate"]
    secs = seconds or DEFAULT_SECONDS.get(model_name, 11.0)
    torch.manual_seed(0)
    wav = torch.randn(1, int(sr * secs)) * 0.1

    plan = plan_adapters(encoder, model_name)
    with torch.no_grad():
        reference = encoder(wav)
        if isinstance(reference, tuple):
            reference = reference[-1]
        boundary = capture_boundary(
            encoder, model_name, plan.cut_index, lambda: encoder(wav))

    tail = build_tail(encoder, model_name, plan.cut_index).eval()
    with torch.no_grad():
        bare = tail(boundary)  # the tail before any adapter is injected
    probe = LoRATailProbe(tail, plan.target_paths(), num_labels=1, dropout=0.05)
    probe.eval()

    with torch.no_grad():
        adapted = probe.tail(boundary)
    # B = 0 must be an EXACT no-op on the tail. The tail itself is compared
    # with the frozen encoder (within tolerance) by check() above.
    zero_delta = torch.equal(bare, adapted)
    print(f"  {'✓' if zero_delta else '✗'} B=0 no-op: "
          f"max|diff|={(bare - adapted).abs().max().item():.3e} "
          f"(vs encoder {(reference - adapted).abs().max().item():.3e})")

    budget_ok = probe.n_trainable_lora == plan.total_params <= 100_000
    print(f"  {'✓' if budget_ok else '✗'} budget: {probe.n_trainable_lora:,} "
          f"LoRA params (plan {plan.total_params:,}, ceiling 100,000)")

    logits = probe([boundary.squeeze(0)])
    logits.pow(2).mean().backward()
    leaked = [n for n, p in probe.tail.named_parameters()
              if p.grad is not None and "lora_" not in n]
    got_grad = all(p.grad is not None for p in probe.lora_parameters())
    isolated = not leaked and got_grad
    print(f"  {'✓' if isolated else '✗'} gradient isolation: "
          f"{len(leaked)} frozen tail params with grads")

    with torch.no_grad():
        for b in lora_branches(probe.tail):
            b.lora_B.normal_(std=0.02)
    probe.eval()
    with torch.no_grad():
        want = probe([boundary.squeeze(0)])
        state = adapter_state_dict(probe.tail)
        for b in lora_branches(probe.tail):
            b.lora_B.zero_()
        load_adapter_state_dict(probe.tail, state)
        got = probe([boundary.squeeze(0)])
    round_trip = torch.allclose(want, got, atol=1e-6)
    print(f"  {'✓' if round_trip else '✗'} checkpoint round trip "
          f"({len(state)} adapter tensors)")

    return zero_delta and budget_ok and isolated and round_trip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", dest="encoders")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--split-only", action="store_true",
                    help="skip the adapter-injected end-to-end phase")
    args = ap.parse_args()

    results = {}
    for name in args.encoders or PANEL:
        print(f"\n=== {name} ===", flush=True)
        try:
            ok = check(name, args.seconds)
            if ok and not args.split_only:
                print("  -- with adapters injected --")
                ok = check_adapted(name, args.seconds)
            results[name] = ok
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            results[name] = False

    print(f"\n{'=' * 60}\nsplit parity + B=0 no-op + budget + grads + checkpoint")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} FAILED — do NOT build caches: {failed}")
        return 1
    print("\nAll tails are exact. Cut layers are safe to cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
