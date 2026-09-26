"""LoRA adaptation track: adapt the encoder's final blocks, not just the head.

Opt-in and self-contained: nothing here is imported by the frozen-probe
benchmark path (``sdx/train*.py`` never loads an encoder), so the published
protocol is unaffected. See ``sdx/lora/README.md`` for the protocol and how to
run it.

Adding an encoder is a spec, not code: add an ``EncoderTargetSpec`` to
``sdx.lora.targets.SPECS`` (block path, separate ``q``/``v`` or one fused
``qkv`` projection, runner, per-block kwargs, and the ``post`` steps between
the last block and the benchmark's tap), then
run ``python scripts/lora_parity.py --encoder <name>``. The encoder is
supported iff that passes — ``tail(capture(x)) == encoder(x)`` at the tap, a
no-op at ``B = 0``, gradients only on adapters. Only blocks that thread state
their parent builds need a new runner in ``sdx.lora.tails.RUNNERS``.
"""

from sdx.lora.targets import (  # noqa: F401
    SPECS,
    AdapterPlan,
    BlockTarget,
    plan_adapters,
)
