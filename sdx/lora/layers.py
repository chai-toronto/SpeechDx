"""LoRA adapter layer and injection.

``W' = W + (alpha/r) B A``, with ``A`` initialised Kaiming-uniform and ``B``
initialised to **zero** — so a freshly injected adapter is an exact no-op and
the adapted tail starts bit-identical to the frozen encoder by construction.

Only query and value projections are wrapped, and only in the suffix chosen by
:func:`sdx.lora.targets.plan_adapters`. Every original parameter is frozen.

Two adapter shapes, one contract:

* :class:`LoRALinear` wraps a separate ``q_proj`` / ``v_proj``.
* :class:`LoRAFusedQKV` wraps a fused ``[q; k; v]`` projection (timm ViT,
  data2vec 2.0) and adapts only the q and v thirds of its output, through two
  :class:`LoRADelta` branches. That is exactly two separate adapters on the
  sliced projections — same factorization, same ``r * (in + d)`` parameters per
  slice, independent dropout masks — so K is never adapted and the
  budget stays comparable across encoders.

Everything downstream (freezing, re-init, checkpointing) addresses adapters
through :func:`lora_branches`: any module holding a ``lora_A``/``lora_B`` pair.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update.

    The base layer is kept as a child module (not copied), so state-dict keys
    for the frozen weights are preserved and a checkpoint of the adapter alone
    is meaningful.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        # Reference init (Hu et al.): A ~ Kaiming-uniform, B = 0 so BA = 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + delta * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """``W + (alpha/r) BA`` — for a merged-weight equivalence check."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.rank}, alpha={self.alpha}, scaling={self.scaling:g}")


class LoRADelta(nn.Module):
    """A bare low-rank update ``(alpha/r) B A x`` with no base layer.

    The branch type :class:`LoRAFusedQKV` adds onto one slice of its fused
    output. Same init contract as :class:`LoRALinear` (``B = 0``).
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:
        return (f"in={self.lora_A.shape[1]}, out={self.lora_B.shape[0]}, "
                f"r={self.rank}, alpha={self.alpha}, scaling={self.scaling:g}")


class LoRAFusedQKV(nn.Module):
    """Wraps a frozen fused ``[q; k; v]`` ``nn.Linear``; adapts q and v only.

    The output is split into equal thirds along the last dim — the layout both
    timm's ``Attention`` and data2vec 2.0's ``AltAttention`` unpack with
    ``reshape(B, N, 3, H, D)``. The k third passes through untouched.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        if base.out_features % 3:
            raise ValueError(
                f"fused projection out_features={base.out_features} is not "
                f"divisible into q/k/v thirds")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d = base.out_features // 3
        self.q = LoRADelta(base.in_features, d, rank, alpha, dropout)
        self.v = LoRADelta(base.in_features, d, rank, alpha, dropout)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.base(x).chunk(3, dim=-1)
        return torch.cat([q + self.q(x), k, v + self.v(x)], dim=-1)

    def merged_weight(self) -> torch.Tensor:
        """``W + [ΔW_q; 0; ΔW_v]`` — for a merged-weight equivalence check."""
        dq, dv = self.q.delta_weight(), self.v.delta_weight()
        return self.base.weight + torch.cat([dq, torch.zeros_like(dq), dv], dim=0)


def lora_branches(module: nn.Module):
    """Every module that owns a trainable ``lora_A``/``lora_B`` pair."""
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRADelta)):
            yield m


def _get_parent(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def inject_adapters(root: nn.Module, target_paths: list[str], *,
                    rank: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                    fused: bool = False) -> dict[str, nn.Module]:
    """Replace each ``target_paths`` entry with an adapter.

    ``root`` is whatever module the paths are relative to (the encoder wrapper
    for planning, or the tail when the paths have been rebased onto it).
    ``fused`` selects :class:`LoRAFusedQKV` (paths name fused qkv projections)
    over :class:`LoRALinear` (paths name separate q/v projections).
    Returns the injected adapters keyed by path.
    """
    adapter = LoRAFusedQKV if fused else LoRALinear
    injected: dict[str, nn.Module] = {}
    for path in target_paths:
        parent, attr = _get_parent(root, path)
        base = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
        if isinstance(base, (LoRALinear, LoRAFusedQKV)):
            raise ValueError(f"{path} already has an adapter")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} is {type(base).__name__}, not nn.Linear")
        wrapped = adapter(base, rank=rank, alpha=alpha, dropout=dropout)
        if attr.isdigit():
            parent[int(attr)] = wrapped
        else:
            setattr(parent, attr, wrapped)
        injected[path] = wrapped
    return injected


def freeze_all_but_adapters(module: nn.Module) -> tuple[int, int]:
    """Freeze everything, then unfreeze only ``lora_A``/``lora_B``.

    Returns ``(trainable, frozen)`` parameter counts.
    """
    for p in module.parameters():
        p.requires_grad = False
    trainable = 0
    for m in lora_branches(module):
        m.lora_A.requires_grad = True
        m.lora_B.requires_grad = True
        trainable += m.lora_A.numel() + m.lora_B.numel()
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen


def adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter tensors — what gets checkpointed."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()
            if k.endswith(("lora_A", "lora_B"))}


def load_adapter_state_dict(module: nn.Module,
                            state: dict[str, torch.Tensor]) -> None:
    missing = module.load_state_dict(state, strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected:
        raise KeyError(f"adapter state has unknown keys: {unexpected[:5]}")
    expected = set(adapter_state_dict(module))
    if set(state) != expected:
        raise KeyError(
            f"adapter state mismatch: missing {sorted(expected - set(state))[:5]}, "
            f"extra {sorted(set(state) - expected)[:5]}"
        )
