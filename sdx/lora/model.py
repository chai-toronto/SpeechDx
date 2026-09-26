"""The trainable object: adapted tail → frozen-parity pooling → linear head.

Also the two-group optimizer and a group-aware LR schedule. SpeechBrain's
``update_learning_rate(optimizer, new_lr)`` (what ``sdx/brain.py`` calls)
takes no ``param_group`` argument and therefore overwrites **every** group's
LR with the head's scheduled value. With two groups that silently discards the
LoRA learning rate after epoch 1, so this track schedules each group itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sdx.lora.layers import freeze_all_but_adapters, inject_adapters, lora_branches
from sdx.lora.tails import EncoderTail


def frozen_parity_mean(chunk_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Reproduce ``sdx.warm._combine_chunk_embs``: concat chunks, mean over frames.

    Deliberately **not** length-masked. The published frozen vector includes the
    frames the encoder produced from ``_chunk_signal``'s ``min_samples`` zero
    padding, so masking them here would move the baseline and confound the
    LoRA delta with a pooling change.
    """
    return torch.cat(chunk_outputs, dim=-2).mean(dim=-2)


class LoRATailProbe(nn.Module):
    """Adapted encoder suffix + the benchmark's linear head.

    Input is a list of per-chunk boundary activations for ONE recording, each
    ``(T_i, d)``. The tail runs per chunk — matching the frozen path, which
    encoded each chunk separately — and the outputs are concatenated before a
    single recording-level mean.
    """

    def __init__(self, tail: EncoderTail, target_paths: list[str],
                 num_labels: int, *, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05, bias: bool = True):
        super().__init__()
        self.tail = tail
        # Paths were planned against the encoder; rebase onto the tail, whose
        # block i corresponds to encoder block cut+i.
        self.target_paths = [self._rebase(p, tail) for p in target_paths]
        self.adapters = inject_adapters(
            self.tail, self.target_paths, rank=rank, alpha=alpha, dropout=dropout,
            fused=bool(tail.spec.qkv))
        self.n_trainable_lora, _ = freeze_all_but_adapters(self.tail)

        width = next(iter(self.adapters.values())).in_features
        self.classifier = nn.Linear(width, num_labels, bias=bias)

    @staticmethod
    def _rebase(path: str, tail: EncoderTail) -> str:
        """``<blocks>.<i>.<rest>`` on the encoder → ``blocks.<i-cut>.<rest>``."""
        prefix = tail.spec.blocks + "."
        if not path.startswith(prefix):
            raise ValueError(f"{path!r} does not start with {prefix!r}")
        rest = path[len(prefix):]
        idx, tail_path = rest.split(".", 1)
        local = int(idx) - tail.cut
        if local < 0:
            raise ValueError(
                f"{path} is at block {idx}, before the cut at {tail.cut}")
        return f"blocks.{local}.{tail_path}"

    def freeze_adapters(self) -> int:
        """Control arm: keep the adapters injected but untrainable.

        With ``B = 0`` at init and no gradient, the tail is EXACTLY the frozen
        encoder, so this trains only the linear head on frozen boundary
        features — a frozen probe under this study's protocol (15 epochs, one
        configuration, this head LR).

        That is the control a LoRA delta needs. Comparing the LoRA arm
        directly against the published board would conflate two changes:
        adaptation, and a training protocol that differs from the board's
        (50 epochs, 5 configurations, plateau stopper). LoRA − control
        isolates the first; control − published board measures the second.
        """
        n = 0
        for p in self.lora_parameters():
            p.requires_grad = False
            n += p.numel()
        return n

    def reset_parameters(self) -> None:
        """Re-initialise adapters and head to their fresh state.

        Needed for cross-validation: each fold must start from scratch, or
        fold *k* inherits fold *k−1*'s adapters and the folds stop being
        independent. Reloading the encoder per fold would also work but costs a
        multi-GB model load five times per cell.

        Reinstates the init contract exactly — ``A`` Kaiming-uniform, ``B``
        zero — so the tail is once again bit-identical to the frozen encoder at
        epoch 0.
        """
        import math

        for m in lora_branches(self.tail):
            nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
            nn.init.zeros_(m.lora_B)
        self.classifier.reset_parameters()

    def lora_parameters(self):
        for m in lora_branches(self.tail):
            yield m.lora_A
            yield m.lora_B

    def head_parameters(self):
        return self.classifier.parameters()

    def forward(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """``chunks``: list of ``(T_i, d)`` (or ``(1, T_i, d)``) boundary tensors."""
        outs = []
        for c in chunks:
            if c.dim() == 2:
                c = c.unsqueeze(0)
            outs.append(self.tail(c).squeeze(0))
        pooled = frozen_parity_mean(outs)
        return self.classifier(pooled)

    def forward_batch(self, batch: list[list[torch.Tensor]]) -> torch.Tensor:
        return torch.stack([self(chunks) for chunks in batch])


def build_param_groups(model: LoRATailProbe, *, lora_lr: float, head_lr: float,
                       head_weight_decay: float,
                       lora_weight_decay: float = 0.0) -> list[dict]:
    """Two groups, tagged so the schedule can address them."""
    return [
        {"name": "lora", "params": list(model.lora_parameters()),
         "lr": lora_lr, "weight_decay": lora_weight_decay, "initial_lr": lora_lr},
        {"name": "head", "params": list(model.head_parameters()),
         "lr": head_lr, "weight_decay": head_weight_decay, "initial_lr": head_lr},
    ]


class GroupAwareLinearSchedule:
    """Per-group linear decay to ``initial_lr / final_ratio`` over the run.

    Same family and shape as the benchmark's ``LinearScheduler``
    (``lr_s → lr_s/10`` across ``number_of_epochs``), but applied to each param
    group against **its own** initial LR instead of collapsing them.

    Use this INSTEAD of ``speechbrain.nnet.schedulers.update_learning_rate``,
    which iterates every group and assigns one scalar.
    """

    def __init__(self, optimizer, epoch_count: int, final_ratio: float = 10.0):
        if epoch_count < 1:
            raise ValueError("epoch_count must be >= 1")
        self.optimizer = optimizer
        self.epoch_count = epoch_count
        self.final_ratio = final_ratio
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])

    def lr_at(self, initial: float, epoch: int) -> float:
        """``epoch`` is 1-based, matching SpeechBrain's epoch counter."""
        final = initial / self.final_ratio
        if self.epoch_count == 1:
            return final
        frac = min(max(epoch - 1, 0), self.epoch_count - 1) / (self.epoch_count - 1)
        return initial + (final - initial) * frac

    def step(self, epoch: int) -> dict[str, float]:
        out = {}
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr_at(g["initial_lr"], epoch)
            out[g.get("name", str(len(out)))] = g["lr"]
        return out
