"""End-to-end evaluation of sparse FFN variants inside the full model.

`SwitchableMLP` wraps each original `Qwen3_5MLP` and can run, per forward:
  dense              original FFN
  oracle             E-expert partition, k experts per token picked by true output norm (upper bound)
  static             same k experts for every token, chosen on calibration data (structured pruning)
  neuron             top fraction of neurons per token by contribution (unstructured upper bound)
All variants reuse the original weights, so one model load serves every configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dense_to_moe.partitioning.strategies import Partition


@dataclass(frozen=True)
class Mode:
    kind: str = "dense"  # dense | oracle | static | neuron
    E: int = 0
    k: int = 0
    keep: float = 1.0


class SwitchableMLP(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp
        self.mode = Mode()
        self.partitions: dict[int, Partition] = {}
        self.static_sets: dict[tuple[int, int], torch.Tensor] = {}
        self.stats: list[float] = []  # optional per-forward relative error, for diagnostics

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp.act_fn(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)

    def expert_outputs(self, h: torch.Tensor, E: int) -> torch.Tensor:
        """h: [T, I] -> per-expert outputs [E, T, d]."""
        perm = self.partitions[E].groups.flatten().to(h.device)
        T = h.shape[0]
        hp = h[:, perm].view(T, E, -1)
        wd = self.mlp.down_proj.weight[:, perm].view(-1, E, hp.shape[-1])
        return torch.einsum("tem,dem->etd", hp, wd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mode
        if m.kind == "dense":
            return self.mlp(x)
        shape = x.shape
        h = self.intermediate(x.reshape(-1, shape[-1]))
        if m.kind == "neuron":
            score = h.abs() * self.mlp.down_proj.weight.norm(dim=0)
            n = int(m.keep * h.shape[1])
            mask = torch.zeros_like(h).scatter_(1, score.topk(n, dim=1).indices, 1)
            y = self.mlp.down_proj(h * mask)
        else:
            out = self.expert_outputs(h, m.E)  # [E, T, d]
            if m.kind == "oracle":
                idx = out.float().norm(dim=-1).topk(m.k, dim=0).indices  # [k, T]
            elif m.kind == "static":
                idx = self.static_sets[(m.E, m.k)].to(h.device).view(-1, 1).expand(-1, h.shape[0])
            else:
                raise ValueError(m.kind)
            y = out.gather(0, idx.unsqueeze(-1).expand(-1, -1, out.shape[-1])).sum(0)
        return y.view(shape)


def wrap_model(model: nn.Module) -> list[SwitchableMLP]:
    wrappers = []
    for layer in model.model.layers:
        w = SwitchableMLP(layer.mlp)
        layer.mlp = w
        wrappers.append(w)
    return wrappers


def set_mode(wrappers: list[SwitchableMLP], mode: Mode, layers: set[int] | None = None) -> None:
    for i, w in enumerate(wrappers):
        w.mode = mode if layers is None or i in layers else Mode()


@torch.no_grad()
def compare_to_dense(dense_logits: torch.Tensor, logits: torch.Tensor,
                     labels: torch.Tensor) -> dict[str, float]:
    """Per-sequence sums for next-token prediction. logits: [L, V] (positions 0..L-2 used)."""
    lp_d = F.log_softmax(dense_logits[:-1].float(), -1)
    lp = F.log_softmax(logits[:-1].float(), -1)
    tgt = labels[1:]
    kl = (lp_d.exp() * (lp_d - lp)).sum(-1)  # KL(dense || sparse) per position
    return {
        "n": tgt.numel(),
        "kl_sum": kl.sum().item(),
        "nll_sum": -lp.gather(1, tgt.view(-1, 1)).sum().item(),
        "top1_agree_sum": (lp.argmax(-1) == lp_d.argmax(-1)).sum().item(),
    }
