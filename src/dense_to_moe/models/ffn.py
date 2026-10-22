"""Dense SwiGLU FFN and its exact expert-partitioned equivalent.

Dense:        y = W_down (silu(W_gate x) * (W_up x))
Partitioned:  y = sum_e  w_e * W_down[:, G_e] (silu(W_gate[G_e] x) * (W_up[G_e] x))

With every w_e = 1 the two are mathematically identical: the dense output is an additive sum
over intermediate neurons, and the groups G_e are a disjoint cover of those neurons. Only the
floating-point summation order differs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dense_to_moe.partitioning.strategies import Partition


class DenseFFN(nn.Module):
    """Reference SwiGLU FFN matching `Qwen3_5MLP` (no biases, SiLU)."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError(f"bad shapes gate={gate.shape} up={up.shape} down={down.shape}")
        self.gate = nn.Parameter(gate, requires_grad=False)  # [I, d]
        self.up = nn.Parameter(up, requires_grad=False)  # [I, d]
        self.down = nn.Parameter(down, requires_grad=False)  # [d, I]

    @property
    def hidden_size(self) -> int:
        return self.gate.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.gate.shape[0]

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(F.linear(x, self.gate)) * F.linear(x, self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.intermediate(x), self.down)


class PartitionedFFN(nn.Module):
    """FFN stored as `num_experts` disjoint neuron groups, in a contiguous per-expert layout.

    w_gate, w_up: [E, m, d]   (rows of the dense gate/up for that expert's neurons)
    w_down:       [E, d, m]   (matching columns of the dense down projection)
    """

    def __init__(self, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
                 partition: Partition) -> None:
        super().__init__()
        self.w_gate = nn.Parameter(w_gate, requires_grad=False)
        self.w_up = nn.Parameter(w_up, requires_grad=False)
        self.w_down = nn.Parameter(w_down, requires_grad=False)
        self.partition = partition

    @classmethod
    def from_dense(cls, dense: DenseFFN, partition: Partition) -> PartitionedFFN:
        partition.validate()
        if partition.intermediate_size != dense.intermediate_size:
            raise ValueError("partition size does not match FFN intermediate size")
        idx = partition.groups.to(dense.gate.device)
        w_gate = dense.gate.data[idx].contiguous()  # [E, m, d]
        w_up = dense.up.data[idx].contiguous()  # [E, m, d]
        w_down = dense.down.data[:, idx].permute(1, 0, 2).contiguous()  # [E, d, m]
        return cls(w_gate, w_up, w_down, partition)

    @property
    def num_experts(self) -> int:
        return self.w_gate.shape[0]

    def to_dense(self) -> DenseFFN:
        """Inverse mapping, used to verify that weight mapping is lossless."""
        idx = self.partition.groups.flatten().to(self.w_gate.device)
        E, m, d = self.w_gate.shape
        gate = torch.empty(E * m, d, dtype=self.w_gate.dtype, device=self.w_gate.device)
        up = torch.empty_like(gate)
        down = torch.empty(d, E * m, dtype=self.w_down.dtype, device=self.w_down.device)
        gate[idx] = self.w_gate.reshape(E * m, d)
        up[idx] = self.w_up.reshape(E * m, d)
        down[:, idx] = self.w_down.permute(1, 0, 2).reshape(d, E * m)
        return DenseFFN(gate, up, down)

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Every expert's contribution for every token: [E, T, d] for x of shape [T, d]."""
        g = torch.einsum("td,emd->etm", x, self.w_gate)
        u = torch.einsum("td,emd->etm", x, self.w_up)
        return torch.einsum("etm,edm->etd", F.silu(g) * u, self.w_down)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor | None = None) -> torch.Tensor:
        """Weighted sum of expert outputs.

        `expert_weights` is [T, E] (zeros for unselected experts). None means all experts
        with weight 1, which must reproduce the dense FFN.
        """
        out = self.expert_outputs(x)  # [E, T, d]
        if expert_weights is None:
            return out.sum(0)
        return torch.einsum("etd,te->td", out, expert_weights.to(out.dtype))
