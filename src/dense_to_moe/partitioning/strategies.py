"""Expert partitioning strategies: assign each FFN intermediate neuron to exactly one expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Partition:
    """A disjoint, equal-size assignment of `intermediate_size` neurons to `num_experts` experts.

    `groups[e]` holds the (sorted) neuron indices of expert e. Sorting keeps gathers as
    contiguous as the strategy allows and makes partitions comparable across runs.
    """

    groups: torch.Tensor  # [num_experts, expert_size], int64
    strategy: str

    @property
    def num_experts(self) -> int:
        return self.groups.shape[0]

    @property
    def expert_size(self) -> int:
        return self.groups.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.groups.numel()

    def validate(self) -> None:
        flat = self.groups.flatten()
        n = flat.numel()
        if not torch.equal(torch.sort(flat).values, torch.arange(n, device=flat.device)):
            raise ValueError(f"partition '{self.strategy}' is not a permutation of 0..{n - 1}")

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "groups": self.groups.tolist()}


def _check_divisible(intermediate_size: int, num_experts: int) -> int:
    if intermediate_size % num_experts:
        raise ValueError(f"intermediate_size {intermediate_size} not divisible by {num_experts}")
    return intermediate_size // num_experts


def contiguous_partition(intermediate_size: int, num_experts: int) -> Partition:
    """Strategy A: expert e owns neurons [e*m, (e+1)*m)."""
    m = _check_divisible(intermediate_size, num_experts)
    groups = torch.arange(intermediate_size).view(num_experts, m)
    return Partition(groups=groups, strategy="contiguous")


def random_partition(intermediate_size: int, num_experts: int, seed: int) -> Partition:
    """Strategy B: uniformly random equal-size groups, deterministic given `seed`."""
    m = _check_divisible(intermediate_size, num_experts)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(intermediate_size, generator=gen)
    groups = torch.sort(perm.view(num_experts, m), dim=1).values
    return Partition(groups=groups, strategy=f"random(seed={seed})")


def balanced_kmeans_partition(
    features: torch.Tensor,
    num_experts: int,
    seed: int,
    iters: int = 20,
    strategy: str = "kmeans",
) -> Partition:
    """Strategy C building block: balanced k-means over per-neuron feature vectors.

    `features` is [intermediate_size, feat_dim] (e.g. gate weight rows, or activation
    statistics). Equal cluster sizes are enforced with a greedy capacity-constrained
    assignment: (neuron, cluster) pairs are visited in order of increasing distance and a
    neuron is placed in the nearest cluster that still has room. This approximates the
    exact balanced assignment (Jonker-Volgenant, as in CMoE) at a fraction of the cost.
    """
    n = features.shape[0]
    m = _check_divisible(n, num_experts)
    x = features.float()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    centroids = x[torch.randperm(n, generator=gen)[:num_experts].to(x.device)].clone()
    assign = torch.empty(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, centroids)  # [n, E]
        order = torch.argsort(dist.flatten())
        neuron = order // num_experts
        cluster = order % num_experts
        fill = [0] * num_experts
        placed_in = [-1] * n
        placed = 0
        for nu, cl in zip(neuron.tolist(), cluster.tolist()):
            if placed_in[nu] >= 0 or fill[cl] >= m:
                continue
            placed_in[nu] = cl
            fill[cl] += 1
            placed += 1
            if placed == n:
                break
        new_assign = torch.tensor(placed_in, device=x.device)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for e in range(num_experts):
            centroids[e] = x[assign == e].mean(0)
    groups = torch.stack([torch.nonzero(assign.cpu() == e).flatten() for e in range(num_experts)])
    return Partition(groups=groups, strategy=f"{strategy}(seed={seed})")
