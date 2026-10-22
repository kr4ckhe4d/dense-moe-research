"""Oracle expert selection: upper bounds on what any router could achieve.

All oracles see the true expert outputs, which a deployable router cannot. They answer
"if routing were perfect, how much error does dropping (E - k) experts cost?"
"""

from __future__ import annotations

import itertools
import math

import torch


def select_by_norm(expert_out: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k experts by output norm ||E_i(x)||. expert_out: [E, T, d] -> indices [T, k]."""
    norms = expert_out.float().norm(dim=-1)  # [E, T]
    return norms.topk(k, dim=0).indices.T


def select_greedy(expert_out: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy forward selection minimizing ||y - sum_S E_i(x)|| with unit weights."""
    out = expert_out.float()
    y = out.sum(0)  # [T, d]
    E, T, _ = out.shape
    approx = torch.zeros_like(y)
    chosen = torch.zeros(T, E, dtype=torch.bool, device=out.device)
    picks = []
    for _ in range(k):
        cand = approx.unsqueeze(0) + out  # [E, T, d]
        err = (y.unsqueeze(0) - cand).pow(2).sum(-1).T  # [T, E]
        err = err.masked_fill(chosen, float("inf"))
        best = err.argmin(1)  # [T]
        picks.append(best)
        chosen[torch.arange(T, device=out.device), best] = True
        approx = approx + out[best, torch.arange(T, device=out.device)]
    return torch.stack(picks, dim=1)


def select_exhaustive(expert_out: torch.Tensor, k: int, max_combos: int = 20000,
                      chunk: int = 1024) -> torch.Tensor:
    """Exact best k-subset with unit weights, via the Gram matrix of expert outputs.

    ||y - sum_S o_i||^2 = ||y||^2 - 2 sum_S <y, o_i> + sum_{i,j in S} <o_i, o_j>
    """
    E = expert_out.shape[0]
    if math.comb(E, k) > max_combos:
        raise ValueError(f"C({E},{k}) too large for exhaustive search")
    out = expert_out.float()
    y = out.sum(0)
    combos = torch.tensor(list(itertools.combinations(range(E), k)), device=out.device)  # [C, k]
    onehot = torch.zeros(combos.shape[0], E, device=out.device)
    onehot.scatter_(1, combos, 1.0)  # [C, E]
    best_all = []
    for s in range(0, y.shape[0], chunk):
        o = out[:, s:s + chunk]  # [E, t, d]
        b = torch.einsum("etd,td->te", o, y[s:s + chunk])  # [t, E]
        G = torch.einsum("etd,ftd->tef", o, o)  # [t, E, E]
        lin = b @ onehot.T  # [t, C]
        quad = torch.einsum("ce,tef,cf->tc", onehot, G, onehot)  # [t, C]
        best_all.append(combos[(quad - 2 * lin).argmin(1)])
    return torch.cat(best_all)


def select_static(expert_out: torch.Tensor, k: int) -> torch.Tensor:
    """Control: one fixed expert subset for all tokens (= structured pruning), chosen by
    total output energy. Shows how much of any gain is due to *dynamic* routing."""
    energy = expert_out.float().pow(2).sum(dim=(1, 2))  # [E]
    idx = energy.topk(k).indices
    return idx.unsqueeze(0).expand(expert_out.shape[1], k)


def select_random(expert_out: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Control: uniformly random k experts per token."""
    E, T, _ = expert_out.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(T, E, generator=gen).argsort(1)[:, :k].to(expert_out.device)


ORACLES = {
    "norm": select_by_norm,
    "greedy": select_greedy,
    "exhaustive": select_exhaustive,
    "static": select_static,
    "random": select_random,
}


def indices_to_weights(indices: torch.Tensor, num_experts: int,
                       scores: torch.Tensor | None = None, mode: str = "sum") -> torch.Tensor:
    """Convert selected expert indices [T, k] into dense combination weights [T, E].

    mode:
      "sum"            selected experts weight 1 (preserves the dense invariant at k = E)
      "softmax_renorm" softmax over `scores` restricted to the selected experts, summing to 1
                       (llama.cpp qwen35moe / Qwen-MoE semantics)
      "scaled"         weight E/k for each selected expert (DSMoE-style rescaling)
    """
    T, k = indices.shape
    w = torch.zeros(T, num_experts, device=indices.device)
    if mode == "sum":
        w.scatter_(1, indices, 1.0)
    elif mode == "scaled":
        w.scatter_(1, indices, num_experts / k)
    elif mode == "softmax_renorm":
        if scores is None:
            raise ValueError("softmax_renorm needs router scores")
        sel = scores.float().gather(1, indices).softmax(dim=1)
        w.scatter_(1, indices, sel)
    else:
        raise ValueError(f"unknown combination mode {mode}")
    return w
