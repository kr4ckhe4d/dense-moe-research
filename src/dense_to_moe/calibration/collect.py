"""Capture FFN inputs (post-attention-norm hidden states) from every layer of the dense model.

Memory bound: only a random subset of token positions per sequence is kept, and captured
rows are moved to CPU immediately, so GPU memory stays at model + one forward pass.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


def find_mlps(model: nn.Module) -> list[nn.Module]:
    layers = model.model.layers
    return [layer.mlp for layer in layers]


@torch.no_grad()
def collect_ffn_inputs(model: nn.Module, sequences: list[tuple[str, list[int]]],
                       tokens_per_seq: int, seed: int, skip_first: int = 4,
                       ) -> tuple[list[torch.Tensor], list[str]]:
    """Returns per-layer [N, d] bf16 CPU tensors of FFN inputs and a domain label per row.

    The first `skip_first` positions of each sequence are excluded: attention-sink tokens have
    extreme activations that would otherwise dominate the statistics.
    """
    mlps = find_mlps(model)
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    store: list[list[torch.Tensor]] = [[] for _ in mlps]
    pos: torch.Tensor | None = None

    def make_hook(i):
        def hook(_mod, args):
            store[i].append(args[0][0, pos].to("cpu", torch.bfloat16))
        return hook

    handles = [m.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(mlps)]
    domains: list[str] = []
    try:
        for n, (domain, ids) in enumerate(sequences):
            L = len(ids)
            pos = (torch.randperm(L - skip_first, generator=gen)[:tokens_per_seq] + skip_first)
            pos = pos.sort().values.to(device)
            model(torch.tensor([ids], device=device), use_cache=False)
            domains += [domain] * pos.numel()
            if n % 8 == 0:
                log.info("collected %d/%d sequences", n + 1, len(sequences))
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(s) for s in store], domains
