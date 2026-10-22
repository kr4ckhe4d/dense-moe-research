"""Approximation metrics between a dense FFN output and its sparse approximation."""

from __future__ import annotations

import torch


def approximation_metrics(ref: torch.Tensor, approx: torch.Tensor) -> dict[str, float]:
    """ref, approx: [T, d]. All metrics computed in float32.

    rel_err          aggregate ||approx - ref||_F / ||ref||_F
    rel_err_median   median per-token relative error
    cosine_mean      mean per-token cosine similarity
    mse              mean squared error per element
    """
    r = ref.float()
    a = approx.float()
    diff = a - r
    per_tok = diff.norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a, r, dim=-1)
    return {
        "rel_err": (diff.norm() / r.norm().clamp_min(1e-12)).item(),
        "rel_err_median": per_tok.median().item(),
        "rel_err_p90": per_tok.quantile(0.9).item(),
        "cosine_mean": cos.mean().item(),
        "cosine_p10": cos.quantile(0.1).item(),
        "mse": diff.pow(2).mean().item(),
    }


def expert_utilization(indices: torch.Tensor, num_experts: int) -> dict[str, float | list]:
    """Load statistics for selected expert indices [T, k]."""
    counts = torch.bincount(indices.flatten(), minlength=num_experts).float()
    frac = counts / counts.sum()
    entropy = -(frac[frac > 0] * frac[frac > 0].log()).sum().item()
    return {
        "load_fraction": frac.tolist(),
        "max_over_uniform": (frac.max() * num_experts).item(),
        "normalized_entropy": entropy / torch.log(torch.tensor(float(num_experts))).item(),
    }
