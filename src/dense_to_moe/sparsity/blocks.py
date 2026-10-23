"""Phase 3: token-dependent FFN sparsity over blocks of B consecutive intermediate neurons.

SwiGLU FFN: h = silu(W_g x) * (W_u x)   [I],   y = W_d h = sum_b W_d[:, b] h_b   over blocks b.

Selection methods differ in what has to be computed *before* the choice is made, and that fixes
what can be skipped. Single-token decode is memory bound, so weight bytes read is the cost:

  method      needs before choosing      can skip                        FFN bytes read
  oracle      h (all of gate and up)     down columns of dropped blocks  (2 + keep) / 3
  gate        W_g x (all of gate)        up rows + down columns          (1 + 2 keep) / 3
  teal, wina  |x|, then |h|              gate/up input columns,          (2 x_keep + keep) / 3
                                         down columns
  (a Stage C predictor needs only a small projection of x: keep + predictor cost)

`oracle` and `gate` are selection-quality bounds, not deployable speedups: they show how good
the choice could be with information a cheap predictor would have to approximate.

Oracle objectives (per token):
  contrib  sum_i |h_i| ||W_d[:, i]||    (the Phase 1 per-neuron oracle at B = 1)
  norm     ||W_d[:, b] h_b||            (identical to contrib at B = 1)
  greedy   forward selection minimising ||y - y_S||; accounts for blocks cancelling each other.
           A heuristic: the exact optimum is a subset-selection problem and is not computed.
TEAL (Liu et al. 2024) thresholds |x| per layer; here top-k per token at the same budget is used.
WINA (Chen et al., arXiv 2505.19427) weights |x_j| by the norm of the weight column it multiplies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

METHODS = ("oracle", "gate", "teal", "wina")
OBJECTIVES = ("norm", "contrib", "greedy")


@dataclass(frozen=True)
class SparseSpec:
    method: str  # oracle | gate | teal | wina
    keep: float  # fraction of intermediate neurons kept (rounded to whole blocks)
    block: int = 1  # neurons per block
    objective: str = "norm"  # oracle only
    x_keep: float = 1.0  # teal/wina: fraction of FFN input channels kept for gate/up
    x_block: int = 1  # teal/wina: input channels per block (32 = GGUF quant-block aligned)

    def __post_init__(self) -> None:
        if self.method not in METHODS or self.objective not in OBJECTIVES:
            raise ValueError(f"bad spec {self}")

    @property
    def name(self) -> str:
        s = f"{self.method} B={self.block} keep={self.keep}"
        if self.method == "oracle" and self.objective != "norm":
            s += f" {self.objective}"
        if self.method in ("teal", "wina"):
            s += f" x_keep={self.x_keep} xB={self.x_block}"
        return s


def n_keep(n: int, keep: float) -> int:
    return max(1, min(n, round(keep * n)))


def topk_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """score [T, n] -> bool mask [T, n] with exactly k True per row."""
    return torch.zeros_like(score, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True)


def expand(mask: torch.Tensor, block: int) -> torch.Tensor:
    """Block mask [T, n] -> element mask [T, n * block]."""
    return mask.repeat_interleave(block, dim=1)


def block_sum(v: torch.Tensor, block: int) -> torch.Tensor:
    return v.view(v.shape[0], -1, block).sum(-1)


def ffn_bytes_fraction(spec: SparseSpec, kept: float, x_kept: float = 1.0) -> float:
    """Fraction of the dense FFN's weight bytes a decode step must read (gate, up, down equal)."""
    if spec.method == "oracle":
        return (2 + kept) / 3
    if spec.method == "gate":
        return (1 + 2 * kept) / 3
    return (2 * x_kept + kept) / 3


@torch.no_grad()
def greedy_blocks(h: torch.Tensor, down: torch.Tensor, block: int, k: int,
                  chunk: int = 16) -> torch.Tensor:
    """Greedy forward selection of k blocks per token minimising ||y - sum_{b in S} y_b||.

    With r = y - y_S and c_b = <y_b, r>, adding block b lowers the squared error by
    2 c_b - ||y_b||^2, and afterwards c <- c - G[:, b] (G = per-token Gram of block outputs).
    """
    T, inter = h.shape
    n, d = inter // block, down.shape[0]
    wd = down.float().view(d, n, block)
    out = torch.zeros(T, n, dtype=torch.bool, device=h.device)
    for s in range(0, T, chunk):
        hb = h[s:s + chunk].float().view(-1, n, block)
        t = hb.shape[0]
        Y = torch.einsum("tnb,dnb->tnd", hb, wd)
        G = Y @ Y.transpose(1, 2)  # [t, n, n]
        del Y
        c = G.sum(-1)
        diag = G.diagonal(dim1=1, dim2=2)
        sel = torch.zeros(t, n, dtype=torch.bool, device=h.device)
        rows = torch.arange(t, device=h.device)
        for _ in range(k):
            b = (2 * c - diag).masked_fill(sel, float("-inf")).argmax(-1)
            sel[rows, b] = True
            c = c - G[rows, b]
        out[s:s + chunk] = sel
    return out


class SparseFFN:
    """Emulates block-sparse execution of one SwiGLU FFN (masks, not skipped work).

    Works on raw weight tensors so it can wrap a `DenseFFN` or a HF `Qwen3_5MLP` alike. Per-layer
    statistics that depend on data (`up_absmean`, for `gate`) must come from calibration data.
    """

    @torch.no_grad()  # HF parameters require grad; without this each norm keeps an fp32 copy alive
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        self.gate, self.up, self.down = gate, up, down
        self.wd_norm = down.float().norm(dim=0)  # [I]
        self.x_colnorm = gate.float().norm(dim=0).hypot(up.float().norm(dim=0))  # [d]
        self.up_absmean: torch.Tensor | None = None
        self._grams: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def calibrate(self, x_cal: torch.Tensor) -> None:
        self.up_absmean = F.linear(x_cal.to(self.up.dtype), self.up).abs().float().mean(0)

    @torch.no_grad()
    def gram(self, block: int) -> torch.Tensor:
        """Per-block Gram W_d[:, b]^T W_d[:, b]: [n, B, B]."""
        if block not in self._grams:
            wd = self.down.float().view(self.down.shape[0], -1, block)
            self._grams[block] = torch.einsum("dnb,dnc->nbc", wd, wd)
        return self._grams[block]

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(F.linear(x, self.gate)) * F.linear(x, self.up)

    @torch.no_grad()
    def block_scores(self, h: torch.Tensor, block: int, objective: str) -> torch.Tensor:
        c = h.abs().float() * self.wd_norm
        if objective == "contrib" or block == 1:
            return block_sum(c, block)
        hb = h.float().view(h.shape[0], -1, block)
        # Not einsum("tnb,nbc,tnc->tn"): that lowers to a bmm with t*n batches, and hipBLASLt bmm
        # returns garbage for batch index >= 65536 (scripts/repro_bmm_batch_limit.py). Batch = n here.
        sq = (torch.einsum("tnb,nbc->tnc", hb, self.gram(block)) * hb).sum(-1)
        return sq.clamp_min(0).sqrt()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, spec: SparseSpec) -> tuple[torch.Tensor, dict]:
        """x [T, d] -> (y [T, d], info) with info['mask'] the kept-block mask [T, n] (and, for
        teal/wina, info['x_mask'] the kept input-channel-block mask)."""
        inter = self.gate.shape[0]
        n = inter // spec.block
        k = n_keep(n, spec.keep)
        info: dict = {"kept": k / n, "x_kept": 1.0}
        if spec.method == "oracle":
            h = self.intermediate(x)
            if spec.objective == "greedy":
                mask = greedy_blocks(h, self.down, spec.block, k)
            else:
                mask = topk_mask(self.block_scores(h, spec.block, spec.objective), k)
        elif spec.method == "gate":
            if self.up_absmean is None:
                raise RuntimeError("gate method needs calibrate() first")
            a = F.silu(F.linear(x, self.gate))
            s = a.abs().float() * self.up_absmean * self.wd_norm
            mask = topk_mask(block_sum(s, spec.block), k)
            h = a * F.linear(x, self.up)
        else:  # teal / wina: sparsify the FFN input, then the down-projection input
            xs = x.abs().float()
            if spec.method == "wina":
                xs = xs * self.x_colnorm
            nx = x.shape[1] // spec.x_block
            kx = n_keep(nx, spec.x_keep)
            info["x_mask"] = topk_mask(block_sum(xs, spec.x_block), kx)
            xm = expand(info["x_mask"], spec.x_block)
            info["x_kept"] = kx / nx
            h = self.intermediate(x * xm)
            hs = h.abs().float() * (self.wd_norm if spec.method == "wina" else 1.0)
            mask = topk_mask(block_sum(hs, spec.block), k)
        y = F.linear(h * expand(mask, spec.block).to(h.dtype), self.down)
        info["mask"] = mask
        info["bytes"] = ffn_bytes_fraction(spec, info["kept"], info["x_kept"])
        return y, info
