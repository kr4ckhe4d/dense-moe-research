"""Inference implementations of dense and top-k expert FFNs, for latency benchmarking.

Expert layout used by every sparse path (contiguous per expert, so selecting an expert
reads one contiguous slab of memory):
    w_gu:   [E, 2m, d]  rows [0, m) are gate, rows [m, 2m) are up
    w_down: [E, d, m]

Decode (one token) is memory-bandwidth bound. The only way to be faster than dense is to
read fewer weight bytes, so each path is judged by achieved GB/s on the bytes it must read.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN


def pack_dense(dense: DenseFFN) -> dict[str, torch.Tensor]:
    return {"w_gu": torch.cat([dense.gate.data, dense.up.data], 0).contiguous(),
            "w_down": dense.down.data.contiguous()}


def pack_experts(moe: PartitionedFFN) -> dict[str, torch.Tensor]:
    return {"w_gu": torch.cat([moe.w_gate.data, moe.w_up.data], 1).contiguous(),
            "w_down": moe.w_down.data.contiguous()}


def dense_ffn(x: torch.Tensor, w_gu: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    g, u = F.linear(x, w_gu).chunk(2, dim=-1)
    return F.linear(F.silu(g) * u, w_down)


def route_topk(x: torch.Tensor, w_router: torch.Tensor, k: int) -> torch.Tensor:
    """Linear router + top-k. Returns expert indices [T, k] (int32, stays on device)."""
    return F.linear(x, w_router).topk(k, dim=-1).indices.to(torch.int32)


@triton.jit
def _router_topk_kernel(x_ptr, w_ptr, idx_ptr, D, E: tl.constexpr, K: tl.constexpr,
                        BLOCK_D: tl.constexpr):
    """One program: logits = W_router x for all E experts, then K rounds of argmax."""
    experts = tl.arange(0, E)
    acc = tl.zeros((E,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        xv = tl.load(x_ptr + cols).to(tl.float32)
        w = tl.load(w_ptr + experts[:, None] * D + cols[None, :]).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    for j in tl.static_range(K):
        best = tl.argmax(acc, axis=0)
        tl.store(idx_ptr + j, best.to(tl.int32))
        acc = tl.where(experts == best, float("-inf"), acc)


def route_topk_triton(x: torch.Tensor, w_router: torch.Tensor, k: int,
                      block_d: int = 1024) -> torch.Tensor:
    """Fused decode router (T = 1): avoids ~30 us of hipBLAS GEMV + topk launch overhead.
    Selection order may differ from torch.topk on exact ties."""
    E, D = w_router.shape
    assert E & (E - 1) == 0 and D % block_d == 0, "E must be a power of two"
    idx = torch.empty(1, k, device=x.device, dtype=torch.int32)
    _router_topk_kernel[(1,)](x, w_router, idx, D, E=E, K=k, BLOCK_D=block_d)
    return idx


# ---------------------------------------------------------------------------------------
# Decode paths (T = 1)
# ---------------------------------------------------------------------------------------

def moe_decode_loop(x, w_gu, w_down, idx):
    """Python loop over selected experts. `idx.tolist()` forces a GPU->CPU sync per layer,
    which is what a naive PyTorch MoE does."""
    m = w_down.shape[2]
    y = None
    for e in idx[0].tolist():
        gu = F.linear(x, w_gu[e])
        h = F.silu(gu[..., :m]) * gu[..., m:]
        o = F.linear(h, w_down[e])
        y = o if y is None else y + o
    return y


def moe_decode_gather(x, w_gu, w_down, idx):
    """No sync, but `w[idx]` materializes a copy of the selected expert weights, so it moves
    the selected bytes twice (read + write) before computing on them."""
    e = idx[0].long()
    m = w_down.shape[2]
    gu = torch.bmm(w_gu[e], x.view(1, -1, 1).expand(e.numel(), -1, -1)).squeeze(-1)  # [k, 2m]
    h = F.silu(gu[:, :m]) * gu[:, m:]
    return torch.bmm(w_down[e], h.unsqueeze(-1)).sum(0).view(1, -1)


@triton.jit
def _expert_gate_up_kernel(x_ptr, w_ptr, idx_ptr, h_ptr, D, M,
                           BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    """h[j, r] = silu(Wg[e_j, r] . x) * (Wu[e_j, r] . x)   for r in this block."""
    j = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.load(idx_ptr + j).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    base = w_ptr + e * (2 * M * D)
    acc_g = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        xv = tl.load(x_ptr + cols).to(tl.float32)
        wg = tl.load(base + rows[:, None] * D + cols[None, :]).to(tl.float32)
        wu = tl.load(base + (rows[:, None] + M) * D + cols[None, :]).to(tl.float32)
        acc_g += tl.sum(wg * xv[None, :], axis=1)
        acc_u += tl.sum(wu * xv[None, :], axis=1)
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(h_ptr + j * M + rows, h.to(h_ptr.dtype.element_ty))


@triton.jit
def _expert_down_kernel(h_ptr, w_ptr, idx_ptr, y_ptr, D, M, K,
                        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, SPLIT: tl.constexpr):
    """y[n] += sum_j sum_r Wd[e_j, n, r] * h[j, r], split over the reduction for occupancy."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    chunk = M // SPLIT
    for j in range(0, K):
        e = tl.load(idx_ptr + j).to(tl.int64)
        base = w_ptr + e * (D * M)
        for m0 in range(pid_s * chunk, (pid_s + 1) * chunk, BLOCK_M):
            cols = m0 + tl.arange(0, BLOCK_M)
            hv = tl.load(h_ptr + j * M + cols).to(tl.float32)
            w = tl.load(base + rows[:, None] * M + cols[None, :]).to(tl.float32)
            acc += tl.sum(w * hv[None, :], axis=1)
    tl.atomic_add(y_ptr + rows, acc)


def moe_decode_triton(x, w_gu, w_down, idx, block_m=16, block_d=256, block_n=16, block_mr=128,
                      split=None):
    """Fused gathered GEMVs that read expert ids on device: no sync, no weight copy."""
    E, D, M = w_down.shape
    k = idx.shape[1]
    if split is None:  # largest split (more programs, better occupancy) that tiles exactly
        split = next(s for s in (4, 2, 1) if M % s == 0 and (M // s) % block_mr == 0)
    # Kernels have no masks: every block must tile its range exactly.
    assert M % block_m == 0 and D % block_d == 0 and D % block_n == 0
    assert M % split == 0 and (M // split) % block_mr == 0, (M, split, block_mr)
    h = torch.empty(k, M, device=x.device, dtype=x.dtype)
    _expert_gate_up_kernel[(k, M // block_m)](x, w_gu, idx, h, D, M,
                                              BLOCK_M=block_m, BLOCK_D=block_d)
    y = torch.zeros(D, device=x.device, dtype=torch.float32)
    _expert_down_kernel[(D // block_n, split)](h, w_down, idx, y, D, M, k,
                                               BLOCK_N=block_n, BLOCK_M=block_mr, SPLIT=split)
    return y.to(x.dtype).view(1, D)


# ---------------------------------------------------------------------------------------
# Prefill paths (T > 1)
# ---------------------------------------------------------------------------------------

def moe_prefill_loop(x, w_gu, w_down, idx):
    """Sort tokens by expert, one GEMM per expert over its tokens, scatter-add back."""
    T, D = x.shape
    E, _, M = w_down.shape
    k = idx.shape[1]
    flat = idx.flatten().long()
    order = flat.argsort()
    tok = order // k
    counts = torch.bincount(flat, minlength=E).tolist()  # sync
    y = torch.zeros(T, D, device=x.device, dtype=torch.float32)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        t = tok[start:start + c]
        gu = F.linear(x[t], w_gu[e])
        h = F.silu(gu[:, :M]) * gu[:, M:]
        y.index_add_(0, t, F.linear(h, w_down[e]).float())
        start += c
    return y.to(x.dtype)


def moe_prefill_grouped(x, w_gu, w_down, idx):
    """Same as the loop but with torch._grouped_mm: one launch per projection."""
    T, D = x.shape
    E, _, M = w_down.shape
    k = idx.shape[1]
    flat = idx.flatten().long()
    order = flat.argsort()
    tok = order // k
    offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
    xs = x[tok]
    gu = torch._grouped_mm(xs, w_gu.transpose(1, 2), offs=offs)
    h = F.silu(gu[:, :M]) * gu[:, M:]
    o = torch._grouped_mm(h, w_down.transpose(1, 2), offs=offs)
    y = torch.zeros(T, D, device=x.device, dtype=torch.float32)
    y.index_add_(0, tok, o.float())
    return y.to(x.dtype)
