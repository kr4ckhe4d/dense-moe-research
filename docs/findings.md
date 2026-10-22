# Findings (measured)

Every number here is measured on this workstation unless labelled otherwise. The raw data is in
`experiments/results/`.

## F1. Reconstruction invariant holds (2026-10-22, commit 3423c31)

`pytest`: 61 passed. These cover:
- Weight mapping dense -> experts -> dense is **bitwise exact** (bf16) for every strategy and E ∈ {4, 8, 16}.
- All-experts output matches dense to fp64 1e-12 and fp32 rel < 1e-6 (synthetic). On real layer-15 weights and real dims, fp32 rel < 1e-5.
- bf16 partitioned error vs an fp32 reference is no worse than dense bf16 error (within 1.5x).
- Dropping expert e removes exactly that expert's neurons' contribution.
- Every inference kernel (loop, gather, Triton, prefill loop, grouped_mm) matches the reference for k ∈ {1, 2, 4, 8}.
- The fused Triton router selects the same experts as `torch.topk`.

## F2. Single-layer kernel latency (real layer-15 weights, BF16, batch 1)

`benchmarks/kernel_latency.py`, 8 rotating layer copies, 50 repeats, CUDA-graph replay.
Std < 1% everywhere (see JSON). Router: random-init linear router fused with top-k in Triton,
included in every sparse time. Files: `kernel_latency_9070xt.json`, `kernel_latency_r9700.json`.

### Decode (T = 1), RX 9070 XT

| Path | µs/layer | GB/s achieved | vs best dense |
|---|---|---|---|
| dense, torch `F.linear` (hipBLASLt) | 680 | 444 | 0.76x |
| dense, Triton GEMV (E=1) | 527 | 573 | 0.98x |
| **dense-equivalent, Triton E=4 k=4 (best dense)** | **516** | **585** | **1.00x** |
| E=8 k=8 (all experts) | 565 | 535 | 0.91x |
| E=4 k=3 | 398 | 570 | 1.30x (ideal 1.33x) |
| E=8 k=6 | 433 | 523 | 1.19x (ideal 1.33x) |
| E=4 k=2 | 279 | 541 | 1.85x (ideal 2.0x) |
| E=8 k=4 | 301 | 502 | 1.71x (ideal 2.0x) |
| E=16 k=8 | 279 | 542 | 1.85x (ideal 2.0x) |
| E=8 k=2 | 157 | 482 | 3.29x (ideal 4.0x) |
| E=16 k=4 | 146 | 518 | 3.53x (ideal 4.0x) |

The R9700 is within 1-2% on every row (same gfx1201 die, same memory bandwidth class).

**Observations**
- At batch 1 the sparse FFN layer gets 82-98% of the ideal bytes-saved speedup. Kernel quality
  isn't the obstacle for decode.
- The fixed cost per layer is ~6 µs for the fused router, plus lower efficiency on small slabs.
  A torch router (`F.linear` + `topk` + cast) costs 28-33 µs, which would erase ~20% of the top-2/8 gain.
- Naive paths are far worse. A host-sync loop gets 271-358 GB/s. Gathering weights with `w[idx]`
  copies them first, getting 170-190 GB/s and running **slower than dense at k=8**.
- **Cache artefact**: timing one layer copy shows top-2/8 at 157 µs instead of 189 µs (an extra +20%),
  because the 64 MB Infinity Cache holds most of a 75 MB expert pair. Benchmarks must rotate layers.

### Prefill (T = 512, 2048), torch `_grouped_mm` (best available), RX 9070 XT

| Config | T=512 µs | vs dense | T=2048 µs | vs dense |
|---|---|---|---|---|
| dense (hipBLASLt, ~106 TFLOP/s) | 1,466 | 1.00x | 5,962 | 1.00x |
| E=8 k=8 (same FLOPs as dense) | 4,427 | 0.33x | 16,412 | 0.36x |
| E=8 k=4 | 2,695 | 0.54x | 8,733 | 0.68x |
| E=8 k=2 | 2,135 | 0.69x | 4,584 | 1.30x |
| E=16 k=4 | 2,596 | 0.56x | 7,682 | 0.78x |

**Prefill is an implementation bottleneck.** At equal FLOPs (k = E), the grouped path is 2.7-3x
slower than dense. Token gather/scatter, `index_add`, and ROCm `_grouped_mm` on short ragged groups
make up the overhead. Only top-2/8 at T=2048 beats dense. That calls for a proper fused grouped GEMM
(e.g. Triton), or for llama.cpp's `mul_mat_id`, which should be measured separately.

## F3. Oracle bound on real activations

Pending: `scripts/oracle_analysis.py` (Gate 1).
