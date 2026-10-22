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

## F0. Tooling hazard: expandable_segments corrupts tensors (ROCm)

With `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, torch 2.14.1+rocm7.2 changes tensors that
are never written, after unrelated large frees plus `torch.cuda.empty_cache()`
(`scripts/repro_expandable_segments.py` reproduces it in ~20 lines on the 9070 XT). I set this flag
in an early `scripts/env.sh`. It corrupted the first oracle run: E=8/16 results looked
partition-independent. Now removed, and `import dense_to_moe` refuses to run with it. Cached
activations were re-collected without it and are bit-identical (64/64 layer files). Collection
never calls `empty_cache`. Kernel timings don't depend on weight values. All oracle results below
come from clean runs.

## F3. Gate 1: oracle bound on real activations — FAILS for MoE without training

Data: 12,288 calibration tokens per layer and 4,096 held-out validation tokens (disjoint by
document hash). Six domains: prose, math, chat instructions, C++, Python, JSON. Partitions are fit
on calibration and scored on validation. "Oracle" selection sees the true expert outputs, so no
learned router can beat it. Files: `oracle_analysis.json` (32 layers), `oracle_e2e.json`,
`oracle_granularity.json`.

### Per-layer FFN relative error (mean of 32 layers, best oracle = exhaustive or greedy)

| E/k (kept) | contiguous | random | kmeans gate W | kmeans co-activation | kmeans contribution |
|---|---|---|---|---|---|
| 4/3 (75%) | 0.402 | 0.402 | 0.336 | 0.314 | 0.315 |
| 8/6 (75%) | 0.382 | 0.382 | 0.315 | 0.299 | **0.300** |
| 8/4 (50%) | 0.575 | 0.575 | 0.493 | 0.464 | **0.462** |
| 16/8 (50%) | 0.548 | 0.548 | 0.466 | 0.445 | 0.444 |
| 8/2 (25%) | 0.754 | 0.754 | 0.686 | 0.637 | 0.631 |
| 16/4 (25%) | 0.726 | 0.726 | 0.650 | 0.614 | 0.609 |
| **neuron-level oracle** | 75%: **0.031** | 50%: **0.111** | 25%: **0.264** | | |

- Activation-informed partitions help (contribution / co-activation k-means beats contiguous by
  ~20%), but nowhere near enough. Contiguous and random are indistinguishable.
- **Selection barely matters.** At 8/4, oracle 0.462, static (one fixed subset = pruning) 0.511,
  random 0.689. Most of the oracle's gain is static: at 8/4 it picks one expert for nearly every
  token (max load 1.95x uniform, against a ceiling of 2.0).
- **Sum-to-1 gating (llama.cpp `qwen35moe` semantics) is much worse** than unweighted sum. At
  8/6: 0.785 vs 0.300. This confirms the research doc's concern.
- Layer sensitivity: middle layers (3-17) are hardest (~0.52-0.56 at 8/4). Layers 0-2 and
  26-31 are easier; layer 31 is at 0.13.
- SwiGLU contributions are spread out: the top 25% of neurons per token carry 64% of |contribution|,
  the top 50% carry 87%.
- **Granularity doesn't rescue it** (layers 4/15/26, norm oracle, kmeans contribution). At 50%
  kept, E=8: 0.50 / 0.50 / 0.33 -> E=256 (48-neuron experts): 0.44 / 0.43 / 0.28, against
  neuron-level 0.11-0.12. Important neurons are token-specific and don't form stable groups.

### End-to-end: every FFN replaced, 12 held-out sequences x 1024 tokens (12,276 predictions)

| Variant (all 32 layers) | FFN kept | KL(dense‖variant) | PPL (dense 4.111) | top-1 agree |
|---|---|---|---|---|
| oracle 8/6 | 75% | 0.142 | 4.591 | 0.874 |
| oracle 4/3 | 75% | 0.178 | 4.721 | 0.852 |
| oracle 8/4 | 50% | 0.459 | 6.042 | 0.763 |
| oracle 16/8 | 50% | 0.362 | 5.593 | 0.793 |
| oracle 16/4 | 25% | 1.296 | 13.64 | 0.611 |
| oracle 8/2 | 25% | 1.668 | 19.45 | 0.568 |
| static 8/6 (pruning) | 75% | 0.395 | 5.724 | 0.784 |
| static 8/4 (pruning) | 50% | 1.212 | 12.59 | 0.638 |
| neuron oracle | 75% | 0.0014 | 4.106 | 0.987 |
| neuron oracle | 50% | 0.0129 | 4.133 | 0.964 |
| neuron oracle | 25% | 0.0784 | 4.343 | 0.908 |

**Reference point (not measured on this model):** in `~/code/local-llm-benchmarks`, Qwen3.8-27B
KLD vs BF16 is 0.018 at IQ4_XS and 0.002 at Q6_K. Weight quantization to ~4 bits roughly halves
decode bytes for the *whole* model at ~1/8 the KL of oracle 8/6. Oracle 8/6 cuts only FFN bytes by
25%, about a 1.18x model-level ceiling. **It has to be re-measured on Qwen3.5-9B before being
quoted as a comparison.**

### Verdict on Gate 1

- **Training-free dense-to-MoE fails on Qwen3.5-9B**, even with a perfect router. The mildest useful
  config (8/6, ~1.18x model-level ceiling) costs KL 0.14 and +12% perplexity.
- The configs that would give real speed (8/2, 16/4: ~1.8x ceiling) are 3-5x worse in perplexity.
- A learned router can only do worse than these numbers.
- The loss is caused by block structure, not by sparsity itself. Per-token neuron selection at the
  same 50% budget is 35x better in KL (0.013 vs 0.459). Recovering MoE quality would need training
  that reshapes the FFN into block-sparse form. The literature puts that at 10B-200B tokens
  (DSMoE, LLaMA-MoE), which this workstation can't do in reasonable time.
- **Caveat on the neuron oracle:** it ranks neurons by their true activation h = silu(W_g x)·(W_u x).
  Computing h reads all of gate/up, so with exact selection only `down_proj` (1/3 of FFN bytes)
  can be skipped. Getting more needs predicting active neurons *before* gate/up (Deja Vu-style
  predictors), or TEAL-style input thresholding, which has its own, different error profile.
