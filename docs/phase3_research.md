# Phase 3 research: predictive neuron sparsity for Qwen3.5-9B

Written 2026-10-23, before the full-model Phase 3 results, after a one-layer smoke test (layer 15).
The gates in section 4 were fixed at that point. Results: `docs/phase3_findings.md`.

## 1. What the Phase 1 per-neuron oracle is, and what it proves

Code: `src/dense_to_moe/evaluation/e2e.py` (`Mode("neuron")`) and `scripts/oracle_analysis.py`.

- **Objective:** per token, keep the top `keep * 12288` neurons by `|h_i| * ||W_down[:, i]||`, where
  `h = silu(W_gate x) * (W_up x)`. That is the norm of neuron i's own output vector. It is a
  per-neuron magnitude heuristic, not a reconstruction optimum (outputs of different neurons can
  cancel). The scores are computed in bf16 in the Phase 1 code.
- **Information used:** the exact `h` of the current token, i.e. the *entire* gate and up
  projections (2/3 of the FFN weight bytes). Nothing about the choice is available before the FFN runs.
- **What it proves:** the FFN output of this model is well approximated by ~50% of its neurons,
  chosen per token (layer rel. error 0.11, full-model KLD 0.0103 at 50%).
- **What it does not prove:** that any speedup exists. Used as-is it can only skip down-projection
  columns: FFN bytes read = (2 + keep) / 3 = 0.83 at 50% kept. It is a selection-quality bound.

Phase 3 keeps three things apart:
1. **Selection quality** (oracle): how good a choice can be, given perfect information.
2. **Predictor quality**: how good a choice is from information available *before* the FFN.
3. **Execution efficiency**: what the kernel actually saves, including the predictor's cost.

## 2. Literature, and what transfers to a SwiGLU model

| Work | Mechanism | Needs training | Reported speed | Transfer to Qwen3.5-9B on this box |
|---|---|---|---|---|
| Deja Vu (ICML 2023) [arXiv 2310.17157](https://arxiv.org/abs/2310.17157) | Per-layer MLP predictor of active neurons/heads, lookahead from previous layer | Predictors only | >2x vs FasterTransformer, OPT-175B | Built on ReLU (exact zeros, ~90% sparsity). SwiGLU has no zeros, so "active" becomes "large", and a predictor must rank magnitudes. |
| PowerInfer (2023) [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) | Hot neurons on GPU, cold on CPU, online predictors | Predictors; ReLU models | 13.2 tok/s avg on a 4090, OPT-175B etc. | The CPU/GPU split solves "model doesn't fit". Ours fits in VRAM, so that part is irrelevant. Needs ReLU sparsity (TurboSparse/ProSparse ReLUfication for SwiGLU models: continued pretraining). |
| TurboSparse / ProSparse [arXiv 2406.05955](https://arxiv.org/abs/2406.05955) | Replace SiLU with dReLU/ReLU, retrain to ~90% sparsity | Yes, ~100B+ tokens | (with PowerInfer) | Far outside the training budget allowed. |
| CATS (COLM 2024) [arXiv 2404.08763](https://arxiv.org/abs/2404.08763) | Threshold the SiLU(gate) output | No | ~15% end-to-end latency | Needs the full gate projection first: skips at most up + down (bytes (1 + 2 keep)/3). Tested as `gate`. |
| TEAL (ICLR 2025) [arXiv 2408.14690](https://arxiv.org/abs/2408.14690) | Magnitude-threshold the *input* of every projection | No (calibrated thresholds) | 1.53x at 40%, 1.8x at 50% (whole model, BF16 decode) | Directly applicable. Speedups are vs fp16 decode, not vs a 4-bit quant. Skipping input channels of gate/up crosses GGUF quant blocks (32-256 values along the input dimension). Tested as `teal`. |
| WINA (ICLR 2026) [arXiv 2505.19427](https://arxiv.org/abs/2505.19427) | TEAL weighted by the column norm of the weight each input multiplies | No | (claims accuracy vs TEAL, not speed) | Tested as `wina`. |
| R-Sparse (ICLR 2025), LaRoSA (ICML 2025) [arXiv 2507.01299](https://arxiv.org/abs/2507.01299) | Input sparsity + low-rank compensation / layerwise rotations before top-k | No | LaRoSA 1.30x at 40% on LLaMA2-7B | Refinements of TEAL. Rotations change the weights (`W R`), which re-quantization would have to absorb. Not tested here. |
| Prox (2026) [arXiv 2607.27591](https://arxiv.org/abs/2607.27591) | Rank SwiGLU intermediate channels with input sparsity + quantized proxy weights, then compute only the selected channels exactly | No | up to 1.99x at 70% FFN sparsity | The closest to the Phase 3 hypothesis: a cheap approximation of `h` as the predictor. The proxy weights cost extra bytes every token. |
| DSMoE (EMNLP 2025) [arXiv 2502.12455](https://arxiv.org/abs/2502.12455) | FFN split into 8 blocks (experts) of 1024-1376 neurons, sigmoid router + STE, sparsity loss | Yes, 10B tokens continued pretraining | None reported (FLOPs only) | Same block layout that failed training-free in Phase 1 (oracle 8/4 KLD 0.38). Quality comes back only with 10B tokens of training, and still ~58% of parameters are active. |
| BlockFFN (2025) [arXiv 2507.08771](https://arxiv.org/abs/2507.08771) | ReLU+RMSNorm-routed MoE trained for *chunk-level* sparsity, kernels combining sparsity with speculative decoding | Yes (architecture trained for it) | up to 3.67x on end-side devices | Names the problem that matters for us: the union of the experts that consecutive tokens use is large, so speculative decoding (our MTP) erases token-level sparsity unless training targets chunk sparsity. |
| ReLU Strikes Back (2023) [arXiv 2310.04564](https://arxiv.org/abs/2310.04564) | ReLU models; aggregated sparsity over tokens | Yes | | Defines aggregated (union) sparsity over consecutive tokens for speculative decoding. |

### Points that matter for this model and hardware

1. **SwiGLU is not sparse, only compressible.** Selection is by magnitude, and quality falls
   smoothly with budget. No method can find "the zero neurons" because there are none.
2. **Block granularity is a trade, not a requirement for batch-1 decode.** A neuron's gate and up
   rows are contiguous 8 KB runs (bf16), ~2.2 KB at 4.25 bpw, and quantized independently along
   the input dimension. The down projection stored transposed ([I, d]) makes each neuron's down
   column contiguous too. So neuron-granular (B = 1) skipping is coalescing-friendly for one token.
   Blocks matter for (a) batched/prefill execution with matrix cores, (b) a predictor that scores
   n = 12288 / B blocks instead of 12288 neurons, and (c) the union over a verify batch.
3. **Input-channel sparsity (TEAL) conflicts with GGUF layouts.** It skips columns of gate/up, i.e.
   elements *inside* every quantized row. A quantized kernel can only skip whole quant blocks
   (32 values; 256 for k-quants super-blocks), or the weights have to be re-laid out transposed and
   quantized along the output dimension. The 32-aligned variant is tested.
4. **The decode bar is quantized + MTP, not BF16.** The measured FFN share of IQ4_XS decode time
   is only ~24% by the FFN-proxy fit (~55% of bytes). MTP verify steps check ~4 tokens at once
   (measured: 3.96-3.99 tokens per step), so the FFN must be sparse for the *union* of 4 masks.
5. **TEAL-class speedups were measured against fp16.** They do not carry over to a model that is
   already quantized to 4 bits, where weight bytes are already 4x smaller and fixed costs dominate.

## 3. Recommended approach for the first milestone

Training-free, measured before anything is trained or any kernel is written:

1. Reproduce the Phase 1 numbers and test the oracle code (done: B=1 `contrib` equals the Phase 1
   neuron mode exactly in fp64; tests in `tests/test_block_sparsity.py`).
2. Block-size sweep B = 1 ... 512 (+ Phase 1 controls) x budgets 90 ... 25%, for three objectives
   (norm, contribution sum, greedy reconstruction), contiguous vs k-means-clustered blocks.
3. Methods by information available: oracle (gate+up), gate-only (CATS-like), TEAL/WINA (input
   only), at matched FFN bytes, including the quant-block-aligned TEAL variant.
4. Full-model KLD against the same llama.cpp BF16 base as the GGUF quants, and per-layer mask
   unions over 2 and 4 consecutive tokens.
5. Speed ceilings from measured llama.cpp numbers for the surviving settings, with and without MTP.

A learned predictor (Stage C) and a sparse kernel (Stage D) only if the ceilings leave room.

## 4. Gates (fixed before the full results)

- **G1, oracle viability:** some block size B >= 8 reaches full-model KLD <= 0.0184 (IQ4_XS) at
  <= 50% of neurons kept. (B = 1 is known to pass at 0.0103.)
- **G2, predictability:** a method that does not need the full gate+up (gate, teal, wina) reaches
  KLD <= 0.0184 while reading <= 60% of FFN bytes.
- **G3/G5 ceiling, single token:** for some base quant q and sparse setting s, the bound on
  tok/s of q+s exceeds the measured tok/s of the fastest dense quant whose KLD is <= that of q+s
  (KLD of q+s estimated as KLD_q + KLD_s; additivity is an assumption).
- **Stretch, with MTP:** the same bound, using the 4-token union, exceeds IQ4_XS + MTP (148 tok/s).

If G2 or G3 fails, no predictor training or kernel work, and the failure is analysed instead.
