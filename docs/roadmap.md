# Roadmap

## Status after milestone 1 (2026-10-22)

| Gate | Question | Result |
|---|---|---|
| Kernel (Gate 2) | Can k-of-E expert FFNs approach the bytes-saved speedup at batch 1? | **Yes.** 82-98% of ideal in Triton (8/2: 3.3x per layer). Prefill is slower than dense with current grouped GEMM (implementation-limited) |
| Quality (Gate 1) | With a perfect router, is quality acceptable? | **No.** Oracle 8/6 (1.18x ceiling) costs KL 0.14 / PPL +12%; 8/4 costs KL 0.46 / PPL +47% |

The original hypothesis, training-free dense-to-MoE for faster inference, is not supported for
Qwen3.5-9B. The block-structured approximation is the problem. Neuron-level dynamic sparsity at
the same budget is >10x better in KL.

## Options for the next milestone

### A. Establish the quantization bar (recommended first, ~30-60 min)
Convert Qwen3.5-9B to GGUF and measure Q8_0 / Q6_K / Q4_K_M / IQ4_XS: KLD vs BF16 on our eval
text, plus decode/prefill tok/s on both GPUs. Every alternative has to beat this curve. MTP
speculative decoding goes on the same table.

### B. Neuron-level dynamic sparsity (most promising sparse direction)
1. Exact down-proj column skipping: no quality loss beyond the neuron oracle (KL 0.0014 at 75%,
   0.013 at 50%). Saves at most 1/3 of FFN bytes, a ~1.1-1.2x model ceiling.
2. TEAL input thresholding on gate/up/down (+ attention): training-free, published 1.2-1.6x batch-1.
3. Low-rank predictor for gate/up skipping (Deja Vu / SVD predictors): small training.
Reuses our Triton gathered-GEMV infrastructure (column-sparse instead of expert-sparse).
llama.cpp has no kernel for it, so integration would mean writing a GGML op.

### C. MoE with training recovery (requirements stages 2-3)
Shared + routed experts, distillation from the dense teacher. The literature suggests 10B+ tokens
(DSMoE) for ~40% sparsity with a visible PPL hit. Needs a measured tokens/hour estimate on the
R9700 before committing. The expected value is low given Gate 1.

### D. Combine: quantization + B
If B gives a gain on top of Q4/Q6, that's the real deliverable.
