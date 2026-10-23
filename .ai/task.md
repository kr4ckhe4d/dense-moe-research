# Current task

## Done (milestone 1: single-layer prototype + Gate 1), 2026-10-22
- Environment, model verification, research (docs/environment.md, docs/research.md)
- Partitioned FFN with exact reconstruction, 63 tests passing
- Oracle/control routing, Triton decode kernels + fused router, latency benchmarks on both GPUs
- Calibration pipeline (6 domains, hash-disjoint splits), activations cached for all 32 layers
- Gate 1 (oracle bound): per-layer (32 layers), granularity sweep, end-to-end KL/PPL
- Result: decode kernels hit 82-98% of the ideal speedup, but **even oracle routing loses too
  much quality** (8/6: KL 0.14, PPL +12%; 8/4: KL 0.46, PPL +47%). See docs/findings.md F2-F3.

## Done (milestone 2: quantization bar), 2026-10-23
- BF16 GGUF from the HF snapshot, imatrix on calib, Q8_0/Q6_K/Q5_K_M/Q4_K_M/IQ4_XS/Q3_K_M
  (all in /mnt/fast/models/Qwen3.5-9B-*.gguf, plus two FFN-proxy files)
- KLD vs llama.cpp BF16 on val + wikitext-2 (benchmarks/gguf_kld.sh); PyTorch sparse variants scored
  against the same base file (scripts/sparse_kld_vs_gguf.py); llama-bench on both GPUs; MTP
  (benchmarks/gguf_mtp.sh); FFN-bytes speed proxy. Summary: scripts/gguf_summary.py. See F4.
- Result: IQ4_XS + MTP 148 tok/s at KLD 0.018; oracle MoE 8/6 has KLD 0.117; at most ~1.2x
  from sparse FFN on top of IQ4_XS.

## Waiting on the user
The decision on the next milestone (options in docs/roadmap.md).

## Next candidate experiments (cheapest first)
1. Measure the real bar: Qwen3.5-9B GGUF at Q8_0 / Q6_K / Q4_K_M / IQ4_XS in llama.cpp (KLD vs BF16
   on the same eval text, plus tok/s). Needs `convert_hf_to_gguf.py` + `llama-quantize` (~30 min).
2. Realistic neuron-level sparsity:
   a. exact `down_proj` skipping (h is computed anyway): quality free at 75-50%, saves <= 1/3 FFN bytes
   b. TEAL-style input thresholding on all projections (training-free)
   c. Deja Vu-style low-rank predictor of active neurons before gate/up (small training)
3. Only if the user wants it: MoE + distillation recovery. Needs a training-budget estimate first.
