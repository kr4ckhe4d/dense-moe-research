# Roadmap

## Reopened (2026-11-11): candidate tests beyond the closed findings

The sparsity results stand. These are the tests that could still change decode speed, on
llama.cpp b12164 (4b77dee05). Only ideas with a plausible gain are listed; a trained tiny router
and other fixed neuron groupings (including overlapping ones) are not, because the oracle router
and the 8-neuron block result already bound them.

Done (`docs/depth.md`, F10):
- **Patch re-check on b12164.** Upstream's own MMVQ rework (#30517) made stock faster, and the
  patch still helps on top (UD-Q6_K: 6-token batch 56.7 -> 48.8 ms, 8-token 70.8 -> 54.3 ms;
  1 token unchanged).
- **Decode against context depth** (27B UD-Q6_K, R9700, 2K-120K). Acceptance holds at depth; the
  time goes to attention. The q8_0 KV cache takes 4.2x its bandwidth-ideal time, f16 runs at the
  ideal. Switching to f16 KV: +22-31% at 120K for 3.3 GiB.

- **Quantized-KV attention kernel** (`docs/depth.md` section 6, F11,
  `patches/llama.cpp-rdna4-fattn-q8-tile.patch`). The tile kernel now reads q8_0 K/V directly:
  one layer at 120K 2073 -> 944 us; 27B with q8_0 KV at 120K +24% plain, +14-17% with MTP.

- **Patched build on the two-card daily preset** (`docs/depth.md` section 8, F12). Same gains as
  on one card (MTP +15% at 120K), no measurable change in code quality over 12 seeds. f16 KV on
  the two-card layout: +22% over stock q8_0 at 120K for 2.8 GiB.

- **Attention patch extended** (`docs/depth.md` section 9, F13): q4_0 KV and head sizes 64 / 128.
  27B with q4_0 KV at 120K: plain +94%, MTP +11%. Open follow-up: f16 decode at head sizes
  64 / 128 is slower than the patched quantized path, so the tile kernel may help there too.

Candidates, in the order I would run them:

| # | Test | Why it could matter | Cheap first step |
|---|---|---|---|
| 1 | **Small-batch matrix-core attention kernel** | A 4-token verify batch costs 2.0x one token at depth because the tile kernel is compute-bound. Block-count tuning gives only ~4% per MTP step at 120K (tried, `docs/depth.md` section 7); the WMMA kernel is 2.6x the bandwidth ideal at small batches even with more blocks. A proper fix is a new kernel: up to ~+20% for MTP at 120K, uncertain | Profile where the WMMA kernel's flat per-call cost goes |
| 2 | **Lenient draft acceptance** | Verification accepts only exact matches. Accepting near-ties spends a small quality budget on acceptance instead of on bits; aimed at prose and instructions (~49% acceptance) | Log the target's probability of each rejected draft token; no patch needed to size the gain |
| 3 | **Per-context neuron subsets** | One neuron subset per kind of text (code, prose, JSON), chosen per neuron from recorded usage. Unlike per-token masks it is stable across the verify batch, so it could stack with MTP / DFlash. Ceiling ~1.2x on top of quantization. Untested: only whole-expert static pruning was measured (KLD 0.395 at 75%) | Offline: per-domain subsets at 50% / 75% kept, same KLD protocol, plus routing of held-out prompts. Needs activations regenerated (~half a day in all). Pass bar: near IQ4_XS's 0.018 at 50% |
| 4 | **Shared-dequant verify kernel** | The multi-column MMVQ loop dequantizes each weight block once per drafted token. Smaller than first estimated: with the patch a 6-token batch costs 1.18x one token on UD-Q6_K (Q3_K_M 1.47x, Q8_0 1.06x), so ~10% of a verify step on Q6_K. Upstream merged the same idea for OpenCL (#31758) | Prototype for Q6_K only, time with `llama-bench -p 1..8` |
| 5 | **Drafter fine-tuned on the user's workload** | DFlash drafters are 1.3-2B parameters and trainable on the R9700; output stays lossless. Targets the same weak domains as 2 | Measure acceptance on real agent transcripts first |

## Final status (2026-10-24, F9): project closed

All open items from the 27B milestone were done (`docs/dflash.md`). The closing summary,
recommendations per GPU size, and limits are in `docs/conclusions.md`.
- MMVQ patch: neutral or faster on both RDNA4 cards, the 9B, Gemma 4 31B and two MoE models.
  Upstream evidence on b11719 is in `docs/upstream_mmvq.md`. Opening the PR is left to the user:
  llama.cpp requires the contributor to write it and disclose AI use.
- DFlash at long context (+3-6% over MTP), sampling (+14-18%), editing with n-gram (5.4-5.7x),
  4 users (+3-6%).
- The 9B has its own DFlash drafter (+12-29% over MTP).
- Under offload MTP wins, except for a 27B that is mostly in RAM.

Left open, none planned:
- Real smaller GPUs and slower host RAM. Budgets were simulated on a fast card with DDR5.
- Shrinking the DFlash drafter's VRAM use with `-ngld 0`.
- Prefill-heavy workloads under offload.
- Training a sparse model (requirements stages 2-3).

## Status after the 27B speed milestone (2026-10-24, F8)

On a 32 GB RDNA4 card the 27B runs at 62.6 tok/s at UD-Q6_K quality (was 51.2) and 81.9 at
IQ4_XS: DFlash2 drafter at draft length 5, plus a tuned RDNA4 MMVQ kernel for 2-8 token batches.
See `docs/fast27b.md`.

Open:
- Upstream the MMVQ patch (needs RDNA4 numbers on more shapes and MoE models).
- DFlash2 at long context and at temperature 1.
- DFlash2 + n-gram drafting for editing; DFlash2 under offload on 16 GB cards.
- The same kernel check on the 9070 XT and for the 9B.

## Status after the offload milestone (2026-10-23, F7)

For limited-VRAM users: MTP always (1.7-2.5x when offloaded), quantize until it fits, offload
whole layers when nothing fits. CPU-side activation sparsity is capped at +14-26% without MTP and
~0 with it. See `docs/offload.md`.

Open:
- Real smaller GPUs (lower VRAM bandwidth).
- Slower host RAM (DDR4, single channel).
- Prefill-heavy workloads under offload.
- MTP + n-gram for editing under offload.


## Status after MTP tuning (2026-10-23, F6)

MTP defaults are already near-optimal on the R9700 (tuning: <= +2% except BF16 +5%). Recommended
setups: Q6_K + MTP (114 tok/s, KLD 0.0028) or IQ4_XS + MTP (136 tok/s, KLD 0.018), with
`--spec-draft-sampling probabilistic` at temperature > 0. See `docs/mtp_tuning.md`.
Open: long-context and repetitive-editing workloads, the 9070 XT, concurrent requests.


## Status after Phase 3 milestone 1 (2026-10-23): predictive neuron sparsity (F5)

| Gate | Result |
|---|---|
| G1: blocks of >= 8 neurons at <= 50% kept reach IQ4_XS quality (oracle) | **Fail.** B=8: KLD 0.115 |
| G2: a deployable selector reaches IQ4_XS quality at <= 60% of FFN bytes | **Fail (narrow).** TEAL needs ~70% |
| G3: single-token ceiling beats the dense quant ladder | **Paper pass only.** +4-12%, ideal kernel + re-layout + additivity assumed |
| Stretch: beat IQ4_XS + MTP | **Fail.** <= +0.3%: 4-token mask unions cover the whole FFN |

Recommendation: stop FFN-sparsity work on this model (`docs/phase3_findings.md` G). Remaining
options, if any:
1. **Close the project** with the Phase 1-3 negative results and the measured bar.
2. **MTP tuning** (draft length, per-domain acceptance, R9700 runs): the only lever that moved
   decode speed by more than 1.2x.
3. **TEAL in llama.cpp without MTP** (transposed, output-dim-quantized FFN + column-sparse kernel):
   at most +5-12%, multi-week. Measure quant+sparsity KLD additivity first (cheap).


## Status after milestone 1 (2026-10-22)

| Gate | Question | Result |
|---|---|---|
| Kernel (Gate 2) | Can k-of-E expert FFNs approach the bytes-saved speedup at batch 1? | **Yes.** 82-98% of ideal in Triton (8/2: 3.3x per layer). Prefill is slower than dense with current grouped GEMM (implementation-limited) |
| Quality (Gate 1) | With a perfect router, is quality acceptable? | **No.** Oracle 8/6 (1.18x ceiling) costs KL 0.14 / PPL +12%; 8/4 costs KL 0.46 / PPL +47% |

The original hypothesis, training-free dense-to-MoE for faster inference, is not supported for
Qwen3.5-9B. The block-structured approximation is the problem. Neuron-level dynamic sparsity at
the same budget is >10x better in KL.

## Status after milestone 2 (2026-10-23): the quantization bar (F4)

| Option | Quality (KLD vs BF16, val) | Decode 9070 XT |
|---|---|---|
| Q8_0 | 0.0016 | 57.5 tok/s |
| Q8_0 + MTP (lossless) | 0.0016 | 115.8 tok/s |
| IQ4_XS + MTP | 0.0184 | 148.0 tok/s |
| best training-free MoE (oracle 8/6, BF16) | 0.117 | ceiling ~1.18x of its base |
| FFN bytes cut to 37% on top of IQ4_XS (speed proxy) | n/a | 105.8 tok/s (1.18x) |

Option A is done. Sparse FFN is dominated on quality and can add at most ~1.2x on top of
quantization, while MTP gives 1.67-2x losslessly. Anything further has to beat
**IQ4_XS + MTP** or **Q8_0 + MTP** and stack with MTP.

## Options for the next milestone (updated)

Remaining sparse directions, re-ranked by expected value against the bar:
1. **Close the project with a write-up** (negative result plus the measured bar). Lowest cost.
2. **B1, exact down-proj skipping in llama.cpp**: lossless-ish at 75% kept (KLD 0.0013), but saves
   ~8% of bytes, a few percent of decode at best, and it interacts badly with MTP batch-verify.
   Not recommended.
3. **B2/B3, TEAL or a predictor**: could reach the 1.1-1.2x the proxy allows, but needs a GGML
   kernel, and the quality cost has to stay below IQ4_XS's 0.018. Research-grade, multi-week.
4. **C, MoE + distillation**: expected value is low (oracle 8/6 is already 6x IQ4_XS's KLD before
   any router error, and the speed ceiling is ~1.2x on top of quantization).

## Original options (milestone 1)

### A. Establish the quantization bar: DONE, see F4
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
