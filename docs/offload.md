# Running dense models on smaller GPUs: offload, quantization, MTP (2026-10-23)

Question: on a GPU too small for the model, what is the fastest way to run a good dense model,
and could CPU-side activation sparsity help? Models: Qwen3.5-9B and Qwen3.8-27B (same `qwen35`
hybrid architecture). Smaller cards are simulated on the RX 9070 XT (16 GB) by limiting what
llama.cpp places in VRAM. Host: Ryzen 7 9800X3D, 2 x 16 GB DDR5-6000 (53 GB/s STREAM Triad,
limited by the single-CCD link), PCIe 5.0 x16, llama.cpp b11434, 8 threads.

Code: `benchmarks/offload_bench.py` (budget search + llama-bench), `benchmarks/offload_mtp_fit.py`
(MTP at the same budget), `benchmarks/build_27b_quants.sh` (27B quants + KLD),
`scripts/offload_summary.py` -> `experiments/results/offload/summary.md`.

## Method

- **Budget rule:** llama.cpp's GPU buffers (weights + KV cache + recurrent state + compute, from a
  1-token debug run) must fit in budget - 0.5 GiB at 8k context. For each budget, a binary search
  finds the most GPU-resident placement under two strategies:
  - `layers`: `-ngl k` puts whole layers in RAM.
  - `ffn`: `-ot` keeps everything on the GPU except some layers' FFN weights.
- **Speed:** llama-bench pp512 / tg128 (2 reps). MTP runs use llama-server with 3 prompts (code,
  prose, JSON) x 256 tokens at temperature 0.
  - MTP's draft context needs ~0.4 GiB (9B) / ~0.8 GiB (27B) extra VRAM, so its -ngl was lowered
    until it fits the same budget.
  - The card's VRAM was checked before and after every run; no other process used it.
- **Quality:**
  - 9B quants: KLD vs llama.cpp BF16 on our val text (F4).
  - 27B quants: KLD vs Q8_0 on wikitext-2 test (100 chunks), against the existing Q8_0 base
    file. The 27B IQ4_XS and Q3_K_M were requantized here from Q8_0 with an imatrix from our
    calibration text ("-rq").
  - KLDs are not comparable across the two models (different references and text).

| 27B quant | Size | KLD vs Q8_0 | PPL vs Q8_0 | same top-1 |
|---|---:|---:|---:|---:|
| Q8_0 | 27.0 GiB | 0 | | |
| UD-Q6_K | 20.5 GiB | 0.0023 | +0.02% | 97.9% |
| IQ4_XS-rq | 14.3 GiB | 0.0185 | +1.1% | 94.0% |
| Q3_K_M-rq | 12.6 GiB | 0.054 | +4.3% | 90.0% |

## Results

### 1. A GiB in RAM costs ~10x a GiB in VRAM

Fitting decode time against weight bytes left in RAM, per quant, over all placements:

| | ms per token per GiB in RAM | implied bandwidth |
|---|---:|---:|
| 9B Q8_0 / Q6_K / IQ4_XS | 17.3 / 18.5 / 20.5 | 62 / 58 / 52 GB/s |
| 27B Q8_0 / UD-Q6_K / IQ4_XS / Q3_K_M | 15.4 / 17.1 / 18.2 / 18.4 | 70 / 63 / 59 / 58 GB/s |

A GiB read from VRAM costs ~1.6 ms (F6/Phase 3 fit). So small offloads hurt a lot: 9B Q6_K with
1.5 GiB in RAM decodes at 24.5 tok/s instead of 70.

**Whole layers vs FFN only:** moving whole layers to RAM is equal or 3-12% faster than moving only
FFN weights at the same budget, although the FFN strategy keeps attention and the KV cache on the
GPU. Prefill also suffers from offloading: the 27B at 8 GB processes prompts at 420-820 tok/s,
vs ~1,400 fully on the GPU.

### 2. Best setup per VRAM budget (decode tok/s, MTP on, same budget)

**Qwen3.5-9B** (KLD vs BF16):

| VRAM | Highest quality that runs | Fastest |
|---|---|---|
| 4 GB | Q6_K, 12 of 34 layers on GPU: **31.6** (0.0028) | IQ4_XS, 18 layers: **41.5** (0.018) |
| 6 GB | Q6_K, 23 layers: **48.2** (0.0028) | **IQ4_XS fully on GPU: 136.2** (0.018) |
| 8 GB | Q8_0, 26 layers: **46.9** (0.0016) | **Q6_K fully on GPU: 111.1** (0.0028) |

**Qwen3.8-27B** (KLD vs Q8_0):

| VRAM | UD-Q6_K (0.0023) | IQ4_XS-rq (0.0185) | Q3_K_M-rq (0.054) |
|---|---:|---:|---:|
| 8 GB | 9.4 | 12.5 | 14.7 |
| 12 GB | 12.7 | 21.6 | 27.9 |
| 16 GB | 18.2 | **69.0** (fits) | (fits; MTP not run, 33.3 without) |

Plain-decoding numbers for every placement, prefill, and the non-MTP rows are in `summary.md`.

### 3. MTP is worth more when part of the model is in RAM

| | MTP speedup at the same budget |
|---|---:|
| fully on GPU (9B IQ4_XS 6 GB, Q6_K 8 GB) | 1.54-1.60x |
| 9B partly in RAM | 1.70-2.29x |
| 27B IQ4_XS / Q3_K_M partly in RAM | 1.65-1.90x |
| 27B UD-Q6_K, most in RAM | 2.40-2.46x |
| 27B IQ4_XS fully on GPU (16 GB) | 2.13x |

Verifying ~3 drafted tokens reads the RAM-resident weights once, the same as one token. The
slower the per-token weight read, the more MTP's extra tokens are worth, even after giving up 2-3
GPU layers for the draft context. Acceptance is 66-76%, higher on the 27B.

### 4. CPU-side activation sparsity: small without MTP, nothing with it

Ceiling for TEAL-style skipping of the RAM-resident FFN rows (FFN strategy, ideal kernel, free
mask, from the per-quant fits). The added-KLD estimate scales the 9B's all-layer TEAL KLD by the
share of sparsified layers (assumed additive; untested on the 27B):

| | 90% of rows read | 75% of rows read | est. KLD added (90% / 75%) |
|---|---:|---:|---|
| 9B, 4-8 GB budgets | +6-9% | +16-26% | 0.0003-0.0010 / 0.0026-0.0087 |
| 27B, 8-16 GB budgets | +5-9% | +14-26% | 0.0002-0.0010 / 0.0019-0.0083 |

- **With MTP:** the verify step reads the union of ~4 tokens' masks, which covers 99-100% of the
  FFN at these budgets (Phase 3, F5), so the gain disappears.
- **Without MTP:** +14-26% at most, against MTP's 1.7-2.5x.
- **Hot neurons:** this model has none at useful budgets (F5: at 50% kept, ~1% of neurons are used
  by >90% of tokens). PowerInfer-style GPU/CPU neuron placement has nothing to exploit.

Not worth building for these models.

## Recommendations (for people with limited VRAM)

1. **Always turn on MTP** (`--spec-type draft-mtp`). It costs ~0.4-0.8 GiB of VRAM and gives
   1.7-2.5x when part of the model is in RAM. Lower `-ngl` by 2-3 layers to make room.
2. **Quantize until it fits, if a quant with acceptable quality fits.** On 6 GB, 9B IQ4_XS on the GPU
   (136 tok/s) beats Q6_K half in RAM (48) by 2.8x. On 16 GB, 27B IQ4_XS (69 tok/s) beats UD-Q6_K
   (18) by 3.8x.
3. **When nothing fits, pick the quality level and accept the RAM cost.** Each GiB in RAM costs
   ~17 ms per token on DDR5-6000 (more on slower RAM). Offload whole layers (`-ngl`), not just
   FFNs.
4. **Bigger model vs smaller model** at the same VRAM is a quality question:
   - At 8 GB, the 9B Q8_0 (47 tok/s) runs 3-5x faster than any 27B (9-15 tok/s).
   - At 16 GB, the 27B IQ4_XS (69 tok/s) is close to the 9B Q6_K's speed on 8 GB (111).
5. **Leave threads at llama.cpp's default (8 on this CPU).** RAM bandwidth saturates at ~4 threads,
   and 16 threads collapses (local-llm-benchmarks/ddr5-9800x3d.md).

Example, 27B on a 12 GB card, fastest acceptable quality:

```bash
llama-server -m Qwen3.8-27B-IQ4_XS.gguf -ngl 47 -fa on -c 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

## Caveats

- **Simulated budgets on one card.**
  - A real 8 GB card has less memory bandwidth than the 9070 XT, so its GPU-side time is longer.
  - The RAM-side cost dominates offloaded setups and was measured for real.
  - On DDR4 or single-channel RAM, offloaded speeds fall roughly in proportion to bandwidth.
- **The 27B IQ4_XS / Q3_K_M were requantized from Q8_0.** Official quants from BF16 may be
  slightly better.
- **MTP was measured on 3 short prompts at temperature 0;** the per-domain spread is large
  (docs/mtp_tuning.md).
- **The TEAL quality numbers come from the 9B,** applied to the 27B by assumption.
