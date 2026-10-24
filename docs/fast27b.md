# Making Qwen3.8-27B decode faster on one 32 GB card (2026-10-24)

Question: with the whole dense 27B on the GPU, what is the fastest way to decode it at a given
quality? Card: AMD Radeon AI PRO R9700 (32 GB, gfx1201, ~640 GB/s), llama.cpp b11434, ROCm.

Code: `benchmarks/fast27b_drafters.sh` (drafters, stock build), `benchmarks/fast27b_quants.sh`
(quants, batch cost, KLD), `benchmarks/mmvq_batch_bench.sh` (kernel tuning),
`benchmarks/fast27b_mmvq_drafters.sh` (drafters, tuned build), `scripts/fast27b_summary.py` ->
`experiments/results/fast27b/summary.md`. Patch: `patches/llama.cpp-rdna4-mmvq-multicol.patch`.

## Method

- **Speed:** llama-server, 6 domain prompts (code Python/C++, prose, math, instruction, JSON) x 512
  tokens, temperature 0, ctx 8192, `-np 1`. Harmonic mean over the domains, as in F6.
- **Drafters:**
  - The in-file MTP head (`--spec-type draft-mtp`).
  - DFlash2, a block-diffusion drafter trained for this model ([incoai/Qwen3.8-27B-DFlash2]; GGUF
    Q8_0 from [Anbeeld/Qwen3.8-27B-DFlash2-GGUF], 2.06 GB): `--spec-type draft-dflash -md <file>`.
    Stock b11434 supports it.
- **Quants:** Q8_0 and UD-Q6_K (unsloth), plus Q5_K_M, Q4_K_M, Q4_0, IQ4_XS, Q3_K_M requantized
  here from Q8_0 with the calibration imatrix ("-rq"). KLD vs Q8_0 on wikitext-2 test, 100 chunks.
- **Verify cost:** `llama-bench -p 1,2,3,4,5,6,8 -n 0`: the time to process n tokens in one batch,
  which is what a speculative verify step does.
- The card's VRAM was checked before and after every run; no other process used it. Peak use was
  30.9 GB (Q8_0 + DFlash2).

[incoai/Qwen3.8-27B-DFlash2]: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
[Anbeeld/Qwen3.8-27B-DFlash2-GGUF]: https://huggingface.co/Anbeeld/Qwen3.8-27B-DFlash2-GGUF

## Results

### 1. Plain decoding is already near the bandwidth limit

UD-Q6_K decodes at 24.4 tok/s, about 537 GB/s of weight reads (84% of peak); Q8_0 19.5 tok/s.
There is little to gain at batch 1. The gains are in how many tokens each weight read produces.

### 2. Checking drafts was slow for K-quants on RDNA4, and a small kernel change fixes it

Time for one batch of n tokens, stock b11434 (ms):

| quant | 1 | 4 | 6 | 8 | t(6)/t(1) |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 54.2 | 57.8 | 60.7 | 66.6 | 1.12 |
| UD-Q6_K | 43.2 | 52.9 | 66.4 | 79.4 | 1.53 |
| Q4_K_M | 35.1 | 47.4 | 60.9 | 75.2 | 1.73 |
| IQ4_XS | 32.3 | 37.2 | 43.5 | 53.2 | 1.35 |

Checking 6 drafted tokens on UD-Q6_K took longer than on Q8_0, although the file is 25% smaller.
The cause is llama.cpp's MMVQ kernel parameters for RDNA4 (`ggml-cuda/mmvq.cu`). For 2-8 columns
it launches one 32-thread warp per output row (`calc_nwarps` returns 1, `calc_rows_per_block` 1),
where the generic table uses 2-4 warps and 2 rows. Types with cheap dequantization (Q8_0) hide
this; K-quants slow down roughly linearly with the number of columns. Upstream (b11719) still has
the same values.

Sweep over warps (for 2-4 / 5-8 columns) and rows per block, ms at 6 columns:

| warps | rows | UD-Q6_K | Q4_K_M |
|---|---:|---:|---:|
| 1 / 1 (stock) | 1 | 66.4 | 60.9 |
| 4 / 4 | 1 | 68.9 | 68.0 |
| 8 / 4 | 1 | 70.2 | 68.0 |
| 1 / 1 | 2 | 57.2 | 52.5 |
| 4 / 2 | 2 | 55.7 | 52.9 |
| 2 / 2 | 2 | 56.0 | 52.9 |
| 4 / 4 | 2 | 60.3 | 58.2 |
| 1 / 1 | 4 | 54.6 | 51.1 |
| **2 / 2** | **4** | **53.4** | **50.5** |
| 4 / 2 | 4 | 52.8 | 50.0 |
| 2 / 1 | 4 | 54.9 | 50.8 |
| 2 / 2 | 8 | 53.8 | 55.6 |

More rows per block is what helps; more warps alone hurts. 2 warps x 4 rows ties with 4/2 x 4
and is the simpler choice. With it (ms at 6 / 8 columns; 1 column is unchanged):

| quant | stock | tuned | change at 6 |
|---|---|---|---:|
| Q8_0 | 60.7 / 66.6 | 58.1 / 60.7 | -4% |
| UD-Q6_K | 66.4 / 79.4 | 53.4 / 63.5 | -20% |
| Q5_K_M | 65.5 / 79.8 | 52.3 / 63.0 | -20% |
| Q4_K_M | 60.9 / 75.2 | 50.5 / 60.8 | -17% |
| IQ4_XS | 43.5 / 53.2 | 39.2 / 44.7 | -10% |
| Q4_0 | 48.2 / 64.6 | 38.2 / 40.6 | -21% |
| Q3_K_M | 61.7 / 76.7 | 51.5 / 62.8 | -17% |

`test-backend-ops -o MUL_MAT` (1,893 cases) and `-o MUL_MAT_ID` pass on the patched build. The
patch only changes the RDNA4 path; RDNA3, CDNA and NVIDIA are untouched.

### 3. Decode speed per quant, drafter and build

Best configuration per quant (harmonic mean tok/s, 6 domains):

| quant | GiB | KLD vs Q8_0 | plain | stock: best | tuned: best | tuned vs your MTP-3 on stock |
|---|---:|---:|---:|---|---|---:|
| Q8_0 | 27.1 | 0 | 19.5 | DFlash2 n=5: 54.7 | DFlash2 n=7: **61.9** | +37% (vs 45.1) |
| UD-Q6_K | 20.5 | 0.0023 | 24.4 | DFlash2 n=5: 52.6 | DFlash2 n=5: **62.6** | +22% (vs 51.2) |
| Q5_K_M-rq | 18.2 | 0.0066 | 26.8 | | DFlash2 n=5: **64.5** | |
| Q4_K_M-rq | 15.7 | 0.0157 | 30.2 | | DFlash2 n=5: 66.2 | |
| IQ4_XS-rq | 14.3 | 0.0185 | 33.0 | DFlash2 n=5: 71.5 | DFlash2 n=5: **81.9** | +23% (vs 66.6) |
| Q4_0-rq | 14.7 | 0.0297 | 32.0 | | DFlash2 n=5: **85.0** | |

- **DFlash2 beats MTP at every quant**, by 4-28% (each at its best draft length, tuned build). It is longer
  drafts that pay: acceptance per drafted token is lower (58% at n=5 vs 72% for MTP at n=3), but
  more tokens are accepted per verify.
- **The tuned kernel lifts every drafter config,** most where the verify batch is largest and the
  type is a K-quant: UD-Q6_K +7% (MTP n=3) to +23% (DFlash2 n=7). Q8_0 gains 2-5%.
- **Q8_0 is now as fast as UD-Q6_K** under DFlash2 (61.9 at n=7 vs 62.6), because a Q8_0 verify
  batch costs almost the same as one token (t(8)/t(1) = 1.12). Q8_0 at n=7 needs 30.9 GB of VRAM.
- **Q4_0 is the fastest file (85.0)** but has 1.6x IQ4_XS's KLD for the same size. IQ4_XS (81.9) is
  the better 4-bit choice.
- **Drafting gains depend on the text:** code, math and JSON reach 78-114 tok/s; prose and
  instructions 44-68. On prose and instructions DFlash2 is no better than MTP n=3.

Per domain, tuned build (tok/s):

| quant | drafter | Python | C++ | prose | math | instruction | JSON |
|---|---|---:|---:|---:|---:|---:|---:|
| UD-Q6_K | MTP n=3 | 60 | 52 | 49 | 62 | 44 | 66 |
| UD-Q6_K | DFlash2 n=5 | 78 | 58 | 49 | 82 | 46 | 86 |
| Q8_0 | DFlash2 n=7 | 81 | 52 | 44 | 90 | 46 | 101 |
| Q5_K_M-rq | DFlash2 n=5 | 81 | 62 | 52 | 86 | 45 | 88 |
| IQ4_XS-rq | DFlash2 n=5 | 105 | 72 | 68 | 108 | 59 | 109 |
| Q4_0-rq | DFlash2 n=5 | 103 | 86 | 63 | 107 | 65 | 114 |

## Recommendations (27B on a 32 GB RDNA4 card)

1. **Use DFlash2 with `--spec-draft-n-max 5`** instead of MTP. It costs ~2 GB of VRAM.

   ```bash
   llama-server -m Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -c 32768 -np 1 \
     --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 --spec-draft-n-max 5
   ```

2. **Apply the MMVQ patch** (`patches/llama.cpp-rdna4-mmvq-multicol.patch`) on RDNA4. Worth
   +10-23% with drafters on K-quants and IQ4_XS, nothing without drafters.
3. **Pick the quant by quality:** UD-Q6_K (62.6) for near-lossless, Q8_0 at n=7 (61.9) if it fits,
   IQ4_XS (81.9) for speed.

## Caveats

- **Temperature 0, 512 tokens, short prompts.** Acceptance falls at depth and with sampling (F6:
  long context keeps ~1.5x with MTP); DFlash2 at long context and temperature 1 was not measured.
- **The kernel was tuned on one GPU (R9700) and one model shape** (hidden 5120, FFN 17408). Other
  RDNA4 cards (9070 XT: same gfx1201) and other matrix sizes may prefer other values; MoE
  (`MUL_MAT_ID`) speed was not measured.
- **The -rq quants were requantized from Q8_0,** so their KLD is slightly worse than quants made from
  BF16.
- **Batch-1 numbers vary by ~3-7% between runs** (e.g. Q5_K_M 42.2 vs 39.4 ms at 1 column with the
  1-column path unchanged); the stock row for some quants is from a 3-rep run.
