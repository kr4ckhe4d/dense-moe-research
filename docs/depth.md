# Decode speed at context depth, and the MMVQ patch on b12164 (2026-11-11)

The project closed on 2026-10-24 (`docs/conclusions.md`). This reopens one question it left out:
its speeds were measured in short contexts, but agent sessions run at 100K+ tokens, where the
27B keeps well under half its empty-context rate. Where does that time go?

**Prior work (added 2026-11-12):** the fork llama-cpp-rdna-boosts found and fixed the section 10
problem (GQA 6 scanned three times on the tile kernel) on 2026-09-26 and the MMVQ rows-per-block
setting on 2026-09-29, both before this work, which did not know of it. See
`docs/prior_work.md`. The fork was not run here.

Model: Qwen3.8-27B UD-Q6_K, fully on the Radeon AI PRO R9700 (32 GB). llama.cpp b12164
(`4b77dee05`), 731 commits after the b11434 used for everything else in this repo.

Code: `benchmarks/mmvq_b12164.sh`, `benchmarks/depth_profile.py`, `scripts/depth_summary.py` ->
`experiments/results/depth/summary.md`; `scripts/fattn_summary.py` ->
`experiments/results/fattn/summary.md`; `scripts/fattn_wide_summary.py` ->
`experiments/results/fattn/summary_wide.md`. Patches: `patches/llama.cpp-rdna4-mmvq-multicol.patch`,
`patches/llama.cpp-rdna4-fattn-q8-tile.patch`.

## Method

- **Kernel re-check:** `llama-bench -p 1,2,3,4,5,6,8,512 -n 128 -r 5`, stock (`~/llama.cpp/build`)
  against the same commit with `patches/llama.cpp-rdna4-mmvq-multicol.patch`
  (`~/llama.cpp-b12164-mmvq/build`, same cmake flags). The patch applies cleanly.
- **Depth profile:** one llama-server per config, context 131,072. The prompt is 2K / 32K / 64K /
  120K tokens of llama.cpp C++ source followed by a task; two tasks per depth (explain the last
  functions; write a unit test), 256 tokens each, temperature 0, thinking off.
- **Derived numbers:** a *target evaluation* is one pass of the 27B: one token for plain decoding,
  one verify step for MTP. `ms per evaluation = decode time / (generated - accepted drafts)`. This
  is the clean measure; tok/s also moves with how many drafts happen to be accepted in a run.
- **Configs:** plain and MTP (draft length 3), KV cache q8_0 (what the daily presets use) and f16
  (llama.cpp's default), stock and patched builds.
- The card's VRAM was checked before and after every run; no other process used it.

## Results

### 1. The MMVQ patch still helps on b12164

Upstream reworked MMVQ since b11434 (#30517), and stock got faster: a 6-token batch on UD-Q6_K
went from 66.4 ms to 56.7 ms. The RDNA4 multi-column parameters are still 1 warp x 1 row, and the
patch still helps on top. Time per batch of n tokens (ms):

| GPU | model | build | 1 | 4 | 6 | 8 | t(6)/t(1) |
|---|---|---|---:|---:|---:|---:|---:|
| R9700 | 27B UD-Q6_K | stock | 41.4* | 48.3 | 56.7 | 70.8 | 1.37 |
| R9700 | 27B UD-Q6_K | patched | 41.4 | 45.5 | 48.8 | 54.3 | 1.18 |
| R9700 | 27B Q8_0 | stock | 53.5 | 56.4 | 59.0 | 64.7 | 1.10 |
| R9700 | 27B Q8_0 | patched | 53.8 | 54.9 | 56.7 | 58.6 | 1.06 |
| RX 9070 XT | 27B Q3_K_M | stock | 30.0 | 38.8 | 54.4 | 63.1 | 1.82 |
| RX 9070 XT | 27B Q3_K_M | patched | 29.9 | 35.4 | 44.0 | 54.3 | 1.47 |

\* The stock 1-token run was noisy (47.9 ms, 39% stddev: first test after load); the 1-token path
is the same code in both builds, so the patched value is shown.

- 6 tokens: -14% (UD-Q6_K), -19% (Q3_K_M), -4% (Q8_0). 8 tokens: -23%, -14%, -10%.
- 1 token, 512 tokens and tg128 are within 2%.
- `test-backend-ops -o MUL_MAT`: 2516 of 2517 pass; the one failure (f16, n=2.1M) fails
  identically on stock. `-o MUL_MAT_ID` passes.
- **What a shared-dequant kernel could still add is smaller than it looked.** With the patch, a
  6-token batch costs 1.18x one token on UD-Q6_K, against 1.06x on Q8_0. Closing that gap is worth
  about 10% of a verify step on Q6_K, more on Q3_K (1.47x).

### 2. At depth the time goes to attention, not to lost acceptance

Tokens per target evaluation under MTP stay at 2.9-3.4 from 2K to 120K. The cost of each
evaluation is what grows (ms, mean of the two tasks):

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain, q8_0 KV | 41.7 | 49.2 | 56.2 | 69.4 |
| plain, f16 KV | 40.9 | 44.1 | 47.4 | 53.4 |
| MTP, q8_0 KV | 60.1 | 71.6 | 84.0 | 106.8 |
| MTP, f16 KV | 59.4 | 66.9 | 74.4 | 88.2 |
| MTP, f16 KV, patched | 56.4 | 63.5 | 70.8 | 84.3 |

This differs from the two-card table in local-llm-benchmarks, where acceptance fell from 85% to
50-60% at 128K. That test asked different questions of different text; here the tasks are the
same at every depth.

### 3. The q8_0 KV cache is slow to attend over on RDNA4; f16 is at the bandwidth limit

ms added per evaluation relative to 2K. `ideal` is one read of the KV cache at 640 GB/s (16
attention layers x 4 KV heads x 256, K and V: 65.5 kB per token at f16, 34.8 kB at q8_0):

| | 32K | 64K | 120K |
|---|---:|---:|---:|
| plain, q8_0 KV | 7.4 | 14.5 | 27.7 |
| ideal, q8_0 KV | 1.7 | 3.4 | 6.6 |
| plain, f16 KV | 3.2 | 6.5 | 12.5 |
| ideal, f16 KV | 3.2 | 6.4 | 12.3 |
| MTP, q8_0 KV | 11.6 | 24.0 | 46.7 |
| MTP, f16 KV | 7.5 | 15.0 | 28.8 |

- **f16 decodes at the bandwidth limit** (12.5 ms measured, 12.3 ideal at 120K).
- **q8_0 is half the bytes and takes 2.2x as long:** 4.2x its own ideal. The 1-2 token path for
  quantized K/V is the vector attention kernel (`ggml-cuda/fattn-vec.cuh`), which has RDNA-specific
  thread counts for quantized K (`nthreads_KQ_q = 2`). Which kernel ran was read from the
  selection code, not traced.
- **A verify step pays for depth 1.7-2.3x what one token does** (f16: 28.8 vs 12.5 ms at 120K),
  although its 4 tokens read the same cache. Ideal would be close to 1x.

### 4. Switching the KV cache to f16 is worth +22-31% at 120K today

Decode tok/s:

| config | 2K | 32K | 64K | 120K | peak VRAM |
|---|---:|---:|---:|---:|---:|
| plain, q8_0 KV | 23.9 | 20.3 | 17.7 | 14.3 | 24.7 GiB |
| plain, f16 KV | 24.4 | 22.6 | 21.0 | 18.7 | 28.0 GiB |
| MTP, q8_0 KV | 55.5 | 43.7 | 37.3 | 29.1 | 26.3 GiB |
| MTP, f16 KV | 55.8 | 49.4 | 38.8 | 35.5 | 29.6 GiB |
| MTP, f16 KV, patched | 59.1 | 49.9 | 44.3 | 36.9 | 29.6 GiB |

- Plain at 120K: 14.3 -> 18.7 (+31%). MTP: 29.1 -> 35.5 (+22%), 36.9 with the patch (+27%).
- The gain grows with depth: nothing at 2K, about +7% of step time at 32K, +11% at 64K, +17% at
  120K.
- f16 costs 3.3 GiB more at 131,072 context, and is exact where q8_0 KV costs KLD 0.0007.
- MTP runs differ in tokens per evaluation by up to 10% from one run to the next (the greedy
  texts diverge), so compare the ms table above rather than single tok/s cells.

### 5. Prefill slows with depth too, and the KV type does not matter there

Prefill tok/s over each depth range: 820-870 up to 2K, 750-780 up to 32K, 610-625 up to 64K,
480-490 up to 120K, the same for q8_0 and f16 (the batch kernels convert K/V to f16).

### 6. A kernel change that makes q8_0 KV nearly as fast as f16

Why q8_0 was slow: for 1-2 token batches with quantized K/V, llama.cpp uses the vector attention
kernel, which runs one block per query head. This model has 24 query heads over 4 K/V heads, so
each K/V head is scanned 6 times. For 3-8 tokens it uses the tile kernel, but first converts the
whole K/V of the layer to an f16 copy on every call.

`patches/llama.cpp-rdna4-fattn-q8-tile.patch` (on top of b12164) lets the tile kernel
read q8_0 directly: each tile is dequantized to f16 as it is loaded into shared memory, so there
is no conversion pass and query heads that share a K/V head share one scan. It applies to q8_0
K and V with head size 256, batches of 1-8 tokens, on HIP; the routing of 1-2 token batches to it
is limited to RDNA4. Section 9 extends it to q4_0 and to head sizes 64 and 128.

One attention layer of this shape at 122,624 KV rows, R9700 (us per call;
`experiments/results/fattn/summary.md` has 32K / 64K and the RX 9070 XT):

| tokens | f16 | q8_0 stock | q8_0 patched | change |
|---:|---:|---:|---:|---:|
| 1 | 809 | 2073 | 944 | -54% |
| 2 | 1324 | 2077 | 1729 | -17% |
| 4 | 1608 | 3061 | 2064 | -33% |
| 6 | 2467 | 3924 | 2831 | -28% |

- The RX 9070 XT gains the same (1 token -57%, 4 tokens -33%). f16 timings are unchanged.
- `test-backend-ops -o FLASH_ATTN_EXT`: 4072 of 4072 pass on both cards, including new cases for
  this shape (q8_0 and q4_0, with and without sinks, several sequences).
- Patched q8_0 is still 1.1-1.4x f16's time. The rest is dequantization and the tile kernel's own
  batch scaling (next point).

End to end, 27B UD-Q6_K with q8_0 KV (ms per target evaluation; both MTP rows include the MMVQ
patch):

| config | 2K | 32K | 64K | 120K | tok/s at 120K | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| plain, stock | 41.7 | 49.2 | 56.2 | 69.4 | 14.3 | 24.7 GiB |
| plain, attention patch | 41.0 | 44.8 | 48.6 | 56.1 | 17.8 (+24%) | 24.7 GiB |
| plain, f16 KV (for reference) | 40.9 | 44.1 | 47.4 | 53.4 | 18.7 | 28.0 GiB |
| MTP, MMVQ patch | 57.0 | 68.4 | 80.3 | 104.7 | 30.0 | 26.3 GiB |
| MTP, MMVQ + attention patch | 56.8 | 67.1 | 74.7 | 92.3 | 34.1 (+14%) | 26.3 GiB |
| MTP, MMVQ patch, f16 KV (for reference) | 56.4 | 63.5 | 70.8 | 84.3 | 36.9 | 29.6 GiB |

- Against the fully stock build, MTP with q8_0 KV at 120K goes from 29.1 to 34.1 tok/s (+17%).
- q8_0 KV now costs 5% (plain) to 9% (MTP) against f16 at 120K, down from 24-30%, for 3.3 GiB
  less VRAM.
- Greedy output: all 8 MTP generations are identical to the build without the attention patch
  (verify batches used the tile kernel before too; only the conversion moved). Plain decoding:
  4 of 8 identical, 4 diverge after 200-570 characters. q8_0 against f16 KV diverges at the same
  rate (4 of 8).
- Numerical error of one attention call against the CPU reference (`test-backend-ops`, NMSE,
  head size 256, 1-2 tokens; `experiments/results/fattn/nmse_*.txt`): q8_0 stock 2.3e-6, q8_0
  patched 5.5e-6, f16 K/V (either build) 1.3e-5, q4_0 6.4e-6. The pass threshold is 5e-4. The
  patched q8_0 path is less exact than the vector kernel it replaces (which works in integers)
  and more exact than the f16 path every build uses by default.

### 7. Why a verify batch costs more than one token at depth, and what was tried

In the tile kernel, for f16 as well, a 2-token batch costs 1.6x one token and a 4-token batch
2.0x, although all tokens read the same K/V (809 / 1324 / 1608 us at 120K). This is compute, not
memory: the kernel reaches about 7.5 TFLOP/s at 4+ tokens and one token happens to sit at the
bandwidth limit. Three things were tried on the patched build (temporary switches, not in the
patch; `experiments/results/fattn/blocks_sweep.txt`):

| change | 1 token | 2 tokens | 4 tokens | 6 tokens |
|---|---:|---:|---:|---:|
| baseline, f16 / q8_0 (us per layer at 120K, R9700) | 812 / 912 | 1310 / 1703 | 1588 / 2028 | 2442 / 2785 |
| fewer columns per block (2 instead of 4-16), f16 | 813 | 1482 | 2524 | 3831 |
| WMMA (matrix-core) kernel for small batches, f16 | 4265 | 4196 | 4152 | 4081 |
| WMMA with twice the blocks, f16 | 2265 | | 2174 | 2122 |
| tile kernel with twice the parallel blocks, f16 / q8_0 | 995 / 896 | 1002 / 1512 | 1450 / 1847 | 2489 / 2824 |

- Narrower blocks are slower, and the WMMA kernel has a flat cost per call that is too high below
  about 6 tokens.
- The launcher sizes the number of parallel blocks from an occupancy query that looks low for the
  128-thread tile configurations (4 and 8 columns). Doubling the blocks helps 2 tokens by 11-23%
  and 4 tokens by 9%, slows 1-token f16 by 22%, and does nothing at 6 tokens or for prefill.
- End to end with MTP and q8_0 KV, doubled blocks: 92.3 -> 88.8 ms per step at 120K (-4%),
  74.7 -> 73.3 at 64K (-2%) (`experiments/results/fattn/depth_blocks2x.jsonl`). Not added to the
  patch: the gain is small and it would need per-configuration rules measured on more shapes.
- Getting a verify batch down to the cost of one token would need a matrix-core kernel that is
  efficient at 2-8 tokens. With twice the blocks the existing one is still 2.6x the bandwidth
  ideal.

### 8. Check before daily use: both cards, and code quality

The daily presets split the 27B over both cards and use q8_0 KV with MTP. Two checks on that
layout, stock against the patched build (MMVQ + attention patches). Tables:
`experiments/results/adopt/summary.md` (`scripts/adopt_summary.py`).

**Speed on both cards** (`depth_profile.py --all-gpus`, same prompts as above):

| config | 2K | 32K | 64K | 120K | peak VRAM, both cards |
|---|---:|---:|---:|---:|---:|
| plain, q8_0 KV, stock | 23.5 | 19.9 | 17.1 | 13.8 | 27.2 GiB |
| plain, q8_0 KV, patched | 23.7 | 21.7 | 20.1 | 17.1 | 27.2 GiB |
| MTP, q8_0 KV, stock | 55.0 | 43.3 | 36.9 | 28.5 | 29.4 GiB |
| MTP, q8_0 KV, patched | 57.7 | 45.6 | 41.5 | 32.7 | 29.4 GiB |
| MTP, f16 KV, stock | 55.1 | 48.9 | 38.5 | 34.9 | 32.2 GiB |
| MTP, f16 KV, patched | 57.7 | 48.8 | 43.3 | 35.6 | 32.2 GiB |

- The two-card numbers match the one-card ones (MTP, q8_0, 120K: 28.5 here, 29.1 on the R9700
  alone), and so do the gains: plain +24% at 120K, MTP +15% at 120K and +5% at 2K.
- Per verify step, which removes the run-to-run change in accepted tokens: 109.1 -> 96.2 ms at
  120K (-12%), 60.6 -> 58.1 at 2K (-4%).
- f16 KV on stock is +22% over q8_0 at 120K for 2.8 GiB more. With the patch that gap is 9%.

**Code quality** (`benchmarks/adopt_check.sh`, `adopt_check_seeds.sh`): the 7-task, 50-check suite
of local-llm-benchmarks on the preset `qwen3.8-27B-q6-128k-nothink`.

| run | build | scores | mean | sd |
|---|---|---|---:|---:|
| temperature 0, 3 repeats | stock | 44 44 44 | 44.0 | 0 |
| temperature 0, 3 repeats | patched | 41 41 41 | 41.0 | 0 |
| temperature 1.0 (preset sampler), seeds 1-12 | stock | 41 43 42 35 39 40 31 39 36 32 29 41 | 37.3 | 4.7 |
| temperature 1.0 (preset sampler), seeds 1-12 | patched | 36 42 34 41 35 35 41 39 35 40 35 43 | 38.0 | 3.3 |

- At temperature 0 each build gives one fixed answer per task, so 44 against 41 is one sample
  each. 3 of the 7 answers are byte-identical; the 3 points come from one `textwrap` answer that
  timed out (0/6 against 4/6) and one more `parse_qsl` check passed.
- Over 12 seeds at the preset's sampler the means are 37.3 and 38.0 with a spread of 3-5 points
  between seeds. No per-task mean differs by more than 0.5 checks, and answers that failed to
  run occur 3 times in 12 on both builds. No quality difference is measurable.
- The suite's spread between seeds is large: a difference under about 4 points between two
  single runs of it means nothing.

### 9. The attention patch extended to q4_0 KV and head sizes 64 and 128

The same patch file now also reads q4_0 K/V directly and covers head sizes 64, 128 and 256. For
these shapes all batches of 1-8 tokens go to the tile kernel on RDNA4. Before, 1-2 tokens used
the vector kernel, and larger ones the tile or the matrix-core kernel after converting all of
K/V to f16. Tried both routings for the batches the matrix-core kernel used to take: the tile
kernel without conversion was faster in every quantized case measured.

One attention layer at 122,624 KV rows, R9700, us per call, stock -> patched
(`experiments/results/fattn/summary.md` has 32K and the RX 9070 XT, which is the same within 6%):

| head size (K/V heads x Q per K/V) | KV type | 1 token | 2 tokens | 4 tokens | 8 tokens |
|---|---|---|---|---|---|
| 256 (4 x 6) | q4_0 | 4355 -> 964 | 9162 -> 1785 | 2982 -> 2088 | 3849 -> 2836 |
| 128 (8 x 4) | q8_0 | 915 -> 453 | 1679 -> 591 | 3048 -> 1322 | 3095 -> 2659 |
| 128 (8 x 4) | q4_0 | 1516 -> 441 | 1869 -> 600 | 3027 -> 1539 | 3081 -> 2644 |
| 64 (8 x 2) | q8_0 | 266 -> 231 | 351 -> 248 | 2621 -> 362 | 2473 -> 794 |
| 64 (8 x 2) | q4_0 | 252 -> 192 | 329 -> 235 | 2581 -> 437 | 2439 -> 781 |

- q4_0 at head size 256 was the worst stock case (2 tokens: 9.2 ms per layer); it now costs the
  same as q8_0.
- At head sizes 64 and 128 the stock 4-8 token cost was dominated by the f16 conversion
  (~2.5-3 ms per call at this depth). Without it a 4-token batch is 2.3-7x faster.
- At 8 tokens and head size 128 the gain is small (3095 -> 2659): the tile kernel is
  compute-bound there (section 7), and f16 K/V on the matrix-core kernel takes 907 us.
- For 1-2 tokens at head sizes 64 and 128 the patched quantized path is faster than stock f16
  K/V (128: 453 against 808 us; 64: 231 against 404). The f16 path was not changed; routing f16
  decode at these head sizes to the tile kernel is an untested follow-up.
- `test-backend-ops -o FLASH_ATTN_EXT`: 4232/4232 on both cards (160 new cases). Error against
  the CPU reference for the new shapes is at most 1.4e-5 (threshold 5e-4;
  `experiments/results/fattn/nmse_ext_*.txt`).
- f16 cases, patched / stock: 0.97-1.03 on the RX 9070 XT. On the R9700 0.92-1.21 over two
  passes, with every case outside 0.95-1.05 either a 4096-token batch or a 20-60 us call; the
  code path for them is the same in both builds.

**End to end, q4_0 KV on the 27B** (UD-Q6_K, R9700, same protocol as section 2;
`experiments/results/fattn/depth_q4kv.jsonl`). Decode tok/s:

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain, q4_0 KV, stock | 23.2 | 17.0 | 13.1 | 9.1 |
| plain, q4_0 KV, patched | 24.2 | 22.1 | 20.4 | 17.7 |
| MTP, q4_0 KV, stock | 55.2 | 44.9 | 38.2 | 30.0 |
| MTP, q4_0 KV, patched | 58.5 | 48.6 | 42.3 | 33.3 |

- Plain decoding at 120K: +94%. With MTP: +11% (104.0 -> 92.4 ms per verify step); verify
  batches of 4 already used the tile kernel, so only the conversion is saved. The patched build
  also has the MMVQ patch, worth 3-4 ms of each MTP step.
- q4_0 KV now runs at the speed of q8_0 KV (plain 17.7 against 17.8 at 120K) and uses 2.0 GiB
  less than q8_0 at 131K context (24.3 against 26.3 GiB peak with MTP). Its quality cost was
  not measured here.
- No model on disk has head size 64 or 128 (Qwen3.5-9B and Qwen3.8-27B use 256, Gemma 4 12B
  512 / 256), so those shapes have kernel timings only.

### 10. Verify batches at the cost of one token: the matrix-core kernel with one scan per K/V head

Section 7 ended with "needs a new kernel". It did not: the existing matrix-core (WMMA) kernel was
slow on small batches for two reasons that are both in how it is launched.

- **Three scans per K/V head.** This model has 6 Q heads per K/V head. On RDNA the launcher groups
  Q heads only in exact divisors (8, 4 or 2), so it used groups of 2 and read each K/V head three
  times. Grouping all 6 in one 8-wide tile (2 columns unused) reads it once.
- **Too few blocks.** The block count comes from an occupancy query that is low on RDNA4. Twice
  as many is the optimum for these kernels; 1.5x, 3x and 4x were all slower or equal
  (f16, 4 tokens at 120K: 1272 / 854 / 987 / 895 us at 1x / 2x / 3x / 4x).
- **q8_0 and q4_0 read directly.** The kernel's loader now dequantizes K/V as it fills a tile, as
  the tile kernel does since section 6, so the f16 copy of the whole cache per call is gone.

The rule (RDNA4, head size 256, more than 4 Q heads per K/V head, 1-32 tokens) is in the same
patch file. f16 K/V keeps the tile kernel for 1 token, and 1-2 token batches under 8,192 KV rows
keep the old paths, because those were faster there.

One attention layer of the 27B shape, R9700, us per call, before -> after (before = the patch as
of section 9; `experiments/results/fattn/summary_wide.md` has all depths and the RX 9070 XT):

| KV type | KV rows | 1 token | 2 tokens | 4 tokens | 8 tokens | 16 tokens | 32 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| q8_0 | 4,096 | 36 -> 37 | 55 -> 48 | 68 -> 33 | 97 -> 51 | 159 -> 64 | 202 -> 107 |
| q8_0 | 32,768 | 249 -> 153 | 418 -> 154 | 524 -> 138 | 750 -> 219 | 1420 -> 388 | 1896 -> 743 |
| q8_0 | 122,624 | 945 -> 561 | 1711 -> 567 | 2055 -> 546 | 2839 -> 746 | 5093 -> 1451 | 6352 -> 2769 |
| f16 | 4,096 | 27 -> 27 | 41 -> 42 | 53 -> 41 | 85 -> 55 | 117 -> 69 | 159 -> 113 |
| f16 | 32,768 | 225 -> 225 | 324 -> 267 | 412 -> 256 | 666 -> 304 | 995 -> 461 | 1443 -> 870 |
| f16 | 122,624 | 811 -> 809 | 1315 -> 854 | 1606 -> 851 | 2486 -> 933 | 3633 -> 1686 | 4916 -> 3209 |
| q4_0 | 122,624 | 968 -> 366 | 1779 -> 369 | 2081 -> 476 | 2839 -> 714 | | |

- **A 4-token verify batch now costs the same as one token**, and for q8_0 less than one token
  cost before (546 against 945 us at 120K).
- **q8_0 is now faster than f16** (546 against 851 us): it is half the bytes and the kernel is
  bound by memory bandwidth. One read of q8_0 K/V for this layer at 640 GB/s would be 417 us.
- No measured shape is slower: 6 depths x 7 batch sizes x f16 / q8_0 on both cards. The RX 9070 XT
  is within 7% of the R9700.
- `test-backend-ops -o FLASH_ATTN_EXT`: 4295 of 4295 on both cards, with 63 new cases at 8,192
  rows (1-33 tokens, f16 / q8_0 / q4_0, with sinks, two sequences).

**End to end, 27B UD-Q6_K** (same protocol as section 2; `off` is the same build with the rule
disabled and reproduces the earlier patched numbers;
`experiments/results/fattn/depth_wide.jsonl`, `depth_wide_2gpu.jsonl`). ms per target evaluation:

| config | 2K | 32K | 64K | 120K | tok/s at 120K |
|---|---:|---:|---:|---:|---:|
| R9700: plain, q8_0 KV, before | 41.1 | 44.9 | 48.7 | 56.2 | 17.7 |
| R9700: plain, q8_0 KV, after | 41.1 | 43.8 | 45.8 | 49.7 | 20.0 (+13%) |
| R9700: MTP, q8_0 KV, before | 56.9 | 66.0 | 75.0 | 92.6 | 34.0 |
| R9700: MTP, q8_0 KV, after | 56.9 | 59.9 | 62.8 | 68.5 | 44.8 (+32%) |
| R9700: MTP, f16 KV, before | 56.5 | 63.7 | 70.8 | 84.4 | 36.8 |
| R9700: MTP, f16 KV, after | 56.7 | 60.9 | 65.0 | 72.8 | 42.9 (+17%) |
| R9700: MTP, q4_0 KV, before | 56.9 | 66.1 | 75.2 | 92.4 | 33.3 |
| R9700: MTP, q4_0 KV, after | 56.8 | 59.5 | 62.2 | 67.3 | 47.4 |
| both cards: plain, q8_0 KV, before | 42.1 | 45.8 | 49.6 | 58.4 | 17.1 |
| both cards: plain, q8_0 KV, after | 42.0 | 44.7 | 46.8 | 50.7 | 19.6 (+15%) |
| both cards: MTP, q8_0 KV, before | 58.1 | 67.3 | 77.7 | 96.0 | 32.8 |
| both cards: MTP, q8_0 KV, after | 58.0 | 61.2 | 64.0 | 69.8 | 44.0 (+34%) |

- **MTP with q8_0 KV at 120K: 34.0 -> 44.8 tok/s on the R9700, 32.8 -> 44.0 on both cards.**
  Against the fully stock build (29.1 and 28.5) that is +54% on either layout.
- Going from 2K to 120K now adds 11.6 ms to an MTP step with q8_0 KV. It added 35.7 ms before
  this section and 46.7 ms on stock; one read of the cache at 640 GB/s is 6.6 ms.
- At 120K the 27B with MTP keeps 77% of its 2K speed (44.8 of 58.3 tok/s). Stock kept 52%.
- q8_0 KV is now the faster choice as well as the smaller one: 68.5 against 72.8 ms for f16.
  The q4_0 tok/s gain is not given as a percentage because the two runs accepted different
  numbers of drafts (3.09 and 3.20 tokens per step); its step time fell 27%.
- The two-card 2K cell is from a repeat (`depth_wide_2gpu_recheck.jsonl`, 58.0 and 58.1 ms in two
  runs each way). The main run had one slow row there (64.8 ms on one of the two tasks, mean 61.5).
- Greedy output: 2-4 of 8 generations per config are identical to the run with the rule off; the
  rest diverge, as they do between q8_0 and f16 KV (section 6). The kernels round differently.
- Peak VRAM is unchanged (within 45 MiB).
- Code quality on this build, same suite and preset as section 8 (50 checks, both cards, MTP,
  q8_0 KV): 12 seeds 36.2 +- 3.5 (43 30 34 34 34 35 41 39 37 35 36 37), against 38.0 +- 3.3 for
  the patch before this section and 37.3 +- 4.7 for stock. Temperature 0: 42 (stock 44, before
  41). The 1.8-point gap to the earlier patch is about 1.3 standard errors, so no difference is
  shown, and none is ruled out below about 3 points. Answers that did not run: 5 of 84 (3 and 3
  before). `experiments/results/adopt/seeds/wide_*.txt`.

## What this means

1. **Without rebuilding llama.cpp: use an f16 KV cache for the 27B at long context** where the
   VRAM allows (+3.3 GiB at 128K): drop `-ctk q8_0 -ctv q8_0`. On the two-card daily layout this
   is +22% at 120K (section 8).
2. **With the attention patch, q8_0 KV is within 5-9% of f16** and keeps its VRAM saving: +24%
   plain and +14-17% with MTP at 120K, nothing lost in short contexts.
3. **The MMVQ patch is still worth carrying**, and worth 3-4 ms per verify step at any depth.
4. **The patched build is safe to use for the daily presets** (section 8): the same gains on
   both cards (+15% with MTP at 120K, +5% at 2K) and no measurable change in code quality.
5. **The extra cost of verify batches at depth is gone** (section 10): the matrix-core kernel
   with one scan per K/V head runs a 4-token batch for the price of one token. With MTP and q8_0
   KV at 120K the 27B does 44.8 tok/s (stock 29.1), and q8_0 KV is now faster than f16, so
   point 1 only applies to an unpatched build.

## Limits

- One model, one build; one card except in section 8. The prompt is C++ source with two tasks; 256 tokens per row.
- Only MTP at draft length 3. DFlash at depth on this build was not measured (the drafter GGUF is
  no longer on disk).
- The "ideal" rows assume the whole cache is read once per evaluation at 640 GB/s.
- The section 10 rule covers head size 256 with more than 4 Q heads per K/V head, which is the
  27B (Qwen3.5-9B has 16 Q heads over 4 K/V heads and is not covered). Whether head sizes 64 and
  128 or 4 Q heads per K/V head gain from the same block count and direct quantized loading was
  not tested. Batches over 32 tokens keep the old path; the best cut-off was not searched.
- The attention patch covers q8_0 and q4_0 K and V (both the same type) at head sizes 64, 128
  and 256; other types and sizes keep the old paths. Head sizes 64 and 128 have kernel timings
  and correctness tests only, no end-to-end run. Measured on RDNA4 only. No llama-perplexity
  check: that tool decodes in large batches, which do not use the changed path.
- Stock is `~/llama.cpp/build`: 4b77dee05 with local changes to two server source files that do
  not touch the kernels.
