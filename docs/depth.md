# Decode speed at context depth, and the MMVQ patch on b12164 (2026-11-11)

The project closed on 2026-10-24 (`docs/conclusions.md`). This reopens one question it left out:
its speeds were measured in short contexts, but agent sessions run at 100K+ tokens, where the
27B keeps well under half its empty-context rate. Where does that time go?

Model: Qwen3.8-27B UD-Q6_K, fully on the Radeon AI PRO R9700 (32 GB). llama.cpp b12164
(`4b77dee05`), 731 commits after the b11434 used for everything else in this repo.

Code: `benchmarks/mmvq_b12164.sh`, `benchmarks/depth_profile.py`, `scripts/depth_summary.py` ->
`experiments/results/depth/summary.md`.

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

## What this means

1. **For the 27B at long context, use an f16 KV cache where the VRAM allows** (+3.3 GiB at 128K):
   drop `-ctk q8_0 -ctv q8_0`. The daily presets use q8_0 and split the model over both cards;
   that layout was not measured here.
2. **Two kernel targets, both lossless, both larger than the remaining MMVQ gap:**
   - *Quantized-KV attention on RDNA4.* At its ideal, q8_0 would add 6.6 ms at 120K instead of
     27.7: plain 48 ms per token instead of 69 (+44%), and better than f16 at half the VRAM.
   - *Attention in the verify batch.* If 4 tokens paid for depth once, an MTP step at 120K with
     f16 would drop from 88 to about 72 ms (+22%).
3. **The MMVQ patch is still worth carrying**, and worth 3-4 ms per verify step at any depth.

## Limits

- One model, one card, one build. The prompt is C++ source with two tasks; 256 tokens per row.
- Only MTP at draft length 3. DFlash at depth on this build was not measured (the drafter GGUF is
  no longer on disk).
- The kernel explanation for q8_0 is from reading the code. The "ideal" rows assume the whole
  cache is read once per evaluation at 640 GB/s.
- Stock is `~/llama.cpp/build`: 4b77dee05 with local changes to two server source files that do
  not touch the kernels.
