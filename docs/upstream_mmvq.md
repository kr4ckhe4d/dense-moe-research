# Evidence pack: RDNA4 MMVQ parameters for 2-8 column batches (llama.cpp)

Material for a possible upstream llama.cpp PR of `patches/llama.cpp-rdna4-mmvq-multicol.patch`.

> **This is not a PR description.** llama.cpp's `CONTRIBUTING.md` (AI usage policy) prohibits
> using AI to write PR descriptions or other posts. It also requires the contributor to disclose
> AI use, review the code by hand, and be able to explain every line. The patch, the measurements
> and this file were produced with an AI assistant. Whoever submits the PR has to write the
> description themselves, state that in the AI disclosure, and own the change.

All numbers: `experiments/results/mmvq_upstream/summary.md` (upstream master),
`experiments/results/mmvq_check/summary.md` and `experiments/results/fast27b/summary.md` (b11434).
Scripts: `benchmarks/mmvq_upstream.sh`, `benchmarks/mmvq_types.sh`, `benchmarks/mmvq_check.sh`.

## The change

On master b11719 (5aeac37a3), file `ggml/src/ggml-cuda/mmvq.cu`, 7 lines added, 1 removed:

```diff
@@ calc_nwarps(), RDNA4 branch, after the ncols_dst == 1 switch
-        return 1;
+        // 2-8 columns (e.g. speculative decoding verify): 2 warps x 4 rows per block (see
+        // calc_rows_per_block). Measured on R9700 against 1 warp x 1 row: Q6_K -20%, Q4_K -17%,
+        // Q4_0 -21%, IQ4_XS -10%, Q8_0 -4% time at 6 columns; 1 column unchanged.
+        return 2;
@@ calc_rows_per_block(), before the final return
+    if (table_id == MMVQ_PARAMETERS_RDNA4 && ncols_dst > 1) {
+        return 4;
+    }
     return 1;
```

The code comment cites b11434 numbers. On master the 6-column gains are somewhat smaller (see
below), so the comment should be updated before submitting.

### What each line does

- **`calc_nwarps`, RDNA4 table.** For `ncols_dst == 1`, the type switch above it is unchanged
  (8 warps for simple types, 1 for the others). For `ncols_dst` 2-8 it returned 1 warp for every
  type; it now returns 2. The two warps split the loop over the K dimension (`blocks_per_iter` =
  `vdr * nwarps * warp_size / qi`), and their partial sums are combined in shared memory at the
  end of the kernel, the same as on the other tables.
- **`calc_rows_per_block`.** RDNA4 was not in the first `if`, so it fell through to `return 1`
  for every column count. It now returns 4 rows per block for `ncols_dst > 1`.
  - The kernel computes `rows_per_cuda_block` rows per block. Inside the K loop (`for j` columns,
    `for i` rows, both unrolled), the same `q8_1` activation chunk `&y[j*stride_col_y + kby]` is
    used for all of the block's rows, so one fetch (kept in registers or served from L1) can feed
    4 rows instead of 1.
  - With 1 row, every activation load serves one row, and each block issues few weight loads at a
    time. With 4 rows, each activation load serves 4 rows, and 4 rows' weight loads are in flight
    at once.
  - This is the same reasoning as the existing `mmvq_two_rows_ncols1` comment ("a second row per
    block shares the q8_1 loads and adds memory parallelism"). The generic table uses 2 rows for
    2-8 columns.
- **`ncols_dst > 1` means 2-8.** The MMVQ path is only taken for up to `MMVQ_MAX_BATCH_SIZE` (8)
  columns. The `MUL_MAT_ID` limits (`get_mmvq_mmid_max_batch_rdna4`) are also <= 8.
- **Grid size.** `calc_launch_params` uses `nblocks = ceil(nrows / rows_per_block)`, so for 2-8
  columns there are 4x fewer blocks, each with 64 threads instead of 32.
- **Row counts not divisible by 4.**
  - Writes past the last row are skipped by the existing check (`row0 + i < stride_col_dst`).
  - Weight reads are not guarded: the last block can read up to 3 rows past the end. The generic
    table's 2 rows per block can read 1, so this is the same pattern as existing code, not a new
    one, but a reviewer may ask.
  - All matrices in the tested models have row counts divisible by 4. test-backend-ops includes
    small and odd shapes and passes.
- **Other tables are untouched**: NVIDIA, GCN, CDNA, RDNA1-3, GB10.

### Why it matters

Speculative decoding (MTP, DFlash, draft models, n-gram) verifies 2-8 tokens per step. On RDNA4
with 1 warp x 1 row, the verify batch cost grew almost linearly with the number of tokens for
K-quants. At b11434 on an R9700 with Qwen3.8-27B, verifying 6 tokens cost 1.53x one token for
Q6_K and 1.73x for Q4_K_M, but only 1.12x for Q8_0. This cancels much of what speculative
decoding should gain.

### How the values were chosen

Sweep on b11434, R9700, Qwen3.8-27B UD-Q6_K and Q4_K_M, ms per 6-column batch:

| warps (2-4 / 5-8 cols) | rows | UD-Q6_K | Q4_K_M |
|---|---:|---:|---:|
| 1 / 1 (stock) | 1 | 66.4 | 60.9 |
| 4 / 4 | 1 | 68.9 | 68.0 |
| 1 / 1 | 2 | 57.2 | 52.5 |
| 2 / 2 | 2 | 56.0 | 52.9 |
| 1 / 1 | 4 | 54.6 | 51.1 |
| **2 / 2** | **4** | **53.4** | **50.5** |
| 4 / 2 | 4 | 52.8 | 50.0 |
| 2 / 2 | 8 | 53.8 | 55.6 |

More rows per block is what helps; more warps alone hurts. 2 x 4 ties with 4/2 x 4 and is simpler
(one value for all of 2-8). Full sweep: `docs/fast27b.md` section 2.

## Results on master b11719

`llama-bench -ngl 99 -fa 1 -r 5`, model fully on the GPU, same commit with and without the patch.
Speedup = patched / stock t/s.

### Dense and MoE models

| GPU | model | pp1 | pp4 | pp6 | pp8 | pp512 | tg128 |
|---|---|---:|---:|---:|---:|---:|---:|
| R9700 | Qwen3.8-27B UD-Q6_K | 1.00 | 1.05 | 1.16 | 1.29 | 0.99 | 1.00 |
| R9700 | Qwen3.8-27B Q8_0 | 1.01 | 1.03 | 1.04 | 1.11 | 1.01 | 1.00 |
| R9700 | Gemma 4 31B Q6_K | 1.00 | 1.06 | 1.19 | 1.38 | 0.99 | 1.00 |
| R9700 | Qwen3.5-9B Q4_K_M | 1.01 | 1.09 | 1.17 | 1.19 | 0.99 | 1.00 |
| R9700 | Qwen3.6-35B-A3B UD-Q6_K (MoE) | 0.99 | 1.01 | 0.98 | 1.02 | 1.02 | 1.00 |
| RX 9070 XT | Qwen3.8-27B IQ4_XS | 1.02 | 1.04 | 1.12 | 1.19 | 1.05 | 1.02 |
| RX 9070 XT | Qwen3.5-9B Q6_K | 1.02 | 1.04 | 1.08 | 1.21 | 0.99 | 1.00 |
| RX 9070 XT | gpt-oss-20b MXFP4 (MoE) | 1.04 | 1.01 | 1.00 | 1.00 | 1.00 | 1.00 |

- pp512 and tg128 are within run-to-run noise (+-2%; the one +5%, 27B IQ4_XS pp512, uses MMQ,
  which the patch does not touch).
- pp1 has high variance in llama-bench (+-10-30% stddev on small models). The 1-column code path
  is unchanged.
- MoE expert matmuls (`MUL_MAT_ID`) mostly see one token per expert, so they are neutral.
- A first stock run of the 9B Q4_K_M showed pp512 at 3,014 t/s (an outlier, likely clocks still
  ramping up at the start of the session). Two reruns gave 3,915-3,928, the same as patched. That
  run was replaced and is kept in `experiments/results/mmvq_upstream/discarded/`.

### Per quant type

Qwen3.5-9B quantized with `llama-quantize --pure`, so every tensor has the tested type. The MTP
layer (blk.32) has no imatrix data, so it was kept at Q8_0; for IQ2_* the token embedding and
output were Q8_0 as well. R9700, ratio = patched / stock time (< 1 is faster):

| type | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| IQ1_M | 1.00 | 0.90 | 0.84 | 0.77 | 0.84 |
| IQ2_XXS | 1.00 | 0.93 | 0.88 | 0.87 | 0.85 |
| IQ2_S | 1.00 | 0.94 | 0.87 | 0.90 | 0.86 |
| IQ3_XXS | 1.00 | 0.95 | 0.85 | 0.90 | 0.88 |
| IQ3_S | 0.97 | 0.95 | 0.85 | 0.90 | 0.80 |
| Q2_K | 0.99 | 0.93 | 0.91 | 0.85 | 1.03 |
| Q3_K | 1.00 | 1.00 | 0.94 | 0.79 | 0.88 |
| Q5_K | 1.01 | 0.97 | 0.94 | 0.85 | 0.85 |
| IQ4_NL | 1.01 | 0.96 | 0.97 | 0.87 | 0.76 |

The existing RDNA4 comment says complex vec_dot types (Q3_K, IQ2_*, IQ3_*) regress at higher
thread counts for 1 column. At 2-8 columns with 2 warps x 4 rows they don't: 10-23% faster at 6
columns. Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS are covered by the model rows above and by the
b11434 runs (`docs/fast27b.md`: Q4_0 -21%, IQ4_XS -10%, Q8_0 -4% time at 6 columns).

### End to end (llama-server)

Qwen3.8-27B UD-Q6_K, R9700, six prompts x 512 tokens, temperature 0, harmonic mean:

| drafter | stock tok/s | patched tok/s | speedup |
|---|---:|---:|---:|
| DFlash2 (`draft-dflash`, n=5) | 57.1 | 67.5 | 1.18 |
| MTP (`draft-mtp`, n=3) | 52.3 | 56.2 | 1.07 |

### Correctness

- `test-backend-ops -o MUL_MAT_ID`: 931/931 pass on both GPUs.
- `test-backend-ops -o MUL_MAT`: 2444/2445 pass on both GPUs. The one failure is
  `MUL_MAT(type_a=f16,type_b=f32,m=16,n=2105344,k=32,...)` (ERR 0.0043 > 0.0005). It fails
  identically on the **stock** b11719 build on both GPUs
  (`test-backend-ops_gpu1_MUL_MAT_large-n_stock.txt`). It is f16 with a 2.1M-column batch, a path
  the patch does not touch, so it is a pre-existing ROCm issue (worth its own issue report).
- `llama-perplexity`, Qwen3.5-9B Q6_K, wikitext-2 test, 50 chunks: 7.4995 +/- 0.16917 with both
  builds (prompt batches use MMQ, so this is expected).

## Related upstream PRs (checked 2026-10-24)

- **#24386** "ggml: tune RDNA4 MMVQ warps for K-quants" (open). Changes the RDNA4 `ncols_dst == 1`
  warps for Q4_K/Q6_K. It is a different branch of the same function, so the two don't overlap
  logically, but they may conflict textually.
- **#20831** "cuda : dynamic MMVQ nwarps for narrow matrices" (open, under maintainer review).
  - It renames `calc_nwarps` to `calc_max_nwarps` and clamps warps by matrix width on RDNA3/4.
  - If it lands first, this patch needs a rebase. With only 2 warps, its clamp should rarely
    apply at 2-8 columns.
- Also check for new PRs touching `mmvq.cu` RDNA4 parameters before opening one (CONTRIBUTING
  rule 2 on AI-assisted PRs).

## Not covered

- **gfx1201 only** (R9700, RX 9070 XT). The RX 9060 series (gfx1200) was not tested; the 9070
  sweep in #24386 suggests gfx1200 can differ.
- **Linux, ROCm 7.2 only.** Windows HIP SDK not tested.
- **The sweep was not repeated per type.** IQ types might prefer other values, but none
  regressed.
- **Not tested in this setup:** multi-GPU (`-sm row`/`tensor`) and NVIDIA. NVIDIA is not
  affected by the code path.

## Checklist from CONTRIBUTING.md

- [ ] Write the PR description yourself (no AI text); fill the template's "AI usage disclosure".
- [ ] Review the diff by hand and be able to explain each line (section "What each line does").
- [ ] Update the code comment's numbers to master's, or make it shorter.
- [ ] Re-check open PRs touching RDNA4 MMVQ (#24386, #20831, anything newer); comment there if one
      overlaps.
- [ ] Run `test-backend-ops` and llama-bench on the latest master before opening.
- [ ] New contributors: one open PR at a time.

## Reproduce

```bash
# builds: stock master and master + patch (both worktrees deleted after the run)
git -C ~/llama.cpp worktree add --detach ~/llama.cpp-b11719 5aeac37a3
git -C ~/llama.cpp worktree add --detach ~/llama.cpp-b11719-mmvq 5aeac37a3
git -C ~/llama.cpp-b11719-mmvq apply ~/code/dense-moe/patches/llama.cpp-rdna4-mmvq-multicol.patch
for d in ~/llama.cpp-b11719 ~/llama.cpp-b11719-mmvq; do
  cmake -S $d -B $d/build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm" \
    -DLLAMA_CURL=OFF
  cmake --build $d/build -j --target llama-bench llama-perplexity test-backend-ops llama-server
done
# models: Qwen3.5-9B quants are made from Qwen/Qwen3.5-9B (BF16 GGUF + calibration imatrix,
# docs/writeup.md); the per-type --pure files go in /mnt/fast/models/tmp-puretypes
benchmarks/mmvq_upstream.sh && benchmarks/mmvq_types.sh
python3 scripts/mmvq_upstream_summary.py > experiments/results/mmvq_upstream/summary.md
```
