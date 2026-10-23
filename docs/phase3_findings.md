# Phase 3 findings: predictive neuron sparsity for Qwen3.5-9B

Milestone 1 of Phase 3, 2026-10-23. Research, method and gates: `docs/phase3_research.md` (the
gates were fixed before these results). Raw data in `experiments/results/phase3/`:
`block_sweep.json` (per layer), `e2e_kld_val.json` (full model), `speed_ceiling.json`,
`gate3_frontier.md`, and the tables in `summary.md`.

**Bottom line: no.** Fine-grained, input-dependent FFN sparsity keeps much more quality than
fixed experts, but only at single-neuron granularity. On this model and hardware it can't beat
quantization + MTP. The verify step of MTP reads the union of ~4 tokens' masks, and that union
covers 89-100% of the FFN at every useful budget. Predictor training and sparse kernels (Stages
C-E) were not started, as the gates require.

## A. What was investigated, and why

Phase 1 showed that a per-neuron oracle keeps 50% of FFN neurons at KLD 0.0103. Fixed 8-expert
MoE routing does 10x worse (0.117 at 75%). The question here was whether that gap can be turned
into real speed. It splits into three parts, each measured separately:

1. **Selection quality:** how good is the choice of neurons or blocks with perfect information?
   Block size and objective both matter here.
2. **Predictability:** how good is a choice made from information available *before* the FFN
   runs? The methods are gate-only (CATS-like) and input magnitude (TEAL, WINA).
3. **Execution:** what fraction of FFN bytes a decode step must still read, the bytes a verify
   batch of several tokens reads, and what that is worth in tok/s on the measured llama.cpp
   baselines.

Setup:
- **Model:** Qwen3.5-9B, revision `c202236`, BF16. All 32 FFNs are sparsified in full-model runs.
- **Per-layer data:** held-out val activations (2,048 tokens per layer; 256 for greedy).
  Calibration activations are used only for `gate` statistics and k-means block clustering.
- **Full-model data:** KLD against the llama.cpp BF16 base file, with the same 100 x 255 val tokens
  and the same KL formula as the GGUF quants in F4.
- **Hardware:** all quality runs on the R9700. The sweep took 61-74 s per layer. The full-model
  run took 34 s per chunk for 18 variants, about 57 minutes.

## B. Experimental results

### B1. Reproduction of the Phase 1 oracle (Stage A)

| Variant | Phase 1/2 KLD | Phase 3 re-run |
|---|---:|---:|
| torch dense (noise floor) | 0.00040 | 0.00040 |
| neuron oracle, 75% kept | 0.00133 | 0.00133 |
| neuron oracle, 50% kept | 0.01030 | 0.01030 |
| new code, `oracle B=1 contrib`, 50% | | 0.01035 |

- The Phase 1 numbers reproduce exactly.
- The new block code at B=1 is identical to the Phase 1 oracle in fp64; this is tested in
  `tests/test_block_sparsity.py`. In the model it differs by 0.00005, within one stderr: Phase 1
  ranks neurons with bf16 scores and the new code uses fp32, so near-ties break differently.
- **What the Phase 1 oracle uses:** `|h_i| * ||W_down[:, i]||`, a per-neuron output-norm
  heuristic. It needs the exact `h`, which means the full gate and up projections. It proves
  that the FFN output is compressible per token, and nothing about speed: by itself it can skip
  only down-projection columns, so it still reads 83% of FFN bytes at 50% kept.

### B2. Block granularity (Stage B), per-layer FFN relative error, mean of 32 layers

Oracle top-k blocks by true block output norm `||W_d[:, b] h_b||`, contiguous neuron blocks:

| Neurons per block | 90% kept | 75% | 62.5% | 50% | 37.5% | 25% |
|---|---|---|---|---|---|---|
| 1 | 0.006 | 0.031 | 0.065 | **0.111** | 0.176 | 0.265 |
| 8 | 0.089 | 0.180 | 0.251 | 0.325 | 0.404 | 0.493 |
| 16 | 0.118 | 0.221 | 0.299 | 0.376 | 0.455 | 0.543 |
| 32 | 0.142 | 0.256 | 0.338 | 0.417 | 0.498 | 0.585 |
| 64 | 0.162 | 0.285 | 0.371 | 0.453 | 0.535 | 0.622 |
| 128 | 0.185 | 0.310 | 0.400 | 0.484 | 0.568 | 0.655 |
| 256 | 0.201 | 0.332 | 0.426 | 0.513 | 0.598 | 0.686 |
| 512 | 0.191* | 0.353 | 0.450 | 0.539 | 0.626 | 0.715 |
| 1536 (= Phase 1 E=8) | 0.261* | 0.383 | 0.484 | 0.577 | 0.666 | 0.757 |

\* whole blocks only: 91.7% / 87.5% actually kept.

- **The loss comes from the first grouping step.** Going from 1 to 8 neurons per block triples the
  error at 50% kept (0.111 -> 0.325). After that it rises slowly up to the Phase 1 expert size.
- **The objective matters little.** Ranking by summed per-neuron contribution is slightly worse
  than block norm (B=16, 50%: 0.384 vs 0.376). Greedy reconstruction, which accounts for blocks
  cancelling, is better (0.332 vs 0.375 on the same 256 tokens), but still 3x the per-neuron
  error. Greedy is a heuristic; the exact optimum (subset selection) was not computed.
- **Clustering helps a little.** Blocks formed by balanced k-means on calibration contribution
  vectors (layers 4, 15 and 26) at 50% kept: B=16 0.347 vs 0.379 contiguous, B=64 0.398 vs 0.458,
  B=256 0.430 vs 0.515. That's 8-16%, not the 3x needed.
- **Selection is token-specific; there are no hot blocks.** At 50% kept, 0.5-1% of blocks are
  chosen by more than 90% of tokens. The 50% most frequent blocks cover 58-61% of all selections
  (50% would be uniform).
- **Layer sensitivity:** the first and last layers are easiest (layer 31: 0.041 at B=1, 50%) and
  layers 7-13 are hardest (0.137). Same ordering for every method; table in `summary.md`.

Full-model KLD, all 32 FFNs:

| Variant | FFN kept | KLD | PPL (base 3.725) | same top-1 |
|---|---|---:|---:|---:|
| oracle B=1 | 50% | 0.0104 | 3.746 | 0.963 |
| oracle B=8 | 50% | 0.115 | 4.050 | 0.883 |
| oracle B=16 | 50% | 0.168 | 4.238 | 0.857 |
| oracle B=64 | 50% | 0.297 | 4.798 | 0.806 |
| oracle B=16 | 75% | 0.0464 | 3.832 | 0.925 |
| *IQ4_XS (reference)* | | *0.0184* | *3.787* | |

Even with a perfect choice, 8-neuron blocks at 50% kept cost 6x the KLD of IQ4_XS.

### B3. What can be chosen before the FFN runs (Stage C, training-free predictors)

Per-layer relative error at **matched FFN bytes read** per decode token (interpolated over budgets):

| Method (B=1) | needs | bytes 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|
| oracle | gate + up | 0.045 | 0.163 | - | - | - |
| gate (CATS-like) | gate | 0.057 | 0.137 | 0.228 | 0.325 | 0.432 |
| TEAL (input + down-input magnitude) | nothing extra | - | **0.066** | **0.120** | **0.189** | **0.273** |
| WINA (TEAL weighted by column norms) | nothing extra | - | 0.066 | 0.120 | 0.189 | 0.273 |
| WINA, quant-block aligned (32) | nothing extra | - | 0.422 | 0.550 | 0.659 | 0.756 |

Full-model KLD of the deployable methods:

| Variant | FFN bytes read | KLD | PPL | same top-1 |
|---|---:|---:|---:|---:|
| TEAL/WINA, 90% of x and h | 0.900 | 0.0011 | 3.728 | 0.985 |
| TEAL, 75% | 0.750 | 0.0093 | 3.757 | 0.964 |
| WINA, 75% | 0.750 | 0.0093 | 3.751 | 0.963 |
| WINA, 62.5% | 0.625 | 0.0316 | 3.836 | 0.936 |
| TEAL, 50% | 0.500 | 0.0913 | 4.030 | 0.892 |
| WINA, 50% x only | 0.667 | 0.0877 | 4.029 | 0.896 |
| gate, 50% | 0.667 | 0.0915 | 3.999 | 0.894 |
| gate B=16, 50% | 0.667 | 0.271 | 4.694 | 0.819 |
| WINA 75%, 32-channel blocks (GGUF-aligned) | 0.750 | 0.400 | 5.315 | 0.787 |

- **Input sparsity (TEAL) beats the gate-only predictor at every byte budget.** Its errors are
  almost all on the input side: sparsifying only x at 50% costs 0.251 of the 0.273.
- **WINA = TEAL on this model.** The column norms of `[W_gate; W_up]` vary by only 6.7% (CV), so
  the two input masks overlap 98%.
- **Quantization-aligned input sparsity fails.** Skipping x in 32-channel groups, the only way
  to skip work inside GGUF row blocks, costs KLD 0.40 at 75%. Important input channels are
  scattered, not grouped.
- **No learned predictor was trained.** The cheap training-free signal already beats
  "gate-only", and B3/B4 below show even a perfect predictor doesn't clear the speed gate under
  MTP.

### B4. Masks across consecutive tokens (what a batched MTP verify step reads)

FFN bytes read by one token vs the union over 2 and 4 consecutive tokens (first 20 val chunks):

| Variant | 1 token | 2-token union | 4-token union |
|---|---:|---:|---:|
| oracle B=1, 50% | 0.833 | 0.904 | 0.964 |
| oracle B=16, 50% | 0.833 | 0.901 | 0.958 |
| gate, 50% | 0.667 | 0.784 | 0.885 |
| TEAL 50% | 0.500 | 0.722 | 0.905 |
| TEAL 75% | 0.750 | 0.926 | 0.992 |
| TEAL 90% | 0.900 | 0.988 | 1.000 |

The MTP runs in F4 verify 3.96-3.99 tokens per step (1 + drafts per step, from the server logs).
At budgets with acceptable quality, the 4-token union reads 99-100% of the FFN: MTP and
token-level FFN sparsity don't stack. This is the chunk-level sparsity problem BlockFFN
describes, measured on a model that wasn't trained for it.

## C. Hardware performance

No sparse kernel was built (gates G1/G2 failed, see F). The speed numbers are **ceilings**:
measured llama.cpp decode times, minus the time to read the FFN bytes a method skips, at zero
cost for the predictor, the gather or the mask.

How much decode time is spent reading FFN weights:

| | RX 9070 XT | R9700 |
|---|---|---|
| FFN share of weight bytes read, IQ4_XS | 55% | 55% |
| Cross-quant fit (all weights): ms per GiB read, fixed cost per token | 1.64 ms (655 GB/s), 4.4 ms | 1.57 ms (686 GB/s), 4.5 ms |
| FFN-only proxy fit (9070 XT only) | 1.12 ms per GiB of FFN | |
| FFN share of IQ4_XS decode time | 24% (proxy) to 35% (cross-quant) | ~35% (cross-quant) |

Ceiling tok/s for single-token decode when a fraction f of FFN bytes is read (proxy fit / cross-quant fit):

| Base | measured | f = 0.833 | f = 0.667 | f = 0.5 | f = 0 (no FFN) |
|---|---:|---:|---:|---:|---:|
| 9070 XT Q8_0 | 57.5 | 60.6 / 62.2 | 64.0 / 67.6 | 67.9 / 74.2 | 83.0 / 104.6 |
| 9070 XT Q6_K | 70.3 | 73.9 / 75.6 | 77.8 / 81.9 | 82.2 / 89.2 | 98.9 / 122.2 |
| 9070 XT IQ4_XS | 89.6 | 93.3 / 95.2 | 97.3 / 101.4 | 101.7 / 108.6 | 117.7 / 138.0 |
| R9700 Q8_0 | 59.4 | 64.2 | 69.7 | 76.4 | 106.9 |
| R9700 Q6_K | 71.6 | 76.9 | 83.0 | 90.2 | 122.1 |
| R9700 IQ4_XS | 92.1 | 97.8 | 104.1 | 111.3 | 140.6 |

Even deleting the whole FFN would give IQ4_XS 118-141 tok/s, below the 148 tok/s that
IQ4_XS + MTP reaches with every weight read. The R9700 has no MTP or proxy runs; its
single-token ceilings are within 0-3% of the 9070 XT's.

## D. Quality analysis

- **Where sparsity works:** per neuron, at 75-90% of x and h kept. TEAL 90% costs KLD 0.0011,
  less than Q8_0's 0.0016. TEAL 75% costs 0.0093, between Q6_K and Q5_K_M. The FFN output of
  this model is compressible per token.
- **Where it fails:**
  1. Any grouping of neurons, even 8. Important neurons are scattered and token-specific; there
     are no hot blocks to pin and no stable groups to exploit, and clustering recovers only 8-16%.
  2. Input-channel sparsity in quant-block groups.
  3. Below ~65% of bytes, all deployable methods: TEAL 50% has KLD 0.091, worse than Q3_K_M (0.058).
- **Why the predictors can't match the oracle:** the oracle's advantage comes from knowing `h`.
  The cheapest exact source of `h` is the gate + up product, which is 2/3 of the FFN. The gate
  alone leaves 2.3x the oracle's error at 50%, and input magnitude leaves 2.5x.
- **Not measured:** downstream tasks (coding, reasoning, instruction following, tool calling,
  long context) were not evaluated. Gate 4 was not reached, and the only methods worth testing
  there (TEAL 75-90%) would be judged by KLD first. KLD and PPL here are next-token metrics on
  prose, code, math, instructions and JSON, 512-token windows.

## E. Practical inference comparison

Sparse variant on top of a base quant, against the fastest *dense* quant with equal or better
estimated quality. Ceilings use the cross-quant fit (optimistic). The combined KLD is estimated
as KLD_quant + KLD_sparse; this additivity is **assumed, not measured**. From `gate3_frontier.md`:

| Candidate (quality <= 0.03) | est. KLD | single-token ceiling | best dense at <= KLD | gain | MTP ceiling | best dense + MTP | gain |
|---|---:|---:|---|---:|---:|---|---:|
| TEAL 90% on Q8_0 | 0.0027 | 60.2 | Q8_0 57.5 | +4.7% | 115.9 | Q8_0 115.8 | +0.0% |
| TEAL 90% on Q6_K | 0.0039 | 73.4 | Q6_K 70.3 | +4.4% | 117.4 | Q6_K 117.4 | +0.0% |
| TEAL 75% on Q6_K | 0.0121 | 78.6 | Q6_K 70.3 | +11.9% | 117.8 | Q6_K 117.4 | +0.3% |
| TEAL 75% on IQ4_XS | 0.0277 | 98.2 | IQ4_XS 89.6 | +9.6% | 148.4 | IQ4_XS 148.0 | +0.3% |
| TEAL 90% on IQ4_XS | 0.0195 | 92.8 | IQ4_XS 89.6 | +3.6% | 148.0 | IQ4_XS 148.0 | +0.0% |

- **Without MTP:** the ceiling is +4-12%. Getting it needs three things that don't exist:
  1. A transposed, output-dim-quantized FFN layout in llama.cpp. Row-block quantization and TEAL
     input sparsity are incompatible (the 32-aligned variant costs KLD 0.40).
  2. A kernel at ideal bandwidth.
  3. A free mask.
  With the pessimistic proxy fit, the gain shrinks to +3-8%.
- **With MTP:** the ceiling is +0.0-0.3% at every acceptable quality. The 4-token union reads the
  whole FFN. Even the unusable TEAL 50% (KLD ~0.11 on IQ4_XS) gets only +3.4%.
- The MTP ceiling assumes the draft acceptance rate is unchanged by sparsifying the target model.
  Any lower acceptance makes it worse.

## F. Conclusions

### Gates (fixed in `docs/phase3_research.md` before the full results)

| Gate | Criterion | Result |
|---|---|---|
| G1 oracle viability | B >= 8 at <= 50% kept, KLD <= 0.0184 | **Fail.** B=8: 0.115, B=16: 0.168 |
| G2 predictability | deployable method, <= 60% of FFN bytes, KLD <= 0.0184 | **Fail (narrow).** Needs ~70% (TEAL 75%: 0.0093 at 0.75; 62.5%: 0.0316 at 0.625) |
| G3 single-token ceiling | beats the fastest dense quant at equal quality | **Pass on paper only:** +4-12%, under ideal-kernel, free-mask, re-layout and additivity assumptions |
| Stretch, MTP | beats IQ4_XS + MTP (148 tok/s) at comparable quality | **Fail.** <= +0.3% |

### Demonstrated
- Per-neuron selection reproduces (KLD 0.0103 at 50%). Grouping neurons into blocks of 8 or more
  destroys most of the advantage (0.115 at B=8), with norm, contribution or greedy objectives,
  contiguous or clustered.
- Training-free input sparsity (TEAL/WINA) at 75-90% of channels costs KLD 0.0093-0.0011 on
  Qwen3.5-9B. WINA's weighting is a no-op on this model.
- Quant-block-aligned input sparsity is unusable (KLD 0.40 at 75%).
- Masks over 4 consecutive tokens cover 89-100% of the FFN at every budget tested, so MTP verify
  erases token-level FFN sparsity.
- **Tooling hazard:** on torch 2.14.1+rocm7.2, hipBLASLt `torch.bmm` silently returns wrong values
  for batch index >= 65536, on both GPUs (`scripts/repro_bmm_batch_limit.py`; correct with
  `TORCH_BLAS_PREFER_HIPBLASLT=0`). It produced wrong block scores until it was found and fixed.
  Phase 1/2 code was audited: all batch dimensions there are <= 20,000.

### Estimated, not measured
- All tok/s for sparse variants are ceilings (section C).
- Additivity of quantization and sparsity KLD.
- MTP acceptance under sparsity.

### Unproven hypotheses
- A *trained* sparsifier could make chunk-level (4-token) sparsity high. BlockFFN does this, but
  by training the architecture, which is outside the budget.
- Rotation-based methods (LaRoSA) or quantized proxy predictors (Prox) might move the TEAL curve
  left by a few points. They can't fix the MTP union.

### Negative findings
- Block-structured predictive sparsity, both the brief's central idea and DSMoE-style blocks,
  fails at the oracle level for this model.
- Hot/cold neuron partitioning: there are no hot neurons at useful budgets.

### Recommended future experiments (only if the goal changes)
- **If MTP is unavailable** (other runtimes, batch-1 without speculation): TEAL at 75-90% on
  Q6_K/Q8_0 in a transposed-layout llama.cpp kernel, for at most +5-12%. Measure additivity first:
  it's cheap, since a dequantized GGUF in torch can be scored with the same harness.
- **Better use of the time:** the MTP configuration itself. Draft length, acceptance per domain,
  and the R9700 (MTP never measured there).

## G. Recommendation

**Is predictive neuron sparsity a technically and economically worthwhile way to accelerate
Qwen3.5-9B on this hardware? No.**

- **Technically:** the quality that survives sparsification lives at single-neuron granularity
  and needs ~70% of FFN bytes to match IQ4_XS quality. That's +4-12% single-token decode even in
  the ideal case, and it requires a custom quantized layout.
- **Economically:** the best existing configuration, IQ4_XS + MTP (148 tok/s, KLD 0.018), verifies
  ~4 tokens per step, and their combined masks cover the whole FFN. So the measured ceiling of
  sparsity on top of it is +0.3%.

The strongest alternative is what's already there: **IQ4_XS + MTP** for speed, **Q8_0 / Q6_K + MTP**
(116-117 tok/s, KLD 0.0016 / 0.0028) for quality. Further investment in FFN sparsity for this
model should stop. That includes predictor training, sparse kernels and full-model conversion.
