# Conclusions: making good dense models fast on limited hardware (2026-10-24)

This closes the project. It started as "convert dense Qwen3.5-9B into a sparse MoE for faster
inference" (`requirements.md`). It became: what actually makes a good dense model fast on
consumer hardware, including GPUs too small for it? Models: Qwen3.5-9B and Qwen3.8-27B (same
`qwen35` hybrid architecture). Hardware: RX 9070 XT (16 GB), Radeon AI PRO R9700 (32 GB), Ryzen 7
9800X3D, DDR5-6000, ROCm 7.2, llama.cpp b11434.

Detailed reports, in order: `docs/writeup.md` (MoE conversion, quantization bar),
`docs/phase3_findings.md` (fine-grained sparsity), `docs/mtp_tuning.md`, `docs/offload.md`,
`docs/fast27b.md`, `docs/dflash.md`. Measured findings: `docs/findings.md` (F0-F9).

## 1. Answers

**Can a dense model be made faster by converting it to a sparse MoE without training?** No.
- The sparse kernels work: top-2-of-8 experts run 3.3x faster per layer.
- The approximation fails: even a perfect router loses more quality (KLD 0.117 at 6 of 8
  experts) than 3-bit quantization (0.058), for at most 1.18x. (F3, F4)
- Training a router and experts back (distillation, 10B+ tokens in the literature) was not
  attempted. The ceiling on top of quantization is ~1.2x, so the expected value is low.

**Can activation sparsity (Deja Vu, TEAL, CATS, PowerInfer-style hot neurons) help instead?**
Not on these models with speculative decoding on.
- TEAL-style input sparsity is nearly free in quality, but gives at most +4-12% on single-token
  decoding, and ~0% with MTP or DFlash. The verify step checks several tokens, and their combined
  masks cover the whole FFN. (F5)
- There are no hot neurons to pin in VRAM: at 50% kept, only ~1% of neurons are used by >90% of
  tokens. (F5, F7)
- On the CPU side of an offloaded model, the ceiling is +5-26% without speculative decoding and
  ~0 with it. (F7)

**Can the "brains" of a dense model stay in VRAM and the rest in RAM?** Not in a useful way.
Every weight of a dense model is read for every token, so the cost is set by how many bytes stay
in RAM, not by which ones. Each GiB in RAM costs ~15-20 ms per token on DDR5-6000, ~10x a GiB in
VRAM. Moving whole layers to RAM is equal or 3-12% faster than moving only the FFNs. (F7)

**What does make them faster?** Three things, all available in llama.cpp today, plus one small
kernel fix:

| Lever | Gain | Quality cost | Where measured |
|---|---|---|---|
| Quantize until the model fits in VRAM | 2.8-3.8x vs a better quant partly in RAM | IQ4_XS: KLD 0.018 | F4, F7 |
| Speculative decoding with the model's MTP head | 1.5-1.8x on the GPU, 1.7-2.5x partly offloaded | none (lossless) | F6, F7 |
| A trained drafter (DFlash) instead of MTP, model fully in VRAM | +4-29% over MTP (short prompts), +14-18% sampling, +3-6% at long context or 4 users | none | F8, F9 |
| n-gram drafting on top, for editing | 5.0-5.7x over plain decoding | none | F6, F9 |
| RDNA4 MMVQ patch (2 warps x 4 rows for 2-8 token batches) | +7-23% with drafters on K-quants; verify batches 14-23% faster | none | F8, F9 |

## 2. What to run, by GPU memory

Fastest measured setup at each size, with the quality cost (KLD vs a lossless reference: the
9B's vs BF16, the 27B's vs Q8_0). Speeds are six-domain averages at temperature 0. Offloaded rows
use the offload milestone's three prompts. Up to 16 GB was measured on the RX 9070 XT at simulated
budgets; above that, on the R9700 (~640 GB/s), with peak VRAM at 8k context. A card with less
memory bandwidth will be proportionally slower.

| VRAM | Highest quality that runs well | Fastest |
|---|---|---|
| 4 GB | 9B Q6_K, 12 layers on GPU, MTP: 31.6 tok/s (0.0028) | 9B IQ4_XS, 18 layers, MTP: 41.5 (0.018) |
| 6 GB | 9B IQ4_XS fully on GPU, MTP: 138.7 (0.018) | same |
| 8 GB | 9B Q6_K fully on GPU, MTP: 118.0 (0.0028) | same; 9B IQ4_XS + DFlash needs 7.8 GiB |
| 12 GB | 9B Q8_0 + DFlash n=7: 131.9 (0.0016), 11.4 GiB | 9B IQ4_XS + DFlash n=5: 162.6 (0.018), 7.8 GiB |
| 16 GB | 27B IQ4_XS fully on GPU, MTP: 70.6 (0.018 vs 27B Q8_0) | 9B IQ4_XS + DFlash n=5: 162.6 (0.018) |
| 24 GB | 27B Q5_K_M + DFlash n=5: 64.5 (0.0066), 21.6 GiB | 27B IQ4_XS + DFlash n=5: 81.9 (0.018), 17.9 GiB; Q4_0 85.0 at 1.6x the KLD |
| 32 GB | 27B UD-Q6_K + DFlash n=5: 62.6 (0.0023), or Q8_0 + DFlash n=7: 61.9 (lossless), 30.2 GiB | 27B IQ4_XS + DFlash n=5: 81.9 |

- Bigger model vs faster model at the same VRAM is a quality decision. At 16 GB, the 27B IQ4_XS
  (70.6) and the 9B Q8_0 (132) are both good options.
- **Partly offloaded models: use MTP, not DFlash.** The drafter costs 9-20 GPU layers, or 2-9
  even with its weights in RAM. The exception is a 27B that is mostly in RAM (8-12 GB): DFlash
  with `-ngld 0 --spec-draft-n-max 7` is +8-28% there. (F9)
- Leave CPU threads at llama.cpp's default. RAM bandwidth saturates at ~4 threads.

Commands:

```bash
# Fully on the GPU, one user: DFlash (draft 5; 7 for Q8_0). Model-matched drafter GGUFs:
#   27B: Qwen3.8-27B-DFlash2 (Anbeeld/Qwen3.8-27B-DFlash2-GGUF); 9B: AtomicChat/Qwen3.5-9B-DFlash-GGUF
llama-server -m Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -c 32768 -np 1 \
  --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 --spec-draft-n-max 5

# Editing workloads (the answer copies most of the input): add n-gram drafting
  --spec-type draft-dflash,ngram-mod

# Model partly in RAM: MTP, with -ngl lowered 2-3 layers for the draft context
llama-server -m Qwen3.8-27B-IQ4_XS.gguf -ngl 47 -fa on -c 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

On RDNA4 (RX 9070 / 9070 XT / R9700), build llama.cpp with
`patches/llama.cpp-rdna4-mmvq-multicol.patch` until it is upstream.

## 3. What was built along the way

- **A training-free dense-to-MoE pipeline with Triton kernels** (`src/dense_to_moe/`).
  - Partitioning: contiguous, random, and k-means on weights, co-activation or contribution.
  - Routing: oracle routers and a fused router.
  - Kernels: gathered GEMV decode kernels for ROCm.
  - Bit-exact reconstruction tests.
- **A quality protocol that matches llama.cpp's KLD exactly**, so PyTorch variants and GGUF quants
  can be compared on one scale.
- **A speculative-decoding benchmark harness** (`benchmarks/mtp_tune.py`):
  - Six domain prompts, long-context, editing and concurrency sets.
  - VRAM tracking; budget fitting for offloaded configs (`offload_mtp_fit.py`,
    `offload_dflash_fit.py`).
- **The RDNA4 MMVQ patch** (7 lines) and its evidence on two RDNA4 cards and five models,
  including on upstream master: `docs/upstream_mmvq.md`.
- **Two ROCm tooling hazards documented with repros:**
  - `expandable_segments` corrupts tensors (F0).
  - hipBLASLt bmm returns garbage past batch 65,535 (F5a).

## 4. Limits

- **Two models of one family**, both with MTP heads and DFlash drafters. Models without them get
  only quantization and n-gram drafting from this list.
- **One machine.**
  - AMD RDNA4 only. NVIDIA and older AMD cards have different MMVQ parameters; the patch only
    touches the RDNA4 table.
  - Smaller budgets were simulated on a fast 16 GB card.
  - DDR5 dual channel. On DDR4 or single-channel RAM, offloaded speeds fall roughly in proportion
    to bandwidth.
- **Speeds are decode speeds on short to medium prompts.** Prompt processing is slower when
  offloaded (420-820 tok/s for the 27B at 8 GB vs ~1,400 on the GPU) and was not optimized.
- **Quality is KLD on wikitext-2 and our validation text**, not task benchmarks. The 27B -rq quants
  were made from Q8_0, not BF16.

## 5. Not done

- **Upstreaming the MMVQ patch.** llama.cpp requires the contributor to write the PR and disclose
  AI use. The evidence is ready (`docs/upstream_mmvq.md`). Two open PRs touch the same function
  (#24386 for the 1-token path, #20831 for narrow MoE matrices).
- **DFlash on a model partly in RAM, done properly.** It would need the drafter's GPU buffers to
  shrink with `-ngld 0` (untraced ~0.4-1.4 GB), or a smaller drafter.
- **Training a sparse model** (requirements stages 2-3). Not started, per the brief and the
  results above.
