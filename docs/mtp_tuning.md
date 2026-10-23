# MTP tuning on the R9700 (2026-10-23)

Qwen3.5-9B GGUF quants with the built-in MTP head as the draft (`llama-server --spec-type
draft-mtp`, llama.cpp b11434), on the Radeon AI PRO R9700 (HIP device 1). Harness:
`benchmarks/mtp_tune.py`; tables: `scripts/mtp_summary.py` ->
`experiments/results/mtp/summary_gpu1.md`; raw: `experiments/results/mtp/mtp_tune_gpu1.jsonl`
(660 generations, with the generated text).

## Protocol

- Six prompts, one per domain of the calibration corpus (`code_python`, `code_cpp`, `prose`,
  `math`, `instruction`, `json`), written for this benchmark. 512 generated tokens each, thinking
  off, no prompt cache, one warmup request per server start, `-c 8192 -fa on -ngl 99 -np 1`.
- Speed: decode tok/s from the server's timings, harmonic mean over the six prompts.
  Acceptance = accepted / drafted. Tokens per step = generated / (generated - accepted).
- **Noise:** repeated runs of the same config agree to within 0.5% (IQ4_XS, no speculation: 91.3 /
  91.0 / 90.8; draft-mtp n_max=3: 136.0 / 135.9). Differences below ~1% are noise.
- **GPU sharing:** other services on this machine can load models onto the R9700 on demand. The
  card's VRAM was recorded before and after every config; nothing else used it during any run.
- Temperature 0.7 runs use one seed per prompt, so their per-prompt numbers are noisier.

## Results

### Default MTP vs no speculation (temperature 0)

| Quant | KLD vs BF16 | No speculation | MTP, default (n_max=3) | Speedup | Range over domains |
|---|---:|---:|---:|---:|---|
| BF16 | 0 | 34.3 | 49.9 | 1.46x | prose 42 - json 62 |
| Q8_0 | 0.0016 | 58.8 | 104.8 | 1.78x | prose 88 - json 129 |
| Q6_K | 0.0028 | 70.9 | 113.7 | 1.60x | prose 96 - json 140 |
| Q4_K_M | 0.0228 | 84.8 | 124.2 | 1.46x | prose 103 - json 152 |
| IQ4_XS | 0.0184 | 91.0 | 135.9 | 1.49x | prose 115 - json 172 |

- **The domain matters more than any setting.** JSON and math run 35-50% faster than prose and chat
  under MTP (IQ4_XS: acceptance 90% on JSON, 49% on prose).
- The Phase 2 figure (IQ4_XS + MTP 148 tok/s, 9070 XT) used only the Python code prompt. The same
  prompt gives 148 tok/s here too, but the six-domain mean is 136.
- **Q8_0 gains the most (1.78x):** its single-token step is slowest, so verifying 3-4 tokens per
  step costs relatively least.
- **BF16 gains the least (1.46x).** This build's BF16 batched matmul is slow on RDNA4 (llama-bench
  pp512 636 tok/s, vs ~4,200 for Q8_0), so the batched verify step is expensive.

### Draft length and confidence cut-off (temperature 0)

IQ4_XS, tok/s (acceptance):

| n_max | p_min 0 | p_min 0.3 | p_min 0.6 | p_min 0.8 |
|---|---|---|---|---|
| 1 | 116.4 (85%) | | | |
| 2 | 134.2 (75%) | | | |
| **3 (default)** | **135.9 (64%)** | | | |
| 4 | 130.4 (55%) | 131.0 (63%) | 127.0 (88%) | 116.9 (96%) |
| 5 | 120.0 (48%) | | | |
| 6 | 111.8 (43%) | 116.8 (52%) | 125.3 (85%) | 116.1 (94%) |
| 8 | 106.3 (35%) | 112.4 (46%) | 124.9 (82%) | 115.0 (94%) |

The same shape holds for every quant (full tables in `summary_gpu1.md`). Best config per quant:

| Quant | Best | tok/s | Default (n_max=3) | Difference |
|---|---|---:|---:|---:|
| BF16 | n_max=6, p_min=0.6 | 52.6 | 49.9 | +5.4% |
| Q8_0 | n_max=3 (n_min=2) | 105.1 | 104.8 | noise |
| Q6_K | n_max=3 | 113.7 | 113.7 | 0 |
| Q4_K_M | n_max=2 | 126.6 | 124.2 | +1.9% |
| IQ4_XS | n_max=3 | 135.9 | 135.9 | 0 |

- Acceptance falls by ~10 points per extra drafted token. Beyond 3, the extra drafts cost more
  verify work than they return.
- `p_min` raises acceptance to 82-96% but cuts tokens per step. It rescues long drafts (n_max=8:
  106 -> 125) without beating n_max=3. It only helps on BF16, where verify is expensive.
- `n_min` (2 or 3): no effect beyond noise.

### Combining MTP with n-gram drafting (temperature 0)

| Q8_0 / IQ4_XS | tok/s |
|---|---|
| draft-mtp | 104.8 / 135.9 |
| draft-mtp + ngram-mod | 104.8 / 135.9 |
| draft-mtp + ngram-map-k | 104.5 / 136.1 |
| ngram-mod alone | 58.9 / 90.8 (= no speculation) |

The n-gram drafters need a long repeated token run (ngram-mod: 48 tokens by default) and these
single-turn prompts have none. They cost nothing when idle, so they may help on repetitive work
(editing a long file, quoting context back), which this benchmark doesn't cover.

### Temperature 0.7

| Quant | No speculation | n_max=3 greedy draft | n_max=3 probabilistic draft | n_max=2 probabilistic |
|---|---:|---:|---:|---:|
| Q8_0 | 58.9 | 102.3 | 103.7 | 98.9 |
| Q6_K | 71.0 | 108.7 | 111.7 | 111.1 |
| IQ4_XS | 90.8 | 131.5 | 136.5 | 133.0 |

- Sampling at 0.7 is 0-4% slower than greedy decoding with the same draft settings.
- `--spec-draft-sampling probabilistic` (rejection sampling) is 1-4% faster than the default greedy
  draft at temperature 0.7 on all three quants. That's a single seed, so treat it as "at least as
  good". It also preserves the sampling distribution exactly.

## Accuracy

MTP changes speed, not the output distribution. The target model verifies every drafted token:
greedy at temperature 0, rejection sampling with `probabilistic` drafts. Quality is therefore
the quant's (F4).

One caveat, measured here: at temperature 0, MTP output is deterministic run to run, but it isn't
token-identical to non-speculative decoding. It diverges after ~400-1,100 characters on most
prompts (BF16 n_max=2 matched on 6/6). Batched verification uses different matmul kernels from
single-token decode, so near-ties flip. llama-perplexity, which produced the KLD numbers, also
evaluates in batches.

## Recommendation (R9700, this build)

- Keep the default draft length (`--spec-draft-n-max 3`). Use 2 for Q4_K_M. Don't use `p_min`
  except on BF16.
- At temperature > 0 add `--spec-draft-sampling probabilistic`.
- **Quality first:** Q6_K + MTP, 114 tok/s (KLD 0.0028, 98.1% same top-1 as BF16). Q8_0 + MTP gives
  105 tok/s at KLD 0.0016.
- **Speed first:** IQ4_XS + MTP, 136 tok/s (KLD 0.018).

```bash
HIP_VISIBLE_DEVICES=1 llama-server -m /mnt/fast/models/Qwen3.5-9B-Q6_K.gguf -ngl 99 -fa on -c 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-sampling probabilistic
```

Tuning moves decode speed by at most ~2-5% over the defaults on this hardware. The quant and the
kind of text matter far more.

## Not covered

- Long contexts (all prompts are short; the KV cache stays < 1k tokens).
- Multi-turn and repetitive editing workloads, where n-gram drafting could help.
- More than one seed at temperature 0.7.
- `-np` > 1 (concurrent requests).
- The same sweep on the 9070 XT.
