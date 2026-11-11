# Cheaper MTP drafting: a vocabulary shortlist for the drafter (2026-11-12)

A dense model reads all of its weights once per evaluation, and on these cards that read is the
whole cost: the 27B at UD-Q6_K takes 40.9 ms per token, about the R9700's memory bandwidth over
20.5 GiB. Speculative decoding is the lossless way past that limit, because one read then yields
several tokens. This test looks at what the drafting itself costs and removes most of it.

**Result:** each drafted token read the full output layer (1.29 GiB) to score all 248,320
tokens. Scoring a 65,536-token shortlist instead cuts a drafted token from 3.13 to 1.56 ms. The
27B's output is unchanged (6 of 6 texts identical), 99.5% of accepted drafts are kept, and
decode is 7% faster at the same draft length. Cheaper drafts make longer drafts pay: with
draft 4 the six-prompt mean goes from 52.4 to 58.7 tok/s (+12%), with draft 8 and a confidence
cutoff to 60.7 (+16%; +36-38% on math and JSON, +5-6% on prose and instructions).
Long context (120K) is not measured yet.

## Where a decode step goes

27B UD-Q6_K, MTP, q8_0 KV, 2K context, R9700, build with the MMVQ and attention patches
(`llama-bench -p 1..8`, `benchmarks/depth_profile.py`, timers around the draft calls):

| Part of one step (draft 3) | ms |
|---|---:|
| Target, one token | 40.9 |
| Extra for a 4-token verify batch | 4.4 |
| Three draft tokens, 3.13 ms each | 9.4 |
| Drafter catch-up on the verified tokens | 1.0 |
| Rest (sampling, server) | 1.7 |
| Step | 57.4 |

- A draft token costs the same at every draft position (3.13-3.15 ms for draft 1 to 6).
- Sampling the draft is not the cost (0.002 ms per token on the host side).
- The draft graph is the MTP block (335 MiB of weights) plus the output layer `output.weight`
  (248,320 x 5,120 at Q8_0, 1,288 MiB). At the card's ~540 GB/s that is 0.65 + 2.5 ms: the
  measured 3.13 ms is the read of those bytes, and 80% of it is the output layer.
- The shared-dequant verify kernel that was next on the roadmap could at best remove part of
  the 4.4 ms (7% of a step). It was not built.

## The shortlist

Drafts are verified by the full model, so the drafter may score any subset of the vocabulary
without changing the output. A draft whose correct token is outside the subset is simply
rejected.

- `patches/llama.cpp-mtp-draft-shortlist.patch` (on b12164): with
  `LLAMA_MTP_DRAFT_VOCAB=<file of token ids>` set, creating an MTP context builds
  `output_draft`, a copy of the listed rows of `output.weight` in the same quantization, plus
  all control tokens. The Qwen3.5-family MTP graph multiplies by that instead. Token ids are
  mapped back where the logits or the GPU sampler's candidates are read. Unset, nothing changes.
  The target's own output layer is untouched.
- Extra VRAM: 340 MiB for 65,536 tokens.
- Ranking: token frequency over 9.5M tokens of local text (C++, Python, Markdown, JSON, earlier
  model answers, Wikipedia, dolly-15k instructions, gsm8k), `scripts/shortlist_coverage.py`.
  Only 76,724 of the 248,320 tokens occur at all.

Coverage on held-out generations (the per-step logs of `docs/lenient.md`, six prompt types,
4,251 accepted draft tokens; `experiments/results/shortlist/coverage.md`):

| Shortlist | Share of vocabulary | Accepted drafts kept | Kept with tokens already in the response added |
|---:|---:|---:|---:|
| 16,384 | 6.6% | 81.8% | 88.2% |
| 32,768 | 13.2% | 90.9% | 94.3% |
| 49,152 | 19.8% | 95.1% | 97.2% |
| 65,536 | 26.4% | 97.4% | 98.8% |
| 76,724 | 30.9% | 99.0% | |

The tail is long: JSON and C++ lose the most at small sizes (identifiers, unusual strings).
The last column is a simulation only; the patch uses a fixed list.

## Speed

Six prompt types, 512 tokens each, temperature 0, 8K context, R9700
(`benchmarks/lenient_probe.py --ratios 0`, tables in `experiments/results/shortlist/summary.md`):

| Variant | Python | C++ | prose | math | instructions | JSON | harmonic mean | accepted / drafted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft 3 (today's default) | 58.0 | 49.2 | 45.3 | 61.5 | 43.8 | 63.8 | 52.4 | 2086 / 2924 |
| draft 4 | 60.9 | 45.2 | 46.3 | 67.6 | 45.2 | 70.5 | 53.9 | 2192 / 3482 |
| shortlist 65K, draft 3 | 62.0 | 53.2 | 49.3 | 66.8 | 46.8 | 66.8 | 56.3 | 2075 / 2960 |
| shortlist 65K, draft 4 | 64.4 | 50.1 | 50.4 | 74.7 | 50.0 | 74.2 | 58.7 | 2177 / 3542 |
| shortlist 65K, draft 6 | 65.6 | 46.9 | 46.1 | 80.4 | 45.4 | 82.5 | 57.2 | 2285 / 4660 |
| shortlist 65K, draft 6, p-min 0.5 | 66.9 | 53.1 | 48.9 | 82.3 | 45.4 | 82.4 | 59.7 | 2234 / 3375 |
| shortlist 65K, draft 8, p-min 0.5 | 67.6 | 54.7 | 47.7 | 83.7 | 46.4 | 88.0 | 60.7 | 2297 / 3839 |
| shortlist 65K, draft 8, p-min 0.75 | 69.7 | 47.0 | 43.8 | 84.1 | 42.8 | 85.6 | 56.9 | 2115 / 2599 |

- Same draft length: +7.4% (52.4 -> 56.3), 2075 of 2086 accepted drafts kept.
- Output text identical with and without the shortlist on 6 of 6 prompts, at draft 3 and at
  draft 4.
- Draft 4 was barely worth it before (+3%) and is +12% with the shortlist.
- Long drafts need the confidence cutoff (`--spec-draft-p-min 0.5`): without it prose, C++ and
  instructions pay for drafts that are rejected. At 0.75 too many good drafts are dropped.
- One run per cell, so differences of 1-2 tok/s between neighbouring variants should not be
  read as real.

Shortlist size at 2K context, two tasks, 256 tokens (`sweep.jsonl`), tok/s explain / code:

| | draft 3 | draft 4 | draft 6 |
|---|---:|---:|---:|
| full vocabulary | 52.9 / 62.6 | 57.2 / 71.4 | 52.3 / 76.0 |
| 32,768 | 58.9 / 63.3 | 60.7 / 72.0 | 62.2 / 77.7 |
| 65,536 | 58.4 / 68.2 | 60.9 / 77.8 | 63.5 / 87.0 |

32,768 saves 0.9 ms more per step than 65,536 but loses drafts on the code task (178 against
185 accepted of the same output), so 65,536 is the better size here.

## At depth

Not measured yet: the 2K / 120K run was still going when this was written. The saving per
draft token does not depend on context, so the relative gain should be smaller at 120K, where
a step takes 68 ms instead of 57.

## Limits

- One GPU, one model, one quantization. The daily two-card preset was not run.
- The token ranking comes from English text and code. Other languages would need their own
  list, or lose their drafts (the output stays correct, only slower).
- Speed cells are single runs. The code-quality suite was not rerun: the target's computation
  is unchanged and the texts are identical.
- The patch is a prototype: the shortlist buffer is allocated once per process and never freed,
  only the Qwen3.5-family MTP graph uses it, and the file path comes from an environment
  variable.
- Not tried: a lower-precision copy of the shortlist rows (Q4 would halve the remaining
  0.9 ms), adding the tokens of the current context to the list, and the same for the DFlash
  drafter.
