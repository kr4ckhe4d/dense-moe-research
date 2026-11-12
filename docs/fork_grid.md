# Stock, this repo's patches and the RDNA fork, on this machine (2026-11-12)

`docs/prior_work.md` found that the public fork
[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) had shipped
this repo's two RDNA4 kernel fixes first. This is the like-for-like run that page left open: the
fork's current release against stock llama.cpp and against this repo's patched build, same
machine, same prompts, the daily two-card preset shape.

**Result:** the fork is faster than this repo's build at every point measured. At the same draft
length its decode step is 10-12% shorter (62.7 against 69.7 ms at 120K, 52.6 against 58.1 ms at
2K, on UD-Q6_K). This repo's shortlist with draft 4 closes most of that gap: it is level with or
ahead of the fork's draft 3 at 2K and behind it in three of four cells at 120K. Against stock,
the fork gives +15-24% at 2K and +62-97% at 120K with the
standard MTP drafter; its adaptive drafter gives +22-38% at 2K and is slower than the fixed
draft at 120K. The fork prefills 12-23% slower than stock.

## Setup

- Both cards (R9700 + RX 9070 XT), 128K context, q8_0 KV, `-ub 512 -b 2048`, MTP, temperature 0,
  thinking off. Two tasks over llama.cpp source (explain, write a test), 256 tokens each, at 2K
  and at 120K of context. `benchmarks/grid_builds.sh`, `benchmarks/depth_profile.py --all-gpus`;
  rows in `experiments/results/grid/`, tables from `scripts/grid_summary.py`.
- Two files: the preset's UD-Q6_K (pulled 2026-10-06) and the Unsloth Dynamic v3.1 UD-Q4_K_XL
  (local-llm-benchmarks `r9700+rx9070.md`: 13% faster than the Q6_K on stock, no loss shown in
  code quality).
- Builds, all HIP for gfx1201 with the same CMake options:

| Name | Source |
|---|---|
| stock-b12164 | llama.cpp `4b77dee05`, what the presets run |
| stock-a691 | llama.cpp `a691b2345`, the fork's base (40 commits later), unpatched |
| ours | b12164 + `patches/` MMVQ and attention patches, draft 3 |
| ours-sl65k-n4 | the same + the 65,536-token drafter shortlist (`docs/shortlist.md`), draft 4 |
| fork-n3 | fork release `v17-a691b2345-r265` (17 patch blocks; the applied tree hash matches its `release.json`), `--spec-type draft-mtp`, draft 3, everything else at the fork's defaults |
| fork-adaptive | the same, `--spec-type draft-mtp-adaptive`, n-max 7 |

## UD-Q6_K

Decode tok/s as explain / code; ms per target evaluation is the mean of the two tasks.

| Build | 2K tok/s | 2K ms | 120K tok/s | 120K ms | Prefill to 120K, tok/s | Peak VRAM, MiB |
|---|---:|---:|---:|---:|---:|---:|
| stock-b12164 | 51.1 / 59.5 | 60.6 | 28.1 / 32.0 | 109.1 | 761 | 30,400 |
| stock-a691 | 51.3 / 59.3 | 60.6 | 28.1 / 32.0 | 109.2 | 755 | 30,403 |
| ours | 53.5 / 62.0 | 58.1 | 40.3 / 50.1 | 69.7 | 748 | 30,656 |
| ours-sl65k-n4 | 59.2 / 75.6 | 57.3 | 43.6 / 56.6 | 72.9 | 716 | 31,026 |
| fork-n3 | 61.4 / 68.2 | 52.6 | 45.6 / 55.7 | 62.7 | 667 | 29,581 |
| fork-adaptive | 62.4 / 81.2 | 59.0 | 41.7 / 51.9 | 77.8 | 658 | 30,218 |

## UD-Q4_K_XL v3.1

| Build | 2K tok/s | 2K ms | 120K tok/s | 120K ms | Prefill to 120K, tok/s | Peak VRAM, MiB |
|---|---:|---:|---:|---:|---:|---:|
| stock-b12164 | 55.5 / 67.0 | 53.5 | 28.9 / 33.8 | 102.2 | 936 | 26,621 |
| stock-a691 | 55.6 / 67.3 | 53.4 | 28.9 / 33.8 | 102.2 | 936 | 26,619 |
| ours | 57.3 / 70.2 | 51.2 | 50.7 / 54.1 | 62.9 | 936 | 26,668 |
| ours-sl65k-n4 | 69.4 / 85.1 | 51.7 | 52.4 / 58.9 | 67.6 | 888 | 27,050 |
| fork-n3 | 68.7 / 78.4 | 45.2 | 56.9 / 59.9 | 55.3 | 719 | 25,780 |
| fork-adaptive | 74.4 / 92.3 | 50.9 | 51.0 / 54.2 | 69.3 | 718 | 26,416 |

## Reading

- **The newer base is not the cause.** The two stock builds give the same numbers to 0.3 tok/s
  and identical text. The fork's gain is its patches.
- **Step time, same drafter and draft length (fork-n3 against ours):** 52.6 against 58.1 ms at
  2K and 62.7 against 69.7 ms at 120K on Q6_K; 45.2 against 51.2 and 55.3 against 62.9 on
  Q4_K_XL. The fork is 10-12% shorter everywhere. At 2K this repo's patches take 4% off stock
  and the fork takes 13-16%, so most of the fork's short-context gain comes from parts this
  repo does not have (its MMVQ parameters per weight type, fused kernels and scheduling).
- **At 120K on two cards stock is at 109 ms per step**, against 68 ms on the R9700 alone
  (`docs/shortlist.md`). The split of that between the cards was not measured. Both patch sets
  remove most of it: this repo's to 70 ms, the fork's to 63 ms.
- **The shortlist with draft 4** adds +11-22% to this repo's build at 2K and +3-13% at 120K.
  At 2K that is ahead of fork-n3 in three of four cells (behind by 4% on Q6_K explain). At 120K
  fork-n3 is ahead by 2-9% in three of four cells (behind by 2% on Q6_K code).
- **The fork's adaptive drafter** is the fastest row at 2K (+22% / +36% over stock on Q6_K,
  +34% / +38% on Q4_K_XL) and the wrong choice at 120K: it drafts deeper (565 tokens against
  477), and at depth each drafted token costs more than it returns, so it is 7-10% slower than
  the fork's fixed draft 3. This is the same effect as `docs/shortlist.md`'s draft 6 at depth.
- **Prefill is slower on the fork:** 667 against 761 tok/s on Q6_K (-12%), 719 against 936 on
  Q4_K_XL (-23%), at its defaults with `-ub 512`. Not investigated; the fork documents its own
  prefill settings and none were tried.
- **Text.** At temperature 0 the stock builds agree on all four outputs. This repo's build
  differs from stock on one of four (Q6_K) and two of four (Q4_K_XL), the fork on three of
  four. The kernels sum in a different order, which moves near-ties; `docs/depth.md`
  section 10 measured no code-quality difference for this repo's build.
- **Both changes together, against the preset today** (stock, UD-Q6_K): Q4_K_XL v3.1 on
  fork-n3 at 120K is 56.9 / 59.9 against 28.1 / 32.0 tok/s (2.0x / 1.9x); at 2K with the
  adaptive drafter 74.4 / 92.3 against 51.1 / 59.5 (+46% / +55%).

## Code quality on the fork

The code-quality test of local-llm-benchmarks (50 checks, executed), 12 seeds at the preset
sampler with thinking off, UD-Q4_K_XL v3.1, both cards, 128K, the fork with its adaptive drafter
(`benchmarks/qwen3.8-27B-UD-Q4_K_XL-v31-rdna-fork-r265/seeds/` in that repo):

| Build, UD-Q4_K_XL v3.1 | Throughput probe, tok/s | Draft accepted | 12 seeds, mean ± sd of 50 |
|---|---:|---:|---:|
| stock b12164, draft 3 | 65.1 | 83.4% | 38.4 ± 3.3 |
| fork r265, adaptive | 77.8 | 84.1% | 37.1 ± 4.0 |

No difference in code quality is shown (seeds range 30 to 46 on the fork, 31 to 44 on stock).
The fork is +20% on this probe, and +35% against the preset's UD-Q6_K on stock (57.8).

## Limits

- One run per cell, two tasks, 256 tokens: differences of 1-2 tok/s are not real.
- The fork ran at its defaults apart from the drafter type. Its draft-vocabulary option
  (`--spec-draft-mtp-vocab`), n-gram drafting, streamed KV and `-sm tensor` split were not
  tried, so its rows are a floor for what it can do here.
- This repo's shortlist was not combined with the fork's kernels; the two are separate builds.
- Only the 27B on two cards. No single-card, 9B or MoE rows.

## What this means for the repo

The fork contains this repo's two kernel fixes in a faster form and is maintained. For daily
use the fork is the build to run, not `patches/`. What remains this repo's own is the
measurement: the negative results on sparsity (F3, F5), the step-time split and the shortlist
numbers, which the fork's own option could now be checked against.
