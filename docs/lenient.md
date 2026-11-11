# Lenient draft acceptance (2026-11-11)

Speculative decoding verifies each drafted token against the target and keeps it only if it is
the token the target picks. This test relaxes that: a draft token the target did not pick is
still kept if the target gives it at least `r` times the probability of its top token. It is the
first change in this project that alters the output, so it is measured for both speed and
quality.

**Result:** at `r = 0.3` the 27B decodes +10% faster at temperature 0 (+14-18% on prose,
instructions and code explanation, +2-3% on math and JSON) and +17% at temperature 1.0. The
code-quality suite shows no loss within its noise. 1-6% of output tokens are then not the
target's first choice.

## Method

- `patches/llama.cpp-spec-lenient.patch` (on b12164, `common/sampling.cpp`): two environment
  variables. `SPEC_ACCEPT_LOG=<file>` logs, per verify step, the target's probability of every
  draft token and of its own top token. `SPEC_LENIENT_RATIO=r` turns the rule on. Probabilities
  are the raw softmax, before the sampler chain. End-of-generation tokens are never accepted or
  replaced leniently. Off by default; with both unset the code path is unchanged.
- Speed: `benchmarks/lenient_probe.py`. Qwen3.8-27B UD-Q6_K, MTP draft 3, R9700, 8K context,
  the six domain prompts of `mtp_tune.py`, 512 tokens each, thinking off. The build also has the
  MMVQ and attention patches. Tables: `experiments/results/lenient/summary.md`
  (`scripts/lenient_summary.py`).
- Quality: the code-quality suite of local-llm-benchmarks (7 tasks, 50 checks) on the daily
  two-card preset, 12 seeds at the preset sampler and one run at temperature 0
  (`benchmarks/adopt_check_seeds.sh` with `BUILDS=lenient0.3`), against the exact runs of
  `docs/depth.md` section 8.

## Results

### 1. Sizing from the log of an exact run

Each verify step's draft is re-scored under the rule. This ignores that the text changes
afterwards. Tokens per verify step, relative to exact:

| domain | exact tok/step | r=0.7 | r=0.5 | r=0.3 | r=0.1 |
|---|---:|---:|---:|---:|---:|
| code_python | 3.45 | 1.03 | 1.06 | 1.07 | 1.10 |
| code_cpp | 2.97 | 1.05 | 1.07 | 1.11 | 1.18 |
| prose | 2.72 | 1.03 | 1.06 | 1.11 | 1.20 |
| math | 3.65 | 1.00 | 1.01 | 1.02 | 1.03 |
| instruction | 2.73 | 1.04 | 1.10 | 1.14 | 1.22 |
| json | 3.83 | 1.01 | 1.01 | 1.02 | 1.03 |

At 48% of first rejections the draft token has under a tenth of the top token's probability:
half of the rejections are real misses, not near-ties.

### 2. Measured speed, temperature 0

| domain | exact | r=0.5 | r=0.3 | r=0.1 |
|---|---:|---:|---:|---:|
| code_python | 58.9 | 62.4 | 63.0 | 64.4 |
| code_cpp | 50.9 | 52.1 | 57.9 | 57.1 |
| prose | 46.5 | 48.6 | 53.6 | 53.6 |
| math | 62.2 | 61.1 | 64.2 | 64.3 |
| instruction | 46.6 | 50.0 | 54.9 | 55.7 |
| json | 65.0 | 65.8 | 66.2 | 66.6 |
| **harmonic mean** | **54.0** | **55.9** | **59.6** | **59.9** |
| vs exact | | +3.4% | +10.3% | +10.8% |

- r=0.3 is the useful setting. r=0.1 is no faster and roughly doubles the share of non-top
  tokens (prose 6.4% -> 11.7% of the output), unlike the projection: after an unlikely token the
  next drafts are accepted less often.
- At r=0.3 the measured gain is at or above the projection (prose +15% against +11%).
- Share of output tokens that are not the target's top choice at r=0.3: code_python 2.5%,
  code_cpp 4.5%, prose 6.4%, math 1.2%, instruction 5.1%, json 0.6%. Their mean probability is
  0.5-0.7 of the top token's. The text departs from the exact run's within 13-240 characters.

### 3. Temperature 1.0 (one seed)

| config | harmonic mean tok/s | vs exact | non-top-rule tokens |
|---|---:|---:|---:|
| exact | 49.4 | | 0 |
| lenient r=0.3 | 57.9 | +17% | 7.3% of output |
| exact, `--spec-draft-sampling probabilistic` (lossless) | 53.2 | +8% | 0 |

- When sampling, the target often picks a token other than the draft, so the rule fires more
  often. Each time it replaces the sampled token with the drafted (most likely) one, which
  narrows the sampling distribution, like a lower temperature.
- One seed per config, and the three runs produce different text, so these have a few percent
  of noise. The lenient rule and probabilistic draft sampling were not combined (the patch only
  covers the matching path).

### 4. Code quality on the daily preset

| run | exact (patched build) | lenient r=0.3 |
|---|---|---|
| temperature 1.0, seeds 1-12, scores | 36 42 34 41 35 35 41 39 35 40 35 43 | 29 42 42 35 44 44 38 41 41 37 40 36 |
| mean, sd | 38.0, 3.3 | 39.1, 4.3 |
| temperature 0, one run | 41 | 40 |

- No loss is measurable. With 12 seeds and a spread of 3-4 points the test would not see a
  difference under about 3 points (6% of the checks).
- Per task the means move both ways (`urljoin` 4.4 -> 5.3, `textwrap` 1.7 -> 2.8, `csv_rows`
  7.0 -> 5.8: two lenient seeds scored 0/7 on it, none of the exact ones did).

## What this means

1. Lenient acceptance at r=0.3 is worth about +10% at temperature 0 and +17% at temperature 1.0
   on mixed text, concentrated in prose, instructions and explanations. On code, math and JSON,
   where drafts already pass, it gives 2-7%.
2. It is not lossless. The code-quality suite cannot tell it from exact verification, but that
   suite is coarse; there is no measurement on prose quality, long outputs or tool calls.
3. It stays an opt-in experiment (environment variable). It is not suggested for the daily
   presets on this evidence.

## Limits

- One model, one quant, MTP at draft length 3, 8K context, six prompts, 512 tokens each. One run
  per cell at temperature 0, one seed at temperature 1.0.
- Quality was measured only with the 50-check code suite. No KLD-style measure exists for this
  rule: it changes which token is taken, not the target's distribution.
- Thinking was off everywhere.
- The rule compares raw probabilities, ignoring the sampler chain (top-k, top-p, grammar). A
  grammar-constrained request could accept a token the grammar forbids; not tested.
