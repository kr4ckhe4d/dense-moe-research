# DFlash beyond short greedy prompts, the 9B, smaller GPUs, and the kernel patch elsewhere (2026-10-24)

Follow-up to `docs/fast27b.md`. There, DFlash2 with draft length 5 plus the RDNA4 MMVQ patch was
the fastest way to decode the 27B on one 32 GB card, on six short prompts at temperature 0. This
milestone tests where that holds:
- Does the patch slow anything else down?
- Does DFlash hold at long context, with sampling, on editing, and with several users?
- Does the 9B have a DFlash drafter, and does it help there too?
- Is DFlash worth it on cards too small for the model (the offload setups of `docs/offload.md`)?

Code:
- Kernel check: `benchmarks/mmvq_check.sh` -> `scripts/mmvq_check_summary.py` ->
  `experiments/results/mmvq_check/summary.md`.
- Upstream evidence: `benchmarks/mmvq_upstream.sh`, `benchmarks/mmvq_types.sh` (b11719 master),
  `scripts/mmvq_upstream_summary.py` -> `docs/upstream_mmvq.md`.
- Drafters, full GPU: `benchmarks/dflash_open_items.sh`.
- Drafters, offloaded: `benchmarks/offload_dflash_fit.py`.
- All drafter tables: `scripts/dflash_summary.py` -> `experiments/results/dflash/summary.md`.

## Method

- **Build:** llama.cpp b11434 with `patches/llama.cpp-rdna4-mmvq-multicol.patch`
  (`~/llama.cpp-mmvq/build`) for every drafter run. The stock control is `~/llama.cpp/build`, the
  same commit without the patch.
- **Drafters:**
  - 27B: DFlash2 Q8_0 (2.06 GB), as before.
  - 9B: DFlash for Qwen3.5-9B ([z-lab/Qwen3.5-9B-DFlash], 6 layers, 1.3B parameters), GGUF Q8_0
    from [AtomicChat/Qwen3.5-9B-DFlash-GGUF] (1.38 GB).
  - MTP: the in-file head, draft length 3.
- **Prompts and settings:**
  - Same harness as F6/F8 (`benchmarks/mtp_tune.py`); tok/s is the harmonic mean over the prompts.
  - Full-GPU runs: six domain prompts, 512 tokens, unless noted.
  - Offloaded runs: the offload milestone's 3 prompts (code, prose, JSON) x 256 tokens and its
    budget rule. llama-server's VRAM after a warmup request, minus the card's idle use, must be
    <= budget - 0.5 GiB. For each spec config, `-ngl` was lowered until it fits.
- The cards' VRAM was checked before and after every run; no other process used them.

[z-lab/Qwen3.5-9B-DFlash]: https://huggingface.co/z-lab/Qwen3.5-9B-DFlash
[AtomicChat/Qwen3.5-9B-DFlash-GGUF]: https://huggingface.co/AtomicChat/Qwen3.5-9B-DFlash-GGUF

## Results

### 1. The MMVQ patch helps or is neutral everywhere measured

Time for one batch of n tokens, patched / stock (llama-bench, 5 reps, model fully on the GPU;
< 1 is faster). Full table: `experiments/results/mmvq_check/summary.md`.

| GPU | model | 1 | 4 | 6 | 8 |
|---|---|---:|---:|---:|---:|
| R9700 | 9B Q8_0 | 1.00 | 1.00 | 0.99 | 0.95 |
| R9700 | 9B Q6_K | 1.00 | 0.93 | 0.83 | 0.82 |
| R9700 | 9B Q4_K_M | 0.99 | 0.91 | 0.86 | 0.85 |
| R9700 | 9B IQ4_XS | 0.98 | 0.99 | 0.94 | 0.93 |
| R9700 | Gemma 4 31B Q6_K (dense, other shape) | 0.98 | 0.88 | 0.78 | 0.77 |
| R9700 | Qwen3.6-35B-A3B UD-Q6_K (MoE) | 0.99 | 0.96 | 0.98 | 0.97 |
| R9700 | gpt-oss-20b MXFP4 (MoE) | 1.00 | 0.99 | 0.99 | 1.00 |
| RX 9070 XT | 9B Q6_K | 1.00 | 0.96 | 0.86 | 0.82 |
| RX 9070 XT | 9B Q4_K_M | 1.02 | 0.94 | 0.86 | 0.84 |
| RX 9070 XT | 27B Q3_K_M | 0.98 | 0.90 | 0.86 | 0.84 |
| RX 9070 XT | gpt-oss-20b MXFP4 (MoE) | 0.98 | 0.99 | 0.99 | 1.00 |

- Batch 1 is within +-2% everywhere (noise). The 1-token path is untouched.
- The RX 9070 XT gains the same as the R9700.
- MoE models gain 0-5%: their expert matmuls go through `MUL_MAT_ID`, mostly with one token per
  expert.
- `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` pass on the RX 9070 XT too.
- The Nemotron 3.5 74B-A9B file on disk does not load in b11434 (pruned to 240 experts), so it
  was skipped.

### 2. The 9B has a DFlash drafter, and it beats MTP by 12-29%

Qwen3.5-9B fully on the GPU, six domains (tok/s):

| quant | GPU | plain | MTP n=3 | DFlash n=3 | DFlash n=5 | DFlash n=7 |
|---|---|---:|---:|---:|---:|---:|
| Q8_0 | R9700 | 58.8 | 104.5 | 114.2 | 131.1 | **133.8** |
| Q6_K | R9700 | 70.8 | 118.6 | 128.5 | **132.6** | 120.5 |
| IQ4_XS | R9700 | 91.0 | 141.1 | 149.0 | **164.5** | 154.0 |
| Q8_0 | RX 9070 XT | 57.4 | 102.3 | 111.5 | 128.7 | **131.9** |
| Q6_K | RX 9070 XT | 69.3 | 117.3 | 126.6 | **132.6** | 122.2 |
| IQ4_XS | RX 9070 XT | 88.4 | 138.2 | 145.5 | **162.6** | 153.0 |

- Draft length 5 is the best default. Q8_0 prefers 7, because its verify batch is nearly free.
- With DFlash, Q8_0 (lossless) is as fast as Q6_K. Both cards perform the same.
- The gain comes from code, math and JSON (JSON up to 244 tok/s). On prose and instructions,
  DFlash n=5 is between 8% slower and 15% faster than MTP.

### 3. Long context: DFlash's lead shrinks to 3-6%

~4k / 16k / 28k tokens of llama.cpp source, then a question; 512 tokens out, R9700:

| model | plain | MTP n=3 | DFlash n=3 | DFlash n=5 |
|---|---:|---:|---:|---:|
| 27B UD-Q6_K | 23.3 | 50.6 | 51.2 | **53.4** |
| 27B IQ4_XS | 31.2 | 65.5 | 65.2 | **67.7** |
| 9B Q6_K | 65.8 | 110.8 | **116.0** | 115.3 |

Both drafters still give 1.6-2.2x at 28k tokens. DFlash's lead over MTP drops from ~20% on
short prompts to 3-6%: its acceptance at n=5 falls to 42-52%.

### 4. Sampling: DFlash keeps its lead at temperature 0.7 and 1.0

Six domains, temperature 0.7 (seeds 0, 1, 2) and 1.0 (seed 0); llama-server's default sampler
(top-k 40, top-p 0.95, min-p 0.05). R9700, tok/s:

| model | MTP n=3, t=0.7 | DFlash n=5, t=0.7 | MTP n=3, t=1.0 | DFlash n=5, t=1.0 |
|---|---:|---:|---:|---:|
| 27B UD-Q6_K | 51.6 | **60.3** (+17%) | 51.2 | **60.4** (+18%) |
| 27B IQ4_XS | 66.3 | **75.6** (+14%) | 66.5 | **75.5** (+14%) |
| 9B Q6_K | 114.9 | **131.5** (+14%) | 110.1 | **126.0** (+14%) |

- Sampled outputs differ per seed and diverge from the greedy text within the first few tokens,
  yet speed is close to temperature 0: MTP the same, DFlash 0-8% slower (27B IQ4_XS 81.9 -> 75.6).
  The default top-k/top-p/min-p cut-offs keep the target's choices close to its argmax, which the
  drafts predict.
- **`--spec-draft-sampling probabilistic` does nothing for DFlash.** Only the simple and MTP
  drafters read it (`common/speculative.cpp`). The runs with it were token-for-token identical.

### 5. Editing: DFlash + n-gram drafting reaches 5.4-5.7x

Return a source file with small additions (1,024 tokens out), R9700, tok/s:

| drafter | 27B UD-Q6_K | 27B IQ4_XS |
|---|---:|---:|
| plain | 24.2 | 32.6 |
| MTP n=3 | 65.4 | 84.6 |
| DFlash n=5 | 75.5 | 96.7 |
| n-gram (`ngram-mod`) | 66.3 | 92.1 |
| MTP + `ngram-mod` | 128.6 | 173.6 |
| **DFlash + `ngram-mod`** | **137.3** (5.7x) | **177.0** (5.4x) |
| DFlash + `ngram-map-k` | 110.5 | 142.2 |

DFlash + `ngram-mod` is 2-7% ahead of MTP + `ngram-mod`. Most of the gain comes from the n-gram
part: whenever the output copies the input, n-gram drafts are long and almost always accepted.

### 6. Four concurrent requests: DFlash's lead shrinks to 3-6%

Four llama-server slots, six domain prompts sent in four-request groups; aggregate tok/s, R9700:

| model | plain | MTP n=3 | DFlash n=5 |
|---|---:|---:|---:|
| 27B UD-Q6_K | 74.7 | 109.8 (1.47x) | **116.1** (1.55x) |
| 27B IQ4_XS | 96.8 | 148.0 (1.53x) | **151.8** (1.57x) |

With four sequences, the verify batch is already 4 x (draft + 1) tokens, past the cheap region
of the kernel. Longer drafts buy less.

### 7. Partly offloaded: MTP is better almost everywhere

RX 9070 XT at a VRAM budget, same budget rule for every column, patched build; tok/s (`-ngl`):

| model | VRAM | plain | MTP n=3 | DFlash n=7, drafter on GPU | DFlash n=7, drafter in RAM |
|---|---|---:|---:|---:|---:|
| 9B IQ4_XS | 4 GB | 24.4 (20) | **41.2** (18) | 23.6 (0) | 29.4 (11) |
| 9B IQ4_XS | 6 GB | 88.3 (34) | **138.7** (34) | 45.6 (16) | 48.6 (26) |
| 9B Q6_K | 6 GB | 24.4 (25) | **48.4** (23) | 26.6 (11) | 31.0 (19) |
| 9B Q6_K | 8 GB | 69.3 (34) | **118.0** (34) | 47.1 (23) | 50.2 (30) |
| 9B Q8_0 | 8 GB | 24.7 (28) | **46.4** (26) | 38.0 (17) | 38.8 (23) |
| 27B IQ4_XS | 8 GB | 6.5 (31) | 12.7 (28) | 15.2 (17) | **16.2** (27) |
| 27B IQ4_XS | 12 GB | 11.5 (50) | 21.2 (47) | 22.0 (36) | **23.1** (45) |
| 27B IQ4_XS | 16 GB | 32.3 (66) | **70.6** (66) | 43.2 (56) | 46.8 (64) |
| 27B Q3_K_M | 12 GB | 16.9 (57) | **28.6** (54) | 19.4 (40) | 22.0 (51) |
| 27B UD-Q6_K | 8 GB | 3.8 (20) | 9.6 (18) | 10.1 (11) | **10.4** (18) |
| 27B UD-Q6_K | 16 GB | 7.6 (48) | **18.4** (45) | 17.6 (37) | 17.9 (44) |

n=5 rows, the 4 GB Q6_K row and the 8 GB Q3_K_M row: `experiments/results/dflash/summary.md`.
n=7 beats n=5 everywhere except the 9B Q6_K (n=5 better by up to 6 tok/s, still far below MTP).

- **The drafter costs VRAM the model needed.**
  - On the GPU, it takes 1.4-2.1 GB of weights plus its buffers: 9-20 fewer target layers than
    plain.
  - With `-ngld 0`, its weights stay in RAM, but llama-server still used ~0.4-1.4 GB more VRAM
    than plain, so 2-9 fewer layers.
  - MTP costs 0-3 layers.
  - At 16 GB, the 27B IQ4_XS stops fitting entirely once DFlash is added, and drops from
    70.6 (MTP) to 46.8.
- **A drafter in RAM is slow to run.** It reads its 1.4-2.1 GB of weights from RAM for every
  draft, at ~53 GB/s, so ~25-40 ms per draft.
- **DFlash wins only where the 27B is mostly in RAM:** IQ4_XS at 8 GB (+28%) and 12 GB (+9%),
  UD-Q6_K at 8 GB (+8%). There each verify already costs ~100+ ms of RAM reads, and DFlash's
  longer accepted runs are worth more than the layers it displaces. These are also the slowest
  setups (10-23 tok/s).
- **The patch does not change MTP under offload** (stock vs patched within +-3% when partly in
  RAM). The RAM reads dominate the verify step. Fully on the GPU it adds 2-6%.

## Recommendations

| Situation | Use |
|---|---|
| Model fully on the GPU, one user, chat/code/JSON | DFlash, `--spec-draft-n-max 5` (7 for Q8_0), plus the MMVQ patch on RDNA4 |
| Long documents in context (16k+) | DFlash n=5 or MTP n=3; within 3-6% of each other |
| Editing / returning code with small changes | `--spec-type draft-dflash,ngram-mod` (or `draft-mtp,ngram-mod`) |
| Several concurrent users | Either; DFlash +3-6% |
| Model partly in RAM (smaller card) | MTP n=3. DFlash only if the 27B is mostly in RAM, with `-ngld 0 --spec-draft-n-max 7` |
| Sampling (temperature 0.7-1.0) | Same choice as greedy. Don't bother with `--spec-draft-sampling probabilistic` for DFlash |

9B on a 16 GB card, fully on the GPU (RX 9070 XT, 162.6 tok/s at IQ4_XS, 131.9 at lossless Q8_0):

```bash
llama-server -m Qwen3.5-9B-Q8_0.gguf -ngl 99 -fa on -c 32768 -np 1 \
  --spec-type draft-dflash -md Qwen3.5-9B-DFlash-Q8_0.gguf -ngld 99 --spec-draft-n-max 7
```

## Caveats

- **The 9B drafter GGUF is a third-party conversion** (AtomicChat). Its acceptance (37-66% by
  draft length) is in line with the 27B's DFlash2, so the conversion works, but it was not
  checked against the original weights.
- **Offloaded runs use 3 prompts x 256 tokens.** The per-prompt spread is large (prose vs JSON),
  so differences under ~10% in the offload table are not significant. The 9B Q6_K n=5 vs n=7
  order flips between budgets, for example.
- **Why `-ngld 0` still needs ~0.4-1.4 GB of VRAM was not traced.** llama-server's default log
  level doesn't print buffer sizes. The number is from rocm-smi.
- **Long-context, sampling and concurrency runs are single runs per prompt.** The sampling table
  averages 3 seeds at t=0.7.
- **Simulated budgets** on the 16 GB card, as in `docs/offload.md`.
