# Two cards working on every layer: `-sm tensor` (2026-11-12)

The daily presets split the 27B over the R9700 and the RX 9070 XT by layer: the cards run one
after the other, each reading its own share of the weights. llama.cpp also has
`--split-mode tensor`, marked experimental in its help text, which splits each weight tensor
and the KV cache over the cards so both work on every layer. It had not been tried here. The
[RDNA fork](https://github.com/stew675/llama-cpp-rdna-boosts) documents it as its normal
multi-GPU mode; that is where this test picked it up.

**Result:** on stock llama.cpp, with nothing but the flag, decode is +43-50% at 2K of context
and +60-78% at 120K. On the fork the same flag gives 91 / 99 tok/s at 120K on UD-Q4_K_XL v3.1,
against 28 / 32 for today's preset (stock, UD-Q6_K, layer split): about 3x. This is the largest
single gain measured in this repo, and it is a flag, not a patch.

## Numbers

Both cards, 128K context, q8_0 KV, `-ub 512`, MTP draft 3, temperature 0, thinking off. Two
tasks over llama.cpp source (explain / code), 256 tokens, one run per cell.
`benchmarks/grid_tensor.sh`; layer-split rows from `benchmarks/grid_builds.sh` and
`grid_fork_tune.sh`; all tables in `experiments/results/grid/summary.md`.

UD-Q4_K_XL v3.1, decode tok/s:

| Build, split | 2K | 16K | 32K | 64K | 120K | ms per step at 120K |
|---|---:|---:|---:|---:|---:|---:|
| stock, layer | 55.5 / 67.0 | 53.6 / 62.0 | 44.2 / 52.8 | 39.4 / 41.1 | 28.9 / 33.8 | 102.2 |
| stock, tensor | 83.4 / 95.8 | 78.7 / 91.0 | 71.7 / 78.5 | 61.1 / 71.2 | 51.3 / 57.7 | 60.6 |
| this repo's patches, layer | 57.3 / 70.2 | | | | 50.7 / 54.1 | 62.9 |
| this repo's patches, tensor | 82.5 / 97.6 | 88.1 / 96.1 | 80.5 / 89.9 | 78.3 / 88.7 | 75.7 / 77.3 | 42.7 |
| fork r265, layer | 68.7 / 78.4 | 65.2 / 77.3 | 64.0 / 70.9 | 57.5 / 68.4 | 56.9 / 59.9 | 55.3 |
| fork r265, tensor | 93.0 / 123.7 | 110.7 / 120.4 | 100.1 / 111.0 | 94.7 / 108.2 | 90.9 / 99.0 | 34.5 |

UD-Q6_K (the preset file):

| Build, split | 2K | 16K | 32K | 64K | 120K | ms per step at 120K |
|---|---:|---:|---:|---:|---:|---:|
| stock, layer (today's preset) | 51.1 / 59.5 | | | | 28.1 / 32.0 | 109.1 |
| stock, tensor | 73.8 / 87.1 | 68.9 / 83.8 | 63.9 / 71.7 | 58.6 / 54.7 | 44.9 / 53.5 | 64.8 |
| fork r265, layer | 61.4 / 68.2 | | | | 45.6 / 55.7 | 62.7 |
| fork r265, tensor | 87.4 / 108.5 | 84.4 / 108.6 | 84.1 / 97.5 | 85.0 / 77.2 | 73.1 / 90.4 | 38.2 |

Prefill, tok/s over each depth range, UD-Q4_K_XL v3.1:

| Build, split | 2K to 16K | 16K to 32K | 32K to 64K |
|---|---:|---:|---:|
| stock, layer | 1,517 | 1,260 | 1,023 |
| stock, tensor | 1,395 | 1,206 | 1,003 |
| fork, layer | 1,254 | 1,013 | 792 |
| fork, tensor | 1,868 | 1,557 | 1,250 |

## Reading

- **The step at 2K goes from 53.5 to 37.5 ms on stock** (UD-Q4_K_XL). With the layer split the
  cards take turns, so a step costs the read of all the weights at one card's bandwidth at a
  time. With the tensor split both cards read at once. This is the bandwidth limit of the
  project's title being raised by using both cards' memory buses together.
- **Stock still loses with depth under the tensor split** (37.5 -> 60.6 ms from 2K to 120K):
  the attention kernel problem of `docs/depth.md` section 10 is unchanged. This repo's
  attention patch removes most of that (36.8 -> 42.7 ms) and so does the fork's
  (29.0 -> 34.5 ms). The kernel patches and the split mode multiply.
- **The fork's tensor split has a 19-21% shorter step than this repo's patches** at every
  depth (29.0 against 36.8 ms at 2K), more than under the layer split (10-12%). The fork
  carries its own work for this mode (an all-reduce between the cards among it,
  `ENVIRONMENT.md` there).
- **Prefill:** the tensor split costs stock 2-8% and gains the fork 49-58% over its own layer
  split. Fork with tensor is 22-24% faster than stock with layer at every range, which
  removes the fork's prefill loss recorded in `docs/fork_grid.md`.
- **VRAM is unchanged** (25.4-25.9 GiB peak on UD-Q4_K_XL, 29.2-29.7 on UD-Q6_K).
- **Draft length under the tensor split** (fork, UD-Q4_K_XL): draft 4 and the adaptive drafter
  are ahead on the code task up to 64K (139.6 and 143.7 against 123.7 tok/s at 2K) and behind
  draft 3 on both tasks at 120K (85.6 / 95.9 and 84.5 / 93.3 against 90.9 / 99.0). The fork's
  draft-vocabulary option does nothing here (92.0 / 100.1 at 120K): the fork's notes say the
  subset is copied to each card under this split while the full output layer is shared between
  them.

## Code quality

The code-quality test of local-llm-benchmarks (50 checks, executed), 12 seeds at the preset
sampler with thinking off, and its throughput probe on the same server
(`benchmarks/*-tensor/seeds/` in that repo):

| Config | Throughput probe, tok/s | Draft accepted | 12 seeds, mean ± sd of 50 |
|---|---:|---:|---:|
| stock, UD-Q6_K, layer (today's preset) | 57.8 | 84.2% | 37.3 ± 4.7 |
| stock, UD-Q6_K, tensor | 86.0 | 85.0% | 39.2 ± 5.3 |
| stock, UD-Q4_K_XL v3.1, layer | 65.1 | 83.4% | 38.4 ± 3.3 |
| stock, UD-Q4_K_XL v3.1, tensor | 92.4 | 84.3% | 35.8 ± 4.8 |
| fork r265, UD-Q4_K_XL v3.1, layer, adaptive | 77.8 | 84.1% | 37.1 ± 4.0 |
| fork r265, UD-Q4_K_XL v3.1, tensor | 121.8 | 85.4% | 37.8 ± 4.7 |

No effect of the split mode on code quality is shown. The largest gap between a tensor row
and its layer row is 2.6 checks (UD-Q4_K_XL on stock, lower with tensor), about 1.5 standard
errors; the UD-Q6_K pair moves the other way by 1.9. Seeds range from 26 to 48.

## What does not work

- **The integrated GPU must be hidden.** With the tensor split the fork takes the Ryzen's
  integrated GPU (gfx1036) as a third device and crashes at load. `HIP_VISIBLE_DEVICES=0,1`
  fixes it. Stock was only run with the variable set.
- **This repo's drafter shortlist patch is broken under the tensor split:** 2 of 2,012 drafts
  accepted, 21-26 tok/s. The patch copies rows of the output layer assuming one buffer on one
  device. The output is still correct (every draft is verified); the speed is not.
- Sampling runs on the CPU in this mode (the server logs that backend sampling is not
  supported with the tensor split). No cost was visible in the step times.
- `--fit` does not work with the mode; layers and context have to be set by hand.

## Limits

- One run per cell, two tasks. Explain-task rows move by several tok/s between runs because
  the text, and with it the accepted drafts, differ.
- Two cards of different size and speed on an x8 link each. Other pairs will split differently.
- The mode is experimental upstream. Only the 27B with MTP was run: no MoE model, no DFlash
  drafter, no parallel slots, no long session with context checkpoints.
