# Environment and model inspection (step 1)

Inspected 2026-10-22. Re-run the commands in the "How measured" notes to refresh.

## Hardware

| Component | Spec | Notes |
|---|---|---|
| GPU 0 | AMD Radeon RX 9070 XT, gfx1201 (RDNA4), 16 GB | ROCm index 0 |
| GPU 1 | AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB | ROCm index 1. Holds full BF16 model (19.3 GB) |
| GPU 2 | iGPU gfx1036, 512 MB | Ignore. Exclude via `HIP_VISIBLE_DEVICES` |
| CPU | Ryzen 7 9800X3D, 8C/16T, AVX-512 | |
| RAM | 30 GiB usable (32 GB DDR5-6000) + 30 GiB swap | Cannot hold FP32 copy of the model (~36 GB) |
| Disk | 606 GB free on `/` (SATA SSD, ~540 MB/s) | Cold-loading 19 GB takes ~35 s |
| Network | huggingface.co reachable | |

## Software

| Item | Version | Status |
|---|---|---|
| OS | CachyOS, kernel 7.2.9 | |
| ROCm | 7.2.4 (`/opt/rocm`) | |
| System Python | 3.14.8 | Too new for PyTorch ROCm wheels |
| uv | installed; Python 3.12.8 available | Use `uv` + Python 3.12 venv |
| PyTorch | 2.14.1+rocm7.2 (HIP 7.2.53211), in `.venv` via `uv sync` | Installed 2026-10-22 |
| Transformers | 5.20.2 | Has `Qwen3_5MLP`, `Qwen3_5GatedDeltaNet`, `Qwen3_5ForCausalLM` |
| triton-rocm | 3.8.0 | |
| flash-linear-attention / causal-conv1d / flash-attn | not installed | GatedDeltaNet uses the slow PyTorch fallback. Fine for calibration; revisit for HF-side benchmarks |
| CUDA | none (AMD only) | Triton/HIP kernels, not CUDA |
| llama.cpp | `~/llama.cpp`, `5e03bdd87` (b11434), ROCm build in `build/bin` | Has `qwen35` and `qwen35moe` archs |

## PyTorch GPU setup (measured 2026-10-22)

**`source scripts/env.sh` first.** With the gfx1036 iGPU visible, `torch.cuda` init segfaults (exit 139).
`HIP_VISIBLE_DEVICES=0,1` fixes it. Each import also prints a few harmless
`(null): No such file or directory` lines from the ROCm native library load.

| torch device | Card | VRAM | BF16 GEMM 8192^3 | Device copy BW (read+write) |
|---|---|---|---|---|
| `cuda:0` | RX 9070 XT | 15.9 GiB | 118.4 TFLOP/s | 496 GB/s |
| `cuda:1` | Radeon AI PRO R9700 | 31.9 GiB | 123.2 TFLOP/s | 503 GB/s |

These are single quick runs, not repeated measurements. They're sanity checks, not benchmarks.

## Selected model: `Qwen/Qwen3.5-9B`

- HF revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0, not gated.
- `Qwen3_5ForConditionalGeneration` (multimodal: text + vision tower + 1 MTP layer).
- BF16 safetensors, 4 shards, 19,306,216,416 bytes total.
- Requires a Transformers version with `qwen3_5` support (config written by 4.57.0.dev0).

### Text model architecture

| Field | Value |
|---|---|
| Layers | 32, hybrid: 3x `linear_attention` (Gated DeltaNet) then 1x `full_attention`, repeating (24 linear, 8 full) |
| hidden_size | 4096 |
| FFN | SwiGLU (`silu(gate_proj(x)) * up_proj(x)` then `down_proj`), **intermediate 12288**, no biases, present in all 32 layers (`mlp_only_layers: []`) |
| Full attention | 16 Q heads, 4 KV heads, head_dim 256, q/k RMSNorm, output gate (`attn_output_gate`), partial RoPE (0.25), mRoPE |
| Linear attention | Gated DeltaNet: 16 K heads x 128, 32 V heads x 128, conv kernel 4 |
| Vocab | 248,320, embeddings **untied** |
| Context | 262,144 native |

Tensor names: `model.language_model.layers.{i}.mlp.{gate,up}_proj.weight` `[12288, 4096]`,
`...mlp.down_proj.weight` `[4096, 12288]`.

### Parameter breakdown (from safetensors headers)

| Group | Params | Share |
|---|---|---|
| FFN (32 layers) | 4.832 B | 50.1% |
| Linear attention (24 layers) | 1.618 B | 16.8% |
| lm_head | 1.017 B | 10.5% |
| embed_tokens | 1.017 B | 10.5% |
| Full attention (8 layers) | 0.470 B | 4.9% |
| Vision tower | 0.456 B | 4.7% |
| MTP head | 0.243 B | 2.5% |
| **Total** | **9.653 B** | |
| Text only (no vision/MTP) | 8.954 B | FFN = 54.0% |

Per FFN layer: 151 M params (302 MB BF16). With 8 experts, each expert is 1536 intermediate neurons.

### Theoretical decode ceiling (NOT a measurement)

Decode is memory-bandwidth bound, so the useful upper bound is weight bytes read per token.
Per token, the model reads everything except the embedding table: lm_head (1.017 B) + attention
(2.088 B) + FFN fraction. That is 7.94 B for the dense model.

| Config | Weights read / token | Max speedup vs dense |
|---|---|---|
| Dense | 7.94 B | 1.00x |
| 8 experts, top-4 | 5.52 B | 1.44x |
| 8 experts, top-2 | 4.31 B | 1.84x |
| 16 experts, top-4 | 4.31 B | 1.84x |

Those are ceilings. They ignore router cost, kernel efficiency and KV-cache reads, which grow with depth.
Prefill batches many tokens and touches every expert, so prefill gains can only come from FLOPs,
and only with good grouped-GEMM kernels.

## Existing measurements to beat (from `~/code/local-llm-benchmarks`)

`Qwen3.5-9B-Uncensored` (a fine-tune of the same architecture), Q8_0, llama.cpp ROCm, RX 9070 XT,
`-fa 1`, `-ub 1024`, q8_0 KV:

| Depth | Prefill tok/s | Decode tok/s |
|---|---|---|
| 0 | 4353.5 | 56.6 |
| 32K | 1742.2 | 49.1 |
| 131K | 630.4 | 35.7 |
| 262K | 343.6 | 25.8 |

Because KV/state reads grow with depth and the FFN share of per-token bytes falls, any MoE decode
gain should shrink at long context. Measure this, don't assume it.

## llama.cpp compatibility (first look)

- `qwen35moe` exists (`src/models/qwen35moe.cpp`). It keeps the same hybrid attention and supports
  routed + optional shared experts. This is the natural target format.
- **Gating is hard-coded**: `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with `norm_w = true`. Top-k
  softmax weights are renormalized to sum to 1, then scaled by `expert_weights_scale`.
- Our exact-reconstruction invariant needs **unweighted** expert sums (weight 1 per selected expert).
  `qwen35moe` cannot express that as shipped. Options:
  1. A small patch so `qwen35moe` reads `expert_gating_func` / `expert_weights_norm` from GGUF, as
     `deepseek2`, `glm4-moe` and others already do. Then add a constant-weight mode.
  2. Train routers/experts to work under softmax-renormalized weights. This breaks the exact
     invariant, so recovery training would be needed.
  3. Fold the gating into experts (only possible for constant weights).
- Decide after the PyTorch prototype shows which gating semantics preserve quality.

## Implications for the plan

1. Create a Python 3.12 venv with uv and install PyTorch ROCm (gfx1201) + Transformers with `qwen3_5`.
   Verify the GPUs are visible before doing anything else.
2. The single-layer prototype doesn't need the full 19 GB download. One layer's FFN weights (302 MB)
   can be fetched by HTTP range request. Realistic activations need the full model, so download it
   once on the calibration step (19.3 GB; disk is fine).
3. Run on the R9700 (32 GB) for BF16 full-model work. Use the 9070 XT for the parallel
   benchmark/teacher process.
4. RAM (30 GiB) is the binding host constraint: load with `device_map`/safetensors streaming, never
   materialize FP32 on the CPU.
