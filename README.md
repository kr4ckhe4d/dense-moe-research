# dense-to-moe

Experiments on converting the dense **Qwen3.5-9B** into a sparse Mixture-of-Experts for faster
local inference on AMD RDNA4 (RX 9070 XT + Radeon AI PRO R9700, ROCm 7.2). The brief is in
`requirements.md`.

**Current result:** see `docs/findings.md`. Batch-1 sparse expert kernels work (3.3x per layer at
top-2/8), but even oracle routing loses too much quality without training. The measured bar
(F4) is IQ4_XS + MTP at 148 tok/s (KLD 0.018) and Q8_0 + MTP at 116 tok/s (KLD 0.0016), on a
9070 XT. Training-free MoE is dominated on both quality and speed. Next options are in
`docs/roadmap.md`.

## Setup
```bash
uv sync                      # Python 3.12 venv, torch 2.14.1+rocm7.2
source scripts/env.sh        # REQUIRED: hides the iGPU; never enable expandable_segments
.venv/bin/python -m pytest   # 63 tests (GPU tests need cuda:0)
```

## Reproduce
```bash
.venv/bin/python benchmarks/kernel_latency.py --device cuda:0          # F2 latency
.venv/bin/python scripts/collect_activations.py --split calib          # needs local model snapshot
.venv/bin/python scripts/collect_activations.py --split val --per-domain 4 --seed 1
.venv/bin/python scripts/oracle_analysis.py --layers all               # F3 per-layer
.venv/bin/python scripts/oracle_e2e.py --device cuda:1                 # F3 end-to-end
# F4 (GGUF build steps in docs/findings.md F4 and experiments/logs/):
.venv/bin/python scripts/export_corpus.py                              # cache/corpus/{calib,val}.txt
benchmarks/gguf_kld.sh base val && benchmarks/gguf_kld.sh score val IQ4_XS
.venv/bin/python scripts/sparse_kld_vs_gguf.py --corpus val            # sparse variants, same base
benchmarks/gguf_mtp.sh 0 Q8_0 IQ4_XS && .venv/bin/python scripts/gguf_summary.py
```

Docs: `docs/environment.md`, `docs/research.md`, `docs/architecture.md`, `docs/findings.md`,
`docs/roadmap.md`. Agent continuity: `.ai/project.md`, `.ai/task.md`.
