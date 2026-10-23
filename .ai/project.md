# Project state (read this first)

Goal: decide, by measurement, whether Qwen3.5-9B can be converted into a sparse MoE that decodes
substantially faster at acceptable quality. Full brief: `requirements.md`.

## Where things are
- `docs/environment.md`: hardware, software, model architecture, theoretical ceilings
- `docs/research.md`: literature, recommended architecture, go/no-go gates
- `docs/architecture.md`: code layout and invariants
- `docs/findings.md`: measured results so far (source of truth for numbers)
- `docs/phase3_research.md`, `docs/phase3_findings.md`: Phase 3 (predictive neuron sparsity; verdict: stop)
- `.ai/task.md`: current task, next steps
- `experiments/results/*.json`: raw results, each with env + git commit

## Ground rules learned so far
- `source scripts/env.sh` before anything (iGPU crash otherwise). cuda:0 = 9070 XT 16 GB, cuda:1 = R9700 32 GB.
- Model snapshot: HF cache (physically on /mnt/fast), revision c202236235762e1c871ad0ccb60c8ee5ba337b9a.
  Load as `AutoModelForCausalLM` (gives text-only `Qwen3_5ForCausalLM`, 16.7 GiB bf16).
- Latency benchmarks must rotate >= 8 layer copies (64 MB Infinity Cache inflates sparse results ~20%).
- Compare against the best dense kernel (Triton GEMV, ~516 us/layer), not torch F.linear (~680 us).
- Activations cached in `cache/activations/{calib,val}` (gitignored, regenerate with scripts/collect_activations.py).
- hipBLASLt `torch.bmm` is wrong for batch index >= 65536 (F5a): keep batch dims small or chunk.
- HF parameters have requires_grad=True: compute cached weight statistics under torch.no_grad().
- Phase 3 KLD protocol: scripts/phase3_e2e_kld.py, same base file as F4 (~57 min for 18 variants).
