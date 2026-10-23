# Dense to MoE: can a pretrained LLM be made sparse for faster local inference?

Experiments on converting the dense **Qwen3.5-9B** LLM into a sparse **Mixture-of-Experts (MoE)**
by partitioning its feed-forward (FFN) neurons, with custom **Triton** kernels on **AMD RDNA4 GPUs
(ROCm)**. The sparse variants are benchmarked against the practical alternatives for local
inference: **llama.cpp GGUF quantization** (Q8_0 to Q3_K_M) and **MTP speculative decoding**.
Quality is measured as **KL divergence** against the BF16 model, using the same protocol as
`llama-perplexity`.

Negative results are reported as negative results.

## Key results

**Training-free dense-to-MoE conversion doesn't pay off on this model.** The sparse kernels are
fast: a top-2-of-8 expert FFN runs 3.3x faster than the best dense kernel per layer at batch 1.
But even with a perfect (oracle) router, the quality loss is larger than 3-bit quantization.
Quantization plus MTP gives more speed at a fraction of the quality loss.

Mean KL divergence against llama.cpp BF16 (held-out text, 25,500 scored tokens), and decode speed on an
RX 9070 XT (llama-bench tg128, MTP via `llama-server --spec-type draft-mtp`):

| Variant | Mean KLD | Decode tok/s | With MTP |
|---|---:|---:|---:|
| PyTorch BF16 (implementation noise floor) | 0.0004 | | |
| Q8_0 | 0.0016 | 57.5 | 115.8 |
| Q6_K | 0.0028 | 70.3 | 117.4 |
| Q4_K_M | 0.0228 | 84.0 | 141.9 |
| **IQ4_XS** | **0.0184** | **89.6** | **148.0** |
| Q3_K_M | 0.0581 | 88.9 | |
| MoE 8 experts, top-6, oracle routing | 0.117 | | |
| MoE 8 experts, top-4, oracle routing | 0.376 | | |
| Per-neuron oracle, 50% of FFN neurons kept | 0.0103 | | |

- Expert partitions fitted on activations (balanced k-means on co-activation and contribution
  features) beat contiguous and random ones by about 20%, but nowhere near enough.
- Per-neuron selection is far better than fixed experts. Important neurons are token-specific
  and don't form stable groups, but the per-neuron oracle needs the full FFN computed first.
- Removing FFN bytes entirely from IQ4_XS would cap out at about 1.2x speedup (measured with
  FFN-only re-quantization proxies), below what MTP already gives.

Full numbers and methods: [docs/findings.md](docs/findings.md). Every command that was run, in
order, with descriptions: [docs/writeup.md](docs/writeup.md).

**Phase 3, predictive fine-grained neuron sparsity (Deja Vu / TEAL / CATS / WINA-style), doesn't
pay off either.**
- Quality only survives at single-neuron granularity: 8-neuron blocks already cost 6x
  IQ4_XS's KLD, even with a perfect choice.
- Training-free input sparsity (TEAL) at 75-90% of channels is nearly free in quality (KLD
  0.0093-0.0011), but the gain is small:
  - At most +4-12% single-token decode, even with an ideal kernel.
  - About 0% with MTP: the verify step checks ~4 tokens, and their combined masks cover the
    whole FFN.

Report: [docs/phase3_findings.md](docs/phase3_findings.md). Pre-registered gates and literature
review: [docs/phase3_research.md](docs/phase3_research.md).

**MTP tuning on the R9700:** llama.cpp's defaults are already near-optimal.
- Six-domain averages: Q6_K + MTP 114 tok/s, IQ4_XS + MTP 136 tok/s (BF16 without MTP: 34 tok/s).
- The kind of text matters more than any setting: 115 tok/s on prose vs 172 on JSON.
- For editing workloads that copy their input, adding n-gram drafting (`--spec-type draft-mtp,ngram-mod`)
  reaches 5x plain decoding (IQ4_XS 451 tok/s on the R9700).
- Details: [docs/mtp_tuning.md](docs/mtp_tuning.md).

## How the experiments fit together

Calibration data fits everything; every quality number is scored on held-out text against one
llama.cpp BF16 reference, so PyTorch sparse variants and GGUF quants share one table. Details and
the FFN-to-experts diagram: [docs/architecture.md](docs/architecture.md).

```mermaid
flowchart TB
    HF[("Qwen3.5-9B · HF checkpoint · BF16")]
    SRC["6 domains: prose, math, instructions, C++, Python, JSON"] --> SPLIT{"hash split per document<br/>(calibration/data.py)"}
    SPLIT --> CAL["calib 60%"]
    SPLIT --> VAL["val 10%"]

    subgraph torch["PyTorch + Triton on ROCm"]
        direction TB
        ACT["FFN-input activations, 32 layers<br/>collect_activations.py"] --> PART["fit partitions<br/>contiguous / random / k-means"]
        PART --> ORA["per-layer oracle bound<br/>oracle_analysis.py"]
        PART --> SPARSE["full model with sparse FFNs<br/>oracle MoE, per-neuron"]
        KER["Triton decode kernels<br/>kernel_latency.py"]
    end

    subgraph llama["llama.cpp on HIP"]
        direction TB
        GGUF["BF16 GGUF"] --> IMAT["imatrix"] --> QUANTS["Q8_0 … Q3_K_M<br/>FFN-only proxies"]
        GGUF --> BASE["KLD base file<br/>BF16 log-probs"]
        QUANTS --> BENCH["llama-bench · llama-server + MTP"]
    end

    HF --> ACT & SPARSE & KER & GGUF
    CAL --> ACT
    CAL --> IMAT
    VAL --> ORA
    VAL --> BASE
    BASE --> SCORE{{"one scorer: same tokens, same KL formula<br/>sparse_kld_vs_gguf.py · gguf_kld.sh"}}
    SPARSE --> SCORE
    QUANTS --> SCORE
    SCORE & BENCH & KER & ORA --> FIND["docs/findings.md: quality vs speed"]
```

## What's in the repo

- **Exact dense-to-MoE partitioning** of SwiGLU FFNs (contiguous, random, balanced k-means
  strategies), with tests proving the partitioned FFN reconstructs the dense output exactly.
- **Oracle and baseline routers:** norm top-k, greedy and exhaustive reconstruction, static
  (structured pruning), per-neuron top-k.
- **Triton decode kernels** for gathered expert GEMVs with a fused router. They are CUDA-graph
  capturable on ROCm, with no host sync and no weight copy.
- **Calibration pipeline:** six domains (prose, math, instructions, C++, Python, JSON) from
  permissively licensed sources, with hash-based disjoint calibration/validation/test splits.
- **Evaluation:** full-model KL divergence scored against llama.cpp's `--kl-divergence-base` files.
  This puts PyTorch sparse variants and GGUF quants in one table.
- **llama.cpp benchmarks:** imatrix quantization, KLD, llama-bench and MTP throughput scripts.
- **ROCm pitfalls with repro scripts:**
  - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` silently corrupts tensors on
    torch 2.14.1+rocm7.2 (`scripts/repro_expandable_segments.py`).
  - hipBLASLt `torch.bmm` silently returns wrong values for batch index >= 65536
    (`scripts/repro_bmm_batch_limit.py`).
  - The iGPU must be hidden or PyTorch segfaults on init.

## Hardware and software

- **GPUs:** AMD Radeon RX 9070 XT (16 GB) and Radeon AI PRO R9700 (32 GB), RDNA4 (gfx1201).
- **Software:** ROCm 7.2, PyTorch 2.14.1+rocm7.2, Triton, llama.cpp b11434 (ROCm/HIP backend).
- **Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), revision `c202236`. It's a
  hybrid Gated DeltaNet / attention model with SwiGLU FFNs and an MTP head.

## Setup

```bash
uv sync                      # Python 3.12 venv, torch 2.14.1+rocm7.2
source scripts/env.sh        # REQUIRED: hides the iGPU; never enable expandable_segments
.venv/bin/python -m pytest   # GPU tests need cuda:0
```

## Reproduce

```bash
.venv/bin/python benchmarks/kernel_latency.py --device cuda:0          # F2 kernel latency
.venv/bin/python scripts/collect_activations.py --split calib          # needs local model snapshot
.venv/bin/python scripts/collect_activations.py --split val --per-domain 4 --seed 1
.venv/bin/python scripts/oracle_analysis.py --layers all               # F3 per-layer oracle bound
.venv/bin/python scripts/oracle_e2e.py --device cuda:1                 # F3 end-to-end
# F4 (GGUF build steps in docs/findings.md F4 and experiments/logs/):
.venv/bin/python scripts/export_corpus.py                              # cache/corpus/{calib,val}.txt
benchmarks/gguf_kld.sh base val && benchmarks/gguf_kld.sh score val IQ4_XS
.venv/bin/python scripts/sparse_kld_vs_gguf.py --corpus val            # sparse variants, same base
benchmarks/gguf_mtp.sh 0 Q8_0 IQ4_XS && .venv/bin/python scripts/gguf_summary.py
```

## Documentation

| Doc | Contents |
|---|---|
| [docs/findings.md](docs/findings.md) | All measured results (F0 to F4) |
| [docs/writeup.md](docs/writeup.md) | Result summary and a runbook of every command run |
| [docs/phase3_findings.md](docs/phase3_findings.md) | Phase 3: block granularity, TEAL/WINA/CATS predictors, MTP union, speed ceilings |
| [docs/mtp_tuning.md](docs/mtp_tuning.md) | MTP draft length, confidence cut-off, n-gram, temperature sweep on the R9700 |
| [docs/phase3_research.md](docs/phase3_research.md) | Phase 3 literature (Deja Vu, PowerInfer, DSMoE, TEAL, WINA, BlockFFN, Prox) and gates |
| [docs/research.md](docs/research.md) | Literature survey: MoE upcycling, MoEfication, CMoE, TEAL, CATS, Deja Vu |
| [docs/architecture.md](docs/architecture.md) | Partitioning, routing, kernels, code map |
| [docs/environment.md](docs/environment.md) | Hardware, ROCm and model inspection |
| [docs/roadmap.md](docs/roadmap.md) | Next options and their expected value |
| [requirements.md](requirements.md) | The original research brief |

`.ai/project.md` and `.ai/task.md` hold the state an AI agent needs to continue the work.

## License

Code and documentation in this repository: [MIT](LICENSE). The Qwen3.5-9B weights are not
included and are covered by their own license, and the calibration text comes from the sources
listed in `src/dense_to_moe/calibration/data.py` under their own licenses.
