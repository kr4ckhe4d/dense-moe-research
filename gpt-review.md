# Research Project — Phase 3: Predictive Neuron Sparsity for Qwen3.5-9B

## Your Role

You are a principal AI research engineer specializing in language model architecture, GPU kernel optimization, structured sparsity, conditional computation, quantization, knowledge distillation, Triton, ROCm and llama.cpp.

You are continuing an existing, experimentally validated research project.

Your task is to **investigate whether learned, input-dependent neuron selection can accelerate a pretrained dense LLM while retaining significantly more quality than fixed-expert MoE partitioning**.

This is a practical research and implementation assignment. Research, write code, run tests, measure results, and make evidence-based decisions.

Do not produce only a proposal.

Do not begin by rewriting the existing repository.

## 1. Existing project

Repository:

`~/code/dense-moe`

Primary model:

`Qwen/Qwen3.5-9B`

Hardware:

- AMD Radeon RX 9070 XT, 16 GB VRAM.
- AMD Radeon AI PRO R9700, 32 GB VRAM.
- ROCm 7.2.4.
- Approximately 30 GiB system RAM.

Software and existing infrastructure:

- PyTorch with ROCm.
- Triton kernels.
- llama.cpp.
- Hugging Face model checkpoint.
- GGUF quantization benchmarks.
- MTP speculative decoding.
- Activation calibration datasets.
- Oracle routing analysis.
- KL divergence evaluation.
- Automated test suite.

Inspect the environment to verify all details before relying on them.

Read these existing project files first:

- `README.md`
- `docs/research.md`
- `docs/architecture.md`
- `docs/findings.md`
- `docs/roadmap.md`
- `.ai/project.md`
- `.ai/task.md`
- Existing experiment results and benchmark scripts.

Also read the experiment write-up if available.

Preserve all previous results and working implementations.

## 2. Previous findings

The original project investigated partitioning Qwen3.5-9B's FFN neurons into conventional MoE experts.

It established that sparse GPU kernels can accelerate individual FFN layers, but quality deteriorates substantially when neuron selection is restricted to large, fixed expert groups.

Important measurements:

| Configuration | Mean KL divergence |
|---|---:|
| PyTorch BF16 versus llama.cpp BF16 | 0.0004 |
| Q8_0 | 0.0016 |
| Q6_K | 0.0028 |
| IQ4_XS | 0.0184 |
| Fixed-expert oracle MoE 8/6 | 0.117 |
| Fixed-expert oracle MoE 8/4 | 0.376 |
| Per-neuron oracle, 75% kept | 0.0013 |
| Per-neuron oracle, 50% kept | 0.0103 |

Existing GPU results:

- Dense optimized FFN kernel: 516 microseconds.
- MoE 8/4: 301 microseconds.
- MoE 8/2: 157 microseconds.

Existing practical inference results:

- IQ4_XS: 89.6 tokens/sec.
- IQ4_XS + MTP: 148.0 tokens/sec.
- Q8_0 + MTP: 115.8 tokens/sec.

These values were measured under particular experimental conditions. Do not assume they apply universally.

### Critical observation

At 50% retained FFN neurons, per-neuron oracle routing achieved dramatically lower KL divergence than fixed-expert routing.

This suggests that useful computational sparsity exists, but that conventional fixed expert groups are a poor representation of it.

### Critical limitation

The per-neuron oracle has access to information that may require computing the full FFN first.

Therefore, oracle quality does not imply an efficient inference implementation.

**Our next objective is to determine whether useful neurons can be predicted cheaply enough to reduce real inference latency.**

## 3. Revised research hypothesis

A pretrained dense FFN may contain token-dependent redundancy that can be exploited through fine-grained, structured, predictable sparsity.

Instead of converting the FFN into a conventional MoE with large experts, investigate smaller blocks of intermediate neurons.

A lightweight predictor should select useful blocks before performing the expensive FFN projections.

Desired properties:

1. Preserve existing pretrained FFN weights.
2. Avoid duplicating the entire FFN.
3. Use token-dependent activation selection.
4. Skip gate, up and down computations for inactive blocks wherever feasible.
5. Maintain GPU-friendly memory layouts.
6. Minimize predictor overhead.
7. Support quantized execution.
8. Preserve coding and reasoning ability.
9. Deliver measurable end-to-end inference gains.

Do not assume the hypothesis is correct.

## 4. Research tasks

Investigate the following research directions:

- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.
- PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU.
- DSMoE: Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs.
- Dynamic FFN sparsity.
- Activation prediction.
- Sparse gated MLP computation.
- Hot/cold neuron partitioning.
- Structured neuron selection.
- Predictive routing.
- Layer-adaptive sparsity.
- Quantization-aware sparse inference.

Also search for more recent relevant research and implementations.

Examine actual algorithms, code, hardware assumptions, and reported limitations.

Do not assume techniques developed for ReLU-based transformers transfer directly to Qwen3.5's gated FFNs.

Produce `docs/phase3_research.md` with findings and a recommended approach.

## 5. Stage A — Re-examine the existing oracle

Before implementing new algorithms, validate and extend the original per-neuron oracle experiments.

Inspect the existing oracle's selection objective.

Determine whether it selects neurons using:

- Activation magnitude.
- Individual neuron contribution norm.
- Per-token output reconstruction.
- Another heuristic or optimization method.

Establish exactly what the previous oracle proves and what it does not.

Reproduce the original measurements.

Add tests to verify the oracle implementation.

Investigate whether the oracle uses information that would be unavailable before executing the full FFN.

Create a clean distinction between:

1. Oracle selection quality.
2. Predictor selection quality.
3. Actual sparse execution efficiency.

Do not conflate these.

## 6. Stage B — Block granularity experiments

Investigate partitioning the FFN's intermediate dimension into small computational blocks.

Suggested block sizes:

- 1 neuron.
- 8 neurons.
- 16 neurons.
- 32 neurons.
- 64 neurons.
- 128 neurons.
- 256 neurons.
- 512 neurons.

Include the existing large-expert configurations as controls.

For each block size, evaluate multiple retained-neuron budgets:

- 90%.
- 75%.
- 62.5%.
- 50%.
- 37.5%.
- 25%.

Implement oracle block selection.

Test alternative objectives, including output contribution norms and reconstruction error.

Be careful: independently choosing blocks with the largest contribution norms is not necessarily equivalent to minimizing reconstruction error, because block outputs can cancel one another.

If exact combinatorial selection is infeasible, distinguish approximate greedy or heuristic oracles from mathematical optima.

Measure:

- FFN reconstruction error.
- Relative output error.
- Cosine similarity.
- End-to-end KL divergence.
- Layer sensitivity.
- Selection consistency across tokens.
- Theoretical FLOP reduction.
- Number and distribution of active blocks.

Use existing calibration and held-out validation datasets without mixing splits.

### Goal

Determine whether moderately sized blocks can approach the quality of per-neuron selection while remaining practical for GPU execution.

## 7. Stage C — Activation predictability

Analyze whether important neuron blocks can be predicted from information available before the FFN executes.

Investigate:

### A. Linear predictor

Train a lightweight linear mapping from the FFN input hidden state to block importance scores.

### B. Low-rank predictor

Investigate whether low-dimensional projections can predict sparse activation patterns at lower computation cost.

### C. Hot/cold blocks

Identify blocks that are frequently useful regardless of the token.

Keep hot blocks always active.

Predict only the remaining conditional blocks.

### D. Temporal and layerwise correlation

Investigate whether activation patterns correlate across consecutive tokens or adjacent layers.

Look for opportunities to reuse or predict activation masks.

### E. Variable computation budgets

Investigate whether different layers or tokens benefit from different amounts of active computation.

Evaluate fixed Top-K versus threshold-based selection.

For every predictor, measure:

- Recall of important blocks.
- Precision of selection.
- Output reconstruction quality.
- Full-model KL divergence.
- Prediction latency.
- Memory overhead.
- Additional parameter count.
- End-to-end computational cost.

Do not optimize prediction accuracy independently of model quality and latency.

A predictor with high mask accuracy may still select the wrong blocks for preserving useful model behavior.

## 8. Stage D — Sparse FFN kernels

Implement efficient GPU execution of selected FFN blocks.

The kernel must avoid expensive dense FFN work wherever possible.

Investigate:

- Prepacked block-major weights.
- Efficient sparse gather.
- Contiguous memory access.
- Quantized weight loading.
- Fused prediction and dispatch where appropriate.
- Kernel launch overhead.
- GPU memory bandwidth.
- Block selection and accumulation.
- Avoiding unnecessary host-device transfers.
- Prefill versus decode performance.
- CUDA Graph-compatible execution where supported by ROCm.

Reuse the existing Triton infrastructure.

Do not introduce a large dependency or runtime replacement without a clear reason.

Benchmark on both GPUs.

Compare against the existing optimized dense FFN kernel, not only PyTorch `F.linear`.

Test realistic cache conditions rather than repeatedly benchmarking a single cache-resident layer.

### Required measurements

For every candidate:

1. Predictor latency.
2. Sparse gate/up latency.
3. Sparse down latency.
4. Total routed FFN latency.
5. Original dense FFN latency.
6. Memory traffic where measurable.
7. Prefill performance.
8. Single-token decode performance.
9. Performance variation across active-block patterns.

The predictor overhead must be included in all end-to-end comparisons.

## 9. Stage E — Full-model evaluation

After identifying a promising single-layer implementation, integrate the approach into a full-model experimental runtime.

Preserve:

- Original attention mechanisms.
- Model normalization.
- Embeddings.
- Existing hybrid architecture components.
- KV-cache semantics.
- MTP compatibility where feasible.

Avoid architectural modifications unrelated to FFN sparsity.

Evaluate:

- Full-model KL divergence.
- Perplexity.
- Coding task quality.
- Reasoning task quality.
- Instruction following.
- Structured output.
- Tool calling.
- Long-context behavior.

Measure actual token generation throughput.

Do not extrapolate full-model speed from an individual FFN kernel without validation.

## 10. Quality recovery

Only if predictive sparsity demonstrates promising initial results, investigate additional training.

Start with:

1. Predictor-only training.
2. Layerwise reconstruction training.
3. Router distillation using the dense model.
4. Optional low-rank expert adaptation.
5. Joint sparse fine-tuning if justified.

Use the original dense model as a teacher.

Investigate multiple objectives:

- FFN reconstruction.
- Intermediate hidden-state matching.
- Final logit KL divergence.
- Task-specific supervised loss.
- Sparsity regularization.

Keep training costs controlled.

Estimate compute, memory and expected dataset requirements before substantial training.

Do not initiate large multi-billion-token training runs without explicit authorization.

## 11. Compare against practical baselines

The goal is not simply to outperform BF16 dense inference.

Compare with the existing usable local inference configurations.

Mandatory baselines:

- BF16 dense.
- Q8_0 dense.
- Q6_K dense.
- IQ4_XS dense.
- IQ4_XS + MTP.
- Q8_0 + MTP.

Where technically feasible, investigate combining predictive sparsity with quantization and MTP.

Preserve the distinction between model quality and decoding acceleration.

Do not report an unfair speedup obtained by comparing an optimized sparse implementation to an unoptimized dense reference.

Use identical contexts, generation settings, batch sizes, and sampling policies when appropriate.

## 12. Realistic success criteria

Define objective gates before running the experiments.

Suggested exploratory targets:

### Gate 1 — Oracle viability

Find a block size and sparsity budget that significantly reduces computation while approaching per-neuron oracle quality.

### Gate 2 — Predictability

Demonstrate that a lightweight predictor can approximate useful oracle selections without executing the full FFN.

### Gate 3 — Kernel viability

Demonstrate faster complete FFN execution including prediction and routing overhead.

### Gate 4 — Model quality

Demonstrate acceptable downstream task quality, not just low reconstruction error.

### Gate 5 — Practical inference

Demonstrate end-to-end performance improvement over an appropriately matched quantized dense baseline.

Stretch goal: beat IQ4_XS + MTP at comparable quality.

This is a research target, not a requirement to fabricate positive results.

If a gate fails, investigate the cause before proceeding.

It is acceptable to conclude that the architecture is unsuitable.

## 13. Additional investigation: Could we avoid a separate router?

Explore whether neuron or block importance can be estimated using cheaper signals already present in the transformer.

Possibilities include:

- Earlier-layer activations.
- Low-dimensional hidden-state projections.
- Previous-token selection patterns.
- Frequently active neuron statistics.
- Approximate or low-precision FFN projections.
- Lightweight learned sketches of gate/up projections.

Evaluate the trade-off between predictive accuracy and prediction cost.

Do not introduce a predictor that performs nearly as much work as the dense FFN.

## 14. Engineering requirements

Maintain the existing repository and history.

Use modular, tested implementations.

Keep experiments configurable.

Save:

- Experiment parameters.
- Hardware configuration.
- Model revision.
- Random seed.
- Dataset splits.
- Timing results.
- Quality results.
- Profiling information.
- Failure descriptions.
- Interpretation of findings.

Update:

- `docs/findings.md`
- `docs/roadmap.md`
- `.ai/project.md`
- `.ai/task.md`

Create new Phase 3 documents where necessary.

Make small, verifiable commits after successful milestones.

Never overwrite successful previous experiments.

Respect the existing environment configuration, including the documented ROCm allocator issue. Do not re-enable `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` without independently resolving and validating that problem.

Do not run multiple large GPU workloads concurrently unless resource availability has been verified.

## 15. Required final report

Produce `docs/phase3_findings.md`.

It must contain:

### A. Research summary

What was investigated and why.

### B. Experimental results

Tables comparing block sizes, sparsity budgets, predictors, output quality and speed.

### C. Hardware performance

Results for both the RX 9070 XT and R9700.

### D. Quality analysis

Where sparsity works, where it fails, and why.

### E. Practical inference comparison

Comparison against the best existing quantization + MTP configurations.

### F. Conclusions

Clearly distinguish:

- Demonstrated results.
- Estimated performance.
- Unproven hypotheses.
- Negative findings.
- Recommended future experiments.

### G. Recommendation

Answer the following question directly:

**Is predictive neuron sparsity a technically and economically worthwhile way to accelerate Qwen3.5-9B on this hardware?**

If the answer is no, explain the strongest alternative and stop investing in the failed architecture.

## 16. Immediate execution plan

Start now.

1. Inspect the existing repository and findings.
2. Reproduce and validate the per-neuron oracle.
3. Review relevant predictive sparsity research.
4. Implement block-level oracle selection.
5. Run a block granularity versus quality sweep.
6. Analyze which blocks are predictably important.
7. Train a minimal lightweight predictor.
8. Implement a single-layer predictive sparse FFN.
9. Benchmark actual latency including routing overhead.
10. Compare the results with the existing dense and MoE implementations.
11. Update research findings and Git history.

Do not begin full-model conversion or expensive distillation until the early experiments justify it.

Work autonomously, use available resources responsibly, and document every important finding.

If a proposed method fails, determine why and test the next most promising hypothesis.

**The central research question is no longer whether a dense model can be converted into a conventional MoE.**

It is:

**Can we exploit token-dependent neuron redundancy in a pretrained dense LLM using sufficiently accurate, low-overhead predictions to achieve meaningful real-world inference acceleration without unacceptable quality loss?**

Build the experiments necessary to answer that question.
