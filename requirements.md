# Project: Dense-to-MoE Conversion for Faster Local LLM Inference

## Role

You are a principal AI research engineer specializing in transformer architectures, Mixture-of-Experts (MoE), model compression, GPU inference optimization, PyTorch, Hugging Face Transformers, and llama.cpp.

You have access to my local development workstation, terminal, filesystem, and potentially the internet.

Your task is to **research, design, implement, train, evaluate, and optimize a system that converts an existing pretrained dense language model into a sparse Mixture-of-Experts model**, with the primary objective of improving local inference performance while preserving as much of the original model's intelligence as possible.

This is an engineering and experimental research project, not a request for a theoretical explanation.

**Build a functioning, reproducible system.**

---

# 1. Project Objective

Convert a pretrained dense 9B language model into an MoE architecture by partitioning existing feed-forward network (FFN) weights into independently selectable experts.

The resulting model should:

1. Preserve the original pretrained weights wherever possible.
2. Maintain approximately the same total parameter count.
3. Activate substantially fewer FFN parameters per token.
4. Reduce inference computation.
5. Preserve coding, reasoning, and instruction-following capabilities.
6. Support additional training to recover quality lost through sparsification.
7. Eventually support efficient local inference, ideally through llama.cpp or another optimized runtime.
8. Be reproducible, configurable, and extensible to larger dense models.

The primary success metric is **actual end-to-end inference speed at comparable model quality**, not simply reduced theoretical FLOPs.

Do not assume that converting a dense model into an MoE will automatically improve performance.

Investigate and experimentally validate whether it does.

# 2. Initial Model

Preferred starting model: **Qwen3.5-9B**, subject to verification of the exact model identifier, availability, architecture, license, and compatibility.

Before implementation:

- Inspect the model's configuration.
- Identify its transformer blocks and FFN implementations.
- Determine the exact dimensions and parameter counts.
- Identify attention, normalization, and any hybrid architectural components.
- Check the installed PyTorch, Transformers, CUDA/ROCm, and inference runtime versions.
- Detect available GPUs, VRAM, system RAM, and compute capabilities.
- Determine whether the full-precision model can be loaded or whether memory-efficient processing is required.

Do not assume the architecture is identical to earlier Qwen generations.

If Qwen3.5-9B presents significant implementation obstacles, explain them and recommend an alternative dense model of approximately 8–9B parameters.

Do not silently change the selected model.

# 3. Research Phase

Before writing the converter, research existing approaches to dense-to-MoE conversion.

Investigate:

- Sparse Upcycling
- Dense-to-MoE conversion
- FFN neuron partitioning
- Activation-based expert clustering
- Shared versus routed experts
- Dynamic FFN sparsity
- Expert routing optimization
- Teacher-student knowledge distillation
- Expert load balancing
- Efficient sparse inference kernels
- Structured sparsity and block-sparse computation

Prioritize peer-reviewed papers, official implementations, and reproducible research.

Relevant starting points include:

- Sparse Upcycling
- DSMoE: Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs
- Research on analytical FFN-to-MoE restructuring
- Existing implementations of dense model upcycling

Verify the exact publications and claims.

Produce `docs/research.md` containing:

1. Relevant research and links.
2. Summary of each technique.
3. Applicability to our model.
4. Expected advantages and limitations.
5. Recommended architecture.
6. Justification for the selected approach.

Avoid implementing an approach merely because it appears promising theoretically.

# 4. Conversion Architecture

The proposed baseline is to partition each dense FFN into multiple disjoint experts.

For a gated FFN:

```python
hidden = activation(gate_proj(x)) * up_proj(x)
output = down_proj(hidden)
```

Partition the intermediate dimension into expert groups.

Each expert receives the corresponding slices of:

- Gate projection output weights.
- Up projection output weights.
- Down projection input weights.
- Relevant biases, if present.

The partition must preserve the original mathematical computation when all experts are active.

For example:

```text
Original dense FFN
        |
        v
  Intermediate neurons
        |
        v
  Partition into experts
        |
        +-- Expert 0
        +-- Expert 1
        +-- Expert 2
        +-- Expert 3
        +-- Expert 4
        +-- Expert 5
        +-- Expert 6
        +-- Expert 7
        |
        v
  Router selects Top-K
        |
        v
  Sum selected contributions
        |
        v
  FFN output
```

Start with:

- 8 experts per FFN.
- Top-2 routing.
- No duplicated expert weights.
- No additional shared expert initially.
- Original attention components unchanged.

These are experimental defaults, not mandatory final settings.

Support configurable expert counts and Top-K values.

## Critical invariant

When all experts are activated with their original contribution weights, the partitioned FFN must reconstruct the original dense FFN output within floating-point tolerance.

Implement automated tests verifying this invariant.

Do not proceed to sparse routing until reconstruction passes.

Also investigate whether routing should use unnormalized expert contributions rather than conventional softmax-weighted MoE outputs, because the original dense FFN is an additive sum of neuron contributions.

# 5. Expert Partitioning Strategies

Implement and compare at least three strategies.

### Strategy A: Contiguous partitioning

Divide the FFN intermediate neurons into equally sized contiguous groups.

Advantages:

- Simple.
- Deterministic.
- Easy to debug.
- Potentially favorable memory access.

Use this as the initial baseline.

### Strategy B: Random partitioning

Randomly assign neurons to experts while preserving group sizes.

Use deterministic random seeds.

This provides a control for determining whether sophisticated grouping actually helps.

### Strategy C: Activation-informed partitioning

Analyze neuron activations using representative calibration data.

Investigate:

- Neuron activation frequency.
- Activation magnitude.
- Co-activation patterns.
- Contribution to FFN output.
- Neuron similarity.
- Token-dependent activation patterns.

Cluster neurons into expert groups using an appropriate algorithm.

Be aware that neurons that activate together are not necessarily optimal expert groups.

Compare grouping strategies empirically.

# 6. Routing

Implement a lightweight router for each converted FFN.

Initial architecture:

```python
router_logits = router(hidden_states)
selected_experts = topk(router_logits, k=2)
```

The router must predict which expert groups are most useful for approximating the original dense FFN output.

Investigate multiple routing approaches:

1. Learned linear router.
2. Router trained from expert contribution statistics.
3. Router trained using output reconstruction loss.
4. Optional shared-expert routing.
5. Alternative routing strategies supported by research.

Initially freeze the original FFN weights.

Train only the routers.

Avoid adding conventional MoE softmax scaling without investigating whether it damages reconstruction.

Measure:

- Reconstruction MSE.
- Cosine similarity to dense FFN outputs.
- Relative output error.
- Expert utilization.
- Router overhead.
- Token-dependent routing behavior.

Investigate expert collapse and routing imbalance.

# 7. Calibration and Data Collection

Build a calibration pipeline that captures representative activations from the original dense model.

Use datasets containing:

- Programming tasks.
- Code completion.
- Software debugging.
- Mathematical reasoning.
- General instructions.
- Natural language.
- Structured outputs.

Use publicly available datasets with appropriate licenses.

Separate calibration, training, validation, and final evaluation data.

Avoid benchmark contamination.

Support configurable sample counts, sequence lengths, batch sizes, and random seeds.

Activation collection must be memory efficient.

Do not retain full-model activations unnecessarily.

Use streaming, hooks, incremental statistics, or layer-by-layer processing where appropriate.

# 8. Training and Distillation

Once sparse routing is functional, implement quality recovery.

Training stages:

### Stage 1: Router-only training

Freeze original model weights.

Train routers to approximate dense FFN outputs.

### Stage 2: Optional adapter training

Investigate LoRA or other parameter-efficient methods to improve expert outputs.

### Stage 3: Teacher-student distillation

Use the original dense model as the teacher.

Train the sparse model to match teacher behavior.

Investigate:

- Logit distillation.
- KL divergence.
- Intermediate hidden-state matching.
- FFN output reconstruction.
- Task-specific supervised fine-tuning.

Evaluate whether the training objective actually improves downstream performance.

Use mixed precision where supported.

Support gradient accumulation, checkpointing, resumable training, and configurable memory budgets.

Do not initiate large, expensive training runs without first estimating resource requirements.

# 9. Progressive Sparsification

Do not assume Top-2 of 8 experts will preserve acceptable quality.

Implement experiments with:

| Total experts | Active experts |
|---|---|
| 4 | 4 |
| 4 | 3 |
| 4 | 2 |
| 8 | 8 |
| 8 | 6 |
| 8 | 4 |
| 8 | 2 |
| 16 | 8 |
| 16 | 4 |

Start with dense-equivalent execution.

Progressively reduce active experts.

Track the trade-off between quality and speed.

Investigate layer-dependent sparsity, since different transformer layers may have different sensitivity to dropped FFN contributions.

Allow certain layers to remain dense if this provides a better performance-quality trade-off.

# 10. Inference Optimization

A reduced active parameter count is not sufficient.

Optimize actual inference execution.

Investigate:

- Efficient expert weight layouts.
- Contiguous expert storage.
- Batched expert execution.
- Fused kernels.
- GPU-friendly routing.
- Avoiding unnecessary CPU/GPU transfers.
- Kernel launch overhead.
- Memory bandwidth.
- KV-cache behavior.
- Quantization compatibility.
- Prefill versus decode performance.

Benchmark the following independently:

1. Original dense FFN.
2. Partitioned FFN with all experts active.
3. Sparse FFN with selected experts.
4. Complete dense model.
5. Complete converted model.

Compare single-token decode and prompt prefill separately.

If PyTorch sparse routing is slower than dense execution, identify whether the bottleneck is architectural or an implementation limitation.

Do not misrepresent theoretical FLOP reductions as measured speedups.

# 11. llama.cpp Integration

Investigate whether the converted model can eventually run through llama.cpp.

Determine:

- Whether an existing supported MoE architecture can represent our design.
- Whether custom routing semantics require changes.
- Whether GGUF can store the new expert layout.
- Whether llama.cpp supports the required expert selection behavior.
- Whether quantization can be performed without invalidating the architecture.
- Whether custom GGML kernels are necessary.

Do not force the converted checkpoint into an incompatible existing MoE format.

If llama.cpp integration requires substantial changes, document the requirements and prioritize validating the PyTorch implementation first.

# 12. Benchmarking

Create an automated benchmarking suite.

Measure:

**Performance**

- Tokens per second.
- Time to first token.
- Prompt processing throughput.
- Per-token latency.
- Peak VRAM usage.
- Peak system RAM usage.
- GPU utilization.
- Memory bandwidth where measurable.
- Active parameter count.
- Estimated and measured compute reduction.

**Quality**

- Perplexity on held-out data.
- Teacher-student logit divergence.
- Coding benchmark accuracy.
- Instruction-following accuracy.
- Reasoning benchmark accuracy.
- Long-context behavior where practical.

**Correctness**

- Full-expert reconstruction.
- Weight mapping correctness.
- Deterministic expert partitioning.
- Checkpoint save/load consistency.
- Router correctness.
- Numerical stability.

Use repeated measurements, warm-up runs, identical generation settings, and consistent quantization where possible.

Report measurement variance.

Include both absolute performance and percentage changes relative to the original dense model.

# 13. Experiment Management

Every experiment must be reproducible.

Track:

- Model revision.
- Dataset revision.
- Git commit.
- Configuration.
- Random seed.
- Expert count.
- Active expert count.
- Partition strategy.
- Router architecture.
- Training parameters.
- Hardware configuration.
- Software versions.
- Performance results.
- Quality results.

Store structured results as JSON or CSV.

Generate Markdown summaries automatically.

Never rely on conversation history as the only record of experiment progress.

# 14. Repository Structure

Use a modular, maintainable Python project.

Suggested structure:

```text
dense-to-moe/
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── routing.yaml
│   └── training.yaml
├── src/
│   └── dense_to_moe/
│       ├── models/
│       ├── conversion/
│       ├── partitioning/
│       ├── routing/
│       ├── calibration/
│       ├── training/
│       ├── inference/
│       ├── evaluation/
│       └── utils/
├── scripts/
├── tests/
├── benchmarks/
├── experiments/
├── docs/
│   ├── research.md
│   ├── architecture.md
│   ├── findings.md
│   └── roadmap.md
└── .ai/
    ├── project.md
    └── task.md
```

Avoid monolithic scripts.

Use type hints, logging, configuration files, and clear interfaces.

# 15. Autonomous Engineering Workflow

Work autonomously, but incrementally.

Follow this sequence:

1. Inspect the environment.
2. Research relevant techniques.
3. Document the proposed architecture.
4. Implement a single-layer prototype.
5. Verify exact reconstruction with all experts active.
6. Implement sparse routing.
7. Measure single-layer approximation error.
8. Benchmark single-layer inference performance.
9. Evaluate whether the approach is promising.
10. Expand to multiple layers.
11. Implement full-model conversion.
12. Add router training.
13. Add distillation.
14. Benchmark quality and performance.
15. Investigate optimized runtime integration.

Do not skip validation steps.

After each milestone:

- Run relevant tests.
- Record results.
- Update documentation.
- Commit successful changes to Git.
- Identify the next experiment.

Maintain project continuity in repository files rather than depending on conversation context.

If interrupted, a fresh agent should be able to recover the current state by reading the repository.

Do not overwrite existing work or destroy model checkpoints.

Do not download very large models, datasets, or launch prolonged training runs without assessing disk space, network requirements, and compute costs.

# 16. Scientific Integrity

This project must be driven by measurements.

Do not claim success based only on:

- Reduced active parameter count.
- Reduced theoretical FLOPs.
- Successful checkpoint conversion.
- Lower FFN reconstruction error.
- Successful text generation.

Success requires measurable improvement in actual inference performance with acceptable quality retention.

If the hypothesis fails, report why.

Consider alternative approaches such as:

- Structured FFN pruning.
- Activation sparsity.
- Dynamic neuron selection.
- Quantization.
- Speculative decoding.
- Optimized dense inference kernels.

Compare these alternatives fairly against the proposed MoE conversion.

A well-supported negative result is preferable to an unsupported performance claim.

# 17. Immediate Task

Begin now.

First:

1. Inspect my workstation and available compute resources.
2. Verify the selected 9B model and its exact architecture.
3. Research the most relevant dense-to-MoE conversion techniques.
4. Produce an implementation plan.
5. Create the repository structure.
6. Implement the original FFN and an equivalent partitioned FFN.
7. Write automated tests demonstrating reconstruction equivalence.
8. Run the tests.
9. Implement a simple sparse Top-K prototype.
10. Benchmark reconstruction error and kernel-level latency.

**Do not begin full-model training yet.**

The first milestone is a working, validated, single-layer dense-to-MoE prototype.

Once that milestone is complete, present:

- What was implemented.
- Which tests passed.
- Measured reconstruction error.
- Measured latency.
- Identified bottlenecks.
- Whether the approach merits further investment.
- Recommended next steps.

Then continue with the next justified milestone, subject to the resource constraints above.

**The ultimate objective is to discover whether a pretrained dense 9B model can be transformed into a substantially faster sparse model without unacceptable loss of intelligence. Build the experiments necessary to answer that question.**
