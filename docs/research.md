# Research: dense-to-MoE conversion for Qwen3.5-9B

Compiled 2026-10-22. Each claim gives its source. Numbers are the authors' own unless marked
"measured here". **Verification level** says how much of each paper I actually read:
*full text* (method and results sections read), *partial* (HTML read, some sections truncated), or
*abstract* (only abstract/listing read; treat details as unverified).

---

## 1. The question this research has to answer

Decoding a 9B dense model on our GPUs is memory-bandwidth bound (see `environment.md`). An MoE
conversion can only speed up decode by **reading fewer weight bytes per token**. So the question
is not "can we make an MoE", but:

> At a fixed bytes-read-per-token budget, does routed FFN sparsity lose less quality than the
> cheaper alternatives (lower-bit quantization, activation sparsity, speculative decoding)?

The literature mostly reports FLOPs or activated-parameter counts. Very few papers report
wall-clock decode speed, and none does so for a hybrid DeltaNet/attention model like Qwen3.5.

---

## 2. Techniques

### 2.1 MoEfication: split a pretrained FFN into experts
Zhang et al., *MoEfication: Transformer Feed-forward Layers are Mixtures of Experts*, Findings of ACL 2022.
[arXiv 2110.01786](https://arxiv.org/abs/2110.01786) · [ACL](https://aclanthology.org/2022.findings-acl.71) · code: thunlp/MoEfication. *Full text.*

- Partitions FFN neurons into experts, then trains routers. Expert construction compared: random,
  parameter clustering (balanced k-means on W1 columns), **co-activation graph partitioning (best)**.
  Routers compared: random, cosine similarity, **MLP router (best practical)**, and a
  **ground-truth oracle**, which is an upper bound.
- Results: 10-30% of FFN params at >95% of original performance on T5 / BERT GLUE-style tasks.
- **Measured speedup (T5-Large, V100): 1.20x at 25% neurons, 1.47x at 12.5%** on GPU, against
  2.0x/2.4x in FLOPs. GPU realised much less than FLOPs suggested; CPU came close.
- **Depends on ReLU sparsity.** GELU BERT had to be converted to ReLU by continued training first.
  This is the biggest concern for us: Qwen3.5 uses SiLU (SwiGLU), which has no exact zeros.

### 2.2 Sparse Upcycling: copy the FFN into experts, then train
Komatsuzaki et al., ICLR 2023. [arXiv 2212.05055](https://arxiv.org/abs/2212.05055). *Abstract.*
He et al. (NVIDIA), *Upcycling Large Language Models into Mixture of Experts*. [arXiv 2410.07524](https://arxiv.org/abs/2410.07524). *Abstract.*

- Each expert starts as a **copy** of the dense FFN. Total params grow by E times, while active params
  stay the same as the dense model. This buys capacity, not speed, and needs large training budgets
  (Nemotron-4 15B: 1T tokens).
- **Not applicable to our goal.** It violates "approximately the same total parameter count" and
  doesn't reduce active compute. NVIDIA's finding that *softmax-then-topK* beats *topK-then-softmax*
  matters for routing design. Fine-grained "virtual group" initialization is relevant as an idea.

### 2.3 LLaMA-MoE (v1, v2): partition, then continued pre-training or post-training
Zhu et al., EMNLP 2024. [arXiv 2406.16554](https://arxiv.org/abs/2406.16554). *Abstract + summary.*
Qu et al., *LLaMA-MoE v2*. [arXiv 2411.15708](https://arxiv.org/abs/2411.15708). *Abstract.*

- v1: neuron partitions compared (random, balanced k-means on up-proj weights, Taylor-importance),
  then **200B tokens** of continued pre-training to reach LLaMA-MoE-3.5B.
- v2: also makes attention heads into experts, adds a "residual" variant (shared expert plus routed
  experts), and recovers with two-stage instruction post-training instead of continued pre-training.
- Takeaway: without training, quality loss from partitioning LLaMA-2 7B was large. Recovery took
  either a very large token budget (v1) or SFT (v2, magnitude of recovery unverified).

### 2.4 DSMoE: matrix-partitioned experts with sigmoid gates
Lv et al., EMNLP 2025. [arXiv 2502.12455](https://arxiv.org/abs/2502.12455). *Partial.*

- Splits gate/up/down along the intermediate dim into **8 experts** (LLaMA-7B: 8 x 1376).
- Gate per expert = sigmoid(h·Y_i), thresholded at τ = 0.5 with a straight-through estimator. The
  number of active experts per token is **dynamic**. Outputs are gate-weighted and rescaled by
  n / active. Loss = LM + L1 on gate values; no load-balancing loss.
- Trained on **10B tokens** (FineWeb-edu, OpenWebMath, StarCoder, Cosmopedia).
- Result: ~58-59% of params active. PPL LLaMA-7B 3.40 dense -> 3.91 DSMoE (LLaMA-MoE 4.12,
  SparseGPT 3.96). **No wall-clock measurement at all.**
- Interesting finding: activation counts are W-shaped across depth, high at input and output
  layers. This supports layer-dependent sparsity.
- Takeaway: even with 10B tokens of training, it gets only ~40% param reduction with a visible PPL
  hit. Dynamic-k complicates kernels and llama.cpp (fixed `n_expert_used`).

### 2.5 CMoE / *Analytical FFN-to-MoE Restructuring*: training-free shared + routed experts
Pei et al. (CUHK / Huawei Noah's Ark), [arXiv 2502.04416](https://arxiv.org/abs/2502.04416), v3
(ACL 2026 per author page). Earlier title *CMoE*. Code: JarvisPei/CMoE. *Full text (v3).*

The closest published match to this project's requirements.

- **Shared experts:** neurons ranked by activation rate μ_i = how often |h_i| is in the token's top-K_a
  (K_a = 10). The top N_s·m neurons become always-on shared experts.
- **Routed experts:** constrained balanced k-means on each neuron's binary activation vector
  (co-activation), solved with Jonker-Volgenant assignment.
- **Analytical router:** for each expert, pick a "representative neuron" nearest the centroid. The
  router is that neuron's SwiGLU response, `Swish(W_gate^R x) * (W_up^R x)`, so there's no training.
- **Combination: unweighted.** `F(x) = E_shared(x) + Σ g_i E_i(x)` with g_i ∈ {0, 1}. For
  fine-tuning, `g_i = 1 + softmax(s)_i · u_i` with u initialised to 0, so the model starts exactly at
  the unweighted sum. **This directly answers the requirements' question about softmax scaling.**
- Load balancing via a bias on scores before top-k (aux-loss-free, γ = 0.001).
- Calibration: just 8 x 2048 tokens of WikiText-2.
- Results, LLaMA-2 7B, S3A3E8 (3 shared + 3 of 5 routed, 25% of FFN neurons dropped):
  WikiText-2 PPL 5.27 dense -> **7.32 training-free** -> 5.92 after LoRA on 2k samples.
  Zero-shot after fine-tune drops 4-7 points (ARC-E 74.6 -> 67.1, HellaSwag 76.0 -> 69.4).
  At 50% / 75% dropped: PPL 7.71 / 12.73 (16 experts).
- **Speed: 1.02-1.08x when memory-bound, up to 1.17x when compute-bound (batch > 400)**.
  Hardware and kernel are not stated. These are modest gains for dropping 25% of FFN neurons.
- Stated limitation: state-space/hybrid architectures are untested. That's us.

### 2.6 ExpertWeaver: GLU-aware training-free partitioning
Zhao et al., [arXiv 2602.15521](https://arxiv.org/abs/2602.15521) (Feb 2026, preprint). *Partial.*

- Uses the **gate** activation `Swish(x W_gate)`, specifically for GLU models. Shared ("universal") neurons
  are the ones with high mean |activation| and low coefficient-of-variation across tasks. Routed
  neurons are split with balanced k-means.
- **Layer-adaptive shared ratio** α_ℓ ∈ [0.2, 0.7], chosen from per-layer specialization.
- Training-free router: **mean of the cluster's W_gate columns**. Training-free mode sums expert
  outputs **unweighted**.
- Training-free at 25% sparsity: LLaMA3-8B avg acc 60.4 vs CMoE 57.2. Qwen2.5-7B 67.0 vs
  LLM-Pruner 65.0. Fails on GSM8K at 25% (appendix).
- "Downcycling" to a 7B / 3.5B-active MoE needs **200B tokens** and retains 87.6% of Qwen2.5-7B.
- vLLM serving at batch 128 (A100): TPOT 40.3 vs 47.8 ms. This is batched serving, not single-stream decode.

### 2.7 D2DMoE: sparsify first, then convert, with dynamic-k
Szatkowski et al., NeurIPS 2024. [arXiv 2310.04361](https://arxiv.org/abs/2310.04361). *Abstract.*

- Regularize activation sparsity in the dense model, then convert neuron groups to experts with
  per-token dynamic k. Reports up to 60% inference-cost reduction on NLP/vision tasks.
- The sparsification stage is the expensive part. See 2.9 for a failed attempt on our exact model.

### 2.8 Activation sparsity without MoE (the main competitor)
- **TEAL**, Liu et al., ICLR 2025. [arXiv 2408.14690](https://arxiv.org/abs/2408.14690) · code FasterDecoding/TEAL. *Full text.*
  Training-free magnitude thresholding of the **inputs** of all 7 projections. Per-layer sparsity
  found by greedy block search in under 1 GPU-hour. Llama-3-8B WikiText PPL 5.87 -> 5.94 (25%) -> 6.21
  (40%) -> 6.67 (50%). Triton column-sparse GEMV, **measured batch-1 decode 1.22/1.42/1.63x on A6000,
  1.11/1.25/1.33x on A100** at 25/40/50%. Gains vanish with batch size and it doesn't help prefill.
- **CATS**, Lee et al., COLM 2024. [arXiv 2404.08763](https://arxiv.org/abs/2404.08763). *Abstract.* Thresholds the SiLU
  gate. Within 1-2% at 50% MLP sparsity without fine-tuning. ~15% end-to-end latency reduction with a
  custom kernel.
- **Deja Vu**, Liu et al., ICML 2023. [arXiv 2310.17157](https://arxiv.org/abs/2310.17157). *Abstract.* Contextual
  sparsity with learned per-layer predictors (OPT, ReLU). >2x vs FasterTransformer on OPT-175B.
- **ProSparse** [arXiv 2402.13516](https://arxiv.org/abs/2402.13516) and **TurboSparse** (dReLU): switch SiLU to ReLU
  variants and train to ~89% sparsity. Too costly for us; ProSparse needed ~134B tokens for 13B
  (figure from the 2.9 write-up, not checked against ProSparse itself).

**Why this matters:** neuron-level dynamic sparsity is a strict generalisation of "route to 2 of 8
expert blocks". The MoE constraint (blocks of 1536 neurons, fixed k) can only be worse in
approximation error at the same active fraction. It wins only if block granularity buys kernel
efficiency or runtime support (llama.cpp has MoE kernels; it has no column-sparse GEMV).

### 2.9 Prior attempt on our exact model (negative result)
[iamboosted/Qwen3.5-9B-Dense-To-Moe](https://github.com/iamboosted/Qwen3.5-9B-Dense-To-Moe). Individual project,
README only, no logs or scripts visible. *Treat as anecdotal.*

- CMoE E16 S2 A4 on Qwen3.5-9B: **output degenerate (repetition loops) training-free**. Router-only
  training plateaued at loss ~4.92. Training experts too (4.83B params, 8-bit AdamW) diverged.
- D2DMoE-style L1 at λ = 1e-3 for 1000 steps: only 23.4% sparsity, and quality degraded.
- Their diagnosis: SwiGLU has no natural activation sparsity, so slicing breaks distributed representations.
- Caveats: E16 S2 A4 drops a lot (6 of 16 experts active = 37.5% of neurons), they report no
  reconstruction error or oracle bound, and the router-init and implementation weren't checked.
  This tells us to **measure the oracle bound first**, before investing in routers.

### 2.10 Kernels and runtime
- **MegaBlocks** (block-sparse) and **ScatterMoE** ([arXiv 2403.08245](https://arxiv.org/abs/2403.08245)) target
  training and large batches. At batch 1, MoE decode reduces to gathering k expert matrices and running
  GEMVs. The cost is router + top-k + k-way gathered GEMV, and launch overhead dominates in eager PyTorch.
- **llama.cpp** `ggml_mul_mat_id` already does batch-1 expert GEMVs for quantized experts, and is
  well tuned on ROCm (Qwen3.6-35B-A3B runs at 72 tok/s here). `qwen35moe` matches our backbone but
  hard-codes softmax + renormalize gating (see `environment.md`).
- **Speculative decoding interaction:** Qwen3.5-9B ships an MTP head. llama.cpp MTP gave 2.6x on
  Qwen3.8-27B / Gemma-31B here (`local-llm-benchmarks/r9700+rx9070.md`). Verifying n draft tokens in
  one batch makes each token pick experts independently. With top-2/8 and 4 tokens, the expected union
  is 8·(1-(6/8)^4) ≈ 5.5 experts, assuming independent routing. **MoE savings and MTP gains partly
  cancel.** This has to be measured, not assumed away.

---

## 3. Applicability to Qwen3.5-9B

| Factor | Implication |
|---|---|
| SwiGLU (SiLU), no exact zeros | MoEfication's premise is weakest here. CMoE / ExpertWeaver are GLU-aware and training-free, but quality loss at 25% drop is already large on 7-8B models. 75% drop (top-2/8) is far beyond anything reported to work without heavy training |
| FFN = 54% of text params, ~61% of decode bytes | Ceiling for top-2/8 is 1.84x, for top-4/8 1.44x (environment.md). CMoE-like 25% drop caps at ~1.15x |
| Hybrid DeltaNet + full attention | Untested by every paper. The attention side is untouched, so the FFN math is the same, but layer sensitivity may differ between linear and full-attention blocks |
| Untied 248K vocab, 1.02B-param lm_head | lm_head alone is 13% of decode bytes and stays dense |
| Ships MTP head; llama.cpp supports MTP | Strong competing speedup, and it interacts with MoE (2.10) |
| Q8_0 -> Q4 quantization | Halves bytes for every weight, not just FFN. **Dense Q4 (IQ4_XS-Q4_K_M, 4.25-4.85 bpw) reads 4.2-4.8 GB/token, about what top-2/8 MoE at Q8_0 reads (4.31B x 8.5 bits = 4.6 GB).** Fair comparison: MoE-Q8 vs dense-Q4 quality at equal bytes, then MoE-Q4 vs dense-Q2/Q3 |

---

## 4. Expected advantages and limitations

**Advantages**
- Exact reconstruction with all experts active, so every quality loss can be attributed to routing.
- Fixed-k block experts map onto existing llama.cpp MoE kernels and quantization (`mul_mat_id`) with
  a small patch, unlike neuron-level sparsity, which needs new kernels.
- Training-free conversion is cheap (minutes). We can sweep partitions, k and per-layer configs quickly.

**Limitations and risks (ranked)**
1. **Approximation error at useful sparsity.** The evidence says 25% drop is roughly the training-free
   limit. Top-2/8 (75% drop) almost certainly needs substantial training, likely more than this
   workstation can provide (DSMoE: 10B tokens; LLaMA-MoE: 200B).
2. **Speed realised << ceiling.** MoEfication on GPU realised 1.2x of a 2x FLOP reduction. CMoE
   realised ~1.05x memory-bound.
3. **Beaten by quantization at equal bytes** unless routing error is smaller than quantization error.
   Our notes show IQ4_XS KLD 0.018, Q6_K 0.002 on Qwen3.8-27B.
4. **MTP interaction** erodes decode savings.
5. **Prefill gains are small or negative.** Large batches touch every expert, so the only gain is FLOPs,
   and only with good grouped GEMMs.

---

## 5. Recommended architecture (for the prototype)

1. **Partition, don't duplicate.** Experts are disjoint slices of gate/up rows and down columns.
   Baseline E = 8, k = 2, no shared expert, as the requirements specify. Make E, k and shared size configurable.
2. **Unweighted combination: `y = Σ_{i∈topk} E_i(x)`** (gate value 1 for selected experts). For
   trainable variants use CMoE's `g_i = 1 + u_i·softmax(s)_i` with u = 0 at init, so the invariant
   holds at initialization. Also implement softmax-renormalized weighting (llama.cpp `qwen35moe`
   semantics) purely to measure how much it hurts.
3. **Shared expert as the first variant after baseline.** CMoE and ExpertWeaver both depend on it.
   Size from activation statistics (CMoE μ_i rate; ExpertWeaver mean |gate| / CV).
4. **Partition strategies** (requirements A/B/C + literature):
   A contiguous · B random (seeded) · C1 co-activation balanced k-means (CMoE) ·
   C2 balanced k-means on W_gate columns (LLaMA-MoE / MoEfication parameter clustering) ·
   C3 contribution-based: cluster on |h_i|·‖W_down[:, i]‖, i.e. actual output contribution, not
   just activation. This is our addition, motivated by the warning that co-activation ≠ good grouping.
5. **Routing, in order of cost:**
   - **Oracle** (top-k experts by true ‖E_i(x)‖ or by least reconstruction error). An upper bound,
     **measured first**.
   - Analytical router: CMoE representative neuron, ExpertWeaver mean gate column.
   - Linear router trained to predict oracle selection (cross-entropy on oracle labels).
   - Linear router trained on reconstruction MSE through soft top-k / straight-through.
6. **Per-layer k**, motivated by the DSMoE W-shape and ExpertWeaver's layer-adaptive ratios: some layers
   stay dense.

## 6. Justification, and the go/no-go gate

The single-layer prototype is cheap and answers the decisive question before any training spend:

> **Gate 1 (oracle bound):** on real Qwen3.5-9B activations, what is the relative FFN output error
> and end-to-end KL when each layer keeps the oracle-best k of E experts?

- If **oracle** top-4/8 already gives large error, no router can fix it and training-free MoE is dead
  for this model. That would replicate 2.9 with an actual measurement. Pivot to the comparison
  against quantization / TEAL / MTP, and report the negative result.
- If oracle error is small but learned routers are far from it, routing is the bottleneck, which is a
  cheap problem.
- **Gate 2 (speed):** batch-1 kernel latency of a k-of-E FFN vs dense in PyTorch, then in llama.cpp.
  It has to approach the bandwidth ceiling, or the quality question is moot.

Training (requirements stage 3, distillation) is justified only after both gates pass, and only
with a token budget estimated against this hardware first.

---

## 7. Not yet verified (to-do)
- ExpertWeaver full appendix (only first 100K chars read). Exact Qwen2.5 configs.
- D2DMoE method details (abstract only).
- LLaMA-MoE v2 quantitative results (abstract only).
- CMoE speedup hardware and kernel (paper doesn't state them).
- Whether Qwen3.5-9B's SwiGLU intermediate activations are as dense as claimed in 2.9.
  **Measure directly**: distribution of |h_i| and the fraction of output norm carried by the top x% of neurons.
