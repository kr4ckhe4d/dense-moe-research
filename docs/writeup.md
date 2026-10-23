# Can Qwen3.5-9B be made faster by converting it to a Mixture-of-Experts?

**Short answer: not without training, and probably not usefully even with it.** On this machine,
plain quantization plus the model's own MTP head is faster and more accurate than any
training-free MoE conversion, even with a perfect router.

Dates: 2026-10-22 (milestone 1, the MoE prototype) and 2026-10-23 (milestone 2, the quantization
bar). Hardware: RX 9070 XT (16 GB) + Radeon AI PRO R9700 (32 GB), ROCm 7.2.4, 30 GiB RAM.
Model: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache-2.0).

Part 1 is the result. Part 2 is **every command that was run**, in order, rewritten so you can
paste it yourself from a fresh shell. Part 3 lists what failed on the way and why.

---

## Part 1. Result

### What was tried

The FFN of every layer (54% of the text parameters) was split into E experts by partitioning its
12,288 neurons, without duplicating any. With all experts on, the result is bit-for-bit the dense
model (63 tests). Each token then uses only k experts. Five partition methods were tried
(contiguous, random, and k-means on gate weights, on co-activation, and on neuron contribution).
Routing was an *oracle*: it sees the true expert outputs, so a learned router can only do worse.

### Speed: the kernels work

Triton decode kernels for batch 1, real layer-15 weights, BF16, RX 9070 XT. The best dense kernel
takes 516 µs per layer.

| Config (experts / active) | µs per layer | Speedup | Ideal |
|---|---|---|---|
| 8 / 6 | 433 | 1.19x | 1.33x |
| 8 / 4 | 301 | 1.71x | 2x |
| 8 / 2 | 157 | 3.29x | 4x |
| 16 / 4 | 146 | 3.53x | 4x |

Prefill is slower than dense with every expert active: PyTorch's grouped matmul on ROCm is about
2.7x slower for the same work.

### Quality: the approximation does not

Everything is scored the same way as a llama.cpp quant: KL divergence against llama.cpp BF16 on
our held-out validation text (100 windows of 512 tokens, last 255 of each scored). Lower is
better. PyTorch BF16 against llama.cpp BF16 is 0.0004, so the two engines agree.

| Variant | Mean KLD | Decode, 9070 XT | + MTP (draft acceptance) |
|---|---|---|---|
| Q8_0 | 0.0016 | 57.5 tok/s | **115.8** (79%) |
| Q6_K | 0.0028 | 70.3 | 117.4 (70%) |
| Q5_K_M | 0.0151 | 77.0 | — |
| IQ4_XS | 0.0184 | 89.6 | **148.0** (75%) |
| Q4_K_M | 0.0228 | 84.0 | 141.9 (80%) |
| Q3_K_M | 0.0581 | 88.9 | — |
| oracle MoE 8/6 | 0.117 | (ceiling ~1.18x of its base) | |
| oracle MoE 4/3 | 0.148 | | |
| oracle MoE 16/8 | 0.295 | | |
| oracle MoE 8/4 | 0.376 | | |
| per-neuron oracle, 75% kept | 0.0013 | | |
| per-neuron oracle, 50% kept | 0.0103 | | |

Wikitext-2 gives the same ordering (oracle 8/6 0.187, IQ4_XS 0.025).

**What sparse FFN could add on top of IQ4_XS:** with only the FFN re-quantized to fewer bits, which
stands in for skipping FFN bytes, cutting the FFN to 37% of its bytes gave 89.6 → 105.8 tok/s
(1.18x). About half of each 11 ms token doesn't depend on weight bytes at this model size. MTP gives
1.67x on the same file, and it doesn't change the output.

### Conclusions

1. Training-free dense-to-MoE fails on Qwen3.5-9B. The mildest useful config (8/6) is worse than
   Q3_K_M, and the configs that would be fast (8/2, 16/4) are much worse again.
2. The loss is from forcing neurons into fixed groups, not from sparsity. Per-token neuron
   selection at the same 50% budget is 35x better. But exact selection can only skip `down_proj`
   (a third of the FFN bytes), because choosing neurons needs the gate/up outputs.
3. Recovering MoE quality needs 10B-200B tokens of training (DSMoE, LLaMA-MoE). The most it could
   then buy on top of quantization is about 1.2x.
4. **What to use instead:** IQ4_XS + MTP (148 tok/s, KLD 0.018) or Q8_0 + MTP (116 tok/s, KLD
   0.0016).

Detailed tables: `docs/findings.md` (F0-F4). Raw data: `experiments/results/`.

---

## Part 2. Every command, in order

### Conventions

- Run everything from the repo root, `~/code/dense-moe`, in your normal zsh.
- `source scripts/env.sh` once per shell before any Python. It hides the integrated GPU (PyTorch
  segfaults otherwise) and unsets an allocator flag that corrupts tensors (Part 3).
  After it, `cuda:0` is the 9070 XT and `cuda:1` is the R9700.
- llama.cpp tools pick their GPU with `HIP_VISIBLE_DEVICES`: `0` is the 9070 XT, `1` the R9700.
- Every PyTorch import prints a few `(null): No such file or directory` lines from the ROCm
  library loader. They're harmless; filter them with `| grep -v '^(null)'`.
- Times are wall-clock times from the actual runs.
- Commits used `git -c user.name="nipuna" -c user.email="hash.crackhead@gmail.com" commit`. Below
  that's written as `git commit`.

### Phase 1. Inspect the machine and the model (read-only, ~2 min)

```bash
# GPUs, CPU, RAM, disk, Python, network. Showed: no torch installed, system Python 3.14,
# llama.cpp in ~/llama.cpp, Hugging Face reachable.
rocm-smi --showproductname --showmeminfo vram | head -30
rocminfo | grep -E "Marketing Name|gfx" | head
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
free -h; df -h ~ ~/.cache
du -sh ~/.cache/huggingface; ls ~/.cache/huggingface/hub
which python3 uv; python3 --version
curl -sI --max-time 5 https://huggingface.co | head -1

# Your earlier research: README sections, Qwen3.5-9B notes, any MoE notes.
cd ~/code/local-llm-benchmarks
grep -nE '^#{1,3} ' README.md | head -80
grep -rniE 'qwen3\.5-9b|qwen3\.5 9b' --include='*.md' --include='*.ini' --include='*.sh' . | head -20
sed -n 919,960p README.md; sed -n 1538,1571p README.md
grep -rliE 'upcycl|dense-to-moe|moefication|expert partition|dsmoe' . | grep -v '^./.git'
head -40 instella-moe.md
grep -nE 'MTP|mtp' r9700+rx9070.md | head -25
grep -nE 'KLD|kld' README.md | grep -iE 'Q4|IQ4|Q8|Q6' | head -15
cd ~/code/dense-moe

# llama.cpp build, local models, ROCm version, available uv Pythons.
git -C ~/llama.cpp log -1 --format='%h %cd %s'          # 5e03bdd87 (b11434)
ls ~/llama.cpp/build/bin
ls -laL ~/llama.cpp/models | head -40                   # symlink to /mnt/fast/models
cat /opt/rocm/.info/version                             # 7.2.4
uv python list | head
```

```bash
# Model config and file list from Hugging Face (small JSON files only).
mkdir -p cache/hf-meta
for f in config.json model.safetensors.index.json; do
  curl -sfL --max-time 20 https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/$f -o cache/hf-meta/$f
done
curl -sf https://huggingface.co/api/models/Qwen/Qwen3.5-9B | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['sha'],d.get('cardData',{}).get('license'))"
cat cache/hf-meta/config.json

# Tensor-name patterns and the size of shard 1.
python3 - cache/hf-meta/model.safetensors.index.json <<'EOF'
import json,sys,re,collections
wm=json.load(open(sys.argv[1]))["weight_map"]
for p,c in sorted(collections.Counter(re.sub(r'\.\d+\.', '.N.', n) for n in wm).items()): print(f"{c:4d}  {p}")
EOF
curl -sfI -L https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/model.safetensors-00001-of-00004.safetensors | grep -i content-length

# Parameter breakdown by component, read from the safetensors headers with HTTP range requests
# (nothing downloaded). Result: 9.653B total, 8.954B text-only, FFN 54% of text params.
python3 scripts/probes/param_breakdown.py
```

```bash
# How llama.cpp's qwen35moe weights its experts. Finding: it hard-codes SOFTMAX gating with
# renormalization, which turns out to be much worse for this conversion (see F3).
cd ~/llama.cpp
grep -nE 'QWEN35|qwen35' src/llama-arch.cpp | head -20
ls src/models | grep -i qwen3
sed -n 495,515p src/models/qwen35moe.cpp
grep -nE 'gating_op|norm_w|scale_w' src/llama-graph.cpp | head -30
sed -n 2085,2110p src/llama-graph.cpp; sed -n 2175,2200p src/llama-graph.cpp
grep -rlE 'hparams.expert_gating_func' src/models | head -20
cd ~/code/dense-moe
```

### Phase 2. Python environment (~3 min)

```bash
# Which PyTorch ROCm wheels exist for Python 3.12. Picked torch 2.14.1+rocm7.2 to match ROCm 7.2.
curl -s https://download.pytorch.org/whl/torch/ \
  | grep -oE 'torch-[0-9.]+%2Brocm[0-9.]+-cp312-cp312-manylinux[^"#<]*' | sed 's/%2B/+/' | sort -uV | tail -12

# Create .venv (Python 3.12) with torch, triton-rocm, transformers and the rest.
# pyproject.toml pulls torch and triton-rocm from the PyTorch ROCm index.
uv sync --python 3.12

# GPUs visible to PyTorch, BF16 GEMM speed, copy bandwidth.
# Result: 118 / 123 TFLOP/s and ~500 GB/s on the 9070 XT / R9700.
source scripts/env.sh
.venv/bin/python scripts/probes/gpu_check.py 2>&1 | grep -v '^(null)'

# Find what prints the "(null)" lines: it's torch._rocm_init (harmless).
.venv/bin/python -X importtime -c "import torch" 2>&1 | grep -n -B2 '(null)' | head -8

git init && git add -A && git commit -m "Inspect environment and model; set up uv venv with PyTorch ROCm"   # 80e74d9
git add -A && git commit -m "Add research survey and recommended architecture"                         # f66e0ef
```

### Phase 3. Single-layer prototype and tests (~10 min, plus a 19 GB download)

```bash
# The code in src/dense_to_moe/ and tests/ was written at this point. Run the test suite.
uv sync -q
.venv/bin/python -m pytest -q 2>&1 | grep -v '^(null)' | tail -3

# Fetch only layer 15's FFN (gate/up/down, ~300 MB) with HTTP range requests, so work could
# start before the full download. Goes to cache/tensors/.
# (Originally an inline call to dense_to_moe.utils.hf_weights.fetch_ffn_layer; same thing.)
.venv/bin/python scripts/fetch_layer.py --layer 15

# Full checkpoint download (19 GB, into the HF cache on /mnt/fast). Run it in the background.
mkdir -p experiments/logs
HF_HUB_ENABLE_HF_TRANSFER=0 .venv/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --max-workers 8 \
  > experiments/logs/download-qwen3.5-9b.log 2>&1 &

# Reconstruction on real layer-15 weights.
.venv/bin/python -m pytest -q tests/test_real_weights.py

# Triton, CUDA graphs and torch._grouped_mm all work on ROCm.
.venv/bin/python scripts/probes/tri_check.py 2>&1 | grep -v '^(null)' | tail -4

# Sparse decode/prefill kernels against the dense reference.
.venv/bin/python -m pytest -q tests/test_sparse_kernels.py
```

### Phase 4. Kernel latency (~15 min)

```bash
# First pass on the 9070 XT, then the configs the first pass crashed on (Part 3, item 4).
.venv/bin/python benchmarks/kernel_latency.py --device cuda:0 --repeats 30 --out experiments/results/kernel_latency_9070xt_v0.json
.venv/bin/python benchmarks/kernel_latency.py --device cuda:0 --repeats 30 --configs 16:4,16:8,4:2,4:3 \
  --out experiments/results/kernel_latency_9070xt_v0b.json

# Router overhead (the PyTorch router took 28-33 µs), and the 64 MB Infinity Cache effect:
# one layer copy vs 8 rotating copies. One copy inflates sparse speedups by ~20%.
.venv/bin/python scripts/probes/router_probe.py 2>&1 | grep ' us$'
for c in 1 8; do
  .venv/bin/python benchmarks/kernel_latency.py --copies $c --configs 8:2 --prefill "" --repeats 20 \
    --out cache/cache_$c.json 2>&1 | grep -E 'dense \[graph\]|triton gathered GEMV \[graph\]'
done

# Fused Triton router: test it, then compare it to PyTorch (6 µs vs 33 µs).
.venv/bin/python -m pytest -q tests/test_sparse_kernels.py
.venv/bin/python scripts/probes/router_probe2.py 2>&1 | grep ' us$'

# Final latency run on both GPUs (the results quoted in Part 1).
for dev in cuda:0 cuda:1; do
  name=$([ $dev = cuda:0 ] && echo 9070xt || echo r9700)
  .venv/bin/python benchmarks/kernel_latency.py --device $dev --repeats 50 \
    --out experiments/results/kernel_latency_$name.json 2>&1 | grep -v '^(null)'
done
```

### Phase 5. Calibration activations (~5 min)

```bash
# Check the download finished, load the full model on the R9700, and test generation.
# Result: loads as Qwen3_5ForCausalLM, 16.7 GiB; 2048-token forward 2.7 s, peak 17.9 GiB.
tail -2 experiments/logs/download-qwen3.5-9b.log
.venv/bin/python scripts/probes/load_probe.py 2>&1 | grep -v '^(null)' | tail -6

# FFN inputs for all 32 layers. Calibration: 48 sequences x 1024 tokens over 6 domains,
# 256 sampled positions each = 12,288 rows/layer (3.1 GB). Validation is a disjoint hash split.
.venv/bin/python scripts/collect_activations.py --split calib
.venv/bin/python scripts/collect_activations.py --split val --per-domain 4 --seed 1

.venv/bin/python -m pytest -q
git add -A && git commit -m "Single-layer prototype: partitioned FFN, oracle routing, Triton decode kernels, benchmarks, calibration pipeline"   # 3423c31
```

### Phase 6. Oracle analysis, and the bug it exposed (~20 min)

```bash
# Smoke test on layer 15.
.venv/bin/python scripts/oracle_analysis.py --layers 15 --out experiments/results/oracle_smoke_l15.json
```

The first smoke test gave identical errors for every partition at E=8 and 16, which is
impossible. These commands tracked it down (scripts in `scripts/debug/`):

```bash
.venv/bin/python scripts/debug/kmeans_check.py     # expert energy shares differ by partition
.venv/bin/python scripts/debug/bug_check.py        # outside analyze_layer, partitions behave
.venv/bin/python scripts/debug/bug_check2.py       # inside it, results are partition-independent
.venv/bin/python scripts/debug/km_debug.py         # no NaNs or ties in the k-means features
.venv/bin/python scripts/debug/km_trace.py         # greedy assignment is fine on synthetic data
.venv/bin/python scripts/debug/km_trace2.py        # uninitialized buffer is not the cause
.venv/bin/python scripts/debug/km_trace3.py        # iterations on real data look normal
.venv/bin/python scripts/debug/km_trace4.py        # repeated calls don't degenerate
.venv/bin/python scripts/debug/bug_check3.py       # the features fed to k-means inside analyze_layer were NaN
```

**Cause:** an allocator flag I had put in the original `scripts/env.sh`,
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. On this ROCm build it silently changes tensors
nobody writes to, after large frees plus `empty_cache()`. Fix and checks:

```bash
# env.sh now unsets the flag (already in the repo). Re-collect activations without it and
# compare bit-for-bit with the old ones: 0 of 64 layer files differed, so the cached data was fine.
mv cache/activations cache/activations_expseg
source scripts/env.sh
.venv/bin/python scripts/collect_activations.py --split calib
.venv/bin/python scripts/collect_activations.py --split val --per-domain 4 --seed 1
.venv/bin/python - <<'EOF'
import torch
from safetensors.torch import load_file
bad = 0
for split in ("calib", "val"):
    for i in range(32):
        a = load_file(f"cache/activations/{split}/layer_{i:02d}.safetensors")["x"]
        b = load_file(f"cache/activations_expseg/{split}/layer_{i:02d}.safetensors")["x"]
        bad += not torch.equal(a, b)
print("layers differing:", bad, "of 64")
EOF
rm -rf cache/activations_expseg

# 20-line repro, without and with the flag. With it: "step 0: CORRUPTED".
for flag in "" "expandable_segments:True"; do
  HIP_VISIBLE_DEVICES=0 PYTORCH_HIP_ALLOC_CONF=$flag .venv/bin/python scripts/repro_expandable_segments.py 2>&1 | grep -v '^(null)'
done

# The package now refuses to import with the flag set (expect RuntimeError).
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True .venv/bin/python -c "import dense_to_moe" 2>&1 | tail -1

source scripts/env.sh
.venv/bin/python -m pytest -q
rm -f experiments/results/oracle_smoke_l15.json
.venv/bin/python scripts/oracle_analysis.py --layers 15 --out experiments/results/oracle_smoke_l15.json
git add -A && git commit -m "Fix tensor corruption: drop expandable_segments (ROCm allocator bug), add guard + repro"   # 42290dc

# Full per-layer analysis: 32 layers x 5 partitions x 7 configs x 5 selection methods (~10 min).
.venv/bin/python scripts/oracle_analysis.py --layers all --out experiments/results/oracle_analysis.json
.venv/bin/python scripts/probes/aggregate_oracle.py

# Does smaller expert size help? E up to 256 on layers 4, 15, 26 (~2 min). Barely.
.venv/bin/python scripts/probes/granularity.py
```

### Phase 7. End-to-end oracle evaluation (~6 min)

```bash
# Every FFN replaced at once, on 12 held-out sequences x 1024 tokens, on the R9700.
.venv/bin/python -m pytest -q tests/test_e2e_wrapper.py
.venv/bin/python scripts/oracle_e2e.py --device cuda:1 2>&1 | grep -E 'KL|wrote'

.venv/bin/python -m pytest -q        # 63 passed
git add -A && git commit -m "Gate 1: oracle routing analysis (32 layers, granularity, end-to-end KL); findings and roadmap"   # 19df9e8
```

### Phase 8. Build the GGUFs (~10 min, 63 GB on /mnt/fast)

```bash
# Look at what already exists: your KLD script, models, disk, llama.cpp tools.
cat ~/code/local-llm-benchmarks/benchmarks/kld-test.sh
ls /mnt/fast/models; df -h / /mnt/fast
ls ~/llama.cpp/build/bin | grep -E 'quantize|perplexity|bench|imatrix|server'
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/

# The converter needs llama.cpp's gguf-py; adding it to PYTHONPATH avoids installing anything.
PYTHONPATH=$HOME/llama.cpp/gguf-py .venv/bin/python -c "import gguf; print(gguf.__file__)"

# HF safetensors -> BF16 GGUF (18.4 GB, 26 s). Includes the MTP head as blk.32.nextn.
S=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*)
PYTHONPATH=$HOME/llama.cpp/gguf-py .venv/bin/python ~/llama.cpp/convert_hf_to_gguf.py "$S" \
  --outtype bf16 --outfile /mnt/fast/models/Qwen3.5-9B-BF16.gguf > experiments/logs/convert_bf16.log 2>&1
grep -iE "nextn" experiments/logs/convert_bf16.log | head -4

# Calibration and validation text as plain files (cache/corpus/calib.txt, val.txt), from the
# same hash-disjoint splits as the PyTorch experiments.
.venv/bin/python scripts/export_corpus.py 2>&1 | grep -E "val:|calib:"

# Importance matrix from the calibration split only, BF16 on the R9700 (295 chunks, 5.7 min).
mkdir -p cache/gguf
HIP_VISIBLE_DEVICES=1 ~/llama.cpp/build/bin/llama-imatrix -m /mnt/fast/models/Qwen3.5-9B-BF16.gguf \
  -f cache/corpus/calib.txt -c 512 -ngl 99 -o cache/gguf/imatrix-calib.gguf > experiments/logs/imatrix.log 2>&1

# Quantize. Q8_0 without the imatrix, everything else with it (~1 min each).
B=~/llama.cpp/build/bin; M=/mnt/fast/models
$B/llama-quantize $M/Qwen3.5-9B-BF16.gguf $M/Qwen3.5-9B-Q8_0.gguf Q8_0 16 > experiments/logs/quantize_Q8_0.log 2>&1
for q in Q6_K Q5_K_M Q4_K_M IQ4_XS Q3_K_M; do
  $B/llama-quantize --imatrix cache/gguf/imatrix-calib.gguf $M/Qwen3.5-9B-BF16.gguf $M/Qwen3.5-9B-$q.gguf $q 16 \
    > experiments/logs/quantize_$q.log 2>&1
  echo "$q rc=$? $(grep 'quant size' experiments/logs/quantize_$q.log)"
done
```

### Phase 9. KL divergence (~1 h of GPU time, mostly unattended)

```bash
# How llama.cpp stores and scores KLD, so the PyTorch scorer could match it exactly.
cd ~/llama.cpp
sed -n 79,105p tools/perplexity/perplexity.cpp        # base file: uint16 log-probs + scale/min per token
sed -n 191,252p tools/perplexity/perplexity.cpp       # KL formula (terms below -16 dropped)
sed -n 1680,1800p tools/perplexity/perplexity.cpp     # header, stored tokens, chunking (first = n_ctx/2)
cd ~/code/dense-moe
PYTHONPATH=$HOME/llama.cpp/gguf-py .venv/bin/python -c "
from gguf import GGUFReader
f=GGUFReader('/mnt/fast/models/Qwen3.5-9B-Q3_K_M.gguf').fields['tokenizer.ggml.add_bos_token']; print(f.parts[f.data[0]])"   # [False]: no BOS

# BF16 reference logits for both corpora, R9700 (2.6 + 3.6 min; 12.7 GB each in ~/llama.cpp/kld).
benchmarks/gguf_kld.sh base wiki
benchmarks/gguf_kld.sh base val
python3 -c "import struct;f=open('$HOME/llama.cpp/kld/qwen3.5-9B-bf16-val.kld','rb');print(f.read(8),struct.unpack('<Iii',f.read(12)))"   # (512, 248320, 100)

# Score each quant against them (12 runs, ~25 min).
for c in wiki val; do for q in Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS Q3_K_M; do
  echo "== $c $q"; benchmarks/gguf_kld.sh score $c $q
done; done 2>&1 | tee experiments/logs/kld_all.log

# PyTorch variants (dense, per-neuron oracle, oracle MoE) scored against the same base files:
# a 2-chunk smoke test, then both corpora (~25 min each).
.venv/bin/python scripts/sparse_kld_vs_gguf.py --corpus val --chunks 2 --out cache/smoke.json 2>&1 | grep KLD
for c in val wiki; do
  .venv/bin/python scripts/sparse_kld_vs_gguf.py --corpus $c 2>&1 | grep -v '^(null)' > experiments/logs/sparse_kld_$c.log
done
grep KLD experiments/logs/sparse_kld_*.log
```

### Phase 10. Speed (~30 min)

```bash
# llama-bench, every quant, 9070 XT, then R9700 (which also fits BF16).
M=/mnt/fast/models
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build/bin/llama-bench \
  -m $M/Qwen3.5-9B-Q8_0.gguf,$M/Qwen3.5-9B-Q6_K.gguf,$M/Qwen3.5-9B-Q5_K_M.gguf,$M/Qwen3.5-9B-Q4_K_M.gguf,$M/Qwen3.5-9B-IQ4_XS.gguf,$M/Qwen3.5-9B-Q3_K_M.gguf \
  -ngl 99 -fa 1 -p 512,2048 -n 128 -r 5 -o md 2>/dev/null | tee experiments/results/gguf/bench_9070xt.md
HIP_VISIBLE_DEVICES=1 ~/llama.cpp/build/bin/llama-bench \
  -m $M/Qwen3.5-9B-BF16.gguf,$M/Qwen3.5-9B-Q8_0.gguf,$M/Qwen3.5-9B-Q6_K.gguf,$M/Qwen3.5-9B-Q5_K_M.gguf,$M/Qwen3.5-9B-Q4_K_M.gguf,$M/Qwen3.5-9B-IQ4_XS.gguf,$M/Qwen3.5-9B-Q3_K_M.gguf \
  -ngl 99 -fa 1 -p 512,2048 -n 128 -r 5 -o md 2>/dev/null > experiments/results/gguf/bench_r9700.md

# MTP: llama-server with and without --spec-type draft-mtp, measured with your
# throughput-test.py (700-token code answer, temperature 0, thinking off, 3 runs). 9070 XT.
benchmarks/gguf_mtp.sh 0 Q8_0 Q6_K Q4_K_M IQ4_XS

# Speed proxy for sparse FFN: IQ4_XS everywhere except the FFN of layers 0-31, re-quantized
# to Q2_K (62% of the bytes) or IQ1_S (37%). Quality is irrelevant; only tok/s matters.
B=~/llama.cpp/build/bin
$B/llama-quantize --imatrix cache/gguf/imatrix-calib.gguf \
  --tensor-type ffn_gate=q2_k --tensor-type ffn_up=q2_k --tensor-type ffn_down=q2_k \
  $M/Qwen3.5-9B-BF16.gguf $M/Qwen3.5-9B-IQ4_XS-ffn_q2_k.gguf IQ4_XS 16 > experiments/logs/quantize_ffn_q2_k.log 2>&1
$B/llama-quantize --imatrix cache/gguf/imatrix-calib.gguf \
  --tensor-type 'blk\.([0-9]|[12][0-9]|3[01])\.ffn_(gate|up|down)=iq1_s' \
  $M/Qwen3.5-9B-BF16.gguf $M/Qwen3.5-9B-IQ4_XS-ffn_iq1_s.gguf IQ4_XS 16 > experiments/logs/quantize_ffn_iq1_s.log 2>&1
HIP_VISIBLE_DEVICES=0 $B/llama-bench \
  -m $M/Qwen3.5-9B-IQ4_XS.gguf,$M/Qwen3.5-9B-IQ4_XS-ffn_q2_k.gguf,$M/Qwen3.5-9B-IQ4_XS-ffn_iq1_s.gguf \
  -ngl 99 -fa 1 -p 512 -n 128 -r 5 -o md 2>/dev/null | tee experiments/results/gguf/bench_ffn_proxy_9070xt.md
```

### Phase 11. Summary and commit

```bash
# One table from all the KLD, llama-bench, MTP and PyTorch outputs.
.venv/bin/python scripts/gguf_summary.py | tee experiments/results/gguf/summary.md
git add -A && git commit -m "Milestone 2: quantization bar (GGUF KLD, llama-bench, MTP) vs sparse FFN variants"   # c37d25d
```

---

## Part 3. What went wrong on the way

Each of these is fixed in the repo, so the commands in Part 2 already work. They're recorded so
you don't hit them again.

1. **`uv sync` couldn't resolve torch.** `torch==2.14.1+rocm7.2` needs `triton-rocm`, which is
   only on the PyTorch index. Fix: add `triton-rocm` to `pyproject.toml`, sourced from the
   `pytorch-rocm` index.
2. **PyTorch segfaulted (exit 139) on the first GPU call.** The gfx1036 integrated GPU was visible.
   Fix: `HIP_VISIBLE_DEVICES=0,1` in `scripts/env.sh`.
3. **Triton refused to compile from a heredoc:** "@jit functions should be defined in a Python
   file". Fix: run Triton code from files (`scripts/probes/tri_check.py`).
4. **The Triton down-projection kernel returned garbage and crashed at k=8.** A block size of 256
   didn't divide the 384-row chunk, so it read out of bounds. Then E=16 failed an assert (768 rows
   with split 4). Fix: block 128, automatic split choice, and asserts on exact tiling.
5. **The R9700 benchmark measured nothing ("CUDA Graph is empty").** The graph was captured on the
   default device. Fix: `torch.cuda.set_device(dev)` in `benchmarks/kernel_latency.py`.
6. **Unfair dense baseline.** `F.linear` (hipBLASLt) takes 680 µs per layer at batch 1, while our
   own Triton kernel with every expert on takes 516 µs. Speedups are quoted against 516 µs.
7. **Corrupted tensors from `expandable_segments`** (Phase 6). The serious one: I had added the
   flag to `env.sh` myself, and it made the first analysis results meaningless. Fixed by removing
   it, adding an import guard, and re-running everything. `scripts/repro_expandable_segments.py`
   is small enough to report upstream if you want.
8. **Quantization with the imatrix failed under zsh.** `IM="--imatrix file"` followed by an
   unquoted `$IM` is one argument in zsh (no word splitting), so llama-quantize printed its usage.
   Fix: pass `--imatrix` directly (as in Part 2), or use `${=IM}` or `bash -c`.
9. **IQ1_S failed on `blk.32.ffn_down`.** That's the MTP layer, which never runs during the imatrix
   pass, and IQ1_S requires an imatrix. Fix: limit the override to layers 0-31 with the regex
   shown in Phase 10.
10. **My grep for "Mean KLD" printed nothing.** llama.cpp writes `Mean    KLD:` with padding. Fixed
    in `benchmarks/gguf_kld.sh`.
11. **A wait loop never ended.** `pgrep -f llama-perplexity` matched its own command line. Not in
    the repo; just don't wait that way.

## Part 4. What's on disk

| Path | Size | Notes |
|---|---|---|
| HF cache `models--Qwen--Qwen3.5-9B` (on /mnt/fast) | 19 GB | the source checkpoint |
| `/mnt/fast/models/Qwen3.5-9B-BF16.gguf` | 18.4 GB | only needed to rebuild quants or KLD bases |
| `/mnt/fast/models/Qwen3.5-9B-{Q8_0,Q6_K,Q5_K_M,Q4_K_M,IQ4_XS,Q3_K_M}.gguf` | 39.8 GB | usable models |
| `/mnt/fast/models/Qwen3.5-9B-IQ4_XS-ffn_{q2_k,iq1_s}.gguf` | 8.0 GB | speed proxies only, **not usable models** |
| `~/llama.cpp/kld/qwen3.5-9B-bf16-{wiki,val}.kld` | 25.3 GB | KLD reference logits |
| `cache/activations` | 4.6 GB | gitignored; regenerate with Phase 5 |

To reclaim space (not done, your call):

```bash
rm /mnt/fast/models/Qwen3.5-9B-IQ4_XS-ffn_q2_k.gguf /mnt/fast/models/Qwen3.5-9B-IQ4_XS-ffn_iq1_s.gguf   # 8 GB
rm ~/llama.cpp/kld/qwen3.5-9B-bf16-{wiki,val}.kld      # 25 GB, only needed to score new quants
rm /mnt/fast/models/Qwen3.5-9B-BF16.gguf               # 18 GB, rebuild with Phase 8 in 30 s
```

## Part 5. Limits of these results

- All quality numbers are KL divergence and perplexity on next-token prediction, with no task
  benchmarks. For ranking variants that's the more sensitive measure, but it isn't a direct
  measure of answer quality.
- MTP speeds were measured at temperature 0 on one coding prompt. Acceptance, and with it the
  speedup, depends on the text and the sampler.
- The oracle MoE results are upper bounds for *training-free* conversion. They say nothing about
  what distillation could recover, only that it starts far behind and its speed ceiling on top of
  quantization is about 1.2x.
- The FFN-bytes proxy uses different dequantization kernels than a real sparse kernel would.
  It's an estimate of the attainable speedup, not a measurement of a sparse kernel.
