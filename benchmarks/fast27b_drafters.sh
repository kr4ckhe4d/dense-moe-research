#!/usr/bin/env bash
# Qwen3.8-27B fully on the R9700 (HIP 1): plain decoding vs the in-file MTP head vs the DFlash2
# block-diffusion drafter (Anbeeld/Qwen3.8-27B-DFlash2-GGUF, Q8_0), over draft lengths, for three
# target quants. Six domain prompts x 512 tokens, temperature 0. Resumable (mtp_tune.py skips
# configs already in the JSONL).
#
#   benchmarks/fast27b_drafters.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
DF=$M/Qwen3.8-27B-DFlash2-Q8_0.gguf
OUT=experiments/results/fast27b/drafters_gpu1.jsonl
mkdir -p "$(dirname $OUT)"

c=()
for q in UD-Q6_K IQ4_XS-rq Q8_0; do
  base="\"quant\": \"27B-$q\", \"model_path\": \"$M/Qwen3.8-27B-$q.gguf\""
  sp='"n_min": 0, "p_min": 0.0'
  c+=("{$base, \"spec\": \"none\"}")
  for n in 2 3 4 5 6; do
    c+=("{$base, \"spec\": \"draft-mtp\", \"n_max\": $n, $sp}")
  done
  for n in 3 5 7; do
    c+=("{$base, \"spec\": \"draft-dflash\", \"n_max\": $n, $sp, \"extra\": [\"-md\", \"$DF\", \"-ngld\", \"99\"]}")
  done
done
IFS=,; cfgs="[${c[*]}]"; unset IFS
python3 benchmarks/mtp_tune.py --gpu 1 --prompt-set domains --max-tokens 512 --configs "$cfgs" --out $OUT
