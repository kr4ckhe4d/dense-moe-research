#!/usr/bin/env bash
# Drafters on the R9700 with the RDNA4 MMVQ tuning (llama.cpp worktree ~/llama.cpp-mmvq, build/ =
# patches/llama.cpp-rdna4-mmvq-multicol.patch:
# 2 warps, 4 rows per block for 2-8 token batches) for six 27B quants. Same prompts and settings as
# fast27b_drafters.sh. Plain decoding is unchanged by the patch (1-token path), so it is only run
# for the quants that fast27b_drafters.sh did not cover.
#
#   benchmarks/fast27b_mmvq_drafters.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
DF=$M/Qwen3.8-27B-DFlash2-Q8_0.gguf
BIN=$HOME/llama.cpp-mmvq/build/bin
OUT=experiments/results/fast27b/drafters_mmvq_gpu1.jsonl

c=()
for q in UD-Q6_K Q8_0 IQ4_XS-rq Q5_K_M-rq Q4_K_M-rq Q4_0-rq; do
  base="\"quant\": \"27B-$q\", \"model_path\": \"$M/Qwen3.8-27B-$q.gguf\", \"bin\": \"$BIN\""
  sp='"n_min": 0, "p_min": 0.0'
  case $q in *-rq) [ $q = IQ4_XS-rq ] || c+=("{$base, \"spec\": \"none\"}");; esac
  for n in 3 4 5; do
    c+=("{$base, \"spec\": \"draft-mtp\", \"n_max\": $n, $sp}")
  done
  for n in 3 5 7; do
    c+=("{$base, \"spec\": \"draft-dflash\", \"n_max\": $n, $sp, \"extra\": [\"-md\", \"$DF\", \"-ngld\", \"99\"]}")
  done
done
IFS=,; cfgs="[${c[*]}]"; unset IFS
python3 benchmarks/mtp_tune.py --gpu 1 --prompt-set domains --max-tokens 512 --configs "$cfgs" --out $OUT
