#!/usr/bin/env bash
# Re-check of patches/llama.cpp-rdna4-mmvq-multicol.patch on llama.cpp b12164 (4b77dee05), which
# has upstream's own MMVQ rework (#30517) since the b11719 check in docs/upstream_mmvq.md.
# Stock: ~/llama.cpp/build (4b77dee05 plus local server-only changes). Patched:
# ~/llama.cpp-b12164-mmvq/build (4b77dee05 + patch, same cmake flags).
#   1. llama-bench: one batch of 1..8 tokens (the verify step), pp512, tg128; 5 reps; R9700.
#   2. test-backend-ops MUL_MAT / MUL_MAT_ID on the patched build.
# Resumable.
#
#   benchmarks/mmvq_b12164.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
OUT=experiments/results/mmvq_b12164
mkdir -p $OUT
declare -A BIN=([stock]=$HOME/llama.cpp/build/bin [patched]=$HOME/llama.cpp-b12164-mmvq/build/bin)

bench() {  # bench GPU NAME FILE
  for b in stock patched; do
    o=$OUT/bench_${b}_gpu$1_$2.json
    [ -s "$o" ] || HIP_VISIBLE_DEVICES=$1 ${BIN[$b]}/llama-bench -m $M/$3 -ngl 99 -fa 1 \
      -p 1,2,3,4,5,6,8,512 -n 128 -r 5 -o json > "$o"
  done
}
bench 1 27B-UD-Q6_K Qwen3.8-27B-UD-Q6_K.gguf
bench 1 27B-Q8_0 Qwen3.8-27B-Q8_0.gguf
bench 0 27B-Q3_K_M Qwen3.5-27B-Uncensored-Q3_K_M.gguf

for op in MUL_MAT MUL_MAT_ID; do
  o=$OUT/test-backend-ops_gpu1_$op.txt
  [ -s "$o" ] || HIP_VISIBLE_DEVICES=1 ${BIN[patched]}/test-backend-ops -o $op > "$o" 2>&1 || true
done
