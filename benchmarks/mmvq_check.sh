#!/usr/bin/env bash
# Does the RDNA4 MMVQ patch (patches/llama.cpp-rdna4-mmvq-multicol.patch) slow anything down?
# llama-bench pp1..8 (one batch of n tokens) with the stock build (~/llama.cpp, same base commit)
# and the patched build (~/llama.cpp-mmvq/build), alternating per model, on both GPUs: the 9B
# quants, another dense shape (Gemma 4 31B) and MoE models (MUL_MAT_ID path), then
# test-backend-ops MUL_MAT / MUL_MAT_ID on the RX 9070 XT. Resumable. The Nemotron 3.5 74B-A9B file
# on disk (pruned to 240 experts) does not load in this llama.cpp, so it is not included.
#
#   benchmarks/mmvq_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
OUT=experiments/results/mmvq_check
mkdir -p $OUT
declare -A BIN=([stock]=$HOME/llama.cpp/build/bin [patched]=$HOME/llama.cpp-mmvq/build/bin)

run() {  # run GPU NAME FILE
  local g=$1 name=$2 f=$3
  for b in stock patched; do
    o=$OUT/${b}_gpu${g}_$name.json
    [ -s "$o" ] || HIP_VISIBLE_DEVICES=$g ${BIN[$b]}/llama-bench -m $M/$f -ngl 99 -fa 1 \
      -p 1,2,3,4,5,6,8 -n 0 -r 5 -o json > "$o"
  done
}

NINE=(Q8_0 Q6_K Q4_K_M IQ4_XS)
for g in 1 0; do
  for q in "${NINE[@]}"; do run $g 9B-$q Qwen3.5-9B-$q.gguf; done
  run $g gpt-oss-20b-mxfp4 gpt-oss-20b-mxfp4.gguf
done
run 0 27B-Q3_K_M-rq Qwen3.8-27B-Q3_K_M-rq.gguf
run 1 gemma-4-31B-Q6_K gemma-4-31B-it-Q6_K.gguf
run 1 Qwen3.6-35B-A3B-UD-Q6_K Qwen3.6-35B-A3B-UD-Q6_K.gguf

for op in MUL_MAT MUL_MAT_ID; do
  o=$OUT/test-backend-ops_gpu0_$op.txt
  [ -s "$o" ] || HIP_VISIBLE_DEVICES=0 ${BIN[patched]}/test-backend-ops -o $op > "$o" 2>&1 || true
  tail -2 "$o"
done
