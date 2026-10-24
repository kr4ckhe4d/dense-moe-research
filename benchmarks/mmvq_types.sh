#!/usr/bin/env bash
# Per-type check of the RDNA4 MMVQ patch on upstream b11719: Qwen3.5-9B quantized with --pure (all
# tensors one type) to the types the existing RDNA4 table treats as "complex vec_dot" (Q3_K, IQ1_*,
# IQ2_*, IQ3_*) and others not covered by the model runs (Q2_K, Q5_K, IQ4_NL). llama-bench pp1..8,
# stock vs patched, R9700. The pure files are temporary (made from Qwen3.5-9B-BF16.gguf with the
# calibration imatrix, see docs/upstream_mmvq.md) and deleted afterwards.
#
#   benchmarks/mmvq_types.sh
set -euo pipefail
cd "$(dirname "$0")/.."
D=/mnt/fast/models/tmp-puretypes
OUT=experiments/results/mmvq_upstream
declare -A BIN=([stock]=$HOME/llama.cpp-b11719/build/bin [patched]=$HOME/llama.cpp-b11719-mmvq/build/bin)
for t in IQ1_M IQ2_XXS IQ2_S IQ3_XXS IQ3_S Q2_K Q3_K_S Q5_K_S IQ4_NL; do
  for b in stock patched; do
    o=$OUT/types_${b}_gpu1_pure-$t.json
    [ -s "$o" ] || HIP_VISIBLE_DEVICES=1 ${BIN[$b]}/llama-bench -m $D/pure-9B-$t.gguf -ngl 99 -fa 1 \
      -p 1,2,3,4,5,6,8 -n 0 -r 5 -o json > "$o"
  done
done
