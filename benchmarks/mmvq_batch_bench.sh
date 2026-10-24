#!/usr/bin/env bash
# Small-batch matmul cost on the R9700 (HIP 1): llama-bench pp1..8 (one batch of n tokens, as a
# speculative verify step) for 27B quants, with a given llama.cpp build. Used to tune the RDNA4
# MMVQ kernel parameters (benchmarks: docs/fast27b.md).
#
#   benchmarks/mmvq_batch_bench.sh BIN_DIR LABEL [quants...]
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$1 LABEL=$2; shift 2
QS=("${@:-UD-Q6_K Q4_K_M-rq Q8_0 IQ4_XS-rq}")
OUT=experiments/results/fast27b/mmvq
mkdir -p $OUT
for q in ${QS[@]}; do
  o=$OUT/${LABEL}_$q.json
  [ -s "$o" ] || HIP_VISIBLE_DEVICES=1 $BIN/llama-bench -m /mnt/fast/models/Qwen3.8-27B-$q.gguf -ngl 99 \
    -fa 1 -p 1,2,3,4,5,6,8 -n 0 -r 5 -o json > "$o"
  python3 -c "
import json; rs=json.load(open('$o')); ms={r['n_prompt']:1000*r['n_prompt']/r['avg_ts'] for r in rs}
print('$LABEL', '$q'.ljust(10), ' '.join(f'{n}:{ms[n]:5.1f}' for n in sorted(ms)), f't6/t1={ms[6]/ms[1]:.2f}')"
done
