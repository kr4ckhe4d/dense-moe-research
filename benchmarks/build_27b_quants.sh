#!/usr/bin/env bash
# Qwen3.8-27B smaller quants for the offload milestone, on the R9700 (HIP 1):
#   1. imatrix from the same calibration text as the 9B (cache/corpus/calib.txt), on the Q8_0
#   2. IQ4_XS and Q3_K_M, requantized from Q8_0 with that imatrix (--allow-requantize)
#   3. KLD of UD-Q6_K, IQ4_XS, Q3_K_M against the existing Q8_0 base (wikitext-2 test, c=512,
#      100 chunks: ~/llama.cpp/kld/qwen3.8-27B-q8_0.kld)
# Requantizing from Q8_0 is slightly worse than quantizing from BF16; step 3 measures by how much
# relative to Q8_0. Outputs: /mnt/fast/models/Qwen3.8-27B-{IQ4_XS,Q3_K_M}-rq.gguf,
# experiments/results/offload/kld27b_*.txt.
#
#   benchmarks/build_27b_quants.sh
set -euo pipefail
cd "$(dirname "$0")/.."
B=~/llama.cpp/build/bin
M=/mnt/fast/models
Q8=$M/Qwen3.8-27B-Q8_0.gguf
IM=cache/gguf/imatrix-27b-calib.gguf
OUT=experiments/results/offload
LOG=experiments/logs/offload
mkdir -p "$OUT" "$LOG" cache/gguf
export HIP_VISIBLE_DEVICES=1

[ -f "$IM" ] || $B/llama-imatrix -m "$Q8" -f cache/corpus/calib.txt -c 512 -ngl 99 -fa on \
  -o "$IM" > $LOG/imatrix27b.log 2>&1

for q in IQ4_XS Q3_K_M; do
  f=$M/Qwen3.8-27B-$q-rq.gguf
  [ -f "$f" ] || $B/llama-quantize --allow-requantize --imatrix "$IM" "$Q8" "$f" $q \
    > $LOG/quantize27b_$q.log 2>&1
done

for f in Qwen3.8-27B-UD-Q6_K Qwen3.8-27B-IQ4_XS-rq Qwen3.8-27B-Q3_K_M-rq; do
  o=$OUT/kld27b_$f.txt
  [ -f "$o" ] || $B/llama-perplexity -m $M/$f.gguf -f ~/llama.cpp/kld/wikitext-2-raw/wiki.test.raw \
    -c 512 --chunks 100 -ngl 99 -fa on --kl-divergence-base ~/llama.cpp/kld/qwen3.8-27B-q8_0.kld \
    --kl-divergence > "$o" 2>&1
  grep -E "Mean PPL\(Q\) |Mean +KLD|Same top p" "$o" | sed "s|^|$f: |"
done
