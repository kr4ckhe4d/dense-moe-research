#!/usr/bin/env bash
# Qwen3.8-27B quant formats on the R9700 (HIP 1): decode speed, the cost of verifying a short
# block of drafted tokens (one batch of 1-8 tokens, as speculative decoding does), and KLD.
#   1. Q5_K_M, Q4_K_M, Q4_0 requantized from Q8_0 with the calibration imatrix (as
#      build_27b_quants.sh did for IQ4_XS / Q3_K_M)
#   2. llama-bench: tg128 and pp1..8 (time to process n tokens in one batch) for every quant
#   3. KLD vs the Q8_0 base (wikitext-2 test, c=512, 100 chunks) for the new quants
# Outputs: experiments/results/fast27b/{bench_*.json,kld27b_*.txt}. Resumable.
#
#   benchmarks/fast27b_quants.sh
set -euo pipefail
cd "$(dirname "$0")/.."
B=~/llama.cpp/build/bin
M=/mnt/fast/models
Q8=$M/Qwen3.8-27B-Q8_0.gguf
IM=cache/gguf/imatrix-27b-calib.gguf
OUT=experiments/results/fast27b
LOG=experiments/logs/fast27b
mkdir -p "$OUT" "$LOG"
export HIP_VISIBLE_DEVICES=1

for q in Q5_K_M Q4_K_M Q4_0; do
  f=$M/Qwen3.8-27B-$q-rq.gguf
  [ -f "$f" ] || $B/llama-quantize --allow-requantize --imatrix "$IM" "$Q8" "$f" $q \
    > $LOG/quantize_$q.log 2>&1
done

for q in Q8_0 UD-Q6_K Q5_K_M-rq Q4_K_M-rq IQ4_XS-rq Q4_0-rq Q3_K_M-rq; do
  o=$OUT/bench_$q.json
  [ -s "$o" ] || $B/llama-bench -m $M/Qwen3.8-27B-$q.gguf -ngl 99 -fa 1 -p 1,2,3,4,5,6,8 -n 128 \
    -r 3 -o json > "$o" 2> $LOG/bench_$q.log
  echo "bench $q done"
done

for q in Q5_K_M-rq Q4_K_M-rq Q4_0-rq; do
  o=$OUT/kld27b_$q.txt
  [ -f "$o" ] || $B/llama-perplexity -m $M/Qwen3.8-27B-$q.gguf \
    -f ~/llama.cpp/kld/wikitext-2-raw/wiki.test.raw -c 512 --chunks 100 -ngl 99 -fa on \
    --kl-divergence-base ~/llama.cpp/kld/qwen3.8-27B-q8_0.kld --kl-divergence > "$o" 2>&1
  grep -E "Mean +KLD|Same top p" "$o" | sed "s|^|$q: |"
done
