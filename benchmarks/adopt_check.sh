#!/usr/bin/env bash
# Regression check before adopting the patched llama.cpp build for daily use: the code-quality
# suite of ~/code/local-llm-benchmarks (7 tasks, 50 hidden checks, temperature 0, thinking off) on
# the daily preset qwen3.8-27B-q6-128k-nothink (both GPUs, MTP, q8_0 KV, 128K), stock vs patched.
# Stock: ~/llama.cpp/build. Patched: ~/llama.cpp-b12164-mmvq/build (MMVQ + q8_0 attention patches).
#
#   benchmarks/adopt_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
B=$HOME/code/local-llm-benchmarks
OUT=experiments/results/adopt
mkdir -p $OUT
declare -A BIN=([stock]=$HOME/llama.cpp/build/bin [patched]=$HOME/llama.cpp-b12164-mmvq/build/bin)
PORT=8099

# Each build is run RUNS times with a fresh server (temperature 0), to show run-to-run spread.
RUNS=${RUNS:-3}
for r in $(seq $RUNS); do for b in stock patched; do
  o=$OUT/codeq_${b}_r$r.txt
  [ -s "$o" ] && continue
  ${BIN[$b]}/llama-server -m /mnt/fast/models/Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -c 131072 \
    --chat-template-file $B/templates/qwen38-late-system.jinja -ub 512 -b 2048 -fa on \
    -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --reasoning-budget 0 --load-mode dio --ctx-checkpoints 4 --host 127.0.0.1 --port $PORT \
    > $OUT/server_${b}_r$r.log 2>&1 &
  pid=$!
  until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "$b: server failed to start"; exit 1; }
    sleep 1
  done
  { echo "# build: $b ($(${BIN[$b]}/llama-server --version 2>&1 | head -1)), $(date -Is)"
    python3 $B/benchmarks/code-quality-test.py --host http://127.0.0.1:$PORT --label $b-r$r-t0 \
      --temperature 0.0 --save-dir $OUT/codeq_${b}_r$r
    grep -E "draft acceptance" $OUT/server_${b}_r$r.log | tail -3
  } > "$o" 2>&1
  kill $pid; wait $pid 2>/dev/null
  sleep 3
done; done
grep -h "checks pass" $OUT/codeq_*_r*.txt
