#!/usr/bin/env bash
# Seeded version of adopt_check.sh. At temperature 0 each build gives one fixed answer per task, so
# 44/50 (stock) vs 41/50 (patched) is one sample each. Here the daily sampler is used
# (temperature 1.0, top-p 0.95, top-k 20, min-p 0) with seeds 1..N, one server per build, to compare
# score distributions. Same preset as adopt_check.sh (both GPUs, MTP, q8_0 KV, 128K, thinking off).
#
# BUILDS selects what runs (default "stock patched"). "lenientR" is the patched build with
# SPEC_LENIENT_RATIO=R (patches/llama.cpp-spec-lenient.patch), e.g. BUILDS=lenient0.3.
# QTEMP=0 runs each build once at temperature 0 instead (output <build>_t0.txt).
#
#   SEEDS=12 benchmarks/adopt_check_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/.."
B=$HOME/code/local-llm-benchmarks
OUT=experiments/results/adopt/seeds
mkdir -p $OUT
declare -A BIN=([stock]=$HOME/llama.cpp/build/bin [patched]=$HOME/llama.cpp-b12164-mmvq/build/bin)
PORT=8099
SEEDS=${SEEDS:-12}
BUILDS=${BUILDS:-stock patched}
QTEMP=${QTEMP:-1.0}
if [ "$QTEMP" = 0 ]; then RUNS=t0; else RUNS=$(seq -f 's%g' $SEEDS); fi

for b in $BUILDS; do
  todo=0; for s in $RUNS; do [ -s $OUT/${b}_$s.txt ] || todo=1; done
  [ $todo = 0 ] && continue
  bin=${BIN[$b]:-${BIN[patched]}}
  case $b in lenient*) export SPEC_LENIENT_RATIO=${b#lenient};; *) unset SPEC_LENIENT_RATIO;; esac
  $bin/llama-server -m /mnt/fast/models/Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -c 131072 \
    --chat-template-file $B/templates/qwen38-late-system.jinja -ub 512 -b 2048 -fa on \
    -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --reasoning-budget 0 --load-mode dio --ctx-checkpoints 4 --host 127.0.0.1 --port $PORT \
    > $OUT/server_$b.log 2>&1 &
  pid=$!
  until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "$b: server failed to start"; exit 1; }
    sleep 1
  done
  for s in $RUNS; do
    o=$OUT/${b}_$s.txt
    [ -s "$o" ] && continue
    python3 $B/benchmarks/code-quality-test.py --host http://127.0.0.1:$PORT --label $b-$s \
      --temperature $QTEMP --top-p 0.95 --extra "{\"top_k\": 20, \"min_p\": 0.0, \"seed\": ${s#[st]}}" > "$o" 2>&1
  done
  kill $pid; wait $pid 2>/dev/null
  sleep 3
done
grep -h "checks pass" $OUT/*_[st]*.txt
