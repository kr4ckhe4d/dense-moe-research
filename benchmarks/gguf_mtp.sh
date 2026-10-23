#!/usr/bin/env bash
# Decode tok/s of Qwen3.5-9B GGUF quants with and without the in-file MTP head
# (--spec-type draft-mtp), via the user's throughput probe (700-token code answer, temp 0,
# thinking off, median of runs): ~/code/local-llm-benchmarks/benchmarks/throughput-test.py
#
#   benchmarks/gguf_mtp.sh 0 Q8_0 Q6_K IQ4_XS      # GPU index (HIP), then quants
set -euo pipefail

GPU=${1:?hip device index}; shift
PORT=8199
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HERE/experiments/results/gguf/mtp_gpu$GPU.txt
PROBE=$HOME/code/local-llm-benchmarks/benchmarks/throughput-test.py
: > "$OUT"

for q in "$@"; do
  for spec in none draft-mtp; do
    args=(-m /mnt/fast/models/Qwen3.5-9B-$q.gguf -ngl 99 -fa on -c 8192 --port $PORT --no-webui)
    [ $spec = none ] || args+=(--spec-type $spec)
    HIP_VISIBLE_DEVICES=$GPU ~/llama.cpp/build/bin/llama-server "${args[@]}" \
      > "$HERE/experiments/logs/server_${q}_$spec.log" 2>&1 &
    pid=$!
    for _ in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 1; done
    { echo "== $q spec=$spec gpu=$GPU"; python3 "$PROBE" --host http://127.0.0.1:$PORT --runs 3; } \
      | tee -a "$OUT" | grep -E "^==|^tg|^draft|^prompt"
    kill $pid; wait $pid 2>/dev/null || true
  done
done
