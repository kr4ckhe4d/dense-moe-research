#!/usr/bin/env bash
# KL divergence of GGUF quants against the BF16 GGUF, with llama.cpp's llama-perplexity.
# Same protocol as ~/code/local-llm-benchmarks/benchmarks/kld-test.sh (-c 512, 255 scored
# tokens per chunk), so numbers are comparable with the user's other models.
#
#   benchmarks/gguf_kld.sh base  wiki            # BF16 reference logits (R9700, ~18 GB VRAM)
#   benchmarks/gguf_kld.sh score wiki Q4_K_M     # one quant against it
#
# Corpora: wiki = wikitext-2 test (the user's convention), val = our hash-disjoint val split
# (cache/corpus/val.txt from scripts/export_corpus.py; the imatrix only saw calib).
set -euo pipefail

BIN=$HOME/llama.cpp/build/bin/llama-perplexity
MODELS=/mnt/fast/models
KLD_DIR=$HOME/llama.cpp/kld
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HERE/experiments/results/gguf
CHUNKS=${CHUNKS:-100}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}   # R9700 (32 GB) alone

mode=${1:?base|score}; corpus=${2:?wiki|val}
case $corpus in
  wiki) TEXT=$KLD_DIR/wikitext-2-raw/wiki.test.raw ;;
  val)  TEXT=$HERE/cache/corpus/val.txt ;;
  *) echo "unknown corpus $corpus" >&2; exit 2 ;;
esac
BASE=$KLD_DIR/qwen3.5-9B-bf16-$corpus.kld
mkdir -p "$OUT"

if [ "$mode" = base ]; then
  "$BIN" -m $MODELS/Qwen3.5-9B-BF16.gguf -f "$TEXT" -c 512 --chunks $CHUNKS -ngl 99 \
    --kl-divergence-base "$BASE" 2>&1 | tee "$OUT/kld_${corpus}_BF16.txt" | grep -E "Final estimate"
else
  q=${3:?quant}
  "$BIN" -m $MODELS/Qwen3.5-9B-$q.gguf -f "$TEXT" -c 512 --chunks $CHUNKS -ngl 99 \
    --kl-divergence --kl-divergence-base "$BASE" > "$OUT/kld_${corpus}_$q.txt" 2>&1
  grep -E "Mean PPL\(Q\) |Mean +KLD|99.0% +KLD|Same top p" "$OUT/kld_${corpus}_$q.txt"
fi
