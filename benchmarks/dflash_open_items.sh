#!/usr/bin/env bash
# DFlash vs MTP beyond short greedy prompts, models fully on the GPU, patched build
# (patches/llama.cpp-rdna4-mmvq-multicol.patch): the 9B with its own DFlash drafter, long context,
# sampling, repetitive editing (+ n-gram drafting), 4 concurrent requests, and the RX 9070 XT.
# Resumable: each experiment appends to its own JSONL and skips what is already there.
#
#   benchmarks/dflash_open_items.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
R=experiments/results/dflash
T=benchmarks/mtp_tune.py
BIN=$HOME/llama.cpp-mmvq/build/bin
mkdir -p $R

cfg() {  # cfg MODEL SPEC N_MAX [json fields...] -> one config; MODEL = 9B-<q> or 27B-<q>
  local m=$1 spec=$2 n=$3; shift 3
  local size=${m%%-*} q=${m#*-} path df
  case $size in
    9B) path=$M/Qwen3.5-9B-$q.gguf; df=$M/Qwen3.5-9B-DFlash-Q8_0.gguf ;;
    27B) path=$M/Qwen3.8-27B-$q.gguf; df=$M/Qwen3.8-27B-DFlash2-Q8_0.gguf ;;
  esac
  local s="\"quant\": \"$m\", \"model_path\": \"$path\", \"bin\": \"$BIN\", \"spec\": \"$spec\""
  [ "$spec" = none ] || s+=", \"n_max\": $n, \"n_min\": 0, \"p_min\": 0.0"
  case $spec in *dflash*) s+=", \"extra\": [\"-md\", \"$df\", \"-ngld\", \"99\"${DF_EXTRA:-}]" ;; esac
  echo "{$s${1:+, $*}}"
}
list() { local IFS=,; echo "[$*]"; }

BIG=(27B-UD-Q6_K 27B-IQ4_XS-rq)
NINE=(9B-Q8_0 9B-Q6_K 9B-IQ4_XS)

# 1. 9B: plain, MTP n=3, DFlash n=3/5/7 on both cards (six domains, 512 tokens, t=0).
for g in 1 0; do
  c=()
  for m in "${NINE[@]}"; do
    c+=("$(cfg $m none 0)" "$(cfg $m draft-mtp 3)")
    for n in 3 5 7; do c+=("$(cfg $m draft-dflash $n)"); done
  done
  python3 $T --gpu $g --configs "$(list "${c[@]}")" --out $R/nine_gpu$g.jsonl
done

# 2. Long context: ~4k / 16k / 28k tokens of llama.cpp source, then a question.
c=()
for m in "${BIG[@]}" 9B-Q6_K; do
  c+=("$(cfg $m none 0 '"ctx": 32768')" "$(cfg $m draft-mtp 3 '"ctx": 32768')"
      "$(cfg $m draft-dflash 3 '"ctx": 32768')" "$(cfg $m draft-dflash 5 '"ctx": 32768')")
done
python3 $T --gpu 1 --prompt-set longctx --configs "$(list "${c[@]}")" --out $R/longctx_gpu1.jsonl

# 3. Sampling: temperature 0.7 (seeds 0-2) and 1.0 (seed 0), greedy and probabilistic drafts.
c=()
for m in "${BIG[@]}" 9B-Q6_K; do
  c+=("$(cfg $m draft-mtp 3)" "$(cfg $m draft-dflash 5)"
      "$(DF_EXTRA=', "--spec-draft-sampling", "probabilistic"' cfg $m draft-dflash 5)")
done
for s in 0 1 2; do
  python3 $T --gpu 1 --temperature 0.7 --seed $s --configs "$(list "${c[@]}")" --out $R/sampling_gpu1.jsonl
done
python3 $T --gpu 1 --temperature 1.0 --configs "$(list "${c[@]}")" --out $R/sampling_gpu1.jsonl

# 4. Repetitive editing: reproduce a file with small additions (1024 tokens).
c=()
for m in "${BIG[@]}"; do
  for s in none draft-mtp draft-dflash ngram-mod draft-mtp,ngram-mod draft-dflash,ngram-mod \
           draft-dflash,ngram-map-k; do
    n=3; case $s in *dflash*) n=5 ;; esac
    c+=("$(cfg $m $s $n '"ctx": 16384')")
  done
done
python3 $T --gpu 1 --prompt-set edit --max-tokens 1024 --configs "$(list "${c[@]}")" \
  --out $R/edit_gpu1.jsonl

# 5. Four concurrent requests (4 server slots), aggregate throughput.
c=()
for m in "${BIG[@]}"; do
  c+=("$(cfg $m none 0 '"ctx": 16384, "np": 4')" "$(cfg $m draft-mtp 3 '"ctx": 16384, "np": 4')"
      "$(cfg $m draft-dflash 5 '"ctx": 16384, "np": 4')")
done
python3 $T --gpu 1 --configs "$(list "${c[@]}")" --out $R/concurrency_gpu1.jsonl
