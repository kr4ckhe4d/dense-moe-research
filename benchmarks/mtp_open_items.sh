#!/usr/bin/env bash
# MTP follow-up: long context, repetitive editing, seeds at temperature 0.7, concurrent requests
# (all on the R9700, HIP 1), then a short sweep on the RX 9070 XT (HIP 0). Resumable: each
# experiment appends to its own JSONL and skips what is already there.
#
#   benchmarks/mtp_open_items.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/mtp
T=benchmarks/mtp_tune.py

cfg() {  # cfg QUANT SPEC [json fields...] -> one config object
  local q=$1 spec=$2; shift 2
  if [ "$spec" = none ]; then
    echo "{\"quant\": \"$q\", \"spec\": \"none\"${1:+, $*}}"
  else
    echo "{\"quant\": \"$q\", \"spec\": \"$spec\", \"n_max\": 3, \"n_min\": 0, \"p_min\": 0.0${1:+, $*}}"
  fi
}
list() { local IFS=,; echo "[$*]"; }

QS=(Q6_K IQ4_XS)

# 1. Long context: ~4k / 16k / 28k tokens of llama.cpp source, then a question.
c=()
for q in "${QS[@]}"; do
  c+=("$(cfg $q none '"ctx": 32768')" "$(cfg $q draft-mtp '"ctx": 32768')"
      "$(cfg $q draft-mtp '"ctx": 32768, "n_max": 2')")
done
python3 $T --gpu 1 --prompt-set longctx --configs "$(list "${c[@]}")" --out $R/mtp_longctx_gpu1.jsonl

# 2. Repetitive editing: reproduce a file with small additions (1024 tokens).
c=()
for q in "${QS[@]}"; do
  c+=("$(cfg $q none '"ctx": 16384')" "$(cfg $q draft-mtp '"ctx": 16384')"
      "$(cfg $q draft-mtp '"ctx": 16384, "n_max": 6')"
      "$(cfg $q draft-mtp,ngram-mod '"ctx": 16384')" "$(cfg $q draft-mtp,ngram-map-k '"ctx": 16384')"
      "$(cfg $q ngram-mod '"ctx": 16384')" "$(cfg $q ngram-map-k '"ctx": 16384')")
done
python3 $T --gpu 1 --prompt-set edit --max-tokens 1024 --configs "$(list "${c[@]}")" \
  --out $R/mtp_edit_gpu1.jsonl

# 3. Temperature 0.7, two more seeds (seed 0 is in mtp_tune_gpu1.jsonl).
c=()
for q in "${QS[@]}"; do
  c+=("$(cfg $q draft-mtp)" "$(cfg $q draft-mtp '"extra": ["--spec-draft-sampling", "probabilistic"]')")
done
for s in 1 2; do
  python3 $T --gpu 1 --temperature 0.7 --seed $s --configs "$(list "${c[@]}")" --out $R/mtp_tune_gpu1.jsonl
done

# 4. Four concurrent requests (4 server slots), aggregate throughput.
c=()
for q in "${QS[@]}"; do
  c+=("$(cfg $q none '"ctx": 16384, "np": 4')" "$(cfg $q draft-mtp '"ctx": 16384, "np": 4')")
done
python3 $T --gpu 1 --configs "$(list "${c[@]}")" --out $R/mtp_concurrency_gpu1.jsonl

# 5. RX 9070 XT: same six domain prompts, draft length 2-4.
c=()
for q in Q8_0 Q6_K IQ4_XS; do
  c+=("$(cfg $q none)" "$(cfg $q draft-mtp '"n_max": 2')" "$(cfg $q draft-mtp)" "$(cfg $q draft-mtp '"n_max": 4')")
done
python3 $T --gpu 0 --configs "$(list "${c[@]}")" --out $R/mtp_tune_gpu0.jsonl
