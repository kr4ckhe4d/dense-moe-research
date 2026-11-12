#!/usr/bin/env bash
# Builds against each other on the daily two-card preset shape (both cards, 128K, q8_0 KV,
# -ub 512 -b 2048, MTP), at 2K and 120K of context, for the preset file and the Unsloth
# Dynamic v3.1 UD-Q4_K_XL (local-llm-benchmarks r9700+rx9070.md, 2026-11-12).
#
#   stock-b12164   ~/llama.cpp                      4b77dee05, what the presets run
#   stock-a691     ~/llama.cpp-a691b2345            a691b2345, the fork's base, unpatched
#   ours           ~/llama.cpp-b12164-mmvq          b12164 + MMVQ + attention patches
#   ours-sl65k-n4  same + LLAMA_MTP_DRAFT_VOCAB (65,536-token shortlist), draft 4
#   fork-n3        ~/llama.cpp-rdna-fork            stew675/llama-cpp-rdna-boosts v17-a691b2345-r265
#   fork-adaptive  same, --spec-type draft-mtp-adaptive (the fork's own drafter), n-max 7
#
#   benchmarks/grid_builds.sh            # rows already in the jsonl are skipped
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/grid
mkdir -p "$OUT"
IDS=$PWD/experiments/results/shortlist/ids_65536.txt
KV='"-ctk","q8_0","-ctv","q8_0","-ub","512","-b","2048"'

run() {  # <tag> <model file> <config name> <bin dir> <spec> <n_max> [VAR=value]
  local tag=$1 model=$2 name=$3 bin=$4 spec=$5 n=$6 var=${7:-GRID_NOENV=1}
  env "$var" python3 benchmarks/depth_profile.py --all-gpus --model "$HOME/llama.cpp/models/$model" \
    --depths 2000 120000 --out "$OUT/$tag.jsonl" \
    --configs "[{\"name\":\"$tag-$name\",\"bin\":\"$bin\",\"spec\":\"$spec\",\"n_max\":$n,\"extra\":[$KV]}]"
}

for pair in q6k:Qwen3.8-27B-UD-Q6_K.gguf q4xl-v31:Qwen3.8-27B-UD-Q4_K_XL-v31.gguf; do
  tag=${pair%%:*}; model=${pair#*:}
  run "$tag" "$model" stock-b12164  "$HOME/llama.cpp/build/bin"             draft-mtp 3
  run "$tag" "$model" stock-a691    "$HOME/llama.cpp-a691b2345/build/bin"   draft-mtp 3
  run "$tag" "$model" ours          "$HOME/llama.cpp-b12164-mmvq/build/bin" draft-mtp 3
  run "$tag" "$model" ours-sl65k-n4 "$HOME/llama.cpp-b12164-mmvq/build/bin" draft-mtp 4 "LLAMA_MTP_DRAFT_VOCAB=$IDS"
  run "$tag" "$model" fork-n3       "$HOME/llama.cpp-rdna-fork/build/bin"   draft-mtp 3
  run "$tag" "$model" fork-adaptive "$HOME/llama.cpp-rdna-fork/build/bin"   draft-mtp-adaptive 7
done
