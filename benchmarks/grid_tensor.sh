#!/usr/bin/env bash
# -sm tensor (both cards work on every layer) against the default layer split, two cards, 128K,
# q8_0 KV, -ub 512, MTP. Stock b12164 has the mode too (EXPERIMENTAL in its help), so every build
# is run with it. HIP_VISIBLE_DEVICES=0,1 hides the integrated GPU, which the mode otherwise
# takes as a third device and crashes on.
# Rows already in the jsonl are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
export HIP_VISIBLE_DEVICES=0,1
OUT=experiments/results/grid/tensor.jsonl
STOCK=$HOME/llama.cpp/build/bin
OURS=$HOME/llama.cpp-b12164-mmvq/build/bin
FORK=$HOME/llama.cpp-rdna-fork/build/bin
IDS=$PWD/experiments/results/shortlist/ids_65536.txt
V=',"--spec-draft-mtp-vocab","98304"'

run() {  # <model file> <name> <bin> <spec> <n_max> <extra> [VAR=value]
  env "${7:-GRID_NOENV=1}" python3 benchmarks/depth_profile.py --all-gpus --model "$HOME/llama.cpp/models/$1" \
    --depths 2000 16000 32000 64000 120000 --out "$OUT" \
    --configs "[{\"name\":\"$2\",\"bin\":\"$3\",\"spec\":\"$4\",\"n_max\":$5,\"extra\":[\"-ctk\",\"q8_0\",\"-ctv\",\"q8_0\",\"-ub\",\"512\",\"-b\",\"2048\",\"-sm\",\"tensor\"$6]}]"
}
Q4=Qwen3.8-27B-UD-Q4_K_XL-v31.gguf
Q6=Qwen3.8-27B-UD-Q6_K.gguf
run $Q4 q4xl-stock-tensor-n3          "$STOCK" draft-mtp 3 ""
run $Q4 q4xl-fork-tensor-n3           "$FORK"  draft-mtp 3 ""
run $Q4 q4xl-ours-tensor-n3           "$OURS"  draft-mtp 3 ""
run $Q6 q6k-stock-tensor-n3           "$STOCK" draft-mtp 3 ""
run $Q6 q6k-fork-tensor-n3            "$FORK"  draft-mtp 3 ""
run $Q4 q4xl-fork-tensor-n3-vocab98k  "$FORK"  draft-mtp 3 "$V"
run $Q4 q4xl-fork-tensor-n4           "$FORK"  draft-mtp 4 ""
run $Q4 q4xl-fork-tensor-adaptive-n7  "$FORK"  draft-mtp-adaptive 7 ""
run $Q4 q4xl-ours-tensor-sl65k-n4     "$OURS"  draft-mtp 4 "" "LLAMA_MTP_DRAFT_VOCAB=$IDS"
