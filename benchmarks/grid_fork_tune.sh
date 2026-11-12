#!/usr/bin/env bash
# Follow-up to grid_builds.sh on the UD-Q4_K_XL v3.1 file, two cards, 128K, q8_0 KV:
#   1. prefill: stock and the RDNA fork at -ub 512 / 2048 / 4096, and the fork with -sm tensor
#   2. drafter on the fork: adaptive with n-max 3 / 4 / 5 (7 is in q4xl-v31.jsonl), fixed draft 4,
#      and the fork's own draft vocabulary subset (--spec-draft-mtp-vocab)
# Rows already in the jsonl are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=experiments/results/grid/fork_tune.jsonl
MODEL=$HOME/llama.cpp/models/Qwen3.8-27B-UD-Q4_K_XL-v31.gguf
STOCK=$HOME/llama.cpp/build/bin
FORK=$HOME/llama.cpp-rdna-fork/build/bin

run() {  # <name> <bin> <spec> <n_max> <extra flags, JSON list body>
  python3 benchmarks/depth_profile.py --all-gpus --model "$MODEL" --depths 2000 120000 --out "$OUT" \
    --configs "[{\"name\":\"ft-$1\",\"bin\":\"$2\",\"spec\":\"$3\",\"n_max\":$4,\"extra\":[\"-ctk\",\"q8_0\",\"-ctv\",\"q8_0\",$5]}]"
}
ub() { echo "\"-ub\",\"$1\",\"-b\",\"$(( $1 > 2048 ? $1 : 2048 ))\""; }

for u in 2048 4096; do
  run "stock-ub$u" "$STOCK" draft-mtp 3 "$(ub $u)"
  run "fork-ub$u"  "$FORK"  draft-mtp 3 "$(ub $u)"
done
for n in 3 4 5; do run "fork-adaptive-n$n" "$FORK" draft-mtp-adaptive $n "$(ub 512)"; done
run fork-n4            "$FORK" draft-mtp 4 "$(ub 512)"
run fork-n4-vocab98k   "$FORK" draft-mtp 4 "$(ub 512),\"--spec-draft-mtp-vocab\",\"98304\""
run fork-n4-vocab65k   "$FORK" draft-mtp 4 "$(ub 512),\"--spec-draft-mtp-vocab\",\"65536\""
run fork-adaptive-n7-vocab98k "$FORK" draft-mtp-adaptive 7 "$(ub 512),\"--spec-draft-mtp-vocab\",\"98304\""

# -sm tensor needs the integrated GPU hidden (it segfaults at load with gfx1036 in the device list)
export HIP_VISIBLE_DEVICES=0,1
run fork-ub512-tensor  "$FORK" draft-mtp 3 "$(ub 512),\"-sm\",\"tensor\""
run fork-ub4096-tensor "$FORK" draft-mtp 3 "$(ub 4096),\"-sm\",\"tensor\""
unset HIP_VISIBLE_DEVICES

# 3. the finalists where a session spends its time: 16K, 32K, 64K of context
OUT=experiments/results/grid/fork_mid.jsonl
mid() {  # <name> <bin> <spec> <n_max> <extra> <depths...>
  local name=$1 bin=$2 spec=$3 n=$4 extra=$5; shift 5
  python3 benchmarks/depth_profile.py --all-gpus --model "$MODEL" --depths "$@" --out "$OUT" \
    --configs "[{\"name\":\"mid-$name\",\"bin\":\"$bin\",\"spec\":\"$spec\",\"n_max\":$n,\"extra\":[\"-ctk\",\"q8_0\",\"-ctv\",\"q8_0\",$(ub 512)$extra]}]"
}
V=',"--spec-draft-mtp-vocab","98304"'
mid stock-n3                  "$STOCK" draft-mtp 3 ""   16000 32000 64000
mid fork-n3                   "$FORK"  draft-mtp 3 ""   16000 32000 64000
mid fork-n3-vocab98k          "$FORK"  draft-mtp 3 "$V" 2000 16000 32000 64000 120000
mid fork-n4-vocab98k          "$FORK"  draft-mtp 4 "$V" 16000 32000 64000
mid fork-adaptive-n7          "$FORK"  draft-mtp-adaptive 7 ""   16000 32000 64000
mid fork-adaptive-n7-vocab98k "$FORK"  draft-mtp-adaptive 7 "$V" 16000 32000 64000
