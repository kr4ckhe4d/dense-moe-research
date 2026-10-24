#!/usr/bin/env bash
# Evidence for upstreaming patches/llama.cpp-rdna4-mmvq-multicol.patch, on upstream master b11719
# (5aeac37a3): stock (~/llama.cpp-b11719) vs patched (~/llama.cpp-b11719-mmvq), same commit.
#   1. llama-bench pp1..8, pp512, tg128 (5 reps), both GPUs, dense and MoE models.
#   2. test-backend-ops MUL_MAT / MUL_MAT_ID on both GPUs.
#   3. llama-perplexity, 9B Q6_K, wikitext-2 test, 50 chunks (expected identical: the patch only
#      changes 2-8 column batches).
#   4. End to end: llama-server, 27B UD-Q6_K + DFlash2 n=5 and MTP n=3, six domains, R9700.
# Resumable.
#
#   benchmarks/mmvq_upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=/mnt/fast/models
OUT=experiments/results/mmvq_upstream
mkdir -p $OUT
declare -A BIN=([stock]=$HOME/llama.cpp-b11719/build/bin [patched]=$HOME/llama.cpp-b11719-mmvq/build/bin)

bench() {  # bench GPU NAME FILE
  for b in stock patched; do
    o=$OUT/bench_${b}_gpu$1_$2.json
    [ -s "$o" ] || HIP_VISIBLE_DEVICES=$1 ${BIN[$b]}/llama-bench -m $M/$3 -ngl 99 -fa 1 \
      -p 1,2,3,4,5,6,8,512 -n 128 -r 5 -o json > "$o"
  done
}
bench 1 27B-UD-Q6_K Qwen3.8-27B-UD-Q6_K.gguf
bench 1 27B-Q8_0 Qwen3.8-27B-Q8_0.gguf
bench 1 gemma-4-31B-Q6_K gemma-4-31B-it-Q6_K.gguf
bench 1 Qwen3.6-35B-A3B-UD-Q6_K Qwen3.6-35B-A3B-UD-Q6_K.gguf
bench 1 9B-Q4_K_M Qwen3.5-9B-Q4_K_M.gguf
bench 0 9B-Q6_K Qwen3.5-9B-Q6_K.gguf
bench 0 27B-IQ4_XS-rq Qwen3.8-27B-IQ4_XS-rq.gguf
bench 0 gpt-oss-20b-mxfp4 gpt-oss-20b-mxfp4.gguf

for g in 0 1; do
  for op in MUL_MAT MUL_MAT_ID; do
    o=$OUT/test-backend-ops_gpu${g}_$op.txt
    [ -s "$o" ] || HIP_VISIBLE_DEVICES=$g ${BIN[patched]}/test-backend-ops -o $op > "$o" 2>&1 || true
  done
done

for b in stock patched; do
  o=$OUT/ppl_$b.txt
  [ -s "$o" ] || HIP_VISIBLE_DEVICES=1 ${BIN[$b]}/llama-perplexity -m $M/Qwen3.5-9B-Q6_K.gguf -ngl 99 \
    -fa 1 -f $HOME/llama.cpp/kld/wikitext-2-raw/wiki.test.raw --chunks 50 > "$o" 2>&1
done

c=()
DF=$M/Qwen3.8-27B-DFlash2-Q8_0.gguf
for b in stock patched; do
  base="\"quant\": \"27B-UD-Q6_K\", \"model_path\": \"$M/Qwen3.8-27B-UD-Q6_K.gguf\", \"bin\": \"${BIN[$b]}\", \"n_min\": 0, \"p_min\": 0.0"
  c+=("{$base, \"spec\": \"draft-mtp\", \"n_max\": 3}"
      "{$base, \"spec\": \"draft-dflash\", \"n_max\": 5, \"extra\": [\"-md\", \"$DF\", \"-ngld\", \"99\"]}")
done
IFS=,; cfgs="[${c[*]}]"; unset IFS
python3 benchmarks/mtp_tune.py --gpu 1 --configs "$cfgs" --out $OUT/server_gpu1.jsonl
