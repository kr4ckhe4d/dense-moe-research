"""Tables for docs/depth.md: the MMVQ patch on b12164 (benchmarks/mmvq_b12164.sh) and decode
speed against context depth (benchmarks/depth_profile.py).

python3 scripts/depth_summary.py > experiments/results/depth/summary.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

K = Path("experiments/results/mmvq_b12164")
D = Path("experiments/results/depth/depth.jsonl")
GPU = {0: "RX 9070 XT", 1: "R9700"}
NS = (1, 2, 3, 4, 5, 6, 8)
# KV cache bytes per token of context, Qwen3.8-27B: 16 attention layers x 4 KV heads x 256 x (K + V)
KV_BYTES = {"f16": 16 * 4 * 256 * 2 * 2, "q8_0": 16 * 4 * 256 * 2 * 34 / 32}
PEAK_GBS = 640  # R9700 memory bandwidth used in docs/fast27b.md


def batch_ms(path: Path) -> dict[int, float]:
    return {r["n_prompt"]: 1000 * r["n_prompt"] / r["avg_ts"]
            for r in json.loads(path.read_text()) if r["n_prompt"]}


def kernel() -> None:
    print("## MMVQ patch on b12164 (4b77dee05): ms per batch of n tokens\n")
    print("llama-bench -ngl 99 -fa 1 -r 5. Ratio = patched / stock time (< 1 is faster).\n")
    print("| GPU | model | build | " + " | ".join(map(str, NS)) + " | 512 | t(6)/t(1) |")
    print("|---|---|---|" + "---:|" * (len(NS) + 2))
    for p in sorted(K.glob("bench_stock_gpu*_*.json")):
        g = int(p.stem.split("_gpu")[1][0])
        name = p.stem.split(f"_gpu{g}_")[1]
        s, t = batch_ms(p), batch_ms(p.with_name(p.name.replace("stock", "patched")))
        t1 = min(s[1], t[1])  # the 1-token path is the same code in both builds
        for label, v in (("stock", s), ("patched", t)):
            print(f"| {GPU[g]} | {name} | {label} | " + " | ".join(f"{v[n]:.1f}" for n in NS)
                  + f" | {v[512]:.0f} | {v[6] / t1:.2f} |")
        print(f"| | | ratio | " + " | ".join(f"{t[n] / s[n]:.2f}" for n in NS)
              + f" | {t[512] / s[512]:.2f} | |")
    print()


def depth() -> None:
    rows = [json.loads(line) for line in D.read_text().splitlines()]
    by = defaultdict(dict)  # config -> target depth -> mean over the two tasks
    for r in rows:
        by[r["config"]["name"]].setdefault(r["target_depth"], []).append(r)
    depths = sorted({r["target_depth"] for r in rows})

    def mean(name: str, d: int, key: str) -> float | None:
        rs = by[name].get(d)
        return sum(r[key] for r in rs) / len(rs) if rs else None

    def fmt(v: float | None, f: str = ".1f") -> str:
        return "" if v is None else format(v, f)

    head = "| config | " + " | ".join(f"{d // 1000}K" for d in depths) + " |"
    sep = "|---|" + "---:|" * len(depths)
    print("## Decode against context depth: Qwen3.8-27B UD-Q6_K on the R9700, b12164\n")
    print("Mean of the two tasks (explain, code), 256 tokens each, temperature 0. Depth is the "
          "prompt length in tokens (actual: " + ", ".join(
              str(by["plain-q8kv"][d][0]["depth"]) for d in depths) + ").\n")
    for title, key, f in (("Decode tok/s", "tg", ".1f"),
                          ("ms per target evaluation (plain: per token; MTP: per verify step)", "step_ms", ".1f"),
                          ("Tokens per target evaluation", "tok_per_step", ".2f")):
        print(f"### {title}\n\n{head}\n{sep}")
        for name in by:
            if key == "tok_per_step" and name.startswith("plain"):
                continue
            print(f"| {name} | " + " | ".join(fmt(mean(name, d, key), f) for d in depths) + " |")
        print()

    print("### Cost of depth: ms added per target evaluation, relative to 2K\n")
    print("`ideal` is the time to read the KV cache once at the card's peak bandwidth "
          f"({PEAK_GBS} GB/s).\n")
    print(head + "\n" + sep)
    for name in by:
        base = mean(name, depths[0], "step_ms")
        print(f"| {name} | " + " | ".join(
            fmt(None if mean(name, d, "step_ms") is None else mean(name, d, "step_ms") - base)
            for d in depths) + " |")
    for kv, b in KV_BYTES.items():
        print(f"| ideal, {kv} KV | " + " | ".join(
            f"{(by['plain-q8kv'][d][0]['depth'] - by['plain-q8kv'][depths[0]][0]['depth']) * b / PEAK_GBS / 1e6:.1f}"
            for d in depths) + " |")
    print()

    print("### Prefill tok/s over each depth range (explain task; the prompt cache covers the rest)\n")
    print("| config | " + " | ".join(f"to {d // 1000}K" for d in depths) + " |\n" + sep)
    for name in by:
        print(f"| {name} | " + " | ".join(
            fmt(next((r["pp"] for r in by[name].get(d, []) if r["task"] == "explain"), None), ".0f")
            for d in depths) + " |")
    print()

    print("### Peak VRAM (MiB) and per-task rows\n")
    print("| config | peak VRAM | task | depth | tok/s | accepted / drafted |")
    print("|---|---:|---|---:|---:|---|")
    for r in rows:
        print(f"| {r['config']['name']} | {r['vram_peak_mib']} | {r['task']} | {r['depth']} | "
              f"{r['tg']:.1f} | {r['draft_accepted']} / {r['draft_n']} |")


if __name__ == "__main__":
    print("# Depth profile and MMVQ re-check on llama.cpp b12164\n")
    kernel()
    depth()
