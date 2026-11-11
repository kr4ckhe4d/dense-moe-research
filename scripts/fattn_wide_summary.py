"""Tables for docs/depth.md section 10: the WMMA attention kernel with one tile per K/V head for
small batches on RDNA4 (patches/llama.cpp-rdna4-fattn-q8-tile.patch), against the same build with
the rule switched off. One attention layer of the 27B shape (head size 256, 4 K/V heads, 6 Q heads
each), `test-backend-ops perf -o FLASH_ATTN_EXT`, plus the end-to-end depth profile.

python3 scripts/fattn_wide_summary.py > experiments/results/fattn/summary_wide.md
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("experiments/results/fattn")
GPUS = {1: "R9700", 0: "RX 9070 XT"}
PAT = re.compile(r"kv=(\d+),nb=(\d+),.*type_K=(\w+),type_V.*?([\d.]+) us/run")


def perf(path: Path) -> dict[tuple[str, int, int], float]:
    out = {}
    for line in path.read_text().splitlines():
        m = PAT.search(line)
        if m:
            out[m.group(3), int(m.group(1)), int(m.group(2))] = float(m.group(4))
    return out


def layer_tables() -> None:
    for g, name in GPUS.items():
        old, new = perf(R / f"perf_wide_old_gpu{g}.txt"), perf(R / f"perf_wide_new_gpu{g}.txt")
        nbs = sorted({k[2] for k in old})
        print(f"## One attention layer, {name}: us per call, before -> after\n")
        for t in ("f16", "q8_0", "q4_0"):
            kvs = sorted({k[1] for k in old if k[0] == t})
            if not kvs:
                continue
            print(f"### {t} K/V\n")
            print("| KV rows | " + " | ".join(f"{n} tok" for n in nbs) + " |")
            print("|---:|" + "---:|" * len(nbs))
            for kv in kvs:
                cells = []
                for n in nbs:
                    a, b = old.get((t, kv, n)), new.get((t, kv, n))
                    cells.append("" if a is None else f"{a:.0f} -> {b:.0f}")
                print(f"| {kv} | " + " | ".join(cells) + " |")
            print()


def depth(fname: str, title: str) -> None:
    f = R / fname
    if not f.exists():
        return
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]["name"]][r["target_depth"]].append(r)
    depths = sorted({r["target_depth"] for r in rows})
    head = "| config | " + " | ".join(f"{d // 1000}K" for d in depths) + " |"
    sep = "|---|" + "---:|" * len(depths)
    print(f"## {title}\n")
    for sub, key, fmt in (("Decode tok/s", "tg", ".1f"), ("ms per target evaluation", "step_ms", ".1f"),
                          ("Tokens per target evaluation", "tok_per_step", ".2f")):
        print(f"### {sub}\n\n{head}\n{sep}")
        for name, d in by.items():
            if key == "tok_per_step" and name.startswith("plain"):
                continue
            print(f"| {name} | " + " | ".join(
                format(st.mean(r[key] for r in d[x]), fmt) if d.get(x) else "" for x in depths) + " |")
        print()
    print("### Peak VRAM (MiB)\n\n| config | peak |\n|---|---:|")
    for name, d in by.items():
        print(f"| {name} | {next(iter(d.values()))[0]['vram_peak_mib']} |")
    print()


if __name__ == "__main__":
    print("# WMMA kernel with one tile per K/V head for small batches (RDNA4)\n")
    layer_tables()
    depth("depth_wide.jsonl", "27B UD-Q6_K on the R9700, end to end (`-off` = same build, rule off)")
    depth("depth_wide_2gpu.jsonl", "27B UD-Q6_K on both cards, end to end")
