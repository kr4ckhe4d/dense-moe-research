"""Tables for docs/shortlist.md: the MTP drafter scoring a vocabulary shortlist
(patches/llama.cpp-mtp-draft-shortlist.patch) against the unchanged drafter.

python3 scripts/shortlist_summary.py > experiments/results/shortlist/summary.md
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("experiments/results/shortlist")
PROBES = ["base-n3", "base-n4", "sl65k-n3", "sl65k-n4", "sl65k-n6-p0", "sl65k-n6-p50", "sl65k-n6-p75",
          "sl65k-n8-p50", "sl65k-n8-p75"]


def load(name: str) -> dict[str, dict]:
    return {r["domain"]: r for r in map(json.loads, (R / f"probe_{name}.jsonl").read_text().splitlines())}


def probes() -> None:
    d = {n: load(n) for n in PROBES if (R / f"probe_{n}.jsonl").exists()}
    doms = list(d["base-n3"])
    print("## Six prompt types, 512 tokens each, temperature 0, 8K context, R9700 (decode tok/s)\n")
    print("`base` = drafter reads the whole output layer; `sl65k` = 65,536-token shortlist; "
          "`nN` = draft length; `pXX` = stop drafting below that draft probability.\n")
    print("| variant | " + " | ".join(doms) + " | harmonic mean | accepted / drafted |")
    print("|---|" + "---:|" * (len(doms) + 2))
    for n, rows in d.items():
        tg = [rows[x]["tg"] for x in doms]
        print(f"| {n} | " + " | ".join(f"{t:.1f}" for t in tg) + f" | {st.harmonic_mean(tg):.1f} | "
              f"{sum(rows[x]['draft_accepted'] for x in doms)} / {sum(rows[x]['draft_n'] for x in doms)} |")
    print()
    for a, b in (("base-n3", "sl65k-n3"), ("base-n4", "sl65k-n4")):
        same = sum(d[a][x]["text"] == d[b][x]["text"] for x in doms)
        print(f"- Output text identical, {a} vs {b}: {same} of {len(doms)} prompts.")
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
            print(f"| {name} | " + " | ".join(
                format(st.mean(r[key] for r in d[x]), fmt) if d.get(x) else "" for x in depths) + " |")
        print()


if __name__ == "__main__":
    print("# MTP drafter with a vocabulary shortlist\n")
    probes()
    depth("sweep.jsonl", "Shortlist size and draft length at 2K context (two tasks, 256 tokens, q8_0 KV)")
    depth("depth.jsonl", "At depth (two tasks, 256 tokens, q8_0 KV)")
