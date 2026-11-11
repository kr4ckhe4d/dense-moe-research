"""Tables for docs/lenient.md from benchmarks/lenient_probe.py.

python3 scripts/lenient_summary.py > experiments/results/lenient/summary.md
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

P = Path("experiments/results/lenient/probe.jsonl")
RATIOS = (0.7, 0.5, 0.3, 0.2, 0.1, 0.05)


def accepted(step: dict, r: float) -> int:
    """Draft tokens a rule with ratio r accepts in this step (r = 1: exact match only)."""
    n = 0
    for d, t, pd, pt in zip(step["draft"], step["top"], step["p_draft"], step["p_top"]):
        if d != t and not (r < 1 and pd >= r * pt):
            break
        n += 1
    return n


def hmean(xs: list[float]) -> float:
    return len(xs) / sum(1 / x for x in xs)


def main() -> None:
    by = defaultdict(dict)  # ratio -> domain -> row
    for row in map(json.loads, P.read_text().splitlines()):
        by[row["ratio"]][row["domain"]] = row
    exact = by[0.0]
    domains = list(exact)
    print("# Lenient draft acceptance: Qwen3.8-27B UD-Q6_K, MTP draft 3, R9700\n")
    print("Six domain prompts, 512 tokens each, temperature 0, thinking off. Ratio r: a draft token "
          "the target did not pick is accepted if p_target(draft) >= r * p_target(top).\n")

    print("## Projected from the exact run's log: tokens per verify step, relative to exact\n")
    print("Each step's draft is re-scored under the rule. Ignores that the text diverges afterwards.\n")
    print("| domain | exact tok/step | " + " | ".join(f"r={r}" for r in RATIOS) + " |")
    print("|---|---:|" + "---:|" * len(RATIOS))
    for d in domains:
        st = exact[d]["steps"]
        base = sum(s["n_acc"] for s in st) + len(st)
        print(f"| {d} | {base / len(st):.2f} | " + " | ".join(
            f"{(sum(accepted(s, r) for s in st) + len(st)) / base:.3f}" for r in RATIOS) + " |")
    print()

    print("## Measured decode tok/s\n")
    ratios = sorted(by, reverse=True)
    ratios.remove(0.0)
    cols = [0.0] + ratios
    print("| domain | " + " | ".join("exact" if r == 0 else f"r={r}" for r in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for d in domains:
        print(f"| {d} | " + " | ".join(f"{by[r][d]['tg']:.1f}" for r in cols) + " |")
    hm = {r: hmean([by[r][d]["tg"] for d in domains]) for r in cols}
    print("| **harmonic mean** | " + " | ".join(f"**{hm[r]:.1f}**" for r in cols) + " |")
    print("| vs exact | " + " | ".join(f"{hm[r] / hm[0.0] - 1:+.1%}" for r in cols) + " |\n")

    print("## What lenient acceptance changes in the output\n")
    print("Lenient tokens: output tokens that are not the target's top choice. Mean ratio: their "
          "p / p_top. First divergence: characters shared with the exact run's answer.\n")
    print("| ratio | domain | lenient tokens | share of output | mean p/p_top | draft accepted / drafted | first divergence |")
    print("|---:|---|---:|---:|---:|---|---:|")
    for r in ratios:
        for d in domains:
            row = by[r][d]
            q = [s["p_draft"][k] / s["p_top"][k] for s in row["steps"]
                 for k in range(min(s["n_acc"], len(s["draft"]))) if s["draft"][k] != s["top"][k]]
            a, b = row["text"], exact[d]["text"]
            div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            mean_q = f"{math.fsum(q) / len(q):.2f}" if q else ""
            print(f"| {r} | {d} | {len(q)} | {len(q) / row['n_gen']:.1%} | {mean_q} | "
                  f"{row['draft_accepted']} / {row['draft_n']} | {div} |")


def sampled() -> None:
    files = {"exact": ("probe_t1.jsonl", 0.0), "lenient r=0.3": ("probe_t1.jsonl", 0.3),
             "exact, probabilistic draft (lossless)": ("probe_t1_rejection.jsonl", 0.0)}
    print("\n## Temperature 1.0 (one seed), decode tok/s\n")
    runs = {}
    for name, (f, ratio) in files.items():
        rows = [r for r in map(json.loads, (P.parent / f).read_text().splitlines()) if r["ratio"] == ratio]
        runs[name] = {r["domain"]: r for r in rows}
    domains = list(runs["exact"])
    print("| config | " + " | ".join(domains) + " | harmonic mean | lenient share of output |")
    print("|---|" + "---:|" * (len(domains) + 2))
    for name, d in runs.items():
        share = sum(s["n_lenient"] for r in d.values() for s in r["steps"]) / sum(r["n_gen"] for r in d.values())
        print(f"| {name} | " + " | ".join(f"{d[x]['tg']:.1f}" for x in domains)
              + f" | {hmean([d[x]['tg'] for x in domains]):.1f} | {share:.1%} |")


if __name__ == "__main__":
    main()
    sampled()
