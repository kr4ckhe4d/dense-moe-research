"""Tables from benchmarks/mtp_tune.py results.

Mean decode speed is the harmonic mean over prompts (total tokens / total time; every prompt
generates max_tokens). Tokens per step = generated / (generated - accepted): each verify step
emits one target token plus the accepted drafts.

python3 scripts/mtp_summary.py experiments/results/mtp/mtp_tune_gpu1.jsonl > experiments/results/mtp/summary_gpu1.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

KLD = {"BF16": 0.0, "Q8_0": 0.0016, "Q6_K": 0.0028, "Q5_K_M": 0.0151, "Q4_K_M": 0.0228,
       "IQ4_XS": 0.0184, "Q3_K_M": 0.0581}  # F4, val, vs llama.cpp BF16
QUANT_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M"]


def label(c: dict) -> str:
    if c["spec"] == "none":
        return "none"
    s = f"{c['spec']} n_max={c['n_max']}"
    if c.get("n_min"):
        s += f" n_min={c['n_min']}"
    if c.get("p_min"):
        s += f" p_min={c['p_min']}"
    if c.get("extra"):
        s += " " + " ".join(c["extra"])
    if c.get("np", 1) > 1:
        s += f" np={c['np']}"
    return s


def hmean(xs: list[float]) -> float:
    return len(xs) / sum(1 / x for x in xs)


def main() -> None:
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    temps = sorted({r["temperature"] for r in rows})
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    print(f"# MTP tuning: {path.name}\n")
    seeds = sorted({r.get("seed", 0) for r in rows})
    print(f"{len(rows)} generations (prompt set {rows[0].get('prompt_set', 'domains')}), up to "
          f"{rows[0]['max_tokens']} tokens each, GPU {rows[0]['gpu']}, seeds {seeds}. Speed is the "
          "harmonic mean over prompts (and seeds); acceptance = accepted / drafted. With np > 1, "
          "tok/s is aggregate throughput of the concurrent group.\n")
    shared = sorted({label(r["config"]) + " " + r["config"]["quant"] for r in rows if r["gpu_shared"]})
    if shared:
        print(f"**GPU shared with another process during:** {', '.join(shared)}. Treat as suspect.\n")

    for temp in temps:
        by = defaultdict(list)
        for r in rows:
            if r["temperature"] == temp:
                by[(r["config"]["quant"], label(r["config"]))].append(r)
        quants = [q for q in QUANT_ORDER if any(k[0] == q for k in by)]
        none = {q: next(lab for (qq, lab) in by if qq == q and lab.startswith("none"))
                for q in quants if any(qq == q and lab.startswith("none") for qq, lab in by)}
        base = {q: hmean([r["tg"] for r in by[(q, lab)]]) for q, lab in none.items()}
        base_text = {(q, r["prompt"]): r["text"] for q, lab in none.items() for r in by[(q, lab)]}
        print(f"## Temperature {temp}\n")
        best = {}
        for q in quants:
            print(f"### {q} (KLD {KLD.get(q, float('nan')):.4f}), no speculation "
                  f"{base.get(q, float('nan')):.1f} tok/s\n")
            print("| config | tok/s | vs none | accept | tokens/step | "
                  + " | ".join(prompts) + " | same text as none |")
            print("|---|---|---|---|---|" + "---|" * len(prompts) + "---|")
            for (qq, lab), rs in sorted(by.items(), key=lambda kv: kv[0][1]):
                if qq != q:
                    continue
                tg = hmean([r["tg"] for r in rs])
                d = sum(r["draft_n"] for r in rs)
                a = sum(r["draft_accepted"] for r in rs)
                n = sum(r["n_gen"] for r in rs)
                per_all = defaultdict(list)  # several seeds: harmonic mean per prompt
                for r in rs:
                    per_all[r["prompt"]].append(r["tg"])
                per = {k: hmean(v) for k, v in per_all.items()}
                same = sum(r["text"] == base_text.get((q, r["prompt"])) for r in rs)
                acc = f"{a / d:.0%}" if d else "-"
                print(f"| {lab} | {tg:.1f} | {tg / base[q]:.2f}x | {acc} | {n / (n - a):.2f} | "
                      + " | ".join(f"{per[p]:.0f}" if p in per else "-" for p in prompts)
                      + f" | {same}/{len(rs)} |")
                if not lab.startswith("none") and (q not in best or tg > best[q][1]):
                    best[q] = (lab, tg, {p: per.get(p) for p in prompts})
            print()
        print("### Best config per quant\n")
        print("| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |")
        print("|---|---|---|---|---|---|---|---|")
        for q, (lab, tg, per) in best.items():
            lo = min((v, p) for p, v in per.items() if v)
            hi = max((v, p) for p, v in per.items() if v)
            print(f"| {q} | {KLD.get(q, float('nan')):.4f} | {base[q]:.1f} | {lab} | {tg:.1f} | "
                  f"{tg / base[q]:.2f}x | {lo[1]} {lo[0]:.0f} | {hi[1]} {hi[0]:.0f} |")
        print()


if __name__ == "__main__":
    main()
