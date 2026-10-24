"""Tables for the 27B speed milestone (R9700, model fully on the GPU).

1. Small-batch matmul cost per quant (llama-bench pp1..8, ms per batch), stock vs tuned MMVQ.
2. Decode tok/s per quant x drafter x draft length, stock vs tuned build (harmonic mean over six
   domain prompts, 512 tokens, temperature 0), with draft acceptance and KLD vs Q8_0.

python3 scripts/fast27b_summary.py > experiments/results/fast27b/summary.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

R = Path("experiments/results/fast27b")
QUANTS = ["Q8_0", "UD-Q6_K", "Q5_K_M-rq", "Q4_K_M-rq", "IQ4_XS-rq", "Q4_0-rq", "Q3_K_M-rq"]
# KLD vs Q8_0, wikitext-2 test, c=512, 100 chunks (offload/kld27b_*.txt, fast27b/kld27b_*.txt)
KLD = {"Q8_0": 0.0, "UD-Q6_K": 0.0023, "Q5_K_M-rq": 0.0066, "Q4_K_M-rq": 0.0157,
       "IQ4_XS-rq": 0.0185, "Q4_0-rq": 0.0297, "Q3_K_M-rq": 0.0542}
# File sizes (GiB); the Q5_K_M / Q4_K_M / Q4_0 trial quants were deleted after the milestone.
GIB = {"Q8_0": 27.05, "UD-Q6_K": 20.47, "Q5_K_M-rq": 18.19, "Q4_K_M-rq": 15.66,
       "IQ4_XS-rq": 14.26, "Q4_0-rq": 14.68, "Q3_K_M-rq": 12.57}


def batch_ms(path: Path) -> dict[int, float]:
    return {r["n_prompt"]: 1000 * r["n_prompt"] / r["avg_ts"]
            for r in json.loads(path.read_text()) if r["n_prompt"]}


def load(path: Path) -> dict[tuple, dict]:
    g = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        c = r["config"]
        g[(c["quant"].removeprefix("27B-"), c["spec"], c.get("n_max"))].append(r)
    out = {}
    for k, rs in g.items():
        dn = sum(r["draft_n"] for r in rs)
        out[k] = {"tg": len(rs) / sum(1 / r["tg"] for r in rs),
                  "acc": sum(r["draft_accepted"] for r in rs) / dn if dn else None,
                  "dom": {r["prompt"]: r["tg"] for r in rs}}
    return out


def main() -> None:
    print("# Qwen3.8-27B fully on the R9700: drafters, quants, RDNA4 MMVQ tuning\n")
    print("## Cost of one batch of n tokens (ms, llama-bench pp-n, -fa 1)\n")
    print("Stock = b11434 (`fast27b_quants.sh`, 3 reps; `mmvq/control_*`, 5 reps, where present). "
          "Tuned = 2 warps x 4 rows per block for 2-8 columns (`patches/`, 5 reps).\n")
    print("| quant | GiB | KLD | build | 1 | 2 | 3 | 4 | 5 | 6 | 8 | t(6)/t(1) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q in QUANTS:
        gib = GIB[q]
        ctrl = R / f"mmvq/control_{q}.json"
        for label, p in (("stock", ctrl if ctrl.exists() else R / f"bench_{q}.json"),
                         ("tuned", R / f"mmvq/v6_{q}.json")):
            if not p.exists():
                continue
            ms = batch_ms(p)
            print(f"| {q} | {gib:.1f} | {KLD[q]:.4f} | {label} | "
                  + " | ".join(f"{ms[n]:.1f}" for n in (1, 2, 3, 4, 5, 6, 8))
                  + f" | {ms[6] / ms[1]:.2f} |")
    print()

    stock = load(R / "drafters_gpu1.jsonl")
    tuned = load(R / "drafters_mmvq_gpu1.jsonl")
    print("## Decode tok/s (harmonic mean of 6 domains, 512 tokens, t=0)\n")
    print("| quant | KLD | drafter | n_max | stock | tuned | tuned/stock | accept | x plain |")
    print("|---|---|---|---|---|---|---|---|---|")
    for q in QUANTS:
        plain = (stock.get((q, "none", None)) or tuned.get((q, "none", None)) or {}).get("tg")
        keys = sorted({k for k in (*stock, *tuned) if k[0] == q},
                      key=lambda k: (k[1] != "none", k[1], k[2] or 0))
        for k in keys:
            s, t = stock.get(k), tuned.get(k)
            st = f"{s['tg']:.1f}" if s else ""
            tt = f"{t['tg']:.1f}" if t else ""
            ratio = f"{t['tg'] / s['tg'] - 1:+.0%}" if s and t and k[1] != "none" else ""
            r = t or s
            acc = f"{r['acc']:.0%}" if r["acc"] is not None else ""
            xp = f"{r['tg'] / plain:.2f}" if plain else ""
            print(f"| {q} | {KLD[q]:.4f} | {k[1]} | {k[2] or ''} | {st} | {tt} | {ratio} | {acc} | {xp} |")
    print()

    print("## Per domain, best tuned configs (tok/s)\n")
    best = ["UD-Q6_K", "Q8_0", "Q5_K_M-rq", "IQ4_XS-rq", "Q4_0-rq"]
    rows = [(q, "draft-mtp", 3) for q in best[:1]] + [
        (q, "draft-dflash", 7 if q == "Q8_0" else 5) for q in best]
    doms = list(next(iter(tuned.values()))["dom"])
    print("| quant | drafter | n_max | " + " | ".join(doms) + " |")
    print("|---|---|---|" + "---|" * len(doms))
    for k in rows:
        d = tuned[k]["dom"]
        print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(f"{d[x]:.0f}" for x in doms) + " |")
    print()


if __name__ == "__main__":
    main()
