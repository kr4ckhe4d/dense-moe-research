"""Phase 3 tables from block_sweep.json (per-layer) and e2e_kld_val.json (full model).

.venv/bin/python scripts/phase3_summary.py > experiments/results/phase3/summary.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path("experiments/results/phase3")
KEEPS = [0.9, 0.75, 0.625, 0.5, 0.375, 0.25]


def main() -> None:
    sweep = json.loads((RES / "block_sweep.json").read_text())["layers"]
    layers = sorted(sweep, key=int)
    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for li in layers:
        for k, r in sweep[li].items():
            for m in ("rel_err", "cosine_mean", "bytes", "static_coverage", "jaccard", "hot90",
                      "cold10", "kept"):
                agg[k][m].append(r[m])
    mean = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in agg.items()}
    n = {k: len(d["rel_err"]) for k, d in agg.items()}
    print(f"# Phase 3 summary\n\nPer-layer metrics are means over {len(layers)} layers "
          f"(val activations, 2048 tokens/layer; greedy and its comparison on 256).\n")

    def grid(title: str, fmt, rows):
        print(f"### {title}\n")
        print("| | " + " | ".join(f"{k:.1%}" for k in KEEPS) + " |")
        print("|---|" + "---|" * len(KEEPS))
        for label, key in rows:
            cells = []
            for keep in KEEPS:
                kk = key(keep)
                if kk not in mean:
                    cells.append("-")
                    continue
                c = fmt(mean[kk]) + ("" if n[kk] == len(layers) else f" ({n[kk]}L)")
                if abs(mean[kk]["kept"] - keep) > 0.01:  # whole blocks only: actual budget differs
                    c += f" [kept {mean[kk]['kept']:.3f}]"
                cells.append(c)
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()

    blocks = [1, 8, 16, 32, 64, 128, 256, 512, 1536, 3072]
    err = lambda r: f"{r['rel_err']:.3f}"
    grid("FFN relative error, oracle top-k by block output norm (columns: neurons kept)", err,
         [(f"B={b}", lambda k, b=b: f"oracle B={b} keep={k}") for b in blocks])
    grid("Objective: contribution sum vs norm vs greedy (relative error)", err,
         [(f"contrib B={b}", lambda k, b=b: f"oracle B={b} keep={k} contrib") for b in (8, 16, 64, 256)]
         + [(f"norm B={b} [256 tok]", lambda k, b=b: f"oracle B={b} keep={k} [subset]")
            for b in (16, 64, 256)]
         + [(f"greedy B={b} [256 tok]", lambda k, b=b: f"oracle B={b} keep={k} greedy [subset]")
            for b in (16, 64, 256)])
    grid("Clustered (k-means) blocks vs contiguous (relative error, layers 4/15/26)", err,
         [(f"clustered B={b}", lambda k, b=b: f"oracle B={b} keep={k} [clustered]") for b in (16, 64, 256)])
    print("Contiguous, same three layers:\n")
    for b in (16, 64, 256):
        vals = [np.mean([sweep[li][f"oracle B={b} keep={k}"]["rel_err"] for li in ("4", "15", "26")
                         if li in sweep]) for k in KEEPS]
        print(f"- B={b}: " + " / ".join(f"{v:.3f}" for v in vals))
    print()
    grid("What can be computed before choosing: relative error by method (B=1 unless noted)", err,
         [("oracle (needs gate+up)", lambda k: f"oracle B=1 keep={k}"),
          ("gate (needs gate)", lambda k: f"gate B=1 keep={k}"),
          ("gate B=16", lambda k: f"gate B=16 keep={k}"),
          ("teal x=h=keep", lambda k: f"teal B=1 keep={k} x_keep={k} xB=1"),
          ("wina x=h=keep", lambda k: f"wina B=1 keep={k} x_keep={k} xB=1"),
          ("wina x=h=keep, 32-blocks", lambda k: f"wina B=32 keep={k} x_keep={k} xB=32"),
          ("wina x only", lambda k: f"wina B=1 keep=1.0 x_keep={k} xB=1"),
          ("wina h only", lambda k: f"wina B=1 keep={k} x_keep=1.0 xB=1")])
    grid("FFN bytes read per decode token (fraction of dense) for the same rows",
         lambda r: f"{r['bytes']:.3f}",
         [("oracle", lambda k: f"oracle B=1 keep={k}"), ("gate", lambda k: f"gate B=1 keep={k}"),
          ("teal/wina x=h=keep", lambda k: f"teal B=1 keep={k} x_keep={k} xB=1"),
          ("wina x only", lambda k: f"wina B=1 keep=1.0 x_keep={k} xB=1")])

    # Error at matched FFN bytes: interpolate rel_err vs bytes for each method family.
    print("### Relative error at matched FFN bytes read (linear interpolation over budgets)\n")
    fams = {"oracle B=1": lambda k: f"oracle B=1 keep={k}",
            "oracle B=16": lambda k: f"oracle B=16 keep={k}",
            "gate B=1": lambda k: f"gate B=1 keep={k}",
            "gate B=16": lambda k: f"gate B=16 keep={k}",
            "teal": lambda k: f"teal B=1 keep={k} x_keep={k} xB=1",
            "wina": lambda k: f"wina B=1 keep={k} x_keep={k} xB=1",
            "wina 32-blocks": lambda k: f"wina B=32 keep={k} x_keep={k} xB=32"}
    targets = [0.9, 0.8, 0.7, 0.6, 0.5]
    print("| method | " + " | ".join(f"bytes {t}" for t in targets) + " |")
    print("|---|" + "---|" * len(targets))
    for fam, key in fams.items():
        b = np.array([mean[key(k)]["bytes"] for k in KEEPS])
        e = np.array([mean[key(k)]["rel_err"] for k in KEEPS])
        o = np.argsort(b)
        cells = [f"{np.interp(t, b[o], e[o]):.3f}" if b.min() <= t <= b.max() else "-" for t in targets]
        print(f"| {fam} | " + " | ".join(cells) + " |")
    print()

    grid("Selection consistency: static coverage (1 = same blocks every token; uniform = kept)",
         lambda r: f"{r['static_coverage']:.2f}",
         [(f"oracle B={b}", lambda k, b=b: f"oracle B={b} keep={k}") for b in (1, 16, 64, 256)])
    grid("Hot blocks: share of blocks chosen by >90% of tokens",
         lambda r: f"{r['hot90']:.3f}",
         [(f"oracle B={b}", lambda k, b=b: f"oracle B={b} keep={k}") for b in (1, 16, 64, 256)])

    print("### Layer sensitivity: relative error per layer at 50% kept\n")
    print("| layer | oracle B=1 | oracle B=16 | oracle B=64 | gate B=1 | wina 50/50 |")
    print("|---|---|---|---|---|---|")
    for li in layers:
        r = sweep[li]
        print(f"| {li} | {r['oracle B=1 keep=0.5']['rel_err']:.3f} | "
              f"{r['oracle B=16 keep=0.5']['rel_err']:.3f} | {r['oracle B=64 keep=0.5']['rel_err']:.3f} | "
              f"{r['gate B=1 keep=0.5']['rel_err']:.3f} | "
              f"{r['wina B=1 keep=0.5 x_keep=0.5 xB=1']['rel_err']:.3f} |")
    print()

    e2e_path = RES / "e2e_kld_val.json"
    if e2e_path.exists():
        res = json.loads(e2e_path.read_text())["results"]
        print("### Full model, all 32 FFNs sparsified: KLD vs llama.cpp BF16 (val, 100 x 255 tokens)\n")
        print("| variant | mean KLD | p99 | PPL | same top | FFN bytes, 1 token | 2-token union "
              "| 4-token union |")
        print("|---|---|---|---|---|---|---|---|")
        for k, r in res.items():
            b = (f"{r['bytes']:.3f} | {r['bytes_union2']:.3f} | {r['bytes_union4']:.3f}"
                 if "bytes" in r else "- | - | -")
            print(f"| {k} | {r['mean_kld']:.4f} ± {r['kld_stderr']:.4f} | {r['kld_p99']:.3f} | "
                  f"{r['ppl']:.3f} | {r['same_top']:.3f} | {b} |")
        print()


if __name__ == "__main__":
    main()
