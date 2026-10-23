"""Tables for the offload milestone from benchmarks/offload_bench.py results.

Per model and VRAM budget: every quant x placement strategy that fits, with GPU / RAM bytes,
pp512, tg128 and the quant's quality. Then a fit of decode time against weight bytes in RAM
(t = t0 + RAM_bytes / BW) and the ceiling for CPU-side activation sparsity: if a fraction f of
the RAM-resident FFN rows is read (TEAL-style), t' = t - (1 - f) * RAM_ffn_bytes / BW. The FFN
strategy puts only FFN weights in RAM, so all its RAM bytes are sparsifiable.

python3 scripts/offload_summary.py > experiments/results/offload/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path("experiments/results/offload")
# Quality: 9B vs llama.cpp BF16 on val text (findings F4); 27B vs Q8_0 on wikitext-2 (kld27b_*.txt)
QUALITY = {
    "Qwen3.5-9B-Q8_0": (0.0016, "BF16"), "Qwen3.5-9B-Q6_K": (0.0028, "BF16"),
    "Qwen3.5-9B-IQ4_XS": (0.0184, "BF16"),
    "Qwen3.8-27B-Q8_0": (0.0, "Q8_0"), "Qwen3.8-27B-UD-Q6_K": (0.0023, "Q8_0"),
    "Qwen3.8-27B-IQ4_XS-rq": (0.0185, "Q8_0"), "Qwen3.8-27B-Q3_K_M-rq": (0.0542, "Q8_0"),
}
TEAL_KLD = {0.9: 0.0011, 0.75: 0.0093}  # 9B, all 32 FFNs sparsified (Phase 3, F5)


def main() -> None:
    rows = []
    for f in sorted(R.glob("offload_[0-9]*b_gpu*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines()]
    print("# Offload: decode and prefill speed per VRAM budget (RX 9070 XT, 8 threads)\n")
    print("Budget = llama.cpp GPU buffers (weights + KV + recurrent state + compute) <= budget - "
          "0.5 GiB, ctx 8192. RAM = weight bytes read from system RAM per decode token.\n")
    for model_prefix in ("Qwen3.5-9B", "Qwen3.8-27B"):
        rs = [r for r in rows if r["model"].startswith(model_prefix) and not r["extra"]]
        if not rs:
            continue
        print(f"## {model_prefix}\n")
        for b in sorted({r["budget_gib"] for r in rs}):
            print(f"### {b:.0f} GiB VRAM\n")
            print("| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |")
            print("|---|---|---|---|---|---|---|---|")
            fit = sorted((r for r in rs if r["budget_gib"] == b and r["fits"]),
                         key=lambda r: -r["tg128"])
            for r in fit:
                q = r["model"].removeprefix(model_prefix + "-")
                kld, ref = QUALITY.get(r["model"], (float("nan"), "?"))
                print(f"| {q} | {r['strategy']} | {r['k']} | {r['gpu_mib'] / 1024:.2f} | "
                      f"{r['cpu_read_bytes'] / 2**30:.2f} | {r['pp512']:.0f} | {r['tg128']:.2f} | "
                      f"{kld:.4f} ({ref}) |")
            nf = [r["model"].removeprefix(model_prefix + "-") + " " + r["strategy"]
                  for r in rs if r["budget_gib"] == b and not r["fits"]]
            if nf:
                print(f"\nDoes not fit: {', '.join(nf)}.")
            print()

        # time per token vs bytes in RAM, per quant (points from all budgets and strategies)
        print(f"### {model_prefix}: cost of weights in RAM\n")
        print("Linear fit per quant of ms/token against GiB in RAM (all placements). Implied RAM "
              "bandwidth = 1 / slope.\n")
        print("| quant | points | ms/token at 0 GiB in RAM | ms per GiB in RAM | implied GB/s |")
        print("|---|---|---|---|---|")
        slopes = {}
        for model in sorted({r["model"] for r in rs}):
            pts = [(r["cpu_read_bytes"] / 2**30, 1000 / r["tg128"]) for r in rs
                   if r["model"] == model and r["fits"]]
            if len({p[0] for p in pts}) < 2:
                continue
            x, y = np.array(pts).T
            s, c = np.polyfit(x, y, 1)
            slopes[model] = (s, c)
            print(f"| {model.removeprefix(model_prefix + '-')} | {len(pts)} | {c:.1f} | {s:.1f} | "
                  f"{2**30 / (s / 1000) / 1e9:.0f} |")
        print()

        print(f"### {model_prefix}: ceiling for CPU-side activation sparsity (FFN strategy)\n")
        print("Only the RAM-resident FFN rows are skipped; the GPU side stays dense. Speed uses "
              "the per-quant fit above; ideal kernel, free mask. KLD added = TEAL KLD (9B, all "
              "layers) x share of layers whose FFN is in RAM: an assumption (additive over "
              "layers; transfer to the 27B unmeasured).\n")
        print("| quant | budget | FFN layers in RAM | tg128 measured | f=0.9 | f=0.75 | "
              "est. KLD added at 0.9 / 0.75 |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted((r for r in rs if r["fits"] and r["strategy"] == "ffn" and r["k"] > 0),
                        key=lambda r: (r["model"], r["budget_gib"])):
            if r["model"] not in slopes:
                continue
            s, _ = slopes[r["model"]]
            t = 1000 / r["tg128"]
            ram = r["cpu_read_bytes"] / 2**30
            n_layers = 32 if model_prefix.endswith("9B") else 64
            share = r["k"] / n_layers
            cells = [f"{1000 / (t - (1 - f) * ram * s):.2f}" for f in (0.9, 0.75)]
            print(f"| {r['model'].removeprefix(model_prefix + '-')} | {r['budget_gib']:.0f} | "
                  f"{r['k']}/{n_layers} | {r['tg128']:.2f} | {cells[0]} | {cells[1]} | "
                  f"{TEAL_KLD[0.9] * share:.4f} / {TEAL_KLD[0.75] * share:.4f} |")
        print()


def mtp_table() -> None:
    """Plain vs MTP at the same VRAM budget (benchmarks/offload_mtp_fit.py; 3 prompts x 256 tokens)."""
    path = R / "offload_mtp_gpu0.jsonl"
    if not path.exists():
        return
    from collections import defaultdict
    g = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        c = r["config"]
        if c["spec"] == "none":
            kind = "plain"
        elif c.get("vram_matched"):
            kind = "mtp"
        else:
            continue  # first MTP pass at the plain -ngl: over budget, superseded
        g[(c["quant"], c["budget_gib"], kind)].append(r)

    def hm(xs: list[float]) -> float:
        return len(xs) / sum(1 / x for x in xs)

    print("## MTP at the same VRAM budget\n")
    print("Harmonic mean over code / prose / JSON prompts, 256 tokens, temperature 0. MTP's draft "
          "context needs ~0.4 GiB (9B) / ~0.8 GiB (27B) of VRAM, so its -ngl is lowered until it "
          "fits the same budget.\n")
    print("| model | budget | plain -ngl | plain tok/s | MTP -ngl | MTP tok/s | speedup | accept | KLD (ref) |")
    print("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(q, b) for q, b, _ in g}, key=lambda t: (t[0][:11], t[1], -hm([r["tg"] for r in g[(t[0], t[1], "plain")]]) if g[(t[0], t[1], "plain")] else 0))
    for q, b in keys:
        p, m = g.get((q, b, "plain")), g.get((q, b, "mtp"))
        if not p or not m:
            continue
        tp, tm = hm([r["tg"] for r in p]), hm([r["tg"] for r in m])
        acc = sum(r["draft_accepted"] for r in m) / sum(r["draft_n"] for r in m)
        kld, ref = QUALITY.get(q, (float("nan"), "?"))
        print(f"| {q} | {b:.0f} | {p[0]['config']['ngl']} | {tp:.2f} | {m[0]['config']['ngl']} | "
              f"{tm:.2f} | {tm / tp:.2f}x | {acc:.0%} | {kld:.4f} ({ref}) |")
    print()


if __name__ == "__main__":
    main()
    mtp_table()
