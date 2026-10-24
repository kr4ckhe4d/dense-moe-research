"""Tables for the DFlash follow-up (benchmarks/dflash_open_items.sh, benchmarks/offload_dflash_fit.py).

Decode tok/s (harmonic mean over the prompts of a set) and draft acceptance, per experiment.

python3 scripts/dflash_summary.py > experiments/results/dflash/summary.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

R = Path("experiments/results/dflash")
OFF = Path("experiments/results/offload")
GPU = {0: "RX 9070 XT", 1: "R9700"}


def hmean(xs: list[float]) -> float:
    return len(xs) / sum(1 / x for x in xs)


def label(c: dict) -> str:
    if c["spec"] == "none":
        return "plain"
    s = c["spec"].replace("draft-", "")
    s += f" n={c['n_max']}"
    extra = c.get("extra", [])
    if "probabilistic" in extra:
        s += " prob"
    if "-ngld" in extra and extra[extra.index("-ngld") + 1] == "0":
        s += " (drafter in RAM)"
    if "budget_gib" in c and not c.get("vram_matched"):
        s += " (budget not matched)"  # early offload MTP runs at the plain -ngl
    return s


def load(path: Path, key=lambda r: ()) -> dict[tuple, dict]:
    """(quant, label, *key(r)) -> tg (harmonic mean), acceptance, per-prompt tg, VRAM peak."""
    g = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        g[(r["config"]["quant"], label(r["config"]), *key(r))].append(r)
    out = {}
    for k, rs in g.items():
        dn = sum(r["draft_n"] for r in rs)
        out[k] = {"tg": hmean([r["tg"] for r in rs]), "n": len(rs), "cfg": rs[0]["config"],
                  "acc": sum(r["draft_accepted"] for r in rs) / dn if dn else None,
                  "dom": defaultdict(list),
                  "vram": max(r.get("vram_peak_mib", 0) - r.get("vram_before_mib", 0) for r in rs)}
        for r in rs:
            out[k]["dom"][r["prompt"]].append(r["tg"])
    return out


def acc(v: dict) -> str:
    return f"{v['acc']:.0%}" if v["acc"] is not None else ""


def table(rows: dict, cols: list[str] | None = None, base: str = "plain") -> None:
    """One row per (quant, label): tok/s, x base, acceptance, and per-prompt tok/s."""
    cols = cols or sorted({d for v in rows.values() for d in v["dom"]})
    print("| model | drafter | tok/s | x " + base + " | accept | " + " | ".join(cols) + " |")
    print("|---|---|---|---|---|" + "---|" * len(cols))
    for (q, lab, *_), v in rows.items():
        b = rows.get((q, base, *_))
        x = f"{v['tg'] / b['tg']:.2f}" if b else ""
        doms = " | ".join(f"{hmean(v['dom'][d]):.0f}" if d in v["dom"] else "" for d in cols)
        print(f"| {q} | {lab} | {v['tg']:.1f} | {x} | {acc(v)} | {doms} |")
    print()


def main() -> None:
    print("# DFlash follow-up: 9B drafter, long context, sampling, editing, concurrency, offload\n")
    print("Patched llama.cpp (patches/llama.cpp-rdna4-mmvq-multicol.patch). tok/s = harmonic mean "
          "over the prompts; accept = accepted / drafted tokens.\n")

    for g in (1, 0):
        p = R / f"nine_gpu{g}.jsonl"
        if p.exists():
            print(f"## Qwen3.5-9B fully on the {GPU[g]} (six domains, 512 tokens, t=0)\n")
            table(load(p))

    p = R / "longctx_gpu1.jsonl"
    if p.exists():
        print("## Long context, R9700 (~4k / 16k / 28k tokens of C++, then a question; 512 tokens)\n")
        table(load(p), ["longctx_4k", "longctx_16k", "longctx_28k"])

    p = R / "sampling_gpu1.jsonl"
    if p.exists():
        print("## Sampling, R9700 (six domains, 512 tokens; t=0.7 over seeds 0-2, t=1.0 seed 0)\n")
        rows = load(p, key=lambda r: (r["temperature"],))
        print("| model | drafter | temperature | runs | tok/s | accept |")
        print("|---|---|---|---|---|---|")
        for (q, lab, t), v in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
            print(f"| {q} | {lab} | {t} | {v['n']} | {v['tg']:.1f} | {acc(v)} |")
        print()

    p = R / "edit_gpu1.jsonl"
    if p.exists():
        print("## Repetitive editing, R9700 (return a file with small additions, 1024 tokens)\n")
        table(load(p))

    p = R / "concurrency_gpu1.jsonl"
    if p.exists():
        print("## Four concurrent requests, R9700 (aggregate tok/s, six domains in groups of 4)\n")
        table(load(p))

    p = OFF / "offload_dflash_gpu0.jsonl"
    if p.exists():
        print("## Partly offloaded, RX 9070 XT at a VRAM budget (code, prose, JSON; 256 tokens)\n")
        print("Plain and stock-build MTP rows are from the offload milestone "
              "(offload_mtp_gpu0.jsonl). All spec rows fit the same budget rule.\n")
        def bk(r: dict) -> tuple:
            return (r["config"]["budget_gib"],)
        old = load(OFF / "offload_mtp_gpu0.jsonl", key=bk)
        new = load(p, key=bk)
        keys = sorted({(q, b) for q, _, b in (*old, *new)})
        print("| model | VRAM | plain | MTP n=3 (stock) | MTP n=3 | DFlash n=5 | DFlash n=7 | "
              "n=5, drafter in RAM | n=7, drafter in RAM |")
        print("|---|---|---|---|---|---|---|---|---|")
        for q, b in keys:
            def cell(d: dict, lab: str, q: str = q, b: int = b) -> str:
                v = d.get((q, lab, b))
                return f"{v['tg']:.1f} (ngl {v['cfg']['ngl']})" if v else ""
            print(f"| {q} | {b} GB | {cell(old, 'plain')} | {cell(old, 'mtp n=3')} | "
                  f"{cell(new, 'mtp n=3')} | {cell(new, 'dflash n=5')} | {cell(new, 'dflash n=7')} | "
                  f"{cell(new, 'dflash n=5 (drafter in RAM)')} | "
                  f"{cell(new, 'dflash n=7 (drafter in RAM)')} |")
        print()


if __name__ == "__main__":
    main()
