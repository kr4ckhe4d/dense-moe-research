"""Stock vs patched RDNA4 MMVQ on upstream b11719 (benchmarks/mmvq_upstream.sh): llama-bench
pp1..8 / pp512 / tg128, test-backend-ops, perplexity, and end-to-end llama-server decode.

python3 scripts/mmvq_upstream_summary.py > experiments/results/mmvq_upstream/summary.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

R = Path("experiments/results/mmvq_upstream")
GPU = {0: "RX 9070 XT", 1: "R9700"}


def load(path: Path) -> dict[str, tuple[float, float]]:
    """test name (pp1..pp512, tg128) -> (avg t/s, stddev)."""
    out = {}
    for r in json.loads(path.read_text()):
        name = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        out[name] = (r["avg_ts"], r["stddev_ts"])
    return out


def batch_ms(path: Path) -> dict[int, float]:
    """ms per batch of n tokens, for the pp tests."""
    return {r["n_prompt"]: 1000 * r["n_prompt"] / r["avg_ts"]
            for r in json.loads(path.read_text()) if r["n_prompt"]}


def main() -> None:
    print("# RDNA4 MMVQ patch on upstream b11719 (5aeac37a3), stock vs patched\n")
    print("llama-bench -ngl 99 -fa 1 -r 5, model fully on the GPU. t/s +- stddev; speedup = "
          "patched / stock t/s. ppN = one batch of N tokens (N <= 8: the speculative verify "
          "path).\n")
    print("| GPU | model | test | stock t/s | patched t/s | speedup |")
    print("|---|---|---|---:|---:|---:|")
    for g in (1, 0):
        for p in sorted(R.glob(f"bench_stock_gpu{g}_*.json")):
            name = p.stem.removeprefix(f"bench_stock_gpu{g}_")
            q = p.with_name(f"bench_patched_gpu{g}_{name}.json")
            if not (p.stat().st_size and q.exists() and q.stat().st_size):
                continue
            s, t = load(p), load(q)
            for test in s:
                sp = t[test][0] / s[test][0]
                mark = f"**{sp:.2f}**" if abs(sp - 1) >= 0.05 else f"{sp:.2f}"
                print(f"| {GPU[g]} | {name} | {test} | {s[test][0]:.2f} +- {s[test][1]:.2f} | "
                      f"{t[test][0]:.2f} +- {t[test][1]:.2f} | {mark} |")
    print()

    types = sorted(R.glob("types_stock_gpu1_*.json"))
    if types:
        print("## Per type: Qwen3.5-9B quantized with --pure, R9700, ms per batch of n tokens\n")
        print("Ratio = patched / stock time (< 1 is faster).\n")
        ns = (1, 2, 3, 4, 5, 6, 8)
        print("| type | build | " + " | ".join(map(str, ns)) + " |")
        print("|---|---|" + "---|" * len(ns))
        for p in types:
            q = p.with_name(p.name.replace("stock", "patched"))
            if not (p.stat().st_size and q.exists() and q.stat().st_size):
                continue
            t = p.stem.removeprefix("types_stock_gpu1_pure-")
            ms = {"stock": batch_ms(p), "patched": batch_ms(q)}
            for b in ("stock", "patched"):
                print(f"| {t} | {b} | " + " | ".join(f"{ms[b][n]:.1f}" for n in ns) + " |")
            print(f"| {t} | ratio | " + " | ".join(
                f"**{r:.2f}**" if abs(r - 1) >= 0.05 else f"{r:.2f}"
                for n in ns for r in [ms["patched"][n] / ms["stock"][n]]) + " |")
        print()

    print("## test-backend-ops (patched build)\n")
    for p in sorted(R.glob("test-backend-ops_*.txt")):
        txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text())
        passed = re.search(r"(\d+)/(\d+) tests passed", txt)
        failing = re.findall(r"^\s+(MUL_MAT\S*\(.*\))$", txt, re.MULTILINE)
        print(f"- `{p.stem}`: {passed[0] if passed else 'no summary'}"
              + (f"; failing: `{'`, `'.join(failing)}`" if failing else ""))
    print("\nThe one failing case (f16 x f32, n=2105344) fails identically on the stock b11719 "
          "build on both GPUs (ERR 0.0043 > 0.0005), so it is a pre-existing ROCm issue, not the "
          "patch: the patch only changes quantized types at 2-8 columns.\n")

    print("## Perplexity, Qwen3.5-9B Q6_K, wikitext-2 test, 50 chunks\n")
    for b in ("stock", "patched"):
        p = R / f"ppl_{b}.txt"
        if p.exists():
            m = re.search(r"Final estimate: PPL = ([\d.]+ \+/- [\d.]+)", p.read_text())
            print(f"- {b}: {m[1] if m else 'n/a'}")
    print()

    p = R / "server_gpu1.jsonl"
    if p.exists():
        print("## End to end: llama-server, Qwen3.8-27B UD-Q6_K, R9700, six domains x 512 tokens, "
              "t=0\n")
        g: dict[tuple, list] = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            c = r["config"]
            build = "patched" if "mmvq" in c["bin"] else "stock"
            g.setdefault((c["spec"], c["n_max"], build), []).append(r)
        print("| drafter | stock tok/s | patched tok/s | speedup | accepted/drafted (patched) |")
        print("|---|---:|---:|---:|---:|")
        for spec, n in sorted({(k[0], k[1]) for k in g}):
            hm = {}
            for b in ("stock", "patched"):
                rs = g.get((spec, n, b), [])
                hm[b] = len(rs) / sum(1 / r["tg"] for r in rs) if rs else None
            rs = g.get((spec, n, "patched"), [])
            acc = sum(r["draft_accepted"] for r in rs) / max(1, sum(r["draft_n"] for r in rs))
            sp = f"{hm['patched'] / hm['stock']:.2f}" if hm["stock"] and hm["patched"] else ""
            print(f"| {spec} n={n} | {hm['stock'] or 0:.1f} | {hm['patched'] or 0:.1f} | {sp} | "
                  f"{acc:.0%} |")
        print()


if __name__ == "__main__":
    main()
