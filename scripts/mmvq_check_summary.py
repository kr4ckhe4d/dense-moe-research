"""Stock vs patched RDNA4 MMVQ (benchmarks/mmvq_check.sh): ms per batch of n tokens, per GPU and
model, and the patched/stock time ratio.

python3 scripts/mmvq_check_summary.py > experiments/results/mmvq_check/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("experiments/results/mmvq_check")
GPU = {0: "RX 9070 XT", 1: "R9700"}
NS = (1, 2, 3, 4, 5, 6, 8)


def batch_ms(path: Path) -> dict[int, float]:
    return {r["n_prompt"]: 1000 * r["n_prompt"] / r["avg_ts"]
            for r in json.loads(path.read_text()) if r["n_prompt"]}


def main() -> None:
    print("# RDNA4 MMVQ patch: stock vs patched, ms per batch of n tokens\n")
    print("llama-bench -p 1..8 -n 0 -r 5 -fa 1, model fully on the GPU. Ratio = patched / stock "
          "time (< 1 is faster).\n")
    print("| GPU | model | build | " + " | ".join(map(str, NS)) + " |")
    print("|---|---|---|" + "---|" * len(NS))
    for g in (1, 0):
        for p in sorted(R.glob(f"stock_gpu{g}_*.json")):
            name = p.stem.removeprefix(f"stock_gpu{g}_")
            q = p.with_name(f"patched_gpu{g}_{name}.json")
            if not (p.stat().st_size and q.exists() and q.stat().st_size):
                continue
            s, t = batch_ms(p), batch_ms(q)
            print(f"| {GPU[g]} | {name} | stock | " + " | ".join(f"{s[n]:.1f}" for n in NS) + " |")
            print(f"| {GPU[g]} | {name} | patched | " + " | ".join(f"{t[n]:.1f}" for n in NS) + " |")
            print(f"| {GPU[g]} | {name} | ratio | "
                  + " | ".join(f"**{t[n] / s[n]:.2f}**" if abs(t[n] / s[n] - 1) >= 0.05
                               else f"{t[n] / s[n]:.2f}" for n in NS) + " |")
    print()
    for p in sorted(R.glob("test-backend-ops_*.txt")):
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        print(f"- `{p.name}`: {lines[-2].strip() if len(lines) > 1 else ''} {lines[-1].strip()}")


if __name__ == "__main__":
    main()
