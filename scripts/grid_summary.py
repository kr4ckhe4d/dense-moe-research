"""Tables for docs/fork_grid.md: builds against each other on the two-card preset shape
(benchmarks/grid_builds.sh).

python3 scripts/grid_summary.py > experiments/results/grid/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("experiments/results/grid")
FILES = {"q6k": "UD-Q6_K (pulled 2026-10-06)", "q4xl-v31": "UD-Q4_K_XL v3.1"}
TASKS = ("explain", "code")


def table(tag: str, title: str) -> None:
    by: dict[str, dict] = {}
    for r in map(json.loads, (R / f"{tag}.jsonl").read_text().splitlines()):
        by.setdefault(r["config"]["name"].removeprefix(tag + "-"), {})[(r["target_depth"], r["task"])] = r
    base = next(iter(by.values()))
    print(f"## {title}\n")
    print("Decode tok/s explain / code, ms per target evaluation, accepted / drafted tokens over both tasks, "
          "prefill tok/s over the first 120K, peak VRAM on both cards, and how many of the four outputs "
          "match the first row's text.\n")
    print("| build | 2K tok/s | 2K ms | 2K drafts | 120K tok/s | 120K ms | 120K drafts | prefill | VRAM MiB | same text |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in by.items():
        cells = []
        for depth in (2000, 120000):
            rows = [d[(depth, t)] for t in TASKS]
            cells += [" / ".join(f"{r['tg']:.1f}" for r in rows),
                      f"{sum(r['step_ms'] for r in rows) / 2:.1f}",
                      f"{sum(r['draft_accepted'] for r in rows)} / {sum(r['draft_n'] for r in rows)}"]
        same = sum(d[k]["text"] == base[k]["text"] for k in d)
        print(f"| {name} | " + " | ".join(cells) + f" | {d[(120000, 'explain')]['pp']:.0f} | "
              f"{d[(120000, 'explain')]['vram_peak_mib']} | {same} of 4 |")
    print()


def depth_table(fname: str, title: str, note: str) -> None:
    """One row per config, decode tok/s explain / code at every depth it ran, ms per step at the deepest."""
    by: dict[str, dict] = {}
    for r in map(json.loads, (R / fname).read_text().splitlines()):
        by.setdefault(r["config"]["name"], {})[(r["target_depth"], r["task"])] = r
    depths = sorted({k[0] for d in by.values() for k in d})
    print(f"## {title}\n\n{note}\n")
    print("| config | " + " | ".join(f"{d // 1000}K" for d in depths) + " | ms at deepest | drafts at deepest | VRAM MiB |")
    print("|---|" + "---:|" * (len(depths) + 3))
    for name, d in by.items():
        cells = [" / ".join(f"{d[(x, t)]['tg']:.1f}" for t in TASKS) if (x, "code") in d else "" for x in depths]
        last = max(k[0] for k in d)
        rows = [d[(last, t)] for t in TASKS]
        print(f"| {name} | " + " | ".join(cells) + f" | {sum(r['step_ms'] for r in rows) / 2:.1f} | "
              f"{sum(r['draft_accepted'] for r in rows)} / {sum(r['draft_n'] for r in rows)} | {rows[0]['vram_peak_mib']} |")
    print()


if __name__ == "__main__":
    print("# Stock, this repo's patches and the RDNA fork on the two-card preset\n")
    for tag, title in FILES.items():
        table(tag, title)
    depth_table("fork_tune.jsonl", "Fork settings on UD-Q4_K_XL v3.1 (benchmarks/grid_fork_tune.sh)",
                "Decode tok/s explain / code. `ubN` = `-ub N`; `tensor` = `-sm tensor`; `vocabNk` = the fork's "
                "`--spec-draft-mtp-vocab`.")
    depth_table("fork_mid.jsonl", "Layer split at 16K-64K, UD-Q4_K_XL v3.1", "Decode tok/s explain / code.")
    depth_table("tensor.jsonl", "-sm tensor on every build (benchmarks/grid_tensor.sh)",
                "Decode tok/s explain / code, draft 3 unless named.")
