"""Tables for docs/depth.md section 8: stock vs patched llama.cpp on the daily two-card preset.
Code-quality scores (benchmarks/adopt_check.sh, adopt_check_seeds.sh) and the two-card depth
profile (benchmarks/depth_profile.py --all-gpus).

python3 scripts/adopt_summary.py > experiments/results/adopt/summary.md
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

A = Path("experiments/results/adopt")
BUILDS = ("stock", "patched")


def scores(paths: list[Path]) -> tuple[list[int], dict[str, list[int]], int]:
    """Total per run, per-task passes per run, number of answers that did not run."""
    total, task, dnr = [], defaultdict(list), 0
    for p in paths:
        t = p.read_text()
        total.append(int(re.search(r"(\d+)/50 checks pass", t).group(1)))
        dnr += t.count("DID NOT RUN")
        for name, a in re.findall(r"^\s+(\w+)\s+(\d+)/\d+", t, re.M):
            task[name].append(int(a))
    return total, task, dnr


def by_seed(b: str) -> list[Path]:
    return sorted(A.glob(f"seeds/{b}_s*.txt"), key=lambda p: int(p.stem.split("_s")[1]))


def code_quality() -> None:
    print("## Code quality, 50 checks over 7 tasks, thinking off\n")
    print("| run | build | scores | mean | sd | answers that did not run |")
    print("|---|---|---|---:|---:|---:|")
    res = {}
    for label, paths in (("temperature 0, 3 repeats", lambda b: sorted(A.glob(f"codeq_{b}_r*.txt"))),
                         ("temperature 1.0, seeds 1-12", by_seed)):
        for b in BUILDS:
            total, task, dnr = scores(paths(b))
            res[label, b] = task
            sd = f"{st.stdev(total):.1f}" if len(set(total)) > 1 else "0"
            print(f"| {label} | {b} | {' '.join(map(str, total))} | {st.mean(total):.1f} | {sd} | {dnr} |")
    print("\n### Mean passes per task at temperature 1.0 (12 seeds)\n")
    s, p = (res["temperature 1.0, seeds 1-12", b] for b in BUILDS)
    print("| task | stock | patched |\n|---|---:|---:|")
    for name in s:
        print(f"| {name} | {st.mean(s[name]):.2f} | {st.mean(p[name]):.2f} |")
    print()


def depth() -> None:
    f = A / "depth_2gpu.jsonl"
    if not f.exists():
        return
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]["name"]][r["target_depth"]].append(r)
    depths = sorted({r["target_depth"] for r in rows})
    head = "| config | " + " | ".join(f"{d // 1000}K" for d in depths) + " |"
    sep = "|---|" + "---:|" * len(depths)
    print("## Depth profile on both cards (model split by llama-server), 27B UD-Q6_K\n")
    for title, key, fmt in (("Decode tok/s", "tg", ".1f"), ("ms per target evaluation", "step_ms", ".1f"),
                            ("Tokens per target evaluation", "tok_per_step", ".2f")):
        print(f"### {title}\n\n{head}\n{sep}")
        for name, d in by.items():
            if key == "tok_per_step" and name.startswith("plain"):
                continue
            print(f"| {name} | " + " | ".join(
                format(st.mean(r[key] for r in d[x]), fmt) if d.get(x) else "" for x in depths) + " |")
        print()
    print("### Peak VRAM over both cards (MiB)\n\n| config | peak |\n|---|---:|")
    for name, d in by.items():
        print(f"| {name} | {next(iter(d.values()))[0]['vram_peak_mib']} |")


if __name__ == "__main__":
    print("# Stock vs patched build on the daily two-card preset\n")
    code_quality()
    depth()
