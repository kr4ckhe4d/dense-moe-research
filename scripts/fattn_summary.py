"""Stock vs patched attention kernel timings for q8_0 / f16 K/V at depth
(patches/llama.cpp-rdna4-fattn-q8-tile.patch; test-backend-ops perf, see docs/depth.md).

python3 scripts/fattn_summary.py > experiments/results/fattn/summary.md
"""

from __future__ import annotations

import re
from pathlib import Path

R = Path("experiments/results/fattn")
GPU = {1: "R9700", 0: "RX 9070 XT"}
LINE = re.compile(r"kv=(\d+),nb=(\d+),.*type_K=(\w+),.* ([\d.]+) us/run")


def load(path: Path) -> dict[tuple[int, int, str], float]:
    """(kv, nb, K/V type) -> us per run."""
    return {(int(m[1]), int(m[2]), m[3]): float(m[4]) for m in map(LINE.search, path.read_text().splitlines()) if m}


def main() -> None:
    print("# Attention over q8_0 K/V at depth: stock vs patched tile kernel (b12164)\n")
    print("`test-backend-ops perf -o FLASH_ATTN_EXT`, one attention layer with the Qwen3.8-27B shape "
          "(head size 256, 4 K/V heads, 24 Q heads). us per call; nb = tokens in the batch. "
          "Ratio = patched / stock time.\n")
    for g in (1, 0):
        s, p = load(R / f"perf_stock_gpu{g}.txt"), load(R / f"perf_patched_gpu{g}.txt")
        print(f"## {GPU[g]}\n")
        print("| KV rows | nb | f16 | q8_0 stock | q8_0 patched | ratio | patched q8_0 / f16 |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for kv, nb, t in sorted(s):
            if t != "q8_0":
                continue
            f16 = s[(kv, nb, "f16")]
            print(f"| {kv} | {nb} | {f16:.0f} | {s[(kv, nb, t)]:.0f} | {p[(kv, nb, t)]:.0f} | "
                  f"{p[(kv, nb, t)] / s[(kv, nb, t)]:.2f} | {p[(kv, nb, t)] / f16:.2f} |")
        f16_ratio = [p[k] / s[k] for k in s if k[2] == "f16"]
        print(f"\nf16 K/V, patched / stock: {min(f16_ratio):.2f}-{max(f16_ratio):.2f} (path unchanged).\n")
        print((R / f"test_patched_gpu{g}.txt").read_text().strip().splitlines()[-1].strip()
              + " (`test-backend-ops -o FLASH_ATTN_EXT`, patched build)\n")


if __name__ == "__main__":
    main()
