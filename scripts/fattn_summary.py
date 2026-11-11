"""Stock vs patched attention kernel timings for quantized / f16 K/V at depth
(patches/llama.cpp-rdna4-fattn-q8-tile.patch; test-backend-ops perf, see docs/depth.md).

python3 scripts/fattn_summary.py > experiments/results/fattn/summary.md
"""

from __future__ import annotations

import re
from pathlib import Path

R = Path("experiments/results/fattn")
GPU = {1: "R9700", 0: "RX 9070 XT"}
LINE = re.compile(r"hsk=(\d+),hsv=\d+,nh=(\d+),nr23=\[(\d+),1\],kv=(\d+),nb=(\d+),.*type_K=(\w+),.* ([\d.]+) us/run")
# (head size, K/V heads, Q heads per K/V head): the Qwen3.8-27B shape, then two common smaller ones
SHAPES = ((256, 4, 6), (128, 8, 4), (64, 8, 2))
KVS = (32768, 65536, 122624)


def load(path: Path) -> dict[tuple, float]:
    """(head size, K/V heads, GQA ratio, kv, nb, K/V type) -> us per run."""
    return {(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), m[6]): float(m[7])
            for m in map(LINE.search, path.read_text().splitlines()) if m}


def nb_ok(nb: int) -> bool:
    return nb <= 8  # decode and verify batches


def main() -> None:
    print("# Attention over quantized K/V at depth: stock vs patched tile kernel (b12164)\n")
    print("`test-backend-ops perf -o FLASH_ATTN_EXT`, one attention layer. us per call; nb = tokens "
          "in the batch. Each quantized cell is stock -> patched.\n")
    for g in (1, 0):
        s, p = load(R / f"perf_stock_gpu{g}.txt"), load(R / f"perf_patched_gpu{g}.txt")
        print(f"## {GPU[g]}\n")
        for hs, nh, gqa in SHAPES:
            print(f"### Head size {hs}, {nh} K/V heads, {nh * gqa} Q heads\n")
            print("| KV rows | nb | f16 | q8_0 | q4_0 | patched q8_0 / f16 |")
            print("|---:|---:|---:|---|---|---:|")
            for kv in KVS:
                for nb in sorted({k[4] for k in s if k[:4] == (hs, nh, gqa, kv) and nb_ok(k[4])}):
                    k = (hs, nh, gqa, kv, nb)
                    cell = lambda t: f"{s[k + (t,)]:.0f} -> {p[k + (t,)]:.0f}" if k + (t,) in s else ""
                    f16 = s.get(k + ("f16",))
                    print(f"| {kv} | {nb} | {'' if f16 is None else format(f16, '.0f')} | {cell('q8_0')} | {cell('q4_0')} | "
                          + ("" if f16 is None or k + ("q8_0",) not in p else f"{p[k + ('q8_0',)] / f16:.2f}") + " |")
            print()
        f16_ratio = [p[k] / s[k] for k in s if k[5] == "f16" and k in p]
        print(f"f16 K/V, patched / stock over all {len(f16_ratio)} f16 cases: "
              f"{min(f16_ratio):.2f}-{max(f16_ratio):.2f} (path unchanged).\n")
        print(re.search(r"\d+/\d+ tests passed", (R / f"test_ext_gpu{g}.txt").read_text())[0]
              + " (`test-backend-ops -o FLASH_ATTN_EXT`, patched build)\n")


if __name__ == "__main__":
    main()
