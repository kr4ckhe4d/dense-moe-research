"""Collect llama.cpp KLD / llama-bench / MTP outputs and the PyTorch sparse-variant KLD into one
table (experiments/results/gguf/summary.json + printed markdown).

.venv/bin/python scripts/gguf_summary.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

R = Path("experiments/results/gguf")
QUANTS = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M"]
BENCH_NAME = {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q5_K_M": "Q5_K - Medium",
              "Q4_K_M": "Q4_K - Medium", "IQ4_XS": "IQ4_XS - 4.25 bpw", "Q3_K_M": "Q3_K - Medium",
              "BF16": "BF16"}


def kld(path: Path) -> dict | None:
    if not path.exists():
        return None
    t = path.read_text()

    def num(pat: str) -> float:
        return float(re.search(pat, t).group(1))

    return {"mean_kld": num(r"Mean +KLD: +([-\d.]+)"),
            "kld_stderr": num(r"Mean +KLD: +[-\d.]+ ± +([-\d.]+)"),
            "kld_p99": num(r"99\.0% +KLD: +([-\d.]+)"),
            "ppl": num(r"Mean PPL\(Q\) +: +([\d.]+)"),
            "same_top": num(r"Same top p: +([\d.]+)") / 100}


def bench(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8 or not cells[0].startswith("qwen35"):
            continue
        name = cells[0].removeprefix("qwen35 9B ").strip()
        out.setdefault(name, {})[cells[6]] = float(cells[7].split("±")[0])
        out[name]["size_gib"] = float(cells[1].split()[0])
    return out


def mtp(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    cur = None
    for line in path.read_text().splitlines():
        if m := re.match(r"== (\S+) spec=(\S+)", line):
            cur = out.setdefault(m.group(1), {}).setdefault(m.group(2), {})
        elif cur is not None and (m := re.match(r"tg +([\d.]+)", line)):
            cur["tg"] = float(m.group(1))
        elif cur is not None and (m := re.match(r"draft +([\d.]+)%", line)):
            cur["accept"] = float(m.group(1)) / 100
    return out


def main() -> None:
    b0, b1 = bench(R / "bench_9070xt.md"), bench(R / "bench_r9700.md")
    m0 = mtp(R / "mtp_gpu0.txt")
    rows = []
    for q in QUANTS:
        n = BENCH_NAME[q]
        rows.append({"variant": q, "kind": "gguf",
                     "wiki": kld(R / f"kld_wiki_{q}.txt") if q != "BF16" else None,
                     "val": kld(R / f"kld_val_{q}.txt") if q != "BF16" else None,
                     "bench_9070xt": b0.get(n), "bench_r9700": b1.get(n), "mtp_9070xt": m0.get(q)})
    sparse = {}
    for corpus in ("wiki", "val"):
        p = R / f"sparse_kld_{corpus}.json"
        if p.exists():
            for k, v in json.loads(p.read_text())["results"].items():
                sparse.setdefault(k, {"variant": k, "kind": "torch"})[corpus] = v
    rows += list(sparse.values())
    (R / "summary.json").write_text(json.dumps(rows, indent=1))

    def f(d, key, fmt):
        return format(d[key], fmt) if d and key in d else "—"

    print("| variant | KLD val | p99 val | same-top val | KLD wiki | same-top wiki | size GiB |"
          " tg 9070XT | pp2048 9070XT | tg R9700 | MTP tg 9070XT (accept) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bb, bb1, mm = r.get("bench_9070xt"), r.get("bench_r9700"), (r.get("mtp_9070xt") or {})
        mt = mm.get("draft-mtp")
        print(f"| {r['variant']} | {f(r.get('val'), 'mean_kld', '.4f')} | "
              f"{f(r.get('val'), 'kld_p99', '.3f')} | {f(r.get('val'), 'same_top', '.3f')} | "
              f"{f(r.get('wiki'), 'mean_kld', '.4f')} | {f(r.get('wiki'), 'same_top', '.3f')} | "
              f"{f(bb or bb1, 'size_gib', '.2f')} | {f(bb, 'tg128', '.1f')} | "
              f"{f(bb, 'pp2048', '.0f')} | {f(bb1, 'tg128', '.1f')} | "
              + (f"{mt['tg']:.1f} ({mt['accept']:.0%})" if mt else "—") + " |")


if __name__ == "__main__":
    main()
