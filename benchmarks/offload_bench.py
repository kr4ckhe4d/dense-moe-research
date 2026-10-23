#!/usr/bin/env python3
"""Partial GPU offload of dense GGUF models: speed per VRAM budget.

Simulates smaller GPUs on one real card by limiting what llama.cpp places in VRAM:
  layers   -ngl k: the last k-1 repeating layers (+ output head) on the GPU, the rest in RAM
  ffn      everything on the GPU except the FFN weights (gate/up/down) of the first k layers,
           which go to RAM via --override-tensor (attention, KV cache, norms stay on the GPU)

For each (model, strategy, budget) a binary search over k finds the most GPU-resident placement
whose llama.cpp GPU buffers (model + KV + recurrent state + compute, read from a 1-token debug
run) fit in `budget - margin` GiB. llama-bench then measures pp512 and tg128 for it. Weight bytes
left in RAM are computed from the GGUF tensor table (excluding the token embedding, of which one
row is read per token, and the MTP block, unused in plain decoding).

    benchmarks/offload_bench.py --gpu 0 --plan 9b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BIN = Path.home() / "llama.cpp/build/bin"
sys.path.insert(0, str(Path.home() / "llama.cpp/gguf-py"))
from gguf import GGUFReader

MODELS = Path("/mnt/fast/models")
PLANS = {
    "9b": {"models": [f"Qwen3.5-9B-{q}" for q in ("Q8_0", "Q6_K", "IQ4_XS")],
           "budgets": [4, 6, 8]},
    "27b": {"models": [f"Qwen3.8-27B-{q}" for q in ("Q8_0", "UD-Q6_K", "IQ4_XS-rq", "Q3_K_M-rq")],
            "budgets": [8, 12, 16]},
}
BUF = re.compile(r"(\S+) (model|KV|RS|compute) buffer size =\s+([\d.]+) MiB")
FFN = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight")


class Model:
    def __init__(self, name: str) -> None:
        self.name, self.path = name, MODELS / f"{name}.gguf"
        r = GGUFReader(self.path)
        arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
        self.n_block = int(r.fields[f"{arch}.block_count"].parts[-1][0])
        self.n_nextn = int(r.fields.get(f"{arch}.nextn_predict_layers", None).parts[-1][0]) \
            if f"{arch}.nextn_predict_layers" in r.fields else 0
        self.tensors = {t.name: int(t.n_bytes) for t in r.tensors}

    def block_of(self, name: str) -> int | None:
        m = re.match(r"blk\.(\d+)\.", name)
        return int(m[1]) if m else None

    def is_mtp(self, name: str) -> bool:
        b = self.block_of(name)
        return b is not None and b >= self.n_block - self.n_nextn

    def read_bytes(self) -> int:
        return sum(v for k, v in self.tensors.items() if k != "token_embd.weight" and not self.is_mtp(k))

    def ffn_bytes(self) -> int:
        return sum(v for k, v in self.tensors.items() if FFN.fullmatch(k) and not self.is_mtp(k))

    def cpu_read_bytes(self, strategy: str, k: int, gpu_blocks: int, output_on_gpu: bool) -> int:
        """Weight bytes a plain decode step reads from RAM for this placement."""
        n = 0
        for name, b in self.tensors.items():
            if name == "token_embd.weight" or self.is_mtp(name):
                continue
            blk = self.block_of(name)
            if strategy == "ffn":
                m = FFN.fullmatch(name)
                n += b if m and int(m[1]) < k else 0
            elif blk is None:
                n += 0 if (output_on_gpu or not name.startswith("output")) else b
            elif blk < self.n_block - gpu_blocks:
                n += b
        return n


def placement(strategy: str, k: int) -> list[str]:
    if strategy == "layers":
        return ["-ngl", str(k)]
    if k == 0:
        return ["-ngl", "99"]
    layers = "|".join(str(i) for i in range(k))
    return ["-ngl", "99", "-ot", rf"blk\.({layers})\.ffn_(gate|up|down)\.weight=CPU"]


def probe(m: Model, strategy: str, k: int, gpu: int, ctx: int, extra: list[str]) -> dict:
    """1-token debug run: GPU buffers (MiB) and what was offloaded."""
    cmd = [str(BIN / "llama-completion"), "-m", str(m.path), *placement(strategy, k), "-fa", "on",
           "-c", str(ctx), "-lv", "4", "-p", "hi", "-n", "1", "-no-cnv", "--no-warmup", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False,
                       env=dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu)))
    log = p.stdout + p.stderr
    bufs: dict[str, float] = {}
    for dev, kind, mib in BUF.findall(log):
        bufs[f"{dev} {kind}"] = bufs.get(f"{dev} {kind}", 0.0) + float(mib)
    rep = re.search(r"offloading (\d+) repeating layers", log)
    off = re.search(r"offloaded (\d+)/(\d+) layers", log)
    gpu_mib = sum(v for kk, v in bufs.items() if kk.startswith("ROCm"))
    ok = p.returncode == 0 and gpu_mib > 0
    return {"ok": ok, "gpu_mib": gpu_mib, "buffers": bufs,
            "gpu_blocks": int(rep[1]) if rep else 0,
            "output_on_gpu": bool(off and rep and int(off[1]) > int(rep[1])),
            "error": None if ok else log[-2000:]}


def search(m: Model, strategy: str, budget_gib: float, margin: float, gpu: int, ctx: int,
           extra: list[str]) -> tuple[int, dict] | None:
    """Most GPU-resident k whose GPU buffers fit in budget - margin."""
    limit = (budget_gib - margin) * 1024
    if strategy == "layers":
        lo, hi = 0, m.n_block + 1  # k GPU layers; more = more VRAM
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            pr = probe(m, strategy, mid, gpu, ctx, extra)
            if pr["ok"] and pr["gpu_mib"] <= limit:
                best, lo = (mid, pr), mid + 1
            else:
                hi = mid - 1
        return best
    lo, hi = 0, m.n_block - m.n_nextn  # k FFNs in RAM; more = less VRAM
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        pr = probe(m, strategy, mid, gpu, ctx, extra)
        if pr["ok"] and pr["gpu_mib"] <= limit:
            best, hi = (mid, pr), mid - 1
        else:
            lo = mid + 1
    return best


def bench(m: Model, strategy: str, k: int, gpu: int, reps: int, extra: list[str]) -> dict:
    cmd = [str(BIN / "llama-bench"), "-m", str(m.path), *placement(strategy, k), "-fa", "1",
           "-p", "512", "-n", "128", "-r", str(reps), "-o", "json", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True,
                       env=dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu)))
    rows = json.loads(p.stdout)
    out = {}
    for r in rows:
        key = "pp512" if r["n_prompt"] else "tg128"
        out[key], out[key + "_sd"] = r["avg_ts"], r["stddev_ts"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="HIP device (0 = RX 9070 XT)")
    ap.add_argument("--plan", choices=list(PLANS), required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--strategies", nargs="+", default=["layers", "ffn"])
    ap.add_argument("--budgets", nargs="*", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.5, help="GiB kept free (driver, display)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--extra", nargs="*", default=[], help="extra llama.cpp flags, e.g. -lm none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    plan = PLANS[args.plan]
    out = Path(args.out or HERE / f"experiments/results/offload/offload_{args.plan}_gpu{args.gpu}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["strategy"], r["budget_gib"], " ".join(r["extra"])))
    for name in args.models or plan["models"]:
        m = Model(name)
        for budget in args.budgets or plan["budgets"]:
            for strategy in args.strategies:
                key = (name, strategy, budget, " ".join(args.extra))
                if key in done:
                    continue
                found = search(m, strategy, budget, args.margin, args.gpu, args.ctx, args.extra)
                row = {"model": name, "strategy": strategy, "budget_gib": budget,
                       "margin_gib": args.margin, "ctx": args.ctx, "gpu": args.gpu,
                       "extra": args.extra, "read_bytes": m.read_bytes(), "ffn_bytes": m.ffn_bytes()}
                if found is None:
                    row.update(fits=False)
                else:
                    k, pr = found
                    cpu = m.cpu_read_bytes(strategy, k, pr["gpu_blocks"], pr["output_on_gpu"])
                    row.update(fits=True, k=k, gpu_mib=pr["gpu_mib"], buffers=pr["buffers"],
                               gpu_blocks=pr["gpu_blocks"], output_on_gpu=pr["output_on_gpu"],
                               cpu_read_bytes=cpu, placement=placement(strategy, k),
                               **bench(m, strategy, k, args.gpu, args.reps, args.extra))
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{name:22s} {strategy:6s} {budget:4.0f} GiB: " + (
                    f"k={row['k']:3d} GPU {row['gpu_mib'] / 1024:5.2f} GiB, RAM weights "
                    f"{row['cpu_read_bytes'] / 2**30:5.2f} GiB, pp512 {row['pp512']:7.1f}, "
                    f"tg128 {row['tg128']:6.2f}" if row["fits"] else "does not fit"), flush=True)


if __name__ == "__main__":
    main()
