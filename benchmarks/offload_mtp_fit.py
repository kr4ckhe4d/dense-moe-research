#!/usr/bin/env python3
"""MTP at the same VRAM budget as plain decoding, for partially offloaded models.

The MTP draft context allocates its own GPU buffers (~0.4 GiB for the 9B, ~0.8 GiB for the 27B),
so at the -ngl that fits a budget without MTP, MTP uses more VRAM or fails to allocate. This runs
the plain configs (if not done yet), then for each lowers -ngl until llama-server with MTP stays
within the same budget rule as the sweep (VRAM used per rocm-smi after a warmup request, minus
the card's idle use, <= budget - margin), then benchmarks MTP there with benchmarks/mtp_tune.py.

    benchmarks/offload_mtp_fit.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mtp_tune as mt

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "experiments/results/offload/offload_mtp_gpu0.jsonl"


def server_vram(c: dict, gpu: int, port: int) -> int | None:
    """VRAM (MiB) used by llama-server with config c after one short request; None if it fails."""
    before = mt.vram_used_mib(gpu)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(mt.server_args(c, port), env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        mt.wait_healthy(port, proc)
        mt.generate(port, "Say hello.", 32, 0.0, 0)
        return mt.vram_used_mib(gpu) - before
    except (RuntimeError, TimeoutError):
        return None
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--plain", required=True, help="JSON list of plain (spec none) configs")
    args = ap.parse_args()
    tune = [sys.executable, str(HERE / "benchmarks/mtp_tune.py"), "--gpu", str(args.gpu),
            "--prompt-set", "short3", "--max-tokens", "256", "--out", str(RES)]
    subprocess.run([*tune, "--configs", args.plain], check=True)
    rows = [json.loads(line) for line in RES.read_text().splitlines()]
    plain = {}
    for r in rows:
        c = r["config"]
        if c["spec"] == "none":
            plain[(c["quant"], c["budget_gib"])] = (c, r["vram_peak_mib"] - r["vram_before_mib"])
    fitted = []
    for (quant, budget), (c, used) in sorted(plain.items()):
        limit = (budget - args.margin) * 1024
        ngl = c["ngl"]
        while ngl > 0:
            m = dict(c, spec="draft-mtp", n_max=3, n_min=0, p_min=0.0, ngl=ngl, vram_matched=True)
            v = server_vram(m, args.gpu, args.port)
            print(f"{quant} {budget} GiB: ngl {ngl} -> {v} MiB (plain {used}, limit {limit:.0f})",
                  flush=True)
            if v is not None and v <= limit:
                fitted.append(m)
                break
            ngl -= 1
    subprocess.run([*tune, "--configs", json.dumps(fitted)], check=True)


if __name__ == "__main__":
    main()
