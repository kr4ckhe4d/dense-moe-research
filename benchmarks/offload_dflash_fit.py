#!/usr/bin/env python3
"""DFlash vs MTP for partially offloaded models, at the same VRAM budget (RX 9070 XT).

Takes the plain placements from the offload milestone (offload_mtp_gpu0.jsonl, spec "none") and,
with the patched build (patches/llama.cpp-rdna4-mmvq-multicol.patch), runs:
  - MTP n=3 at the -ngl already fitted for MTP (vram_matched), to separate the kernel effect;
  - DFlash with the drafter on the GPU (-ngld 99) and in RAM (-ngld 0), n=5 and n=7, each at the
    highest -ngl that keeps llama-server within the same budget rule (VRAM used after a warmup
    request, minus the card's idle use, <= budget - margin).
Same prompts as the offload milestone: short3 (code, prose, JSON) x 256 tokens, temperature 0.

    benchmarks/offload_dflash_fit.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from offload_mtp_fit import server_vram

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "experiments/results/offload/offload_mtp_gpu0.jsonl"
RES = HERE / "experiments/results/offload/offload_dflash_gpu0.jsonl"
FITS = HERE / "experiments/results/offload/offload_dflash_fits.json"
BIN = str(Path.home() / "llama.cpp-mmvq/build/bin")
M = Path("/mnt/fast/models")
DRAFTER = {"Qwen3.5-9B": M / "Qwen3.5-9B-DFlash-Q8_0.gguf",
           "Qwen3.8-27B": M / "Qwen3.8-27B-DFlash2-Q8_0.gguf"}
N_LAYER = {"Qwen3.5-9B": 33, "Qwen3.8-27B": 65}  # incl. the MTP layer


def fit(c: dict, model: str, limit_mib: float, gpu: int, port: int) -> int | None:
    """Highest -ngl <= c["ngl"] at which config c stays within limit_mib; None if none does.
    Tries c["ngl"], jumps down by the estimated excess, then steps by one layer."""
    per_layer = os.path.getsize(c["model_path"]) / 2**20 / N_LAYER[model]

    def ok(ngl: int) -> tuple[bool, int | None]:
        v = server_vram(dict(c, ngl=ngl), gpu, port)
        print(f"  ngl {ngl}: {v} MiB (limit {limit_mib:.0f})", flush=True)
        return v is not None and v <= limit_mib, v

    top = c["ngl"]
    good, v = ok(top)
    if good:
        return top
    excess = v - limit_mib if v is not None else os.path.getsize(DRAFTER[model]) / 2**20 + 512
    ngl = max(0, top - max(1, math.floor(excess / per_layer)))
    if ok(ngl)[0]:
        while ngl + 1 < top and ok(ngl + 1)[0]:
            ngl += 1
        return ngl
    while ngl > 0:
        ngl -= 1
        if ok(ngl)[0]:
            return ngl
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8198)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()
    rows = [json.loads(line)["config"] for line in SRC.read_text().splitlines()]
    plain = {json.dumps(c, sort_keys=True): c for c in rows if c["spec"] == "none"}.values()
    mtp = {json.dumps(c, sort_keys=True): c for c in rows if c.get("vram_matched")}.values()
    fits = json.loads(FITS.read_text()) if FITS.exists() else {}

    runs = [dict(c, bin=BIN) for c in mtp]
    for c in sorted(plain, key=lambda c: (c["quant"], c["budget_gib"])):
        model = "Qwen3.5-9B" if c["quant"].startswith("Qwen3.5") else "Qwen3.8-27B"
        for ngld in (99, 0):
            key = f"{c['quant']} {c['budget_gib']} ngld{ngld}"
            d = dict(c, bin=BIN, spec="draft-dflash", n_max=5, n_min=0, p_min=0.0,
                     vram_matched=True, extra=["-md", str(DRAFTER[model]), "-ngld", str(ngld)])
            if key not in fits:
                print(key, flush=True)
                fits[key] = fit(d, model, (c["budget_gib"] - args.margin) * 1024, args.gpu,
                                args.port)
                FITS.write_text(json.dumps(fits, indent=1))
            if fits[key] is not None:
                runs += [dict(d, ngl=fits[key], n_max=n) for n in (5, 7)]

    subprocess.run([sys.executable, str(HERE / "benchmarks/mtp_tune.py"), "--gpu", str(args.gpu),
                    "--prompt-set", "short3", "--max-tokens", "256", "--port", str(args.port),
                    "--out", str(RES), "--configs", json.dumps(runs)], check=True)


if __name__ == "__main__":
    main()
