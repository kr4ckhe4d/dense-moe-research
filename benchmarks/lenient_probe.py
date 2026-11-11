#!/usr/bin/env python3
"""Lenient draft acceptance: how much would it gain, and what does it cost?

Verification accepts a draft token only if it is the token the target picks. With
patches/llama.cpp-spec-lenient.patch the server can (a) log, for every verify step, the target's
probability of each draft token (SPEC_ACCEPT_LOG) and (b) also accept a draft token whose target
probability is at least r times the target's top probability (SPEC_LENIENT_RATIO=r).

For each ratio this starts one server (27B UD-Q6_K, MTP draft 3, one GPU), runs the six domain
prompts of mtp_tune.py at temperature 0 with thinking off, and writes one row per prompt with the
speed, the acceptance counts and the per-step log. Ratio 0 is exact verification; its log sizes
the gain of any rule offline (scripts/lenient_summary.py).

    benchmarks/lenient_probe.py --ratios 0 0.5 0.3 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_tune as mt  # noqa: E402

BIN = Path.home() / "llama.cpp-b12164-mmvq/build/bin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--model", default=str(mt.MODELS / "Qwen3.8-27B-UD-Q6_K.gguf"))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0])
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra", action="append", default=[], help="one further llama-server flag (repeatable, use --extra=--flag)")
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--out", default=str(mt.HERE / "experiments/results/lenient/probe.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["ratio"], r["n_max"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    logs = mt.HERE / "experiments/logs/lenient"
    logs.mkdir(parents=True, exist_ok=True)
    for ratio in args.ratios:
        if (ratio, args.n_max) in done:
            continue
        steps_path = logs / f"steps_r{ratio}_n{args.n_max}.jsonl"
        steps_path.unlink(missing_ok=True)
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu), SPEC_ACCEPT_LOG=str(steps_path))
        if ratio > 0:
            env["SPEC_LENIENT_RATIO"] = str(ratio)
        c = {"model_path": args.model, "ctx": 8192, "spec": "draft-mtp", "n_max": args.n_max,
             "n_min": 0, "p_min": 0.0, "bin": str(BIN), "extra": args.extra}
        rows = []
        with open(logs / f"server_r{ratio}_n{args.n_max}.log", "w") as log:
            proc = subprocess.Popen(mt.server_args(c, args.port), env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                mt.wait_healthy(args.port, proc)
                mt.generate(args.port, "Say hello.", 16, 0.0, 0)  # warmup
                for domain, prompt in mt.PROMPTS.items():
                    n0 = len(steps_path.read_text().splitlines()) if steps_path.exists() else 0
                    r = mt.generate(args.port, prompt, args.max_tokens, args.temperature, 0)
                    time.sleep(0.2)
                    steps = [json.loads(line) for line in steps_path.read_text().splitlines()[n0:]]
                    rows.append({"ratio": ratio, "n_max": args.n_max, "temperature": args.temperature, "extra": args.extra, "domain": domain,
                                 "model": Path(args.model).name, "gpu": args.gpu, **r, "steps": steps})
                    print(f"ratio {ratio:4.2f} {domain:12s} tg {r['tg']:6.1f} accepted "
                          f"{r['draft_accepted']}/{r['draft_n']} lenient "
                          f"{sum(s['n_lenient'] for s in steps)}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        time.sleep(2)
        with open(out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
