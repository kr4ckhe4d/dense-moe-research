#!/usr/bin/env python3
"""MTP speculative-decoding tuning sweep for Qwen3.5-9B GGUF quants on one GPU.

For each config (quant, spec type, draft n-max / n-min / p-min) this starts llama-server, runs one
warmup request, then every prompt in PROMPTS once at temperature 0 (thinking off, no prompt
cache), and records decode tok/s, prompt tok/s, drafted / accepted tokens and the generated
text. Results are appended to a JSONL file, one line per (config, prompt), so the sweep resumes.

Greedy speculative decoding should reproduce the non-speculative output; batched verification
can flip near-ties in floating point, so the texts are kept for comparison afterwards.

Other services on this machine can load models onto the same GPU on demand. Each config records
the card's VRAM in use (MiB) before our server starts and after it stops; anything above ~1 GiB
means another process shared the GPU and the timings of that config are suspect.

    benchmarks/mtp_tune.py --gpu 1 --quants Q8_0 IQ4_XS --grid nmax
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SERVER = Path.home() / "llama.cpp/build/bin/llama-server"
MODELS = Path("/mnt/fast/models")

# One prompt per domain of the calibration corpus. Written for this benchmark, not taken from any
# split. Each asks for a long answer so that max_tokens is reached.
PROMPTS = {
    "code_python": "Write a complete Python implementation of a thread-safe LRU cache class with "
                   "get, put, and delete methods, full docstrings, and type hints. Then write "
                   "pytest unit tests for it. Output only code.",
    "code_cpp": "Explain, step by step and in detail, how a lock-free single-producer "
                "single-consumer ring buffer works in C++. Include a full implementation with "
                "std::atomic and memory orders, and justify every memory order you use.",
    "prose": "Write a long, detailed essay on the history of the printing press and its effects "
             "on European science, religion and politics between 1450 and 1650.",
    "math": "Solve the following step by step, showing all work. A tank is filled by pipe A in 6 "
            "hours and by pipe B in 9 hours, and drained by pipe C in 12 hours. All three start "
            "together on an empty tank; after 2 hours pipe B is closed. How long does it take to "
            "fill the tank? Then generalise the method and solve three more variants of your own.",
    "instruction": "I am planning a two-week trip to Japan in spring for the first time. Give me "
                   "a detailed day-by-day itinerary with transport between cities, what to book "
                   "in advance, budget estimates in yen, and etiquette tips.",
    "json": "Produce a JSON array of 15 fictional employee records. Each record has id, "
            "full_name, email, department, title, hire_date (ISO 8601), salary, manager_id, "
            "skills (array of strings) and address (object with street, city, postcode, "
            "country). Output only valid JSON.",
}


def post(port: int, path: str, obj: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:  # not listening yet (URLError, connection refused, timeout)
            time.sleep(1)
            continue
        time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def generate(port: int, prompt: str, max_tokens: int, temperature: float, seed: int) -> dict:
    r = post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    t = r["timings"]
    return {"text": r["choices"][0]["message"]["content"], "n_gen": t["predicted_n"],
            "tg": t["predicted_per_second"], "pp": t["prompt_per_second"],
            "prompt_n": t["prompt_n"], "draft_n": t.get("draft_n", 0),
            "draft_accepted": t.get("draft_n_accepted", 0)}


def vram_used_mib(gpu: int) -> int:
    """VRAM in use on rocm-smi card<gpu> (same order as HIP indices on this machine)."""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], check=True,
                         capture_output=True, text=True).stdout
    return int(json.loads(out)[f"card{gpu}"]["VRAM Total Used Memory (B)"]) // 2**20


def config_key(c: dict) -> str:
    return json.dumps(c, sort_keys=True)


def server_args(c: dict, port: int) -> list[str]:
    a = [str(SERVER), "-m", str(MODELS / f"Qwen3.5-9B-{c['quant']}.gguf"), "-ngl", "99",
         "-fa", "on", "-c", "8192", "--port", str(port), "--no-webui", "-np", "1"]
    if c["spec"] != "none":
        a += ["--spec-type", c["spec"], "--spec-draft-n-max", str(c["n_max"]),
              "--spec-draft-n-min", str(c["n_min"]), "--spec-draft-p-min", str(c["p_min"])]
    return a + c.get("extra", [])  # any further llama-server flags, e.g. draft sampling


def grid(name: str, quants: list[str]) -> list[dict]:
    out = []
    for q in quants:
        out.append({"quant": q, "spec": "none"})
        def mtp(q=q, **kw) -> dict:
            return {"quant": q, "spec": "draft-mtp", "n_max": 3, "n_min": 0, "p_min": 0.0, **kw}
        if name in ("nmax", "full"):
            out += [mtp(n_max=n) for n in (1, 2, 3, 4, 5, 6, 8)]
        if name in ("pmin", "full"):
            out += [mtp(n_max=n, p_min=p) for n, p in itertools.product((4, 6, 8), (0.3, 0.6, 0.8))]
        if name == "default":
            out.append(mtp())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1, help="HIP device index (1 = R9700)")
    ap.add_argument("--quants", nargs="+", default=["Q8_0", "Q6_K", "IQ4_XS"])
    ap.add_argument("--grid", default="nmax", choices=["default", "nmax", "pmin", "full"])
    ap.add_argument("--configs", default=None, help="JSON list of configs instead of --grid")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or HERE / f"experiments/results/mtp/mtp_tune_gpu{args.gpu}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((config_key(r["config"]), r["prompt"], r["temperature"], r["max_tokens"]))

    configs = json.loads(args.configs) if args.configs else grid(args.grid, args.quants)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu))
    logs = HERE / "experiments/logs/mtp"
    logs.mkdir(parents=True, exist_ok=True)
    for c in configs:
        todo = [p for p in PROMPTS if (config_key(c), p, args.temperature, args.max_tokens) not in done]
        if not todo:
            continue
        tag = re.sub(r"[^\w.,=-]+", "", "_".join(f"{k}{v}" for k, v in sorted(c.items())))
        vram_before = vram_used_mib(args.gpu)
        rows = []
        with open(logs / f"server_gpu{args.gpu}_{tag}.log", "w") as log:
            proc = subprocess.Popen(server_args(c, args.port), env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                wait_healthy(args.port, proc)
                generate(args.port, "Say hello.", 32, 0.0, 0)  # warmup: graph capture, allocator
                for p in todo:
                    r = generate(args.port, PROMPTS[p], args.max_tokens, args.temperature, args.seed)
                    rows.append({"config": c, "prompt": p, "temperature": args.temperature,
                                 "max_tokens": args.max_tokens, "gpu": args.gpu, **r})
                    acc = r["draft_accepted"] / r["draft_n"] if r["draft_n"] else float("nan")
                    print(f"{tag:55s} {p:12s} tg {r['tg']:6.1f} n {r['n_gen']:4d} "
                          f"draft {r['draft_accepted']}/{r['draft_n']} ({acc:.0%})", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        time.sleep(2)
        vram_after = vram_used_mib(args.gpu)
        shared = max(vram_before, vram_after) > 1024
        if shared:
            print(f"WARNING {tag}: GPU {args.gpu} shared (VRAM {vram_before} / {vram_after} MiB)")
        with open(out, "a") as f:
            for row in rows:
                row.update(vram_before_mib=vram_before, vram_after_mib=vram_after, gpu_shared=shared)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
