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
    benchmarks/mtp_tune.py --gpu 1 --prompt-set longctx --configs '[{"quant": "Q6_K", ...}]'

Prompt sets: `domains` (six short prompts), `longctx` (llama.cpp C++ source as ~4k / 16k / 28k
tokens of context, then a question about it; needs "ctx": 32768 in the config), `edit` (return a
whole source file with small additions: highly repetitive output). A config with "np": N runs N
server slots and sends groups of N different prompts at once; tg is then aggregate throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


LLAMA = Path.home() / "llama.cpp"
QUESTION = ("Explain in detail what the code above does, function by function, including how the "
            "functions interact and any edge cases they handle.")
EDIT = ("Reproduce the following file in full, unchanged, except add a one-line comment directly "
        "above every function or method definition that says what it does. Output only the code.")


def head_lines(path: Path, chars: int) -> str:
    """The first lines of a file, up to `chars` characters (cut at a line boundary)."""
    out, n = [], 0
    for line in path.read_text().splitlines(keepends=True):
        if n + len(line) > chars:
            break
        out.append(line)
        n += len(line)
    return "".join(out)


def prompt_set(name: str) -> dict[str, str]:
    if name == "domains":
        return PROMPTS
    if name == "short3":  # for slow (offloaded) configs: code, prose, json
        return {k: PROMPTS[k] for k in ("code_python", "prose", "json")}
    if name == "longctx":  # llama.cpp C++ is ~3.37 chars/token for this tokenizer
        src = LLAMA / "src/llama-kv-cache.cpp"
        return {f"longctx_{k}k": f"```cpp\n{head_lines(src, int(k * 1000 * 3.37))}```\n\n{QUESTION}"
                for k in (4, 16, 28)}
    if name == "pp":  # prompt processing: ~4k tokens of C++ in, a few tokens out
        src = LLAMA / "src/llama-kv-cache.cpp"
        return {"pp_4k": f"```cpp\n{head_lines(src, int(4000 * 3.37))}```\n\nSummarise this file in one line."}
    if name == "edit":
        return {"edit_python": f"{EDIT}\n\n```python\n{(LLAMA / 'gguf-py/gguf/lazy.py').read_text()}```",
                "edit_cpp": f"{EDIT}\n\n```cpp\n{(LLAMA / 'src/llama-impl.cpp').read_text()}```"}
    raise ValueError(name)


def post(port: int, path: str, obj: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 600) -> None:
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


def generate_group(port: int, prompts: list[str], max_tokens: int, temperature: float,
                   seed: int) -> dict:
    """Send the prompts concurrently; aggregate tok/s = all generated tokens / wall time."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(prompts)) as ex:
        rs = list(ex.map(lambda pr: generate(port, pr, max_tokens, temperature, seed), prompts))
    wall = time.perf_counter() - t0
    n = sum(r["n_gen"] for r in rs)
    return {"text": [r["text"] for r in rs], "n_gen": n, "wall_s": wall, "tg": n / wall,
            "tg_per_request": [r["tg"] for r in rs], "pp": min(r["pp"] for r in rs),
            "prompt_n": sum(r["prompt_n"] for r in rs),
            "draft_n": sum(r["draft_n"] for r in rs),
            "draft_accepted": sum(r["draft_accepted"] for r in rs)}


BUFFER = re.compile(r"(\w+(?:_\w+)?) (model|KV|compute|RS) buffer size =\s+([\d.]+) MiB")


def buffers_from_log(path: Path) -> dict[str, float]:
    """MiB per (device, kind) from llama-server's load log, e.g. {"ROCm0 model": 5021.3}."""
    out: dict[str, float] = {}
    for m in BUFFER.finditer(path.read_text(errors="replace")):
        k = f"{m[1]} {m[2]}"
        out[k] = out.get(k, 0.0) + float(m[3])
    return out


def vram_used_mib(gpu: int) -> int:
    """VRAM in use on rocm-smi card<gpu> (same order as HIP indices on this machine)."""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], check=True,
                         capture_output=True, text=True).stdout
    return int(json.loads(out)[f"card{gpu}"]["VRAM Total Used Memory (B)"]) // 2**20


def config_key(c: dict) -> str:
    return json.dumps(c, sort_keys=True)


def model_path(c: dict) -> str:
    return c.get("model_path") or str(MODELS / f"Qwen3.5-9B-{c['quant']}.gguf")


def server_args(c: dict, port: int) -> list[str]:
    """Optional placement fields: "ngl" (GPU layers, default 99), "ot" (list of --override-tensor
    patterns, e.g. FFN weights of some layers to CPU), "threads", "bin" (another llama.cpp build's
    bin directory)."""
    server = Path(c["bin"]) / "llama-server" if "bin" in c else SERVER
    a = [str(server), "-m", model_path(c), "-ngl", str(c.get("ngl", 99)),
         "-fa", "on", "-c", str(c.get("ctx", 8192)), "--port", str(port), "--no-webui",
         "-np", str(c.get("np", 1))]
    for pat in c.get("ot", []):
        a += ["-ot", pat]
    if "threads" in c:
        a += ["-t", str(c["threads"])]
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
    ap.add_argument("--prompt-set", default="domains", choices=["domains", "short3", "longctx", "edit", "pp"])
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or HERE / f"experiments/results/mtp/mtp_tune_gpu{args.gpu}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((config_key(r["config"]), r["prompt"], r["temperature"], r["max_tokens"],
                      r.get("seed", 0)))

    configs = json.loads(args.configs) if args.configs else grid(args.grid, args.quants)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu))
    logs = HERE / "experiments/logs/mtp"
    logs.mkdir(parents=True, exist_ok=True)
    prompts = prompt_set(args.prompt_set)
    for c in configs:
        np_ = c.get("np", 1)
        # With np > 1 slots, request groups of np different prompts are sent at once ("groupN").
        units = (list(prompts) if np_ == 1 else
                 [f"group{i}" for i in range(len(prompts))])
        todo = [u for u in units
                if (config_key(c), u, args.temperature, args.max_tokens, args.seed) not in done]
        if not todo:
            continue
        tag = re.sub(r"[^\w.,=-]+", "", "_".join(f"{k}{v}" for k, v in sorted(c.items())))
        if len(tag) > 200:  # file names are limited to 255 bytes
            tag = tag[:180] + "_" + hashlib.sha1(tag.encode()).hexdigest()[:12]
        vram_before = vram_used_mib(args.gpu)
        rows = []
        log_path = logs / f"server_gpu{args.gpu}_{tag}.log"
        peak_vram = vram_before
        with open(log_path, "w") as log:
            proc = subprocess.Popen(server_args(c, args.port), env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                wait_healthy(args.port, proc)
                peak_vram = max(peak_vram, vram_used_mib(args.gpu))
                generate(args.port, "Say hello.", 32, 0.0, 0)  # warmup: graph capture, allocator
                names = list(prompts)
                for p in todo:
                    if np_ == 1:
                        r = generate(args.port, prompts[p], args.max_tokens, args.temperature,
                                     args.seed)
                    else:
                        i = int(p[5:])
                        group = [names[(i + j) % len(names)] for j in range(np_)]
                        r = generate_group(args.port, [prompts[g] for g in group], args.max_tokens,
                                           args.temperature, args.seed)
                        r["group"] = group
                    text = prompts[p] if np_ == 1 else "".join(prompts[g] for g in group)
                    rows.append({"config": c, "prompt": p, "prompt_set": args.prompt_set,
                                 "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                                 "temperature": args.temperature, "seed": args.seed,
                                 "max_tokens": args.max_tokens, "gpu": args.gpu, **r})
                    peak_vram = max(peak_vram, vram_used_mib(args.gpu))
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
        buffers = buffers_from_log(log_path)
        if shared:
            print(f"WARNING {tag}: GPU {args.gpu} shared (VRAM {vram_before} / {vram_after} MiB)")
        with open(out, "a") as f:
            for row in rows:
                row.update(vram_before_mib=vram_before, vram_after_mib=vram_after, gpu_shared=shared,
                           vram_peak_mib=peak_vram, buffers_mib=buffers)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
