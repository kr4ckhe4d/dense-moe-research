#!/usr/bin/env python3
"""Decode speed against context depth, to split the loss at depth into its parts.

For each config this starts llama-server once, then for each depth (ascending) sends a prompt of
that many tokens of llama.cpp C++ source followed by a task, and generates at temperature 0
(thinking off). The prompt cache is on, so each depth only prefills what the previous one did not
cover; the prompt tok/s of a row is therefore the prefill speed over that depth range.

Per row: decode tok/s, drafted / accepted tokens, and derived from them
  steps      = generated - accepted   (target evaluations: each yields one token plus the accepted drafts)
  step_ms    = decode time / steps    (for a plain config this is the time per token)
  tok_per_step = generated / steps

Comparing a plain config across depths gives the cost of attention over the KV cache; comparing
MTP's tok_per_step across depths gives the acceptance loss; step_ms(MTP) - step_ms(plain) at the
same depth is what drafting and the wider verify batch cost there.

    benchmarks/depth_profile.py --configs '[{"name": "plain-q8kv", "spec": "none",
        "extra": ["-ctk", "q8_0", "-ctv", "q8_0"]}]'
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

SRC = Path.home() / "llama.cpp-b12164-mmvq/src"  # a clean checkout of 4b77dee05
FILES = ["llama-model.cpp", "llama-context.cpp", "llama-graph.cpp", "llama-sampler.cpp"]
TASKS = {
    "explain": "Explain in detail what the last three functions in the code above do, including how "
               "they interact with the rest of the file and any edge cases they handle.",
    "code": "Write a self-contained C++ unit test file (plain asserts, a main function) that "
            "exercises the last function in the code above. Output only code.",
}


def corpus() -> str:
    return "".join((SRC / f).read_text() for f in FILES)


def cut(text: str, chars: int) -> str:
    """`text` up to `chars` characters, cut at a line boundary."""
    return text[:text.rfind("\n", 0, chars) + 1]


def generate(port: int, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = mt.post(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 0, "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=3600)
    t = r["timings"]
    n, acc = t["predicted_n"], t.get("draft_n_accepted", 0)
    steps = n - acc
    return {"depth": r["usage"]["prompt_tokens"], "prompt_n": t["prompt_n"],
            "pp": t["prompt_per_second"], "n_gen": n, "tg": t["predicted_per_second"],
            "draft_n": t.get("draft_n", 0), "draft_accepted": acc, "steps": steps,
            "step_ms": t["predicted_ms"] / steps, "tok_per_step": n / steps,
            "wall_s": time.perf_counter() - t0, "text": r["choices"][0]["message"]["content"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--all-gpus", action="store_true",
                    help="let llama-server split the model over every GPU (VRAM is then summed)")
    ap.add_argument("--model", default=str(mt.MODELS / "Qwen3.8-27B-UD-Q6_K.gguf"))
    ap.add_argument("--configs", required=True, help="JSON list; each needs name and spec")
    ap.add_argument("--depths", type=int, nargs="+", default=[2000, 32000, 64000, 120000])
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=8198)
    ap.add_argument("--out", default=str(mt.HERE / "experiments/results/depth/depth.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["config"]["name"] for line in out.read_text().splitlines()} if out.exists() else set()
    logs = mt.HERE / "experiments/logs/depth"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ) if args.all_gpus else dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu))
    gpus = (0, 1) if args.all_gpus else (args.gpu,)
    vram = lambda: sum(mt.vram_used_mib(g) for g in gpus)
    text = corpus()
    for c in json.loads(args.configs):
        if c["name"] in done:
            continue
        full = {"model_path": args.model, "ctx": c.get("ctx", args.ctx), "n_min": 0, "p_min": 0.0, **c}
        full["extra"] = ["--ctx-checkpoints", "4"] + c.get("extra", [])
        vram_before = vram()
        log_path = logs / f"server_gpu{'all' if args.all_gpus else args.gpu}_{c['name']}.log"
        rows, peak = [], vram_before
        with open(log_path, "w") as log:
            proc = subprocess.Popen(mt.server_args(full, args.port), env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                mt.wait_healthy(args.port, proc)
                # chars per token of this corpus, from one tokenize call on its first 200k chars
                sample = cut(text, 200_000)
                cpt = len(sample) / len(mt.post(args.port, "/tokenize", {"content": sample})["tokens"])
                generate(args.port, "Say hello.", 16)  # warmup
                for d in c.get("depths", args.depths):
                    code = cut(text, int(d * cpt))
                    for task, q in TASKS.items():
                        r = generate(args.port, f"```cpp\n{code}```\n\n{q}", args.max_tokens)
                        peak = max(peak, vram())
                        rows.append({"config": c, "task": task, "target_depth": d, "gpu": "all" if args.all_gpus else args.gpu,
                                     "model": Path(args.model).name, **r})
                        print(f"{c['name']:24s} {task:8s} depth {r['depth']:6d} pp {r['pp']:7.1f} "
                              f"({r['prompt_n']:6d}) tg {r['tg']:5.1f} step {r['step_ms']:6.1f} ms "
                              f"tok/step {r['tok_per_step']:.2f} "
                              f"draft {r['draft_accepted']}/{r['draft_n']}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        time.sleep(2)
        vram_after = vram()
        buffers = mt.buffers_from_log(log_path)
        with open(out, "a") as f:
            for row in rows:
                row.update(vram_before_mib=vram_before, vram_after_mib=vram_after,
                           vram_peak_mib=peak, buffers_mib=buffers,
                           gpu_shared=max(vram_before, vram_after) > 1024 * len(gpus))
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    sys.exit(main())
