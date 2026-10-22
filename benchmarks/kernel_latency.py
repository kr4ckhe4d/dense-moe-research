"""Single-layer FFN latency: dense vs partitioned-all-experts vs top-k sparse.

Methodology
- Real Qwen3.5-9B layer-15 FFN weights (BF16), replicated into `--copies` distinct buffers.
  Each timed step runs the FFN once per copy, the way a forward pass walks through layers.
  This defeats the 64 MB Infinity Cache: one top-2/8 expert pair is 75 MB, and re-running a
  single copy would serve part of it from cache and overstate the sparse speedup.
- Routing is done by a real (random-init) linear router on device, so router + top-k cost is
  included. Routing decisions vary per copy because each copy has its own router.
- CUDA events, `--warmup` untimed steps, `--repeats` timed steps; report median and std of
  per-layer time. Paths that need a host sync cannot be graph-captured and run eager only.

Usage:  source scripts/env.sh && .venv/bin/python benchmarks/kernel_latency.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from dense_to_moe.inference import sparse_ffn as S
from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.partitioning.strategies import contiguous_partition
from dense_to_moe.utils.env import environment
from dense_to_moe.utils.hf_weights import fetch_ffn_layer

REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BYTES = 2  # bf16


def time_step(fn, warmup: int, repeats: int, graph: bool, device) -> list[float]:
    """Return per-step times in microseconds."""
    torch.cuda.synchronize(device)
    if graph:
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream(device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    for _ in range(warmup):
        run()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return times


def summarize(name, times, copies, bytes_read, extra=None):
    per_layer = [t / copies for t in times]
    med = statistics.median(per_layer)
    rec = {
        "path": name,
        "us_per_layer_median": med,
        "us_per_layer_std": statistics.stdev(per_layer),
        "us_per_layer_min": min(per_layer),
        "weight_bytes_per_layer": bytes_read,
        "effective_GBps": bytes_read / (med * 1e-6) / 1e9,
    }
    if extra:
        rec.update(extra)
    print(f"  {name:38s} {med:8.1f} us  ±{rec['us_per_layer_std']:6.1f}   "
          f"{rec['effective_GBps']:6.0f} GB/s  ({bytes_read / 1e6:.0f} MB)")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--configs", default="1:1,4:4,4:3,4:2,8:8,8:6,8:4,8:2,16:8,16:4")
    ap.add_argument("--prefill", default="512,2048")
    ap.add_argument("--out", default="experiments/results/kernel_latency.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)  # graphs and events use the current device
    torch.manual_seed(0)

    w = fetch_ffn_layer(REPO, REV, 15)
    dense0 = DenseFFN(*(w[p].to(dev) for p in ("gate", "up", "down")))
    d, inter = dense0.hidden_size, dense0.intermediate_size
    dense_bytes = 3 * d * inter * BYTES
    packed_dense = [S.pack_dense(dense0)]
    packed_dense += [{k: v.clone() for k, v in packed_dense[0].items()} for _ in range(args.copies - 1)]
    x1 = torch.randn(1, d, device=dev, dtype=torch.bfloat16)
    results = {"env": environment(), "args": vars(args), "device_name": torch.cuda.get_device_name(dev),
               "decode": [], "prefill": []}
    print(f"device {args.device} ({torch.cuda.get_device_name(dev)}), copies={args.copies}")

    print("decode (T=1)")
    for graph in (False, True):
        t = time_step(lambda: [S.dense_ffn(x1, **p) for p in packed_dense], args.warmup,
                      args.repeats, graph, dev)
        results["decode"].append(summarize(f"dense{' [graph]' if graph else ''}", t, args.copies,
                                           dense_bytes, {"E": 1, "k": 1, "graph": graph}))

    for cfg in args.configs.split(","):
        E, k = map(int, cfg.split(":"))
        moe = PartitionedFFN.from_dense(dense0, contiguous_partition(inter, E))
        p0 = S.pack_experts(moe)
        del moe
        layers = [p0] + [{kk: v.clone() for kk, v in p0.items()} for _ in range(args.copies - 1)]
        routers = [torch.randn(E, d, device=dev, dtype=torch.bfloat16) * d ** -0.5
                   for _ in range(args.copies)]
        sparse_bytes = dense_bytes * k // E + E * d * BYTES
        meta = {"E": E, "k": k}
        print(f"E={E} k={k}")

        def run(fn, router, layers=layers, routers=routers, k=k):
            for p, r in zip(layers, routers):
                fn(x1, idx=router(x1, r, k), **p)

        variants = [("loop (host sync)", S.moe_decode_loop, S.route_topk, (False,)),
                    ("gather (weight copy)", S.moe_decode_gather, S.route_topk, (False, True)),
                    ("triton GEMV + torch router", S.moe_decode_triton, S.route_topk, (True,)),
                    ("triton GEMV + triton router", S.moe_decode_triton, S.route_topk_triton,
                     (False, True))]
        for name, fn, router, graphs in variants:
            for graph in graphs:
                t = time_step(lambda fn=fn, router=router: run(fn, router), args.warmup,
                              args.repeats, graph, dev)
                results["decode"].append(summarize(f"{name}{' [graph]' if graph else ''}", t,
                                                   args.copies, sparse_bytes,
                                                   {**meta, "graph": graph}))
        for rname, router in (("torch", S.route_topk), ("triton", S.route_topk_triton)):
            t = time_step(lambda router=router: [router(x1, r, k) for r in routers],
                          args.warmup, args.repeats, True, dev)
            results["decode"].append(summarize(f"router only ({rname}) [graph]", t, args.copies,
                                               E * d * BYTES, {**meta, "graph": True}))
        del layers

        for T in map(int, args.prefill.split(",")) if args.prefill else []:
            xT = torch.randn(T, d, device=dev, dtype=torch.bfloat16)
            pc = min(args.copies, 2)
            t = time_step(lambda: [S.dense_ffn(xT, **p) for p in packed_dense[:pc]], 3,
                          max(10, args.repeats // 5), False, dev)
            rec_d = summarize(f"prefill T={T} dense", t, pc, dense_bytes, {**meta, "T": T})
            for name, fn in (("loop", S.moe_prefill_loop), ("grouped_mm", S.moe_prefill_grouped)):
                idx = torch.rand(T, E, device=dev).argsort(1)[:, :k].to(torch.int32)
                t = time_step(lambda fn=fn: [fn(xT, idx=idx, **p0) for _ in range(pc)], 3,
                              max(10, args.repeats // 5), False, dev)
                rec = summarize(f"prefill T={T} {name}", t, pc, dense_bytes, {**meta, "T": T})
                rec["speedup_vs_dense"] = rec_d["us_per_layer_median"] / rec["us_per_layer_median"]
                results["prefill"].append(rec)
            results["prefill"].append(rec_d)
        del p0
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
