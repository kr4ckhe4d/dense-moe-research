"""Compare the PyTorch router with the fused Triton router (route_topk_triton) across block sizes.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/router_probe2.py
"""
import sys; sys.path.insert(0, "benchmarks")
import torch
from kernel_latency import time_step
from dense_to_moe.inference.sparse_ffn import route_topk, route_topk_triton
dev=torch.device("cuda:0")
x=torch.randn(1,4096,device=dev,dtype=torch.bfloat16)
for E,k in [(8,2),(16,4)]:
    r=torch.randn(E,4096,device=dev,dtype=torch.bfloat16)
    for name,fn in [("torch",route_topk),("triton",route_topk_triton)]:
        for bd in ([None] if name=="torch" else [256,512,1024]):
            f=(lambda: fn(x,r,k)) if bd is None else (lambda bd=bd: fn(x,r,k,block_d=bd))
            t=time_step(lambda: [f() for _ in range(8)],10,30,True,dev); print(f"E={E} k={k} {name} bd={bd}: {sorted(t)[15]/8:5.1f} us")
