"""Time the pieces of a PyTorch router (linear, topk, dtype cast) at batch 1 against an empty op, to find the router overhead.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/router_probe.py
"""
import sys; sys.path.insert(0, "benchmarks")
import torch, torch.nn.functional as F
from kernel_latency import time_step
dev=torch.device("cuda:0"); d=4096
x=torch.randn(1,d,device=dev,dtype=torch.bfloat16); r=torch.randn(8,d,device=dev,dtype=torch.bfloat16)
logits=F.linear(x,r)
for name,fn in [("linear",lambda: F.linear(x,r)),("topk",lambda: logits.topk(2,dim=-1)),
                ("topk+int32",lambda: logits.topk(2,dim=-1).indices.to(torch.int32)),
                ("matmul-fp32 + argsort",lambda: (x.float()@r.float().T).argsort(dim=-1,descending=True)[:, :2]),
                ("empty graph op (x.add 0)",lambda: x.add(0))]:
    t=time_step(lambda: [fn() for _ in range(8)],10,30,True,dev); print(f"{name:28s} {sorted(t)[15]/8:6.1f} us")
