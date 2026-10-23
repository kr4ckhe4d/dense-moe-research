"""Debug: per-expert output energy share of each partition on layer 15.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/kmeans_check.py
"""
import sys; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
from oracle_analysis import build_partitions, contributions
from dense_to_moe.models.ffn import DenseFFN
from dense_to_moe.utils.checkpoint import load_ffn
dev="cuda:0"; w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
x=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
c=contributions(dense,x[:4096])
for E in (4,8):
    parts=build_partitions(dense,x,E,0,4096)
    for name,p in parts.items():
        # mean contribution energy per expert, and within-expert co-activation
        energy=torch.stack([c[:,g.to(dev)].pow(2).sum(1).mean() for g in p.groups])
        sizes=[len(g) for g in p.groups]
        print(f"E={E} {name:22s} sizes={set(sizes)} expert energy share={[round(v,3) for v in (energy/energy.sum()).tolist()]}")
