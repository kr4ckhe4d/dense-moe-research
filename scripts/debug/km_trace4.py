"""Debug: repeat build_partitions to catch nondeterministic degeneration.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/km_trace4.py
"""
import sys; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
import oracle_analysis as OA
from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.utils.checkpoint import load_ffn
import dense_to_moe.partitioning.strategies as St
orig=St.balanced_kmeans_partition
def traced(features,E,seed,iters=20,strategy="kmeans"):
    p=orig(features,E,seed,iters,strategy)
    n=p.groups.numel(); contig=torch.equal(p.groups,torch.arange(n).view_as(p.groups))
    runs=(p.groups[:,1:]-p.groups[:,:-1]==1).float().mean().item()
    print(f"  {strategy} E={E}: contiguous={contig} frac-consecutive={runs:.3f} feat dtype={features.dtype} dev={features.device}")
    return p
OA.balanced_kmeans_partition=traced
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
xc=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
for rep in range(3):
    print("rep",rep)
    for E in (4,8,16):
        parts=OA.build_partitions(dense,xc,E,0,4096)
