"""Debug: test whether an uninitialized assignment buffer makes the k-means result contiguous.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/km_trace2.py
"""
import torch
import dense_to_moe.partitioning.strategies as St
from dense_to_moe.partitioning.strategies import balanced_kmeans_partition
n,E=12288,8; m=n//E
# Simulate prior GPU allocations of the same size, as PartitionedFFN.from_dense does with groups.to(cuda)
for trial in range(3):
    prev = (torch.arange(n)//m).cuda()      # a freed contiguous assignment-like tensor
    del prev
    gen=torch.Generator().manual_seed(trial)
    feats=torch.nn.functional.normalize(torch.randn(n,64,generator=gen),dim=1).cuda()
    p=balanced_kmeans_partition(feats,E,seed=0,iters=15)
    contig=torch.equal(p.groups, torch.arange(n).view(E,m))
    print("trial",trial,"result contiguous:",contig)
