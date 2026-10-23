"""Debug: compare k-means inputs inside analyze_layer vs a direct call. Running it once with PYTORCH_HIP_ALLOC_CONF=expandable_segments:True exposed the allocator bug (F0). The dense_to_moe import guard now refuses that flag.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/bug_check3.py
"""
import sys, types; sys.path.insert(0,"scripts")
import torch, oracle_analysis as OA
orig=OA.balanced_kmeans_partition
def traced(features,E,seed,iters=20,strategy="kmeans"):
    p=orig(features,E,seed,iters,strategy)
    contig=torch.equal(p.groups,torch.arange(p.groups.numel()).view_as(p.groups))
    print(f"  [{CTX}] {strategy} E={E} shape={tuple(features.shape)} sum={features.double().sum().item():.4f} contiguous={contig}")
    return p
OA.balanced_kmeans_partition=traced
OA.CONFIGS=[(4,2),(8,4)]; OA.ORACLES={"static":OA.ORACLES["static"]}
args=types.SimpleNamespace(acts="cache/activations", val_tokens=1024, feat_tokens=4096, seed=0)
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
CTX="analyze"; OA.analyze_layer(15,args,dev)
from safetensors.torch import load_file
from dense_to_moe.models.ffn import DenseFFN
from dense_to_moe.utils.checkpoint import load_ffn
w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
xc=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
CTX="direct"
for E in (4,8): OA.build_partitions(dense,xc,E,0,4096)
