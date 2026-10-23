"""Debug: static-selection results per partition on layer 15, outside oracle_analysis.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/bug_check.py
"""
import sys; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
from oracle_analysis import build_partitions
from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.routing.oracle import ORACLES, indices_to_weights
from dense_to_moe.evaluation.metrics import approximation_metrics
from dense_to_moe.utils.checkpoint import load_ffn
dev="cuda:0"; w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
xc=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
xv=load_file("cache/activations/val/layer_15.safetensors")["x"][:4096].to(dev).float()
ref=dense(xv)
for E in (4,8):
    parts=build_partitions(dense,xc,E,0,4096)
    for name in ("contiguous","kmeans_contribution"):
        moe=PartitionedFFN.from_dense(dense,parts[name]); out=moe.expert_outputs(xv)
        k=E//2; idx=ORACLES["static"](out,k)
        y=torch.einsum("etd,te->td",out,indices_to_weights(idx,E))
        print(E,name,"static idx",idx[0].tolist(),"rel",round(approximation_metrics(ref,y)["rel_err"],3),
              "out energy share",[round(v,3) for v in (out.pow(2).sum((1,2))/out.pow(2).sum()).tolist()])
