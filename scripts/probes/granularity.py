"""Oracle FFN error versus expert size (E = 8 to 256) on layers 4, 15 and 26, random vs k-means partitions. Writes experiments/results/oracle_granularity.json.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/granularity.py
"""
import sys, json, time; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
from oracle_analysis import contributions
from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.partitioning.strategies import balanced_kmeans_partition, random_partition
from dense_to_moe.routing.oracle import select_by_norm, indices_to_weights
from dense_to_moe.evaluation.metrics import approximation_metrics
from dense_to_moe.utils.checkpoint import load_ffn
dev=torch.device("cuda:0"); torch.cuda.set_device(dev); res=[]
for layer in (4,15,26):
    w=load_ffn(layer,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
    xc=load_file(f"cache/activations/calib/layer_{layer:02d}.safetensors")["x"][:4096].to(dev).float()
    xv=load_file(f"cache/activations/val/layer_{layer:02d}.safetensors")["x"][:2048].to(dev).float()
    ref=dense(xv); c=contributions(dense,xc); feats=(c/c.norm(dim=1,keepdim=True)).T
    for E in (8,32,64,128,256):
        t=time.time()
        for pname,part in (("random",random_partition(12288,E,0)),("kmeans_contribution",balanced_kmeans_partition(feats,E,0,iters=8))):
            moe=PartitionedFFN.from_dense(dense,part); out=moe.expert_outputs(xv)
            for frac in (0.75,0.5,0.25):
                k=int(E*frac); y=torch.einsum("etd,te->td",out,indices_to_weights(select_by_norm(out,k),E))
                r=approximation_metrics(ref,y)["rel_err"]; res.append((layer,E,pname,frac,r))
            del out,moe; torch.cuda.empty_cache()
        print(f"layer {layer} E={E} ({time.time()-t:.0f}s): "+" ".join(f"{p[:6]} keep{f}={r:.3f}" for (l,e,p,f,r) in res if l==layer and e==E), flush=True)
json.dump(res,open("experiments/results/oracle_granularity.json","w"))
