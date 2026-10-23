"""Debug: trace co-activation k-means iterations on real layer-15 data.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/km_trace3.py
"""
import sys; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
from oracle_analysis import contributions
from dense_to_moe.models.ffn import DenseFFN
from dense_to_moe.utils.checkpoint import load_ffn
dev="cuda:0"; w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
xc=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
c=contributions(dense,xc[:4096]); x=(c>=c.quantile(0.9,dim=1,keepdim=True)).float().T
n,E=x.shape[0],8; m=n//E
gen=torch.Generator().manual_seed(0)
cent=x[torch.randperm(n,generator=gen)[:E].to(dev)].clone()
assign=None
for it in range(6):
    dist=torch.cdist(x,cent); order=torch.argsort(dist.flatten())
    fill=[0]*E; placed=[-1]*n; cnt=0
    for nu,cl in zip((order//E).tolist(),(order%E).tolist()):
        if placed[nu]>=0 or fill[cl]>=m: continue
        placed[nu]=cl; fill[cl]+=1; cnt+=1
        if cnt==n: break
    new=torch.tensor(placed,device=dev)
    contig=(new==torch.arange(n,device=dev)//m).float().mean().item()
    chg=None if assign is None else (new!=assign).float().mean().item()
    print(f"iter {it}: frac matching contiguous {contig:.3f}, changed {chg}, centroid norms {[round(v,1) for v in cent.norm(dim=1).tolist()]}, dist std per col {[round(v,2) for v in dist.std(0).tolist()]}")
    assign=new
    for e in range(E): cent[e]=x[assign==e].mean(0)
