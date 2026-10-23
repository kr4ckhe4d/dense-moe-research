"""Debug: check k-means features for ties and NaNs.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/km_debug.py
"""
import sys; sys.path.insert(0,"scripts")
import torch
from safetensors.torch import load_file
from oracle_analysis import contributions
from dense_to_moe.models.ffn import DenseFFN
from dense_to_moe.utils.checkpoint import load_ffn
dev="cuda:0"; w=load_ffn(15,dev); dense=DenseFFN(*(w[p].float() for p in ("gate","up","down")))
x=load_file("cache/activations/calib/layer_15.safetensors")["x"].to(dev).float()
c=contributions(dense,x[:4096]); coact=(c>=c.quantile(0.9,dim=1,keepdim=True)).float().T
print("coact rows all-zero:", (coact.sum(1)==0).sum().item(), "of", coact.shape[0])
gen=torch.Generator().manual_seed(0); n=coact.shape[0]
cent=coact[torch.randperm(n,generator=gen)[:8].to(dev)]
d=torch.cdist(coact,cent)
print("nan in dist:", torch.isnan(d).sum().item(), " dist min/max", d.min().item(), d.max().item())
print("unique distance values:", d.unique().numel(), "of", d.numel())
feat=torch.nn.functional.normalize(dense.gate.float(),dim=1)
cent=feat[torch.randperm(n,generator=gen)[:8].to(dev)]; d=torch.cdist(feat,cent)
print("gate feats: nan", torch.isnan(d).sum().item(), "unique", d.unique().numel())
