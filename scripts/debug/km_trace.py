"""Debug: trace the greedy balanced assignment on synthetic data.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/km_trace.py
"""
import torch
n,E=12288,8; m=n//E
gen=torch.Generator().manual_seed(0)
x=torch.nn.functional.normalize(torch.randn(n,64,generator=gen),dim=1).cuda()
cent=x[torch.randperm(n,generator=gen)[:E].cuda()].clone()
dist=torch.cdist(x,cent); order=torch.argsort(dist.flatten())
neuron=order//E; cluster=order%E
fill=[0]*E; placed=[-1]*n; cnt=0; first_forced=None
for i,(nu,cl) in enumerate(zip(neuron.tolist(),cluster.tolist())):
    if placed[nu]>=0 or fill[cl]>=m: continue
    placed[nu]=cl; fill[cl]+=1; cnt+=1
    if cnt==n: break
a=torch.tensor(placed)
print("pairs visited", i, "of", n*E)
for e in range(E):
    g=torch.nonzero(a==e).flatten()
    runs=(g[1:]-g[:-1]==1).float().mean().item()
    print(e, "size", len(g), "frac consecutive", round(runs,3), "mean dist to own centroid", round(dist[g,e].mean().item(),3))
