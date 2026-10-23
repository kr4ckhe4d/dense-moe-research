"""Read the safetensors headers of Qwen3.5-9B over HTTP range requests (no download) and break parameters down by component.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/param_breakdown.py
"""
import json,struct,urllib.request,re,collections
base="https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/model.safetensors-0000{}-of-00004.safetensors"
shapes={}
for i in range(1,5):
    u=base.format(i)
    r=urllib.request.urlopen(urllib.request.Request(u,headers={"Range":"bytes=0-7"}),timeout=30); n=struct.unpack("<Q",r.read())[0]
    r=urllib.request.urlopen(urllib.request.Request(u,headers={"Range":f"bytes=8-{7+n}"}),timeout=30); h=json.loads(r.read())
    for k,v in h.items():
        if k!="__metadata__": shapes[k]=(v["shape"],v["dtype"])
import math
grp=collections.Counter(); ex={}
for k,(s,dt) in shapes.items():
    n=math.prod(s)
    if k.startswith("model.visual"): g="vision"
    elif k.startswith("mtp."): g="mtp"
    elif "embed_tokens" in k: g="embed"
    elif k=="lm_head.weight": g="lm_head"
    elif ".mlp." in k: g="ffn"
    elif ".linear_attn." in k: g="linear_attn (GatedDeltaNet)"
    elif ".self_attn." in k: g="full_attn"
    else: g="norms"
    grp[g]+=n
    kk=re.sub(r'\.\d+\.', '.N.', k)
    if kk not in ex and not k.startswith("model.visual"): ex[kk]=(s,dt)
tot=sum(grp.values())
for g,n in grp.most_common(): print(f"{g:28s} {n/1e9:7.3f} B  {100*n/tot:5.1f}%")
print(f"{'TOTAL':28s} {tot/1e9:7.3f} B")
text=tot-grp['vision']-grp['mtp']; print(f"text-only (no vision/mtp): {text/1e9:.3f} B; FFN share {100*grp['ffn']/text:.1f}%")
print()
for k,(s,dt) in sorted(ex.items()):
    if "layers.N" in k and ("layers.N.mlp" in k or "attn" in k): print(k, s, dt)
