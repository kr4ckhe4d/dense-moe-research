"""Debug: instrument analyze_layer to print which partition reaches PartitionedFFN.from_dense.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/debug/bug_check2.py
"""
import sys, types; sys.path.insert(0,"scripts")
import torch, oracle_analysis as OA
from dense_to_moe.models.ffn import PartitionedFFN
orig = PartitionedFFN.from_dense.__func__
def spy(cls, dense, part):
    g = part.groups
    contig = torch.equal(g, torch.arange(g.numel()).view_as(g))
    print(f"from_dense E={g.shape[0]} {part.strategy:28s} contiguous={contig} first={g[0,:5].tolist()}")
    return orig(cls, dense, part)
PartitionedFFN.from_dense = classmethod(spy)
OA.CONFIGS = [(4,2),(8,4)]; OA.ORACLES = {"static": OA.ORACLES["static"]}
args = types.SimpleNamespace(acts="cache/activations", val_tokens=1024, feat_tokens=4096, seed=0)
torch.cuda.set_device(0)
rec = OA.analyze_layer(15, args, torch.device("cuda:0"))
for r in rec["experts"]:
    if r["combine"]=="sum": print(r["E"], r["k"], r["partition"], round(r["rel_err"],3))
