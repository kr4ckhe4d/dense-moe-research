"""Check that Triton kernels, CUDA graphs and torch._grouped_mm work on ROCm. (Triton refuses @jit functions defined on stdin, so this has to be a file.)

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/tri_check.py
"""
import torch, triton, triton.language as tl
@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0,B); m = i<n
    tl.store(o+i, tl.load(x+i,mask=m)+tl.load(y+i,mask=m), mask=m)
x=torch.randn(10000,device="cuda"); y=torch.randn_like(x); o=torch.empty_like(x)
add[(triton.cdiv(10000,1024),)](x,y,o,10000,B=1024); print("triton ok", torch.allclose(o,x+y), triton.__version__)
g=torch.cuda.CUDAGraph(); s=torch.zeros(1,device="cuda")
with torch.cuda.graph(g): s.add_(1)
g.replay(); g.replay(); torch.cuda.synchronize(); print("graph ok", s.item())
print("grouped_mm:", hasattr(torch, "_grouped_mm"))
p=torch.cuda.get_device_properties(0); print("L2 bytes", getattr(p,"L2_cache_size",None))
