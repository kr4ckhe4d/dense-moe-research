"""List the GPUs, measure BF16 GEMM TFLOP/s and device copy bandwidth on each, and list the transformers qwen3_5 classes.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/gpu_check.py
"""
import torch, time, transformers
print("torch", torch.__version__, "hip", torch.version.hip, "transformers", transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.gcnArchName, f"{p.total_memory/2**30:.1f} GiB", "CUs", p.multi_processor_count)
for i in range(torch.cuda.device_count()):
    d=f"cuda:{i}"
    a=torch.randn(8192,8192,device=d,dtype=torch.bfloat16); b=torch.randn_like(a)
    for _ in range(3): a@b
    torch.cuda.synchronize(d); t=time.perf_counter(); n=20
    for _ in range(n): a@b
    torch.cuda.synchronize(d); dt=(time.perf_counter()-t)/n
    x=torch.empty(2**28,dtype=torch.bfloat16,device=d); y=torch.empty_like(x)
    y.copy_(x); torch.cuda.synchronize(d); t=time.perf_counter()
    for _ in range(10): y.copy_(x)
    torch.cuda.synchronize(d); bw=10*2*x.numel()*2/(time.perf_counter()-t)/1e9
    print(d, f"bf16 GEMM {2*8192**3/dt/1e12:.1f} TFLOP/s, copy BW {bw:.0f} GB/s")
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print([n for n in dir(m) if n.startswith("Qwen3_5")])
import importlib.util
for pkg in ("fla","causal_conv1d","flash_attn"): print(pkg, bool(importlib.util.find_spec(pkg)))
