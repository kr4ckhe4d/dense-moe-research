"""Minimal check for live-tensor corruption with expandable_segments on ROCm.

PYTORCH_HIP_ALLOC_CONF=expandable_segments:True HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/repro_expandable_segments.py
Prints CORRUPTED if a tensor that is never written changes after unrelated frees + empty_cache.
"""
import torch

torch.manual_seed(0)
dev = "cuda:0"
keep = [torch.randn(12288, 4096, device=dev) for _ in range(3)]  # long-lived, never written
ref = [k.double().sum().item() for k in keep]
for step in range(6):
    tmp = [torch.randn(int(n), 4096, device=dev) for n in (4096 * 4, 12288, 6144 * (step + 1))]
    y = sum(t.sum() for t in tmp)
    del tmp, y
    torch.cuda.empty_cache()
    now = [k.double().sum().item() for k in keep]
    if now != ref:
        print(f"step {step}: CORRUPTED  before={ref} after={now}")
        break
else:
    print("no corruption observed")
