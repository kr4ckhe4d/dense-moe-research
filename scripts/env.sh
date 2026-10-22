# Source before running anything: `source scripts/env.sh`
# Hide the gfx1036 iGPU: the PyTorch ROCm wheel segfaults on init when it is visible.
# After this, cuda:0 = RX 9070 XT (16 GB), cuda:1 = Radeon AI PRO R9700 (32 GB).
export HIP_VISIBLE_DEVICES=0,1
# Do NOT set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True. On torch 2.14.1+rocm7.2 it silently
# corrupts live tensors after large frees + torch.cuda.empty_cache() (see docs/findings.md F0).
unset PYTORCH_HIP_ALLOC_CONF
