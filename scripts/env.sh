# Source before running anything: `source scripts/env.sh`
# Hide the gfx1036 iGPU: the PyTorch ROCm wheel segfaults on init when it is visible.
# After this, cuda:0 = RX 9070 XT (16 GB), cuda:1 = Radeon AI PRO R9700 (32 GB).
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
