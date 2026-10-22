"""Dense-to-MoE conversion for faster local LLM inference."""

import os

# torch 2.14.1+rocm7.2 silently corrupts live tensors with expandable segments after large frees
# + empty_cache (scripts/repro_expandable_segments.py, docs/findings.md F0). Refuse to run with it.
for _var in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    if "expandable_segments:true" in os.environ.get(_var, "").replace(" ", "").lower():
        raise RuntimeError(f"{_var} enables expandable_segments, which corrupts tensors on this "
                           "ROCm build. Unset it (see scripts/env.sh).")
