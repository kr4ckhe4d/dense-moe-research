"""Reproduce: on torch 2.14.1+rocm7.2, torch.bmm silently returns wrong values for every batch
index >= 65536 (65535 = the GPU grid-dimension limit). No error, no warning. Anything that lowers
to bmm with a large batch is affected, e.g. torch.einsum("tnb,nbc,tnc->tn") with t*n > 65535.

source scripts/env.sh && .venv/bin/python scripts/repro_bmm_batch_limit.py
"""

import os
import subprocess
import sys

import torch


def check(dev: str) -> None:
    g = torch.Generator().manual_seed(0)
    print(f"{torch.cuda.get_device_name(dev)} ({dev}), "
          f"TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT', '<unset>')}")
    for nb in (65535, 65536, 65537, 131072):
        a, b = torch.randn(nb, 4, 16, generator=g), torch.randn(nb, 16, 4, generator=g)
        err = (torch.bmm(a.to(dev), b.to(dev)).cpu() - torch.bmm(a, b)).abs().amax(dim=(1, 2))
        bad = (err > 1e-3).nonzero().flatten()
        print(f"  bmm batch {nb:6d}: wrong batches {len(bad):6d}"
              + (f", first {bad[0].item()}" if len(bad) else ""))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(sys.argv[1])
    else:
        for dev in range(torch.cuda.device_count()):
            for blaslt in ("1", "0"):
                env = dict(os.environ, TORCH_BLAS_PREFER_HIPBLASLT=blaslt)
                subprocess.run([sys.executable, __file__, f"cuda:{dev}"], env=env, check=True)
