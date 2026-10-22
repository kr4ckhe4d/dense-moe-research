"""Capture the software/hardware context that every experiment record must carry."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

import torch


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "arch": getattr(p, "gcnArchName", ""),
                         "total_memory": p.total_memory})
    import transformers
    import triton

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "triton": triton.__version__,
        "transformers": transformers.__version__,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "gpus": gpus,
        "host": platform.node(),
    }
