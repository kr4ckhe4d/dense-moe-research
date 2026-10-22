"""Read FFN weights from the local Hugging Face snapshot."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PREFIX = "model.language_model.layers"


@lru_cache(maxsize=4)
def snapshot_dir(repo: str = REPO, revision: str = REV) -> Path:
    return Path(snapshot_download(repo, revision=revision, local_files_only=True))


def load_ffn(layer: int, device: str | torch.device = "cpu", repo: str = REPO,
             revision: str = REV) -> dict[str, torch.Tensor]:
    snap = snapshot_dir(repo, revision)
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for p in ("gate", "up", "down"):
        name = f"{PREFIX}.{layer}.mlp.{p}_proj.weight"
        with safe_open(snap / wm[name], framework="pt", device=str(device)) as f:
            out[p] = f.get_tensor(name)
    return out
