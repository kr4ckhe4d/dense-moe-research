"""Fetch individual tensors from a Hugging Face safetensors checkpoint by HTTP range request.

Lets single-layer experiments use real weights without downloading the whole checkpoint.
Tensors are cached as .safetensors files under `cache_dir`.
"""

from __future__ import annotations

import json
import logging
import struct
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "cache" / "tensors"


def _url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _shard_header(repo: str, revision: str, shard: str) -> tuple[dict, int]:
    url = _url(repo, revision, shard)
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    return json.loads(_get(url, 8, 7 + n)), 8 + n


def fetch_tensors(repo: str, revision: str, names: list[str],
                  cache_dir: Path = DEFAULT_CACHE) -> dict[str, torch.Tensor]:
    cache_dir = cache_dir / repo.replace("/", "--") / revision
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, torch.Tensor] = {}
    missing = []
    for name in names:
        f = cache_dir / f"{name}.safetensors"
        if f.exists():
            out[name] = load_file(f)[name]
        else:
            missing.append(name)
    if not missing:
        return out

    index_url = _url(repo, revision, "model.safetensors.index.json")
    with urllib.request.urlopen(index_url, timeout=60) as r:
        weight_map = json.load(r)["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}
    for name in missing:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = _shard_header(repo, revision, shard)
        header, base = headers[shard]
        meta = header[name]
        s, e = meta["data_offsets"]
        t0 = time.perf_counter()
        raw = _get(_url(repo, revision, shard), base + s, base + e - 1)
        dt = time.perf_counter() - t0
        log.info("fetched %s (%.1f MB) in %.1fs = %.1f MB/s", name, len(raw) / 1e6, dt,
                 len(raw) / 1e6 / dt)
        t = torch.frombuffer(bytearray(raw), dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"])
        save_file({name: t}, cache_dir / f"{name}.safetensors")
        out[name] = t
    return out


def fetch_ffn_layer(repo: str, revision: str, layer: int,
                    prefix: str = "model.language_model.layers") -> dict[str, torch.Tensor]:
    names = {p: f"{prefix}.{layer}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")}
    t = fetch_tensors(repo, revision, list(names.values()))
    return {p: t[n] for p, n in names.items()}
