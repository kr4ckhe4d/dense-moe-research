"""Phase 3 Stage B: per-layer block-granularity x budget sweep of token-dependent FFN sparsity.

For each layer, on held-out val activations (calib activations only for `gate` statistics and
block clustering), measure how well each selection method approximates the dense FFN output:
  oracle norm / contrib   top-k blocks by output norm (needs all of gate+up: quality bound)
  oracle greedy           greedy reconstruction (token subset; B >= 16)
  oracle norm, clustered  same, after permuting neurons into k-means blocks (subset of layers)
  gate                    CATS-like, from the gate projection only
  teal / wina             input-magnitude sparsity of x and h (deployable without a predictor)
B = 1536 / 3072 contiguous are the Phase 1 E=8 / E=4 controls.

source scripts/env.sh && .venv/bin/python scripts/phase3_block_sweep.py --layers all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from dense_to_moe.evaluation.metrics import approximation_metrics
from dense_to_moe.partitioning.strategies import balanced_kmeans_partition
from dense_to_moe.sparsity.blocks import SparseFFN, SparseSpec
from dense_to_moe.utils.checkpoint import load_ffn
from dense_to_moe.utils.env import environment

log = logging.getLogger(__name__)
BLOCKS = [1, 8, 16, 32, 64, 128, 256, 512]
CONTROLS = [1536, 3072]
KEEPS = [0.9, 0.75, 0.625, 0.5, 0.375, 0.25]
GREEDY_BLOCKS = [16, 32, 64, 128, 256, 512]
CLUSTER_BLOCKS = [16, 64, 256]


def mask_stats(mask: torch.Tensor, kept: float) -> dict[str, float]:
    """How token-specific the selection is. static_coverage: share of all selections that fall in
    the k most frequently chosen blocks (1.0 = the same blocks for every token, `kept` = uniform).
    jaccard: mean overlap of two random tokens' masks."""
    f = mask.float().mean(0)
    k = int(mask[0].sum())
    cov = f.sort(descending=True).values[:k].sum() / f.sum()
    perm = torch.randperm(mask.shape[0], generator=torch.Generator().manual_seed(0)).to(mask.device)
    a, b = mask, mask[perm]
    jac = ((a & b).sum(1).float() / (a | b).sum(1).float()).mean()
    return {"hot90": (f > 0.9).float().mean().item(), "cold10": (f < 0.1).float().mean().item(),
            "static_coverage": cov.item(), "jaccard": jac.item(), "uniform": kept}


def run(ffn: SparseFFN, x: torch.Tensor, ref: torch.Tensor, spec: SparseSpec) -> dict:
    y, info = ffn.forward(x, spec)
    r = approximation_metrics(ref, y)
    r.update(kept=info["kept"], x_kept=info["x_kept"], bytes=info["bytes"])
    r.update(mask_stats(info["mask"], info["kept"]))
    return r


def permuted(w: dict, x_cal: torch.Tensor, block: int, feat_tokens: int) -> SparseFFN:
    """Neurons regrouped into balanced k-means clusters of size `block` (contribution features)."""
    gate, up, down = w["gate"], w["up"], w["down"]
    c = (F.silu(F.linear(x_cal[:feat_tokens], gate)) * F.linear(x_cal[:feat_tokens], up)).abs()
    c = c * down.norm(dim=0)
    feats = (c / c.norm(dim=1, keepdim=True)).T
    perm = balanced_kmeans_partition(feats, gate.shape[0] // block, seed=0, iters=10).groups.flatten()
    perm = perm.to(gate.device)
    return SparseFFN(gate[perm].contiguous(), up[perm].contiguous(), down[:, perm].contiguous())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tokens", type=int, default=2048, help="val tokens per layer")
    ap.add_argument("--greedy-tokens", type=int, default=256)
    ap.add_argument("--cal-tokens", type=int, default=4096)
    ap.add_argument("--cluster-layers", default="4,15,26")
    ap.add_argument("--out", default="experiments/results/phase3/block_sweep.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    layers = list(range(32)) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    cluster_layers = {int(v) for v in args.cluster_layers.split(",") if v}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"env": environment(), "args": vars(args), "layers": {}}
    if out.exists():  # resume
        result["layers"] = json.loads(out.read_text())["layers"]

    for i in layers:
        if str(i) in result["layers"]:
            continue
        t0 = time.time()
        w = {k: v.to(dev).float() for k, v in load_ffn(i).items()}
        ffn = SparseFFN(w["gate"], w["up"], w["down"])
        x_cal = load_file(f"cache/activations/calib/layer_{i:02d}.safetensors")["x"]
        x_cal = x_cal[: args.cal_tokens].to(dev).float()
        ffn.calibrate(x_cal)
        x = load_file(f"cache/activations/val/layer_{i:02d}.safetensors")["x"][: args.tokens]
        x = x.to(dev).float()
        ref = F.linear(ffn.intermediate(x), ffn.down)
        xg, refg = x[: args.greedy_tokens], ref[: args.greedy_tokens]
        rows = {}

        def add(key: str, f: SparseFFN, xx, rr, spec: SparseSpec, rows=rows) -> None:
            rows[key] = {"spec": spec.__dict__, **run(f, xx, rr, spec)}

        for keep in KEEPS:
            for B in BLOCKS + CONTROLS:
                add(SparseSpec("oracle", keep, B).name, ffn, x, ref, SparseSpec("oracle", keep, B))
            for B in BLOCKS:
                for spec in (SparseSpec("oracle", keep, B, "contrib"), SparseSpec("gate", keep, B)):
                    add(spec.name, ffn, x, ref, spec)
            for spec in (SparseSpec("teal", keep, 1, x_keep=keep),
                         SparseSpec("wina", keep, 1, x_keep=keep),
                         SparseSpec("wina", keep, 32, x_keep=keep, x_block=32),
                         SparseSpec("wina", keep, 1, x_keep=1.0),
                         SparseSpec("wina", 1.0, 1, x_keep=keep)):
                add(spec.name, ffn, x, ref, spec)
            for B in GREEDY_BLOCKS:
                add(SparseSpec("oracle", keep, B, "greedy").name + " [subset]", ffn, xg, refg,
                    SparseSpec("oracle", keep, B, "greedy"))
                add(SparseSpec("oracle", keep, B).name + " [subset]", ffn, xg, refg,
                    SparseSpec("oracle", keep, B))
        if i in cluster_layers:
            for B in CLUSTER_BLOCKS:
                pf = permuted(w, x_cal, B, feat_tokens=2048)
                for keep in KEEPS:
                    add(SparseSpec("oracle", keep, B).name + " [clustered]", pf, x, ref,
                        SparseSpec("oracle", keep, B))
                del pf
        result["layers"][str(i)] = rows
        out.write_text(json.dumps(result, indent=1))
        r = rows
        log.info("layer %02d %.0fs  B=1/16/64 @50%%: %.3f %.3f %.3f  gate B=16: %.3f  wina: %.3f",
                 i, time.time() - t0, r["oracle B=1 keep=0.5"]["rel_err"],
                 r["oracle B=16 keep=0.5"]["rel_err"], r["oracle B=64 keep=0.5"]["rel_err"],
                 r["gate B=16 keep=0.5"]["rel_err"], r["wina B=1 keep=0.5 x_keep=0.5 xB=1"]["rel_err"])
        del w, ffn, x, x_cal, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
