"""Gate 1: how well can k-of-E expert selection approximate each real FFN, at best?

For every layer: fit partitions on calibration activations, then on held-out validation
activations measure the FFN-output error of oracle expert selection (and of controls), plus a
neuron-level oracle that bounds what unstructured dynamic sparsity (TEAL/CATS-style) could do.

source scripts/env.sh && .venv/bin/python scripts/oracle_analysis.py --layers all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from dense_to_moe.evaluation.metrics import approximation_metrics, expert_utilization
from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.partitioning.strategies import (
    balanced_kmeans_partition,
    contiguous_partition,
    random_partition,
)
from dense_to_moe.routing.oracle import ORACLES, indices_to_weights
from dense_to_moe.utils.checkpoint import load_ffn
from dense_to_moe.utils.env import environment

log = logging.getLogger(__name__)
CONFIGS = [(4, 2), (4, 3), (8, 2), (8, 4), (8, 6), (16, 4), (16, 8)]
NEURON_KEEP = [0.75, 0.5, 0.375, 0.25, 0.125]


@torch.no_grad()
def contributions(dense: DenseFFN, x: torch.Tensor) -> torch.Tensor:
    """|h_i| * ||W_down[:, i]||: each neuron's output-norm contribution per token. [T, I]"""
    return dense.intermediate(x).abs() * dense.down.norm(dim=0)


def build_partitions(dense: DenseFFN, x_cal: torch.Tensor, E: int, seed: int,
                     feat_tokens: int) -> dict:
    inter = dense.intermediate_size
    c = contributions(dense, x_cal[:feat_tokens])  # [T, I]
    # CMoE-style co-activation features: is neuron i in the token's top 10% by contribution?
    thresh = c.quantile(0.9, dim=1, keepdim=True)
    coact = (c >= thresh).float().T  # [I, T]
    contrib = (c / c.norm(dim=1, keepdim=True)).T  # [I, T] normalized per token
    return {
        "contiguous": contiguous_partition(inter, E),
        "random": random_partition(inter, E, seed=seed),
        "kmeans_gate_weights": balanced_kmeans_partition(
            torch.nn.functional.normalize(dense.gate.float(), dim=1), E, seed, iters=15,
            strategy="kmeans_gate_weights"),
        "kmeans_coactivation": balanced_kmeans_partition(coact, E, seed, iters=15,
                                                         strategy="kmeans_coactivation"),
        "kmeans_contribution": balanced_kmeans_partition(contrib, E, seed, iters=15,
                                                         strategy="kmeans_contribution"),
    }


@torch.no_grad()
def neuron_oracle(dense: DenseFFN, x: torch.Tensor, ref: torch.Tensor) -> dict:
    """Keep the top fraction of neurons per token by contribution (unstructured upper bound)."""
    h = dense.intermediate(x)
    score = h.abs() * dense.down.norm(dim=0)
    out = {}
    for keep in NEURON_KEEP:
        n = int(keep * h.shape[1])
        idx = score.topk(n, dim=1).indices
        mask = torch.zeros_like(h).scatter_(1, idx, 1.0)
        out[f"keep_{keep}"] = approximation_metrics(ref, (h * mask) @ dense.down.T)
    # Concentration: share of total |contribution| carried by the top x% neurons
    s = score.sort(dim=1, descending=True).values
    cum = s.cumsum(1) / s.sum(1, keepdim=True)
    out["contribution_share_top"] = {f"{q}": cum[:, int(q * h.shape[1]) - 1].mean().item()
                                     for q in (0.05, 0.125, 0.25, 0.5)}
    return out


@torch.no_grad()
def analyze_layer(layer: int, args, device) -> dict:
    t0 = time.time()
    w = load_ffn(layer, device)
    dense = DenseFFN(*(w[p].float() for p in ("gate", "up", "down")))
    x_cal = load_file(Path(args.acts) / "calib" / f"layer_{layer:02d}.safetensors")["x"]
    x_val = load_file(Path(args.acts) / "val" / f"layer_{layer:02d}.safetensors")["x"]
    x_cal = x_cal.to(device).float()
    x_val = x_val[: args.val_tokens].to(device).float()
    ref = dense(x_val)
    rec = {"layer": layer, "val_tokens": x_val.shape[0], "ref_rms": ref.pow(2).mean().sqrt().item(),
           "neuron_oracle": neuron_oracle(dense, x_val, ref), "experts": []}
    for E in sorted({E for E, _ in CONFIGS}):
        parts = build_partitions(dense, x_cal, E, args.seed, args.feat_tokens)
        for pname, part in parts.items():
            moe = PartitionedFFN.from_dense(dense, part)
            out = moe.expert_outputs(x_val)  # [E, T, d] fp32
            full = approximation_metrics(ref, out.sum(0))
            assert full["rel_err"] < 1e-4, (layer, pname, full)  # reconstruction invariant
            for k in [k for (e, k) in CONFIGS if e == E]:
                for oname, fn in ORACLES.items():
                    if oname == "exhaustive" and math.comb(E, k) > 20000:
                        continue
                    idx = fn(out, k)
                    y = torch.einsum("etd,te->td", out, indices_to_weights(idx, E))
                    r = {"E": E, "k": k, "partition": pname, "selection": oname, "combine": "sum",
                         **approximation_metrics(ref, y)}
                    if oname in ("exhaustive", "greedy"):
                        r["utilization"] = expert_utilization(idx, E)
                    rec["experts"].append(r)
                    if oname == "norm":  # what sum-to-one gating does to the same selection
                        scores = out.norm(dim=-1).T.clamp_min(1e-12).log()
                        wsm = indices_to_weights(idx, E, scores=scores, mode="softmax_renorm")
                        y = torch.einsum("etd,te->td", out, wsm)
                        rec["experts"].append({"E": E, "k": k, "partition": pname,
                                               "selection": "norm", "combine": "softmax_renorm",
                                               **approximation_metrics(ref, y)})
            del out, moe
        torch.cuda.empty_cache()
    rec["seconds"] = time.time() - t0
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--acts", default="cache/activations")
    ap.add_argument("--val-tokens", type=int, default=4096)
    ap.add_argument("--feat-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/oracle_analysis.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    layers = range(32) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    result = {"env": environment(), "args": vars(args), "configs": CONFIGS, "layers": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        rec = analyze_layer(layer, args, device)
        result["layers"].append(rec)
        best = {}
        for r in rec["experts"]:
            if r["combine"] == "sum" and r["selection"] in ("exhaustive", "greedy"):
                key = (r["E"], r["k"])
                best[key] = min(best.get(key, 9), r["rel_err"])
        log.info("layer %2d (%.0fs) best oracle rel_err: %s | neuron keep 25%%: %.3f", layer,
                 rec["seconds"], {f"{e}/{k}": round(v, 3) for (e, k), v in sorted(best.items())},
                 rec["neuron_oracle"]["keep_0.25"]["rel_err"])
        out.write_text(json.dumps(result, indent=1))  # incremental, survives interruption
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
