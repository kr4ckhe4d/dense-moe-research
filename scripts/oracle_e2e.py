"""Gate 1, end to end: KL / perplexity of the full model when every FFN uses oracle routing.

source scripts/env.sh && .venv/bin/python scripts/oracle_e2e.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from dense_to_moe.calibration.data import build_sequences, load_documents
from dense_to_moe.evaluation.e2e import Mode, compare_to_dense, set_mode, wrap_model
from dense_to_moe.partitioning.strategies import (
    Partition,
    balanced_kmeans_partition,
    contiguous_partition,
)
from dense_to_moe.utils.env import environment

log = logging.getLogger(__name__)
REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


@torch.no_grad()
def fit_partition(w, x_cal: torch.Tensor, E: int, strategy: str, seed: int) -> Partition:
    if strategy == "contiguous":
        return contiguous_partition(w.mlp.down_proj.weight.shape[1], E)
    h = w.intermediate(x_cal)
    c = (h.abs() * w.mlp.down_proj.weight.norm(dim=0)).float()
    feats = (c / c.norm(dim=1, keepdim=True)).T
    return balanced_kmeans_partition(feats, E, seed, iters=15, strategy="kmeans_contribution")


@torch.no_grad()
def static_set(w, x_cal: torch.Tensor, E: int, k: int) -> torch.Tensor:
    out = w.expert_outputs(w.intermediate(x_cal), E)
    return out.float().pow(2).sum((1, 2)).topk(k).indices.cpu()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--strategy", default="kmeans_contribution")
    ap.add_argument("--cal-tokens", type=int, default=4096)
    ap.add_argument("--out", default="experiments/results/oracle_e2e.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    modes = [Mode("oracle", 8, 6), Mode("oracle", 8, 4), Mode("oracle", 8, 2),
             Mode("oracle", 4, 3), Mode("oracle", 16, 8), Mode("oracle", 16, 4),
             Mode("static", 8, 6), Mode("static", 8, 4),
             Mode("neuron", keep=0.75), Mode("neuron", keep=0.5), Mode("neuron", keep=0.25)]

    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    # Seed 7 + per-domain 2 => different sequences from the activation-collection val run (seed 1).
    seqs = build_sequences(tok, load_documents(tok), args.split, args.seq_len, args.per_domain,
                           args.seed)
    log.info("eval sequences: %s", dict(Counter(d for d, _ in seqs)))
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    wrappers = wrap_model(model)

    for i, w in enumerate(wrappers):
        x_cal = load_file(f"cache/activations/calib/layer_{i:02d}.safetensors")["x"]
        x_cal = x_cal[: args.cal_tokens].to(dev)
        for E in sorted({m.E for m in modes if m.E}):
            w.partitions[E] = fit_partition(w, x_cal, E, args.strategy, seed=0)
        for m in modes:
            if m.kind == "static":
                w.static_sets[(m.E, m.k)] = static_set(w, x_cal, m.E, m.k)
        log.info("layer %d partitions fitted", i)

    totals: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    per_domain: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with torch.no_grad():
        for n, (domain, ids) in enumerate(seqs):
            x = torch.tensor([ids], device=dev)
            set_mode(wrappers, Mode())
            dense_logits = model(x, use_cache=False).logits[0]
            for m in [Mode()] + modes:
                set_mode(wrappers, m)
                logits = dense_logits if m.kind == "dense" else model(x, use_cache=False).logits[0]
                r = compare_to_dense(dense_logits, logits, x[0])
                name = f"{m.kind}" + (f" E={m.E} k={m.k}" if m.E else "") + \
                       (f" keep={m.keep}" if m.kind == "neuron" else "")
                for kk, v in r.items():
                    totals[name][kk] += v
                    per_domain[name][domain][kk] += v
            log.info("seq %d/%d (%s) done", n + 1, len(seqs), domain)

    def finalize(t):
        return {"kl": t["kl_sum"] / t["n"], "ppl": float(torch.exp(torch.tensor(t["nll_sum"] / t["n"]))),
                "top1_agree": t["top1_agree_sum"] / t["n"], "tokens": t["n"]}

    result = {"env": environment(), "args": vars(args),
              "results": {k: finalize(v) for k, v in totals.items()},
              "per_domain": {k: {d: finalize(v) for d, v in dv.items()} for k, dv in per_domain.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for k, v in result["results"].items():
        log.info("%-28s KL %.4f  PPL %.3f  top1 %.3f", k, v["kl"], v["ppl"], v["top1_agree"])


if __name__ == "__main__":
    main()
