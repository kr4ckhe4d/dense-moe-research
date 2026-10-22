"""Collect per-layer FFN inputs from Qwen3.5-9B on calibration data.

source scripts/env.sh && .venv/bin/python scripts/collect_activations.py --split calib
Writes cache/activations/<split>/layer_XX.safetensors + meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from dense_to_moe.calibration.collect import collect_ffn_inputs
from dense_to_moe.calibration.data import build_sequences, load_documents
from dense_to_moe.utils.env import environment

REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calib")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--tokens-per-seq", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/activations")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    docs = load_documents(tok)
    logging.info("documents per domain: %s", dict(Counter(d.domain for d in docs)))
    seqs = build_sequences(tok, docs, args.split, args.seq_len, args.per_domain, args.seed)
    logging.info("sequences per domain: %s", dict(Counter(d for d, _ in seqs)))

    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    acts, domains = collect_ffn_inputs(model, seqs, args.tokens_per_seq, args.seed)

    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(acts):
        save_file({"x": a.contiguous()}, out / f"layer_{i:02d}.safetensors")
    meta = {"model": REPO, "revision": REV, "args": vars(args), "domains": domains,
            "domain_counts": dict(Counter(domains)), "num_layers": len(acts),
            "rows_per_layer": acts[0].shape[0], "env": environment()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info("wrote %d layers x %s to %s", len(acts), tuple(acts[0].shape), out)


if __name__ == "__main__":
    main()
