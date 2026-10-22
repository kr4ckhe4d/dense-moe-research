"""Fetch one layer's FFN weights by HTTP range request (no full download needed).

.venv/bin/python scripts/fetch_layer.py --layer 15
"""

import argparse
import logging

from dense_to_moe.utils.hf_weights import fetch_ffn_layer

REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    a = ap.parse_args()
    for name, t in fetch_ffn_layer(REPO, REV, a.layer).items():
        print(name, tuple(t.shape), t.dtype)
