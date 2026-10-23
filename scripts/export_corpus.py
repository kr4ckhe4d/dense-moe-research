"""Write calibration / validation text as plain files for llama.cpp (imatrix, KLD).

The text comes from the same hash-disjoint splits as the PyTorch experiments, so an imatrix
built from `calib` never sees the `val` text it is scored on.

source scripts/env.sh && .venv/bin/python scripts/export_corpus.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from dense_to_moe.calibration.data import build_sequences, load_documents

log = logging.getLogger(__name__)
REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="cache/corpus")
    ap.add_argument("--seq-len", type=int, default=1024)
    # val: seed 7 like oracle_e2e.py, so its first 2 sequences per domain are the F3 eval set.
    ap.add_argument("--val-per-domain", type=int, default=9)
    ap.add_argument("--calib-per-domain", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    docs = load_documents(tok)
    meta = {}
    for split, per_domain, seed in [("val", args.val_per_domain, 7),
                                    ("calib", args.calib_per_domain, 0)]:
        seqs = build_sequences(tok, docs, split, args.seq_len, per_domain, seed)
        text = "".join(tok.decode(ids) for _, ids in seqs)
        path = out / f"{split}.txt"
        path.write_text(text)
        meta[split] = {"per_domain": per_domain, "seed": seed, "seq_len": args.seq_len,
                       "domains": dict(Counter(d for d, _ in seqs)),
                       "tokens": len(seqs) * args.seq_len, "bytes": path.stat().st_size}
        log.info("%s: %s", split, meta[split])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
