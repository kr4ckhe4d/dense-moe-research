"""Phase 3: full-model KL divergence of block-sparse FFN variants against llama.cpp BF16.

Same protocol and base file as scripts/sparse_kld_vs_gguf.py (Phase 2), so the numbers sit in one
table with the GGUF quants. Every FFN of all 32 layers is sparsified. Per-layer `gate` statistics
come from calib activations only. For every sparse mode the per-layer kept fraction and the union
over 2 / 4 consecutive tokens (an MTP verify batch reads the union) are recorded.

The Phase 1 neuron modes are re-run to reproduce their numbers (0.00133 / 0.01030 on val).

source scripts/env.sh && .venv/bin/python scripts/phase3_e2e_kld.py --corpus val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from dense_to_moe.evaluation.e2e import Mode, set_mode, wrap_model
from dense_to_moe.sparsity.blocks import SparseSpec, ffn_bytes_fraction
from dense_to_moe.utils.env import environment

sys.path.insert(0, str(Path(__file__).parent))
from sparse_kld_vs_gguf import REPO, REV, KLDBase, score

log = logging.getLogger(__name__)

PRESETS = {
    "milestone1": [
        Mode(),
        Mode("neuron", keep=0.75),
        Mode("neuron", keep=0.5),
        Mode("sparse", spec=SparseSpec("oracle", 0.5, 1, "contrib")),
        Mode("sparse", spec=SparseSpec("oracle", 0.5, 16)),
        Mode("sparse", spec=SparseSpec("oracle", 0.5, 64)),
        Mode("sparse", spec=SparseSpec("oracle", 0.75, 16)),
        Mode("sparse", spec=SparseSpec("gate", 0.5, 1)),
        Mode("sparse", spec=SparseSpec("gate", 0.5, 16)),
        Mode("sparse", spec=SparseSpec("teal", 0.5, 1, x_keep=0.5)),
        Mode("sparse", spec=SparseSpec("wina", 0.5, 1, x_keep=0.5)),
        Mode("sparse", spec=SparseSpec("wina", 0.75, 1, x_keep=0.75)),
        Mode("sparse", spec=SparseSpec("oracle", 0.5, 8)),
        Mode("sparse", spec=SparseSpec("teal", 0.75, 1, x_keep=0.75)),
        Mode("sparse", spec=SparseSpec("wina", 0.625, 1, x_keep=0.625)),
        Mode("sparse", spec=SparseSpec("wina", 0.9, 1, x_keep=0.9)),
        Mode("sparse", spec=SparseSpec("wina", 1.0, 1, x_keep=0.5)),
        Mode("sparse", spec=SparseSpec("wina", 0.75, 32, x_keep=0.75, x_block=32)),
    ],
}


def name(m: Mode) -> str:
    if m.kind == "sparse":
        return m.spec.name
    return {"dense": "torch dense (bf16)", "neuron": f"neuron keep={m.keep}"}[m.kind]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--corpus", default="val")
    ap.add_argument("--chunks", type=int, default=0, help="0 = all in the base file")
    ap.add_argument("--preset", default="milestone1", choices=list(PRESETS))
    ap.add_argument("--cal-tokens", type=int, default=4096)
    ap.add_argument("--stat-chunks", type=int, default=20, help="chunks with mask statistics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    base = KLDBase(Path.home() / f"llama.cpp/kld/qwen3.5-9B-bf16-{args.corpus}.kld")
    n_chunk = args.chunks or base.n_chunk
    modes = PRESETS[args.preset]

    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    assert model.get_output_embeddings().weight.shape[0] == base.n_vocab
    wrappers = wrap_model(model)
    for i, w in enumerate(wrappers):
        x_cal = load_file(f"cache/activations/calib/layer_{i:02d}.safetensors")["x"]
        w.sparse.calibrate(x_cal[: args.cal_tokens].to(dev))
    log.info("calibrated")

    totals = {name(m): {"n": 0, "kl_sum": 0.0, "kl2_sum": 0.0, "nll_sum": 0.0, "same_top": 0,
                        "kl": []} for m in modes}
    with torch.no_grad():
        for c in range(n_chunk):
            toks = torch.from_numpy(base.chunk_tokens(c).astype(np.int64)).to(dev)
            base_lp = base.chunk_logprobs(c, dev)
            targets = toks[base.first + 1:]
            for w in wrappers:
                w.record = c < args.stat_chunks
            for m in modes:
                set_mode(wrappers, m)
                logits = model(toks.unsqueeze(0), use_cache=False).logits[0]
                r = score(base_lp, logits[base.first:base.n_ctx - 1], targets)
                t = totals[name(m)]
                for k in ("n", "kl_sum", "kl2_sum", "nll_sum", "same_top"):
                    t[k] += r[k]
                t["kl"].append(r["kl"])
                if m.kind == "sparse" and c < args.stat_chunks:
                    t.setdefault("layer_stats", [[] for _ in wrappers])
                    for i, w in enumerate(wrappers):
                        t["layer_stats"][i] += w.mask_stats
                        w.mask_stats = []
            log.info("chunk %d/%d", c + 1, n_chunk)

    results = {}
    by_name = {name(m): m for m in modes}
    for k, t in totals.items():
        kl = torch.cat(t["kl"])
        mean = t["kl_sum"] / t["n"]
        sd = (t["kl2_sum"] / t["n"] - mean**2) ** 0.5
        r = {"mean_kld": mean, "kld_stderr": sd / (t["n"] - 1) ** 0.5,
             "kld_p99": kl.quantile(0.99).item(), "kld_median": kl.median().item(),
             "ppl": float(np.exp(t["nll_sum"] / t["n"])),
             "same_top": t["same_top"] / t["n"], "tokens": t["n"]}
        if "layer_stats" in t:
            spec = by_name[k].spec
            per_layer = [{s: float(np.mean([d[s] for d in ls])) for s in ls[0]}
                         for ls in t["layer_stats"]]
            r["per_layer"] = per_layer
            for key, suf in (("kept", ""), ("union2", "_union2"), ("union4", "_union4")):
                h = float(np.mean([p[key] for p in per_layer]))
                xk = float(np.mean([p[f"x_{key}"] for p in per_layer])) if "x_kept" in per_layer[0] else 1.0
                r[f"kept{suf}"], r[f"x_kept{suf}"] = h, xk
                r[f"bytes{suf}"] = ffn_bytes_fraction(spec, h, xk)
        results[k] = r
        log.info("%-40s KLD %.5f ± %.5f  p99 %.4f  PPL %.4f  same-top %.4f  bytes %s / union4 %s", k,
                 mean, r["kld_stderr"], r["kld_p99"], r["ppl"], r["same_top"],
                 f"{r['bytes']:.3f}" if "bytes" in r else "-",
                 f"{r['bytes_union4']:.3f}" if "bytes" in r else "-")
    out = Path(args.out or f"experiments/results/phase3/e2e_kld_{args.corpus}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": environment(), "args": vars(args), "n_chunk": n_chunk,
                               "results": results}, indent=1))


if __name__ == "__main__":
    main()
