"""Score PyTorch sparse-FFN variants against the llama.cpp BF16 reference, exactly as
`llama-perplexity --kl-divergence` scores a quant: same tokens, same 512-token windows, same
255 scored positions per chunk, same compressed base log-probs, same KL formula.

That puts MoE/neuron-sparsity variants and GGUF quants on one table. `torch dense` against the
llama.cpp BF16 base gives the implementation noise floor (two BF16 engines, no approximation).

    benchmarks/gguf_kld.sh base val        # writes ~/llama.cpp/kld/qwen3.5-9B-bf16-val.kld
    source scripts/env.sh && .venv/bin/python scripts/sparse_kld_vs_gguf.py --corpus val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from dense_to_moe.evaluation.e2e import Mode, set_mode, wrap_model
from dense_to_moe.utils.env import environment

sys.path.insert(0, str(Path(__file__).parent))
from oracle_e2e import fit_partition  # noqa: E402

log = logging.getLogger(__name__)
REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


class KLDBase:
    """Reader for llama.cpp's --kl-divergence-base file (tools/perplexity/perplexity.cpp)."""

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            assert f.read(8) == b"_logits_", "not a llama.cpp KLD base file"
            self.n_ctx, self.n_vocab, self.n_chunk = np.frombuffer(f.read(12), "<i4").tolist()
        off = 20
        self.tokens = np.memmap(path, "<i4", "r", off, (self.n_chunk * self.n_ctx,))
        off += self.tokens.nbytes
        self.first = self.n_ctx // 2
        self.n_scored = self.n_ctx - 1 - self.first
        self.nv = 2 * ((self.n_vocab + 1) // 2) + 4
        self.lp = np.memmap(path, "<u2", "r", off, (self.n_chunk, self.n_scored, self.nv))

    def chunk_tokens(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[i * self.n_ctx:(i + 1) * self.n_ctx])

    def chunk_logprobs(self, i: int, dev: torch.device) -> torch.Tensor:
        """Base log-probs [n_scored, n_vocab], decoded as llama.cpp does."""
        raw = np.asarray(self.lp[i])
        hdr = raw[:, :4].copy().view("<f4")  # [n_scored, 2]: scale, min_log_prob
        q = torch.from_numpy(raw[:, 4:4 + self.n_vocab].astype(np.float32)).to(dev)
        scale, min_lp = (torch.from_numpy(hdr[:, j].copy()).to(dev).unsqueeze(1) for j in (0, 1))
        return q * scale + min_lp


@torch.no_grad()
def score(base_lp: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    lp = F.log_softmax(logits.float(), -1)
    mask = base_lp > -16.0  # llama.cpp drops base terms below -16
    kl = torch.where(mask, base_lp.exp() * (base_lp - lp), 0.0).sum(-1)
    nll = -lp.gather(1, targets.view(-1, 1)).squeeze(1)
    return {"n": kl.numel(), "kl_sum": kl.sum().item(), "kl2_sum": kl.pow(2).sum().item(),
            "nll_sum": nll.sum().item(),
            "same_top": (lp.argmax(-1) == base_lp.argmax(-1)).sum().item(),
            "kl": kl.cpu()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--corpus", default="val")
    ap.add_argument("--chunks", type=int, default=0, help="0 = all in the base file")
    ap.add_argument("--cal-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    base = KLDBase(Path.home() / f"llama.cpp/kld/qwen3.5-9B-bf16-{args.corpus}.kld")
    n_chunk = args.chunks or base.n_chunk

    modes = [Mode(), Mode("neuron", keep=0.75), Mode("neuron", keep=0.5),
             Mode("neuron", keep=0.25), Mode("oracle", 4, 3), Mode("oracle", 8, 6),
             Mode("oracle", 16, 8), Mode("oracle", 8, 4)]

    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    assert model.get_output_embeddings().weight.shape[0] == base.n_vocab
    wrappers = wrap_model(model)
    for i, w in enumerate(wrappers):
        x_cal = load_file(f"cache/activations/calib/layer_{i:02d}.safetensors")["x"]
        x_cal = x_cal[: args.cal_tokens].to(dev)
        for E in sorted({m.E for m in modes if m.E}):
            w.partitions[E] = fit_partition(w, x_cal, E, "kmeans_contribution", seed=0)
    log.info("partitions fitted")

    def name(m: Mode) -> str:
        return {"dense": "torch dense (bf16)", "neuron": f"neuron keep={m.keep}",
                "oracle": f"oracle E={m.E} k={m.k}"}[m.kind]

    totals = {name(m): {"n": 0, "kl_sum": 0.0, "kl2_sum": 0.0, "nll_sum": 0.0, "same_top": 0,
                        "kl": []} for m in modes}
    with torch.no_grad():
        for c in range(n_chunk):
            toks = torch.from_numpy(base.chunk_tokens(c).astype(np.int64)).to(dev)
            base_lp = base.chunk_logprobs(c, dev)
            targets = toks[base.first + 1:]
            for m in modes:
                set_mode(wrappers, m)
                logits = model(toks.unsqueeze(0), use_cache=False).logits[0]
                r = score(base_lp, logits[base.first:base.n_ctx - 1], targets)
                t = totals[name(m)]
                for k in ("n", "kl_sum", "kl2_sum", "nll_sum", "same_top"):
                    t[k] += r[k]
                t["kl"].append(r["kl"])
            if (c + 1) % 10 == 0:
                log.info("chunk %d/%d", c + 1, n_chunk)

    results = {}
    for k, t in totals.items():
        kl = torch.cat(t["kl"])
        mean = t["kl_sum"] / t["n"]
        sd = (t["kl2_sum"] / t["n"] - mean**2) ** 0.5
        results[k] = {"mean_kld": mean, "kld_stderr": sd / (t["n"] - 1) ** 0.5,
                      "kld_p99": kl.quantile(0.99).item(), "kld_median": kl.median().item(),
                      "ppl": float(np.exp(t["nll_sum"] / t["n"])),
                      "same_top": t["same_top"] / t["n"], "tokens": t["n"]}
        log.info("%-22s KLD %.5f ± %.5f  p99 %.4f  PPL %.4f  same-top %.4f", k, mean,
                 results[k]["kld_stderr"], results[k]["kld_p99"], results[k]["ppl"],
                 results[k]["same_top"])
    out = Path(args.out or f"experiments/results/gguf/sparse_kld_{args.corpus}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": environment(), "args": vars(args), "n_chunk": n_chunk,
                               "results": results}, indent=1))


if __name__ == "__main__":
    main()
