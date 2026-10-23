"""Phase 3: upper bounds on decode speed from FFN sparsity, from measured llama.cpp numbers.

Model: single-token decode time is linear in weight bytes read, t = t_base - s * (bytes skipped).
The slope s (seconds per FFN byte) is fitted on the Phase 2 FFN proxies: IQ4_XS with its FFN
re-quantized to Q2_K and IQ1_S, so only FFN bytes change (bench_ffn_proxy_9070xt.md). Any sparse
scheme that reads a fraction f of the FFN bytes (and costs nothing else) is then bounded by
1 / (t_q - s * (1 - f) * ffn_bytes_q) for a base quant q with measured time t_q.

The proxy fit is pessimistic if low-bit kernels spend more ALU per byte (they do: IQ1_S/Q2_K
dequantization is heavier), so a second bound is fitted across the six quants instead:
t = c' + s' * (all weight bytes read), which gives ~610 GB/s for s' (near the 644 GB/s peak) plus a
fixed ~3.5 ms per token. Skipped FFN bytes then save s' each.

Both are ceilings: a real sparse kernel has gather/irregular-access overhead, a predictor costs
bytes and time, and quantized blocks constrain which weights can be skipped.

MTP: each verify step evaluates ~w tokens (1 + drafts per step, from mtp_gpu0.txt) in one batch,
so it reads the union of their masks. With per-step FFN fraction f_w the step time ratio is
estimated the same way and applied to the measured MTP tok/s.

source scripts/env.sh && PYTHONPATH=~/llama.cpp/gguf-py .venv/bin/python scripts/phase3_speed_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader

MODELS = Path("/mnt/fast/models")
RES = Path("experiments/results/gguf")
FFN = re.compile(r"blk\.([0-9]|[12][0-9]|3[01])\.ffn_(gate|up|down)\.weight")


def tensor_bytes(path: Path) -> dict[str, int]:
    r = GGUFReader(path)
    ffn = sum(int(t.n_bytes) for t in r.tensors if FFN.fullmatch(t.name))
    # Decode reads every weight except the embedding table (one row) and the MTP block.
    read = sum(int(t.n_bytes) for t in r.tensors
               if t.name != "token_embd.weight" and not t.name.startswith("blk.32."))
    return {"ffn": ffn, "read": read}


def bench_tg(path: Path) -> dict[str, float]:
    """tg128 tok/s per row label, in file order (labels repeat for the proxy file)."""
    out = []
    for line in path.read_text().splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 8 and cols[7] == "tg128":
            out.append((cols[1], float(cols[8].split("±")[0])))
    return out


def mtp(path: Path) -> dict[str, dict]:
    res, cur = {}, None
    for line in path.read_text().splitlines():
        if m := re.match(r"== (\S+) spec=(\S+)", line):
            cur = res.setdefault(m[1], {}).setdefault(m[2], {"runs": []})
        elif m := re.match(r"run\d.*tg\s+([\d.]+) tok/s,\s+(\d+) tok(?: \| draft (\d+)/(\d+))?", line):
            cur["runs"].append([float(m[1]), int(m[2])] + ([int(m[3]), int(m[4])] if m[3] else []))
    for d in res.values():
        for spec, v in d.items():
            v["tps"] = float(np.mean([r[0] for r in v["runs"]]))
            if spec == "draft-mtp":
                tok, acc, drafted = (np.mean([r[i] for r in v["runs"]]) for i in (1, 2, 3))
                steps = tok - acc
                v["verify_width"] = float(1 + drafted / steps)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="experiments/results/phase3/e2e_kld_val.json",
                    help="mask statistics for the candidates (optional)")
    ap.add_argument("--out", default="experiments/results/phase3/speed_ceiling.json")
    args = ap.parse_args()

    proxy_files = ["IQ4_XS", "IQ4_XS-ffn_q2_k", "IQ4_XS-ffn_iq1_s"]
    proxy_tg = [v for _, v in bench_tg(RES / "bench_ffn_proxy_9070xt.md")]
    pb = [tensor_bytes(MODELS / f"Qwen3.5-9B-{f}.gguf") for f in proxy_files]
    x = np.array([b["ffn"] for b in pb], float)
    t = 1 / np.array(proxy_tg)
    s, c = np.polyfit(x, t, 1)
    fit_resid = (t - (s * x + c)) / t

    quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M"]
    labels = {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q5_K_M": "Q5_K - Medium", "Q4_K_M": "Q4_K - Medium",
              "IQ4_XS": "IQ4_XS - 4.25 bpw", "Q3_K_M": "Q3_K - Medium"}
    tg = {gpu: {lab.replace("qwen35 9B ", ""): v for lab, v in bench_tg(RES / f)}
          for gpu, f in (("9070xt", "bench_9070xt.md"), ("r9700", "bench_r9700.md"))}
    m = mtp(RES / "mtp_gpu0.txt")
    qb = {q: tensor_bytes(MODELS / f"Qwen3.5-9B-{q}.gguf") for q in quants}

    def tps(q: str, f: float, base: float) -> float:
        return 1 / (1 / base - s * (1 - f) * qb[q]["ffn"])

    xq = np.array([qb[q]["read"] for q in quants], float)
    tq = 1 / np.array([tg["9070xt"][labels[q]] for q in quants])
    s_bw, c_bw = np.polyfit(xq, tq, 1)
    bw_resid = (tq - (s_bw * xq + c_bw)) / tq

    def tps_bw(q: str, f: float, base: float) -> float:
        return 1 / (1 / base - s_bw * (1 - f) * qb[q]["ffn"])

    table = {}
    for q in quants:
        base = tg["9070xt"][labels[q]]
        row = {"ffn_GiB": qb[q]["ffn"] / 2**30, "read_GiB": qb[q]["read"] / 2**30,
               "ffn_share_of_read": qb[q]["ffn"] / qb[q]["read"],
               "tg128": base, "ffn_time_share": s * qb[q]["ffn"] * base,
               "ceiling_f": {f"{f:.3f}": tps(q, f, base) for f in (1.0, 0.833, 0.75, 0.667, 0.5,
                                                                    0.375, 0.25, 0.0)},
               "ceiling_bw_f": {f"{f:.3f}": tps_bw(q, f, base) for f in (0.833, 0.667, 0.5, 0.0)}}
        if q in m:
            row["mtp_tps"] = m[q]["draft-mtp"]["tps"]
            row["verify_width"] = m[q]["draft-mtp"]["verify_width"]
        table[q] = row

    cands = {}
    e2e = Path(args.e2e)
    if e2e.exists():
        for name, r in json.loads(e2e.read_text())["results"].items():
            if "bytes" not in r:
                continue
            c_ = {"mean_kld": r["mean_kld"], "bytes": r["bytes"]}
            if "bytes_union4" in r:
                c_["bytes_union4"] = r["bytes_union4"]
            for q in ("Q8_0", "Q6_K", "IQ4_XS"):
                base = table[q]["tg128"]
                c_[f"{q}_tps"] = tps(q, r["bytes"], base)
                c_[f"{q}_tps_bw"] = tps_bw(q, r["bytes"], base)
                if "bytes_union4" in r:
                    for key, fn in (("", tps), ("_bw", tps_bw)):
                        ratio = fn(q, r["bytes_union4"], base) / base
                        c_[f"{q}_mtp_tps{key}"] = table[q]["mtp_tps"] * ratio
            cands[name] = c_

    # R9700: no FFN proxy or MTP runs there, so only the cross-quant fit.
    tr = 1 / np.array([tg["r9700"][labels[q]] for q in quants])
    s_r, c_r = np.polyfit(xq, tr, 1)
    r9700 = {"slope_s_per_GiB": s_r * 2**30, "intercept_s": c_r,
             "rel_resid": dict(zip(quants, ((tr - (s_r * xq + c_r)) / tr).tolist())), "quants": {}}
    for q in quants:
        base = tg["r9700"][labels[q]]
        r9700["quants"][q] = {"tg128": base, "ceiling_bw_f": {
            f"{f:.3f}": 1 / (1 / base - s_r * (1 - f) * qb[q]["ffn"]) for f in (0.833, 0.667, 0.5, 0.0)}}

    out = {"r9700": r9700, "slope_s_per_GiB": s * 2**30, "intercept_s": c, "proxy_fit_rel_resid": fit_resid.tolist(),
           "quant_fit_slope_s_per_GiB": s_bw * 2**30, "quant_fit_intercept_s": c_bw,
           "quant_fit_rel_resid": dict(zip(quants, bw_resid.tolist())),
           "proxy_tg128": dict(zip(proxy_files, proxy_tg)), "quants": table, "candidates": cands}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"fit: {s * 2**30 * 1e3:.3f} ms per GiB of FFN weights, intercept {c * 1e3:.2f} ms; "
          f"proxy residuals {np.round(fit_resid * 100, 2).tolist()} %")
    print(f"cross-quant fit: {s_bw * 2**30 * 1e3:.3f} ms per GiB read ({1 / s_bw / 1e9:.0f} GB/s), "
          f"intercept {c_bw * 1e3:.2f} ms; residuals {np.round(bw_resid * 100, 1).tolist()} %")
    print("Ceiling tok/s (9070 XT) when a fraction f of FFN bytes is read; proxy fit / cross-quant fit")
    print("| quant | FFN GiB | FFN share: bytes / time (proxy) | tg128 | f=0.833 | f=0.667 | f=0.5 | f=0 "
          "| MTP tok/s | verify width |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for q, r in table.items():
        cf, cb = r["ceiling_f"], r["ceiling_bw_f"]
        cells = " | ".join(f"{cf[f]:.1f} / {cb[f]:.1f}" for f in ("0.833", "0.667", "0.500", "0.000"))
        print(f"| {q} | {r['ffn_GiB']:.2f} | {r['ffn_share_of_read']:.0%} / {r['ffn_time_share']:.0%} | "
              f"{r['tg128']:.1f} | {cells} | "
              f"{r.get('mtp_tps', float('nan')):.1f} | {r.get('verify_width', float('nan')):.2f} |")
    print(f"\nR9700 cross-quant fit: {s_r * 2**30 * 1e3:.3f} ms per GiB ({1 / s_r / 1e9:.0f} GB/s), "
          f"intercept {c_r * 1e3:.2f} ms")
    print("| quant | tg128 | f=0.833 | f=0.667 | f=0.5 | f=0 |")
    print("|---|---|---|---|---|---|")
    for q, r in r9700["quants"].items():
        cb = r["ceiling_bw_f"]
        print(f"| {q} | {r['tg128']:.1f} | " + " | ".join(f"{cb[f]:.1f}" for f in cb) + " |")
    if cands:
        print("\nCandidates (ceilings, proxy fit / cross-quant fit; MTP uses the 4-token union)")
        print("| candidate | KLD | FFN bytes (1 tok / 4-tok union) | IQ4_XS | IQ4_XS+MTP | Q6_K | Q8_0 "
              "| Q8_0+MTP |")
        print("|---|---|---|---|---|---|---|---|")
        nan = float("nan")
        for k, c_ in cands.items():
            def two(key, c_=c_):
                bw = key + "_bw" if "mtp" in key else key.replace("tps", "tps_bw")
                return f"{c_.get(key, nan):.1f} / {c_.get(bw, nan):.1f}"
            print(f"| {k} | {c_['mean_kld']:.4f} | {c_['bytes']:.3f} / {c_.get('bytes_union4', nan):.3f} | "
                  f"{two('IQ4_XS_tps')} | {two('IQ4_XS_mtp_tps')} | {two('Q6_K_tps')} | "
                  f"{two('Q8_0_tps')} | {two('Q8_0_mtp_tps')} |")


if __name__ == "__main__":
    main()
