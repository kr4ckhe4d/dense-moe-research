"""Would a vocabulary shortlist for the MTP drafter keep its drafts?

Each drafted token reads the whole output projection (248,320 rows, 1.29 GiB at Q8_0 in the 27B
UD-Q6_K file), which is 80% of the 3.1 ms a draft token costs. Drafts are verified by the target,
so the drafter may score only a subset of the vocabulary without changing the output; the price is
the drafts whose correct token is outside the subset.

This ranks tokens by frequency in a local corpus (C++, Python, Markdown, JSON, earlier model
answers, Wikipedia, instructions, school maths) and reports, for shortlist sizes K, the share of tokens covered in held-out generations:
the per-step logs of benchmarks/lenient_probe.py (six domains, ratio 0), using the target's own
token at every draft position. Tokenization goes through a running llama-server (/tokenize).

    llama-server -m Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -c 4096 --port 8197 &
    python3 scripts/shortlist_coverage.py > experiments/results/shortlist/coverage.md
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

HOME = Path.home()
N_VOCAB = 248320
SOURCES = {
    "cpp": sorted((HOME / "llama.cpp-b12164-mmvq/src").glob("*.cpp")) + sorted((HOME / "llama.cpp-b12164-mmvq/common").glob("*.cpp")),
    "python": sorted(Path("/usr/lib").glob("python3.*/[a-z]*.py")),
    "markdown": sorted((HOME / "code").glob("*/docs/*.md")) + sorted((HOME / "code").glob("*/*.md")),
    "json": sorted((HOME / "llama.cpp-b12164-mmvq").glob("**/*.json"))[:200],
    "answers": sorted(Path("experiments/results/adopt/seeds").glob("*_s*.txt")),
}
MAX_CHARS = 60_000  # per file
# large plain-text files, read whole in chunks: Wikipedia (wikitext-2 train), instructions
# (dolly-15k) and school maths (gsm8k train), the last two exported from the HF datasets cache
# as plain text (all string fields of each row, rows separated by a blank line); not in the repo
BIG = {
    "wiki": HOME / "llama.cpp/kld/wikitext-2-raw/wiki.train.raw",
    "instructions": HOME / ".cache/dense-moe/shortlist_corpus/dolly.txt",
    "maths": HOME / ".cache/dense-moe/shortlist_corpus/gsm8k.txt",
}
KS = [4096, 8192, 16384, 24576, 32768, 49152, 65536]


def tokenize(port: int, text: str) -> list[int]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", json.dumps({"content": text}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--probe", nargs="+", default=["experiments/results/lenient/probe.jsonl",
                                                   "experiments/results/lenient/probe_t1.jsonl"])
    ap.add_argument("--out-rank", default="experiments/results/shortlist/rank.json")
    args = ap.parse_args()

    random.seed(0)
    freq: Counter[int] = Counter()
    n_src = {}
    for name, files in SOURCES.items():
        files = list(files)
        random.shuffle(files)
        n = 0
        for f in files[:300]:
            try:
                text = f.read_text(errors="ignore")[:MAX_CHARS]
            except OSError:
                continue
            if not text.strip():
                continue
            toks = tokenize(args.port, text)
            freq.update(toks)
            n += len(toks)
        n_src[name] = n
    for name, f in BIG.items():
        if not f.exists():
            continue
        text = f.read_text(errors="ignore")
        n = 0
        for i in range(0, len(text), MAX_CHARS):
            toks = tokenize(args.port, text[i:i + MAX_CHARS])
            freq.update(toks)
            n += len(toks)
        n_src[name] = n
    rank = [t for t, _ in freq.most_common()]
    pos = {t: i for i, t in enumerate(rank)}
    Path(args.out_rank).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rank).write_text(json.dumps({"n_tokens": n_src, "rank": rank}))

    print("# Vocabulary shortlist for the MTP drafter: coverage\n")
    print("Corpus tokens used for the frequency ranking: " + ", ".join(f"{k} {v:,}" for k, v in n_src.items())
          + f". Distinct tokens seen: {len(rank):,} of {N_VOCAB:,}.\n")

    # held-out: the target's token at each draft position, per domain
    rows = []
    for p in args.probe:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r["ratio"] == 0:
                rows.append((Path(p).stem, r))
    doms = sorted({r["domain"] for _, r in rows})
    for label, pick in (("target's token at draft positions (what a draft must match)", "top"),
                        ("the drafter's own picks", "draft")):
        print(f"## Coverage of {label}\n")
        print("| shortlist K | share of vocab | " + " | ".join(doms) + " | all |")
        print("|---:|---:|" + "---:|" * (len(doms) + 1))
        for k in KS + [len(rank)]:
            cells = []
            tot_in = tot = 0
            for d in doms:
                toks = [t for _, r in rows if r["domain"] == d for s in r["steps"] for t in s[pick]]
                inside = sum(pos.get(t, N_VOCAB) < k for t in toks)
                tot_in += inside
                tot += len(toks)
                cells.append(f"{100 * inside / len(toks):.2f}%")
            print(f"| {k:,} | {100 * k / N_VOCAB:.1f}% | " + " | ".join(cells) + f" | {100 * tot_in / tot:.2f}% |")
        print()

    # effect on accepted tokens: a step keeps drafts up to the first one that differs from the
    # target or falls outside the shortlist
    print("## Accepted draft tokens kept (drafts cut at the first token outside the shortlist)\n")
    print("| shortlist K | accepted now | accepted with shortlist | kept |")
    print("|---:|---:|---:|---:|")
    for k in KS + [len(rank)]:
        now = kept = 0
        for _, r in rows:
            for s in r["steps"]:
                n_acc = 0
                for a, b in zip(s["draft"], s["top"]):
                    if a != b:
                        break
                    n_acc += 1
                now += n_acc
                n_k = 0
                for a in s["draft"][:n_acc]:
                    if pos.get(a, N_VOCAB) >= k:
                        break
                    n_k += 1
                kept += n_k
        print(f"| {k:,} | {now} | {kept} | {100 * kept / now:.2f}% |")


if __name__ == "__main__":
    main()
