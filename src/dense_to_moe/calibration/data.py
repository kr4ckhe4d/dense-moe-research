"""Calibration / train / validation / test text, from permissively licensed sources.

Every document is assigned to exactly one split by a hash of its content, so splits are
disjoint and stable across runs and machines. Sources deliberately exclude the benchmarks we
plan to evaluate on (HumanEval, MBPP, MMLU, GSM8K *test*, IFEval, ...).

| domain      | source                                      | license       |
|-------------|---------------------------------------------|---------------|
| prose       | Salesforce/wikitext wikitext-103-raw-v1 train | CC BY-SA 3.0 |
| math        | openai/gsm8k main *train* split             | MIT           |
| instruction | databricks/databricks-dolly-15k (chat-formatted) | CC BY-SA 3.0 |
| code_cpp    | local ~/llama.cpp C/C++ sources            | MIT           |
| code_python | local CPython stdlib (uv-managed 3.12)     | PSF-2.0       |
| structured  | JSON rendered from dolly records            | CC BY-SA 3.0  |
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

SPLITS = {"calib": (0, 60), "train": (60, 80), "val": (80, 90), "test": (90, 100)}


def split_of(text: str) -> str:
    bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 100
    for name, (lo, hi) in SPLITS.items():
        if lo <= bucket < hi:
            return name
    raise AssertionError


@dataclass
class Doc:
    domain: str
    text: str


def _wikitext() -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:20000]")
    # Re-join lines into articles, split at top-level " = Title = " headings.
    docs, cur = [], []
    for line in ds["text"]:
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            docs.append("".join(cur).strip())
            cur = []
        cur.append(line)
    return [Doc("prose", d) for d in docs if len(d) > 2000]


def _gsm8k() -> list[Doc]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [Doc("math", f"Question: {r['question']}\nAnswer: {r['answer']}") for r in ds]


def _dolly() -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset("databricks/databricks-dolly-15k", split="train"))


def _chat(tokenizer, rows: list[dict]) -> list[Doc]:
    out = []
    for r in rows:
        user = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": r["response"]}]
        out.append(Doc("instruction", tokenizer.apply_chat_template(msgs, tokenize=False)))
    return out


def _structured(rows: list[dict]) -> list[Doc]:
    return [Doc("structured", json.dumps({"category": r["category"], "instruction": r["instruction"],
                                          "response": r["response"]}, indent=2)) for r in rows]


def _local_code(pattern_root: Path, patterns: list[str], domain: str) -> list[Doc]:
    docs = []
    for pat in patterns:
        for f in sorted(glob.glob(str(pattern_root / pat), recursive=True)):
            try:
                t = Path(f).read_text(errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            if 1500 < len(t) < 200_000:
                docs.append(Doc(domain, t))
    return docs


def load_documents(tokenizer) -> list[Doc]:
    dolly = _dolly()
    stdlib = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    docs = (_wikitext() + _gsm8k() + _chat(tokenizer, dolly) + _structured(dolly)
            + _local_code(Path.home() / "llama.cpp", ["src/**/*.cpp", "ggml/src/**/*.c",
                                                      "common/**/*.cpp"], "code_cpp")
            + _local_code(stdlib, ["*.py", "json/*.py", "email/*.py", "asyncio/*.py"],
                          "code_python"))
    return docs


def build_sequences(tokenizer, docs: list[Doc], split: str, seq_len: int, per_domain: int,
                    seed: int) -> list[tuple[str, list[int]]]:
    """Pack documents of each domain into `per_domain` sequences of exactly `seq_len` tokens."""
    rng = random.Random(seed)
    by_domain: dict[str, list[Doc]] = {}
    for d in docs:
        if split_of(d.text) == split:
            by_domain.setdefault(d.domain, []).append(d)
    seqs = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        rng.shuffle(pool)
        buf: list[int] = []
        made = 0
        for d in pool:
            buf += tokenizer(d.text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            while len(buf) >= seq_len and made < per_domain:
                seqs.append((domain, buf[:seq_len]))
                buf = buf[seq_len:]
                made += 1
            if made >= per_domain:
                break
    return seqs
