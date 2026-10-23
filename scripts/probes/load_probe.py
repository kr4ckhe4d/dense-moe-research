"""Load the full Qwen3.5-9B in BF16 on the R9700 (cuda:1), generate 40 tokens greedily, and time one 2048-token forward pass.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/load_probe.py
"""
import time, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
p="Qwen/Qwen3.5-9B"; rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
t=time.time()
tok=AutoTokenizer.from_pretrained(p, revision=rev)
try:
    m=AutoModelForCausalLM.from_pretrained(p, revision=rev, dtype=torch.bfloat16, device_map="cuda:1")
except Exception as e:
    print("AutoModelForCausalLM failed:", type(e).__name__, str(e)[:300]); raise
print("loaded", type(m).__name__, f"{time.time()-t:.0f}s", f"mem {torch.cuda.memory_allocated(1)/2**30:.1f} GiB")
print([n for n,_ in m.named_modules() if n.endswith("layers.0.mlp") or n.endswith("layers.3.self_attn") or n.endswith("layers.0.post_attention_layernorm")])
ids=tok("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", return_tensors="pt").to("cuda:1")
t=time.time()
with torch.no_grad(): out=m.generate(**ids, max_new_tokens=40, do_sample=False)
print(f"gen {time.time()-t:.1f}s:", repr(tok.decode(out[0][ids['input_ids'].shape[1]:])))
x=torch.randint(0,200000,(1,2048),device="cuda:1")
torch.cuda.synchronize(1); t=time.time()
with torch.no_grad(): m(x)
torch.cuda.synchronize(1); print(f"forward 2048 tok: {time.time()-t:.2f}s, peak {torch.cuda.max_memory_allocated(1)/2**30:.1f} GiB")
