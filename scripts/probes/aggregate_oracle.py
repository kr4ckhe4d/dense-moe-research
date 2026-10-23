"""Summarize experiments/results/oracle_analysis.json: mean best-oracle error per partition and config, selection comparison, neuron oracle, per-layer errors, expert load balance.

Run from the repo root: source scripts/env.sh && .venv/bin/python scripts/probes/aggregate_oracle.py
"""
import json, statistics as st
R=json.load(open("experiments/results/oracle_analysis.json"))["layers"]
print("layers:", len(R))
cfgs=[(4,3),(8,6),(8,4),(8,2),(16,8),(16,4)]
parts=["contiguous","random","kmeans_gate_weights","kmeans_coactivation","kmeans_contribution"]
def get(r,E,k,p,s,c="sum"):
    m=[x for x in r["experts"] if x["E"]==E and x["k"]==k and x["partition"]==p and x["selection"]==s and x["combine"]==c]
    return m[0]["rel_err"] if m else None
best_sel=lambda r,E,k,p: get(r,E,k,p,"exhaustive") if get(r,E,k,p,"exhaustive") is not None else get(r,E,k,p,"greedy")
print("\nmean over layers of best-oracle rel_err, by partition")
print("cfg     "+"  ".join(f"{p[:14]:>14s}" for p in parts))
for E,k in cfgs: print(f"{E}/{k}".ljust(8)+"  ".join(f"{st.mean(best_sel(r,E,k,p) for r in R):14.3f}" for p in parts))
print("\nkmeans_contribution, mean over layers: selection comparison")
for E,k in cfgs:
    print(f"{E}/{k}", {s: round(st.mean(get(r,E,k,'kmeans_contribution',s) for r in R),3) for s in ["greedy","norm","static","random"]},
          "softmax_renorm:", round(st.mean(get(r,E,k,'kmeans_contribution','norm','softmax_renorm') for r in R),3))
print("\nneuron oracle (mean over layers):", {k: round(st.mean(r["neuron_oracle"][k]["rel_err"] for r in R),3) for k in R[0]["neuron_oracle"] if k.startswith("keep")})
print("contribution share of top neurons:", {q: round(st.mean(r["neuron_oracle"]["contribution_share_top"][q] for r in R),3) for q in R[0]["neuron_oracle"]["contribution_share_top"]})
print("\nper-layer 8/4 best (kmeans_contribution) and neuron keep 0.5:")
print(" ".join(f"L{r['layer']}:{best_sel(r,8,4,'kmeans_contribution'):.2f}/{r['neuron_oracle']['keep_0.5']['rel_err']:.2f}" for r in R))
u=[x["utilization"]["max_over_uniform"] for r in R for x in r["experts"] if x.get("utilization") and x["E"]==8 and x["k"]==4 and x["partition"]=="kmeans_contribution" and x["selection"]=="exhaustive"]
print("\n8/4 kmeans_contribution oracle load max/uniform: mean", round(st.mean(u),2), "max", round(max(u),2))
