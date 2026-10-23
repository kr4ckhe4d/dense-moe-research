# MTP tuning: mtp_concurrency_gpu1.jsonl

24 generations (prompt set domains), up to 512 tokens each, GPU 1, seeds [0]. Speed is the harmonic mean over prompts (and seeds); acceptance = accepted / drafted. With np > 1, tok/s is aggregate throughput of the concurrent group.

## Temperature 0.0

### Q6_K (KLD 0.0028), no speculation 194.0 tok/s

| config | tok/s | vs none | accept | tokens/step | group0 | group1 | group2 | group3 | group4 | group5 | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=3 np=4 | 244.4 | 1.26x | 65% | 2.94 | 240 | 223 | 230 | 291 | 257 | 237 | 0/6 |
| none | 194.0 | 1.00x | - | 1.00 | 195 | 194 | 194 | 194 | 193 | 194 | 6/6 |

### IQ4_XS (KLD 0.0184), no speculation 256.3 tok/s

| config | tok/s | vs none | accept | tokens/step | group0 | group1 | group2 | group3 | group4 | group5 | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=3 np=4 | 314.4 | 1.23x | 66% | 2.96 | 332 | 320 | 284 | 325 | 338 | 294 | 0/6 |
| none | 256.3 | 1.00x | - | 1.00 | 258 | 256 | 253 | 257 | 257 | 257 | 6/6 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| Q6_K | 0.0028 | 194.0 | draft-mtp n_max=3 np=4 | 244.4 | 1.26x | group1 223 | group3 291 |
| IQ4_XS | 0.0184 | 256.3 | draft-mtp n_max=3 np=4 | 314.4 | 1.23x | group2 284 | group4 338 |

