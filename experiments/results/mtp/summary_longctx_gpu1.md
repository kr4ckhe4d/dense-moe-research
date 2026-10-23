# MTP tuning: mtp_longctx_gpu1.jsonl

18 generations (prompt set longctx), up to 512 tokens each, GPU 1, seeds [0]. Speed is the harmonic mean over prompts (and seeds); acceptance = accepted / drafted. With np > 1, tok/s is aggregate throughput of the concurrent group.

## Temperature 0.0

### Q6_K (KLD 0.0028), no speculation 67.0 tok/s

| config | tok/s | vs none | accept | tokens/step | longctx_4k | longctx_16k | longctx_28k | same text as none |
|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 103.7 | 1.55x | 73% | 2.45 | 112 | 100 | 99 | 0/3 |
| draft-mtp n_max=3 | 103.4 | 1.54x | 62% | 2.83 | 116 | 98 | 98 | 0/3 |
| none | 67.0 | 1.00x | - | 1.00 | 70 | 67 | 64 | 3/3 |

### IQ4_XS (KLD 0.0184), no speculation 84.5 tok/s

| config | tok/s | vs none | accept | tokens/step | longctx_4k | longctx_16k | longctx_28k | same text as none |
|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 122.8 | 1.45x | 72% | 2.43 | 131 | 123 | 116 | 0/3 |
| draft-mtp n_max=3 | 124.8 | 1.48x | 62% | 2.84 | 133 | 128 | 115 | 0/3 |
| none | 84.5 | 1.00x | - | 1.00 | 89 | 84 | 80 | 3/3 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| Q6_K | 0.0028 | 67.0 | draft-mtp n_max=2 | 103.7 | 1.55x | longctx_28k 99 | longctx_4k 112 |
| IQ4_XS | 0.0184 | 84.5 | draft-mtp n_max=3 | 124.8 | 1.48x | longctx_28k 115 | longctx_4k 133 |

