# MTP tuning: mtp_edit_gpu1.jsonl

28 generations (prompt set edit), up to 1024 tokens each, GPU 1, seeds [0]. Speed is the harmonic mean over prompts (and seeds); acceptance = accepted / drafted. With np > 1, tok/s is aggregate throughput of the concurrent group.

## Temperature 0.0

### Q6_K (KLD 0.0028), no speculation 70.0 tok/s

| config | tok/s | vs none | accept | tokens/step | edit_python | edit_cpp | same text as none |
|---|---|---|---|---|---|---|---|
| draft-mtp n_max=3 | 144.2 | 2.06x | 93% | 3.79 | 151 | 138 | 1/2 |
| draft-mtp n_max=6 | 147.3 | 2.10x | 84% | 5.99 | 158 | 138 | 1/2 |
| draft-mtp,ngram-map-k n_max=3 | 237.7 | 3.40x | 92% | 6.63 | 896 | 137 | 1/2 |
| draft-mtp,ngram-mod n_max=3 | 353.3 | 5.05x | 83% | 11.51 | 685 | 238 | 1/2 |
| ngram-map-k n_max=3 | 128.5 | 1.84x | 91% | 1.94 | 812 | 70 | 1/2 |
| ngram-mod n_max=3 | 263.6 | 3.77x | 86% | 4.80 | 691 | 163 | 1/2 |
| none | 70.0 | 1.00x | - | 1.00 | 70 | 70 | 2/2 |

### IQ4_XS (KLD 0.0184), no speculation 89.9 tok/s

| config | tok/s | vs none | accept | tokens/step | edit_python | edit_cpp | same text as none |
|---|---|---|---|---|---|---|---|
| draft-mtp n_max=3 | 173.6 | 1.93x | 93% | 3.77 | 181 | 167 | 1/2 |
| draft-mtp n_max=6 | 177.7 | 1.98x | 80% | 5.75 | 192 | 165 | 2/2 |
| draft-mtp,ngram-map-k n_max=3 | 292.7 | 3.26x | 92% | 6.63 | 1195 | 167 | 1/2 |
| draft-mtp,ngram-mod n_max=3 | 450.9 | 5.02x | 83% | 11.57 | 905 | 300 | 2/2 |
| ngram-map-k n_max=3 | 166.0 | 1.85x | 91% | 1.94 | 1107 | 90 | 2/2 |
| ngram-mod n_max=3 | 346.0 | 3.85x | 86% | 4.81 | 942 | 212 | 2/2 |
| none | 89.9 | 1.00x | - | 1.00 | 90 | 90 | 2/2 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| Q6_K | 0.0028 | 70.0 | draft-mtp,ngram-mod n_max=3 | 353.3 | 5.05x | edit_cpp 238 | edit_python 685 |
| IQ4_XS | 0.0184 | 89.9 | draft-mtp,ngram-mod n_max=3 | 450.9 | 5.02x | edit_cpp 300 | edit_python 905 |

