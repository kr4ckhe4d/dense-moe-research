# MTP tuning: mtp_tune_gpu0.jsonl

72 generations (prompt set domains), up to 512 tokens each, GPU 0, seeds [0]. Speed is the harmonic mean over prompts (and seeds); acceptance = accepted / drafted. With np > 1, tok/s is aggregate throughput of the concurrent group.

## Temperature 0.0

### Q8_0 (KLD 0.0016), no speculation 57.4 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 97.7 | 1.70x | 76% | 2.51 | 105 | 92 | 87 | 106 | 90 | 110 | 1/6 |
| draft-mtp n_max=3 | 103.1 | 1.80x | 66% | 2.98 | 112 | 94 | 87 | 118 | 93 | 128 | 1/6 |
| draft-mtp n_max=4 | 103.2 | 1.80x | 59% | 3.33 | 114 | 86 | 87 | 128 | 91 | 134 | 1/6 |
| none | 57.4 | 1.00x | - | 1.00 | 57 | 57 | 57 | 57 | 57 | 57 | 6/6 |

### Q6_K (KLD 0.0028), no speculation 69.2 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 111.4 | 1.61x | 76% | 2.51 | 111 | 108 | 99 | 122 | 105 | 128 | 1/6 |
| draft-mtp n_max=3 | 113.6 | 1.64x | 66% | 2.95 | 114 | 105 | 97 | 133 | 105 | 140 | 1/6 |
| draft-mtp n_max=4 | 109.6 | 1.58x | 58% | 3.31 | 119 | 97 | 90 | 136 | 95 | 144 | 1/6 |
| none | 69.2 | 1.00x | - | 1.00 | 69 | 69 | 69 | 69 | 69 | 69 | 6/6 |

### IQ4_XS (KLD 0.0184), no speculation 88.1 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 131.9 | 1.50x | 75% | 2.48 | 139 | 123 | 116 | 145 | 124 | 152 | 0/6 |
| draft-mtp n_max=3 | 134.5 | 1.53x | 64% | 2.90 | 146 | 122 | 113 | 156 | 118 | 170 | 0/6 |
| draft-mtp n_max=4 | 129.6 | 1.47x | 55% | 3.20 | 152 | 113 | 100 | 160 | 113 | 174 | 0/6 |
| none | 88.1 | 1.00x | - | 1.00 | 88 | 88 | 88 | 88 | 88 | 88 | 6/6 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| Q8_0 | 0.0016 | 57.4 | draft-mtp n_max=4 | 103.2 | 1.80x | code_cpp 86 | json 134 |
| Q6_K | 0.0028 | 69.2 | draft-mtp n_max=3 | 113.6 | 1.64x | prose 97 | json 140 |
| IQ4_XS | 0.0184 | 88.1 | draft-mtp n_max=3 | 134.5 | 1.53x | prose 113 | json 170 |

