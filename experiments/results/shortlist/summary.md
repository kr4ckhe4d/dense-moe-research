# MTP drafter with a vocabulary shortlist

## Six prompt types, 512 tokens each, temperature 0, 8K context, R9700 (decode tok/s)

`base` = drafter reads the whole output layer; `sl65k` = 65,536-token shortlist; `nN` = draft length; `pXX` = stop drafting below that draft probability.

| variant | code_python | code_cpp | prose | math | instruction | json | harmonic mean | accepted / drafted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base-n3 | 58.0 | 49.2 | 45.3 | 61.5 | 43.8 | 63.8 | 52.4 | 2086 / 2924 |
| base-n4 | 60.9 | 45.2 | 46.3 | 67.6 | 45.2 | 70.5 | 53.9 | 2192 / 3482 |
| sl65k-n3 | 62.0 | 53.2 | 49.3 | 66.8 | 46.8 | 66.8 | 56.3 | 2075 / 2960 |
| sl65k-n4 | 64.4 | 50.1 | 50.4 | 74.7 | 50.0 | 74.2 | 58.7 | 2177 / 3542 |
| sl65k-n6-p0 | 65.6 | 46.9 | 46.1 | 80.4 | 45.4 | 82.5 | 57.2 | 2285 / 4660 |
| sl65k-n6-p50 | 66.9 | 53.1 | 48.9 | 82.3 | 45.4 | 82.4 | 59.7 | 2234 / 3375 |
| sl65k-n6-p75 | 68.4 | 52.4 | 45.0 | 78.2 | 43.6 | 79.5 | 57.6 | 2120 / 2526 |
| sl65k-n8-p50 | 67.6 | 54.7 | 47.7 | 83.7 | 46.4 | 88.0 | 60.7 | 2297 / 3839 |
| sl65k-n8-p75 | 69.7 | 47.0 | 43.8 | 84.1 | 42.8 | 85.6 | 56.9 | 2115 / 2599 |

- Output text identical, base-n3 vs sl65k-n3: 6 of 6 prompts.
- Output text identical, base-n4 vs sl65k-n4: 6 of 6 prompts.

## Shortlist size and draft length at 2K context (two tasks, 256 tokens, q8_0 KV)

### Decode tok/s

| config | 2K |
|---|---:|
| sl32768-n3 | 61.1 |
| sl32768-n4 | 66.4 |
| sl32768-n6 | 69.9 |
| sl65536-n3 | 63.3 |
| sl65536-n4 | 69.4 |
| sl65536-n6 | 75.3 |
| full-n1 | 39.0 |
| full-n2 | 51.1 |
| full-n3 | 57.7 |
| full-n4 | 64.3 |
| full-n6 | 64.2 |

### ms per target evaluation

| config | 2K |
|---|---:|
| sl32768-n3 | 51.6 |
| sl32768-n4 | 54.5 |
| sl32768-n6 | 62.0 |
| sl65536-n3 | 52.6 |
| sl65536-n4 | 55.7 |
| sl65536-n6 | 63.7 |
| full-n1 | 48.7 |
| full-n2 | 52.9 |
| full-n3 | 57.4 |
| full-n4 | 61.8 |
| full-n6 | 72.8 |

### Tokens per target evaluation

| config | 2K |
|---|---:|
| sl32768-n3 | 3.16 |
| sl32768-n4 | 3.63 |
| sl32768-n6 | 4.35 |
| sl65536-n3 | 3.34 |
| sl65536-n4 | 3.88 |
| sl65536-n6 | 4.81 |
| full-n1 | 1.90 |
| full-n2 | 2.71 |
| full-n3 | 3.33 |
| full-n4 | 3.98 |
| full-n6 | 4.69 |

## At depth (two tasks, 256 tokens, q8_0 KV)

### Decode tok/s

| config | 2K | 120K |
|---|---:|---:|
| base-n3 | 59.0 | 46.1 |

### ms per target evaluation

| config | 2K | 120K |
|---|---:|---:|
| base-n3 | 56.8 | 68.3 |

### Tokens per target evaluation

| config | 2K | 120K |
|---|---:|---:|
| base-n3 | 3.36 | 3.16 |

