# Offload: decode and prefill speed per VRAM budget (RX 9070 XT, 8 threads)

Budget = llama.cpp GPU buffers (weights + KV + recurrent state + compute) <= budget - 0.5 GiB, ctx 8192. RAM = weight bytes read from system RAM per decode token.

## Qwen3.5-9B

### 4 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| IQ4_XS | layers | 20 | 3.50 | 1.56 | 2582 | 24.60 | 0.0184 (BF16) |
| IQ4_XS | ffn | 23 | 3.47 | 1.72 | 2842 | 21.46 | 0.0184 (BF16) |
| Q6_K | layers | 14 | 3.42 | 3.31 | 1389 | 13.86 | 0.0028 (BF16) |
| Q6_K | ffn | 30 | 3.46 | 3.46 | 1441 | 12.70 | 0.0028 (BF16) |
| Q8_0 | layers | 10 | 3.34 | 5.14 | 1484 | 9.36 | 0.0016 (BF16) |

Does not fit: Q8_0 ffn.

### 6 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| IQ4_XS | ffn | 0 | 5.18 | 0.00 | 4331 | 89.39 | 0.0184 (BF16) |
| IQ4_XS | layers | 34 | 5.18 | 0.11 | 4370 | 89.33 | 0.0184 (BF16) |
| Q6_K | layers | 25 | 5.34 | 1.49 | 1852 | 24.46 | 0.0028 (BF16) |
| Q6_K | ffn | 13 | 5.43 | 1.50 | 1915 | 23.58 | 0.0028 (BF16) |
| Q8_0 | layers | 19 | 5.36 | 3.22 | 1975 | 13.54 | 0.0016 (BF16) |
| Q8_0 | ffn | 22 | 5.43 | 3.29 | 2127 | 13.07 | 0.0016 (BF16) |

### 8 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| IQ4_XS | ffn | 0 | 5.18 | 0.00 | 4359 | 89.41 | 0.0184 (BF16) |
| IQ4_XS | layers | 34 | 5.18 | 0.11 | 4355 | 89.39 | 0.0184 (BF16) |
| Q6_K | layers | 34 | 6.93 | 0.17 | 2494 | 70.18 | 0.0028 (BF16) |
| Q6_K | ffn | 0 | 6.93 | 0.00 | 2483 | 70.13 | 0.0028 (BF16) |
| Q8_0 | layers | 28 | 7.37 | 1.29 | 2969 | 24.89 | 0.0016 (BF16) |
| Q8_0 | ffn | 9 | 7.37 | 1.34 | 3073 | 24.10 | 0.0016 (BF16) |

### Qwen3.5-9B: cost of weights in RAM

Linear fit per quant of ms/token against GiB in RAM (all placements). Implied RAM bandwidth = 1 / slope.

| quant | points | ms/token at 0 GiB in RAM | ms per GiB in RAM | implied GB/s |
|---|---|---|---|---|
| IQ4_XS | 6 | 10.0 | 20.5 | 52 |
| Q6_K | 6 | 13.2 | 18.5 | 58 |
| Q8_0 | 5 | 18.2 | 17.3 | 62 |

### Qwen3.5-9B: ceiling for CPU-side activation sparsity (FFN strategy)

Only the RAM-resident FFN rows are skipped; the GPU side stays dense. Speed uses the per-quant fit above; ideal kernel, free mask. KLD added = TEAL KLD (9B, all layers) x share of layers whose FFN is in RAM: an assumption (additive over layers; transfer to the 27B unmeasured).

| quant | budget | FFN layers in RAM | tg128 measured | f=0.9 | f=0.75 | est. KLD added at 0.9 / 0.75 |
|---|---|---|---|---|---|---|
| IQ4_XS | 4 | 23/32 | 21.46 | 23.21 | 26.46 | 0.0008 / 0.0067 |
| Q6_K | 4 | 30/32 | 12.70 | 13.83 | 15.94 | 0.0010 / 0.0087 |
| Q6_K | 6 | 13/32 | 23.58 | 25.24 | 28.20 | 0.0004 / 0.0038 |
| Q8_0 | 6 | 22/32 | 13.07 | 14.13 | 16.07 | 0.0008 / 0.0064 |
| Q8_0 | 8 | 9/32 | 24.10 | 25.54 | 28.05 | 0.0003 / 0.0026 |

## Qwen3.8-27B

### 8 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| Q3_K_M-rq | layers | 35 | 7.47 | 5.31 | 735 | 7.96 | 0.0542 (Q8_0) |
| Q3_K_M-rq | ffn | 47 | 7.49 | 5.59 | 776 | 7.54 | 0.0542 (Q8_0) |
| IQ4_XS-rq | layers | 31 | 7.47 | 6.81 | 760 | 6.60 | 0.0185 (Q8_0) |
| IQ4_XS-rq | ffn | 54 | 7.47 | 7.14 | 820 | 6.24 | 0.0185 (Q8_0) |
| UD-Q6_K | layers | 20 | 7.29 | 12.63 | 454 | 3.86 | 0.0023 (Q8_0) |
| Q8_0 | layers | 16 | 7.24 | 18.84 | 420 | 2.72 | 0.0000 (Q8_0) |

Does not fit: Q8_0 ffn, UD-Q6_K ffn.

### 12 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| Q3_K_M-rq | layers | 57 | 11.41 | 1.58 | 990 | 16.91 | 0.0542 (Q8_0) |
| Q3_K_M-rq | ffn | 13 | 11.50 | 1.58 | 1020 | 16.81 | 0.0542 (Q8_0) |
| IQ4_XS-rq | layers | 50 | 11.34 | 3.11 | 1017 | 11.56 | 0.0185 (Q8_0) |
| IQ4_XS-rq | ffn | 24 | 11.44 | 3.18 | 1073 | 11.30 | 0.0185 (Q8_0) |
| UD-Q6_K | layers | 34 | 11.36 | 8.69 | 537 | 5.23 | 0.0023 (Q8_0) |
| UD-Q6_K | ffn | 47 | 11.48 | 8.89 | 563 | 5.14 | 0.0023 (Q8_0) |
| Q8_0 | layers | 27 | 11.50 | 14.70 | 482 | 3.34 | 0.0000 (Q8_0) |
| Q8_0 | ffn | 57 | 11.49 | 15.08 | 521 | 3.12 | 0.0000 (Q8_0) |

### 16 GiB VRAM

| quant | strategy | k | GPU GiB | RAM GiB | pp512 | tg128 | KLD (ref) |
|---|---|---|---|---|---|---|---|
| Q3_K_M-rq | layers | 66 | 13.08 | 0.19 | 1154 | 33.33 | 0.0542 (Q8_0) |
| Q3_K_M-rq | ffn | 0 | 13.08 | 0.00 | 1152 | 33.31 | 0.0542 (Q8_0) |
| IQ4_XS-rq | layers | 66 | 14.62 | 0.20 | 1397 | 32.41 | 0.0185 (Q8_0) |
| IQ4_XS-rq | ffn | 0 | 14.62 | 0.00 | 1398 | 32.40 | 0.0185 (Q8_0) |
| UD-Q6_K | layers | 48 | 15.44 | 4.76 | 657 | 8.05 | 0.0023 (Q8_0) |
| UD-Q6_K | ffn | 27 | 15.39 | 4.98 | 670 | 7.83 | 0.0023 (Q8_0) |
| Q8_0 | layers | 37 | 15.36 | 10.93 | 532 | 4.14 | 0.0000 (Q8_0) |
| Q8_0 | ffn | 42 | 15.46 | 11.11 | 458 | 3.92 | 0.0000 (Q8_0) |

### Qwen3.8-27B: cost of weights in RAM

Linear fit per quant of ms/token against GiB in RAM (all placements). Implied RAM bandwidth = 1 / slope.

| quant | points | ms/token at 0 GiB in RAM | ms per GiB in RAM | implied GB/s |
|---|---|---|---|---|
| IQ4_XS-rq | 6 | 29.5 | 18.2 | 59 |
| Q3_K_M-rq | 6 | 29.1 | 18.4 | 58 |
| Q8_0 | 5 | 78.6 | 15.4 | 70 |
| UD-Q6_K | 5 | 42.5 | 17.1 | 63 |

### Qwen3.8-27B: ceiling for CPU-side activation sparsity (FFN strategy)

Only the RAM-resident FFN rows are skipped; the GPU side stays dense. Speed uses the per-quant fit above; ideal kernel, free mask. KLD added = TEAL KLD (9B, all layers) x share of layers whose FFN is in RAM: an assumption (additive over layers; transfer to the 27B unmeasured).

| quant | budget | FFN layers in RAM | tg128 measured | f=0.9 | f=0.75 | est. KLD added at 0.9 / 0.75 |
|---|---|---|---|---|---|---|
| IQ4_XS-rq | 8 | 54/64 | 6.24 | 6.79 | 7.83 | 0.0009 / 0.0078 |
| IQ4_XS-rq | 12 | 24/64 | 11.30 | 12.09 | 13.50 | 0.0004 / 0.0035 |
| Q3_K_M-rq | 8 | 47/64 | 7.54 | 8.17 | 9.35 | 0.0008 / 0.0068 |
| Q3_K_M-rq | 12 | 13/64 | 16.81 | 17.67 | 19.14 | 0.0002 / 0.0019 |
| Q8_0 | 12 | 57/64 | 3.12 | 3.37 | 3.82 | 0.0010 / 0.0083 |
| Q8_0 | 16 | 42/64 | 3.92 | 4.20 | 4.71 | 0.0007 / 0.0061 |
| UD-Q6_K | 12 | 47/64 | 5.14 | 5.58 | 6.39 | 0.0008 / 0.0068 |
| UD-Q6_K | 16 | 27/64 | 7.83 | 8.39 | 9.40 | 0.0005 / 0.0039 |

## MTP at the same VRAM budget

Harmonic mean over code / prose / JSON prompts, 256 tokens, temperature 0. MTP's draft context needs ~0.4 GiB (9B) / ~0.8 GiB (27B) of VRAM, so its -ngl is lowered until it fits the same budget.

| model | budget | plain -ngl | plain tok/s | MTP -ngl | MTP tok/s | speedup | accept | KLD (ref) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B-IQ4_XS | 4 | 20 | 24.39 | 18 | 41.46 | 1.70x | 66% | 0.0184 (BF16) |
| Qwen3.5-9B-Q6_K | 4 | 14 | 13.77 | 12 | 31.60 | 2.29x | 68% | 0.0028 (BF16) |
| Qwen3.5-9B-IQ4_XS | 6 | 34 | 88.27 | 34 | 136.15 | 1.54x | 65% | 0.0184 (BF16) |
| Qwen3.5-9B-Q6_K | 6 | 25 | 24.38 | 23 | 48.17 | 1.98x | 67% | 0.0028 (BF16) |
| Qwen3.5-9B-Q6_K | 8 | 34 | 69.31 | 34 | 111.13 | 1.60x | 63% | 0.0028 (BF16) |
| Qwen3.5-9B-Q8_0 | 8 | 28 | 24.72 | 26 | 46.94 | 1.90x | 67% | 0.0016 (BF16) |
| Qwen3.8-27B-Q3_K_M-rq | 8 | 35 | 7.92 | 32 | 14.74 | 1.86x | 75% | 0.0542 (Q8_0) |
| Qwen3.8-27B-IQ4_XS-rq | 8 | 31 | 6.54 | 28 | 12.45 | 1.90x | 72% | 0.0185 (Q8_0) |
| Qwen3.8-27B-UD-Q6_K | 8 | 20 | 3.83 | 18 | 9.44 | 2.46x | 71% | 0.0023 (Q8_0) |
| Qwen3.8-27B-Q3_K_M-rq | 12 | 57 | 16.93 | 54 | 27.88 | 1.65x | 73% | 0.0542 (Q8_0) |
| Qwen3.8-27B-IQ4_XS-rq | 12 | 50 | 11.47 | 47 | 21.64 | 1.89x | 75% | 0.0185 (Q8_0) |
| Qwen3.8-27B-UD-Q6_K | 12 | 34 | 5.19 | 31 | 12.70 | 2.45x | 75% | 0.0023 (Q8_0) |
| Qwen3.8-27B-IQ4_XS-rq | 16 | 66 | 32.33 | 66 | 69.00 | 2.13x | 73% | 0.0185 (Q8_0) |
| Qwen3.8-27B-UD-Q6_K | 16 | 48 | 7.55 | 45 | 18.15 | 2.40x | 76% | 0.0023 (Q8_0) |

