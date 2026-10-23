| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B BF16                 |  17.13 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |        636.35 ± 5.54 |
| qwen35 9B BF16                 |  17.13 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |        635.39 ± 1.97 |
| qwen35 9B BF16                 |  17.13 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         34.73 ± 0.01 |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4199.69 ± 37.23 |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      4190.31 ± 13.08 |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         59.40 ± 0.10 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |       2429.67 ± 7.61 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |       2263.65 ± 2.89 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         71.58 ± 0.05 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      3481.32 ± 18.45 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      3409.31 ± 10.49 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         78.40 ± 0.07 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      3537.51 ± 30.05 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |       3459.50 ± 8.60 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         85.74 ± 0.07 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4256.28 ± 61.58 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      4280.84 ± 14.01 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         92.13 ± 0.10 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      3605.80 ± 34.42 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |       3590.35 ± 7.91 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         90.74 ± 0.10 |

build: 5e03bdd87 (11434)
