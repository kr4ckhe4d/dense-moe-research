| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4374.66 ± 62.97 |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      4396.63 ± 20.58 |
| qwen35 9B Q8_0                 |   9.10 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         57.49 ± 0.68 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      2537.63 ± 21.87 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      2373.80 ± 13.80 |
| qwen35 9B Q6_K                 |   7.03 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         70.28 ± 0.07 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |     3497.80 ± 150.52 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      3545.72 ± 13.16 |
| qwen35 9B Q5_K - Medium        |   6.18 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         76.99 ± 0.07 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      3645.34 ± 29.59 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      3576.92 ± 10.29 |
| qwen35 9B Q4_K - Medium        |   5.37 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         84.02 ± 0.09 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4411.03 ± 32.36 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      4434.95 ± 21.74 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         89.58 ± 0.63 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      3719.57 ± 30.82 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |          pp2048 |      3666.96 ± 95.15 |
| qwen35 9B Q3_K - Medium        |   4.40 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         88.93 ± 0.13 |

build: 5e03bdd87 (11434)
