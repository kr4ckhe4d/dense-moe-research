| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4434.14 ± 91.90 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.95 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         89.62 ± 0.31 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.01 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      2152.20 ± 73.76 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   4.01 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |         97.24 ± 0.11 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   3.44 GiB |     9.20 B | ROCm       |  99 |   1 |           pp512 |      4133.16 ± 13.45 |
| qwen35 9B IQ4_XS - 4.25 bpw    |   3.44 GiB |     9.20 B | ROCm       |  99 |   1 |           tg128 |        105.84 ± 0.79 |

build: 5e03bdd87 (11434)
