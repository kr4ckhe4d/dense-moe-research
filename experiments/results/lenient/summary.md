# Lenient draft acceptance: Qwen3.8-27B UD-Q6_K, MTP draft 3, R9700

Six domain prompts, 512 tokens each, temperature 0, thinking off. Ratio r: a draft token the target did not pick is accepted if p_target(draft) >= r * p_target(top).

## Projected from the exact run's log: tokens per verify step, relative to exact

Each step's draft is re-scored under the rule. Ignores that the text diverges afterwards.

| domain | exact tok/step | r=0.7 | r=0.5 | r=0.3 | r=0.2 | r=0.1 | r=0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code_python | 3.45 | 1.031 | 1.057 | 1.069 | 1.080 | 1.098 | 1.110 |
| code_cpp | 2.97 | 1.051 | 1.067 | 1.112 | 1.157 | 1.180 | 1.219 |
| prose | 2.72 | 1.029 | 1.057 | 1.110 | 1.143 | 1.202 | 1.270 |
| math | 3.65 | 1.002 | 1.006 | 1.022 | 1.023 | 1.029 | 1.033 |
| instruction | 2.73 | 1.043 | 1.102 | 1.143 | 1.184 | 1.215 | 1.272 |
| json | 3.83 | 1.010 | 1.010 | 1.024 | 1.027 | 1.029 | 1.029 |

## Measured decode tok/s

| domain | exact | r=0.5 | r=0.3 | r=0.1 |
|---|---:|---:|---:|---:|
| code_python | 58.9 | 62.4 | 63.0 | 64.4 |
| code_cpp | 50.9 | 52.1 | 57.9 | 57.1 |
| prose | 46.5 | 48.6 | 53.6 | 53.6 |
| math | 62.2 | 61.1 | 64.2 | 64.3 |
| instruction | 46.6 | 50.0 | 54.9 | 55.7 |
| json | 65.0 | 65.8 | 66.2 | 66.6 |
| **harmonic mean** | **54.0** | **55.9** | **59.6** | **59.9** |
| vs exact | +0.0% | +3.4% | +10.3% | +10.8% |

## What lenient acceptance changes in the output

Lenient tokens: output tokens that are not the target's top choice. Mean ratio: their p / p_top. First divergence: characters shared with the exact run's answer.

| ratio | domain | lenient tokens | share of output | mean p/p_top | draft accepted / drafted | first divergence |
|---:|---|---:|---:|---:|---|---:|
| 0.5 | code_python | 9 | 1.8% | 0.79 | 372 / 416 | 13 |
| 0.5 | code_cpp | 27 | 5.3% | 0.78 | 344 / 499 | 13 |
| 0.5 | prose | 16 | 3.1% | 0.73 | 332 / 536 | 421 |
| 0.5 | math | 7 | 1.4% | 0.73 | 369 / 426 | 683 |
| 0.5 | instruction | 26 | 5.1% | 0.73 | 337 / 517 | 110 |
| 0.5 | json | 5 | 1.0% | 0.77 | 379 / 393 | 650 |
| 0.3 | code_python | 13 | 2.5% | 0.64 | 373 / 412 | 13 |
| 0.3 | code_cpp | 23 | 4.5% | 0.68 | 361 / 448 | 13 |
| 0.3 | prose | 33 | 6.4% | 0.57 | 349 / 486 | 164 |
| 0.3 | math | 6 | 1.2% | 0.55 | 376 / 403 | 88 |
| 0.3 | instruction | 26 | 5.1% | 0.57 | 353 / 472 | 55 |
| 0.3 | json | 3 | 0.6% | 0.48 | 380 / 393 | 240 |
| 0.1 | code_python | 12 | 2.3% | 0.45 | 376 / 402 | 10 |
| 0.1 | code_cpp | 57 | 11.1% | 0.49 | 359 / 455 | 13 |
| 0.1 | prose | 60 | 11.7% | 0.40 | 349 / 485 | 192 |
| 0.1 | math | 12 | 2.3% | 0.30 | 376 / 402 | 88 |
| 0.1 | instruction | 53 | 10.4% | 0.49 | 355 / 465 | 9 |
| 0.1 | json | 4 | 0.8% | 0.32 | 381 / 388 | 240 |

## Temperature 1.0 (one seed), decode tok/s

| config | code_python | code_cpp | prose | math | instruction | json | harmonic mean | lenient share of output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 60.9 | 42.1 | 41.7 | 57.1 | 41.0 | 64.3 | 49.4 | 0.0% |
| lenient r=0.3 | 63.0 | 56.7 | 50.8 | 63.8 | 50.8 | 66.6 | 57.9 | 7.3% |
| exact, probabilistic draft (lossless) | 62.6 | 50.0 | 43.0 | 61.6 | 44.7 | 66.6 | 53.2 | 0.0% |
