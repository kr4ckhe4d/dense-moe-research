# DFlash follow-up: 9B drafter, long context, sampling, editing, concurrency, offload

Patched llama.cpp (patches/llama.cpp-rdna4-mmvq-multicol.patch). tok/s = harmonic mean over the prompts; accept = accepted / drafted tokens.

## Qwen3.5-9B fully on the R9700 (six domains, 512 tokens, t=0)

| model | drafter | tok/s | x plain | accept | code_cpp | code_python | instruction | json | math | prose |
|---|---|---|---|---|---|---|---|---|---|---|
| 9B-Q8_0 | plain | 58.8 | 1.00 |  | 59 | 59 | 59 | 59 | 59 | 59 |
| 9B-Q8_0 | mtp n=3 | 104.5 | 1.78 | 66% | 90 | 115 | 97 | 129 | 121 | 88 |
| 9B-Q8_0 | dflash n=3 | 114.2 | 1.94 | 63% | 104 | 125 | 99 | 147 | 138 | 93 |
| 9B-Q8_0 | dflash n=5 | 131.1 | 2.23 | 51% | 121 | 146 | 101 | 189 | 182 | 101 |
| 9B-Q8_0 | dflash n=7 | 133.8 | 2.27 | 41% | 123 | 153 | 98 | 219 | 206 | 95 |
| 9B-Q6_K | plain | 70.8 | 1.00 |  | 71 | 71 | 71 | 71 | 71 | 71 |
| 9B-Q6_K | mtp n=3 | 118.6 | 1.67 | 65% | 107 | 128 | 111 | 147 | 140 | 95 |
| 9B-Q6_K | dflash n=3 | 128.5 | 1.81 | 65% | 120 | 145 | 105 | 162 | 157 | 106 |
| 9B-Q6_K | dflash n=5 | 132.6 | 1.87 | 48% | 116 | 157 | 102 | 197 | 181 | 102 |
| 9B-Q6_K | dflash n=7 | 120.5 | 1.70 | 37% | 101 | 146 | 89 | 207 | 179 | 88 |
| 9B-IQ4_XS | plain | 91.0 | 1.00 |  | 91 | 91 | 91 | 91 | 91 | 91 |
| 9B-IQ4_XS | mtp n=3 | 141.1 | 1.55 | 66% | 128 | 151 | 130 | 175 | 166 | 115 |
| 9B-IQ4_XS | dflash n=3 | 149.0 | 1.64 | 63% | 132 | 166 | 128 | 188 | 182 | 125 |
| 9B-IQ4_XS | dflash n=5 | 164.5 | 1.81 | 51% | 143 | 196 | 132 | 220 | 225 | 128 |
| 9B-IQ4_XS | dflash n=7 | 154.0 | 1.69 | 41% | 126 | 190 | 119 | 244 | 226 | 113 |

## Qwen3.5-9B fully on the RX 9070 XT (six domains, 512 tokens, t=0)

| model | drafter | tok/s | x plain | accept | code_cpp | code_python | instruction | json | math | prose |
|---|---|---|---|---|---|---|---|---|---|---|
| 9B-Q8_0 | plain | 57.4 | 1.00 |  | 57 | 57 | 57 | 57 | 57 | 57 |
| 9B-Q8_0 | mtp n=3 | 102.3 | 1.78 | 66% | 88 | 112 | 95 | 127 | 118 | 86 |
| 9B-Q8_0 | dflash n=3 | 111.5 | 1.94 | 63% | 102 | 122 | 97 | 144 | 135 | 90 |
| 9B-Q8_0 | dflash n=5 | 128.7 | 2.24 | 51% | 118 | 143 | 99 | 186 | 179 | 99 |
| 9B-Q8_0 | dflash n=7 | 131.9 | 2.30 | 41% | 121 | 151 | 97 | 215 | 203 | 94 |
| 9B-Q6_K | plain | 69.3 | 1.00 |  | 69 | 69 | 69 | 69 | 69 | 69 |
| 9B-Q6_K | mtp n=3 | 117.3 | 1.69 | 65% | 106 | 127 | 110 | 146 | 139 | 94 |
| 9B-Q6_K | dflash n=3 | 126.6 | 1.83 | 65% | 118 | 142 | 104 | 160 | 155 | 105 |
| 9B-Q6_K | dflash n=5 | 132.6 | 1.91 | 48% | 115 | 156 | 102 | 197 | 181 | 102 |
| 9B-Q6_K | dflash n=7 | 122.2 | 1.77 | 37% | 102 | 148 | 90 | 209 | 182 | 89 |
| 9B-IQ4_XS | plain | 88.4 | 1.00 |  | 88 | 88 | 88 | 88 | 88 | 88 |
| 9B-IQ4_XS | mtp n=3 | 138.2 | 1.56 | 66% | 126 | 148 | 127 | 171 | 163 | 113 |
| 9B-IQ4_XS | dflash n=3 | 145.5 | 1.65 | 63% | 129 | 161 | 125 | 183 | 178 | 122 |
| 9B-IQ4_XS | dflash n=5 | 162.6 | 1.84 | 51% | 141 | 193 | 130 | 219 | 222 | 127 |
| 9B-IQ4_XS | dflash n=7 | 153.0 | 1.73 | 41% | 125 | 188 | 118 | 243 | 225 | 113 |

## Long context, R9700 (~4k / 16k / 28k tokens of C++, then a question; 512 tokens)

| model | drafter | tok/s | x plain | accept | longctx_4k | longctx_16k | longctx_28k |
|---|---|---|---|---|---|---|---|
| 27B-UD-Q6_K | plain | 23.3 | 1.00 |  | 24 | 23 | 23 |
| 27B-UD-Q6_K | mtp n=3 | 50.6 | 2.17 | 71% | 51 | 53 | 49 |
| 27B-UD-Q6_K | dflash n=3 | 51.2 | 2.20 | 66% | 51 | 53 | 49 |
| 27B-UD-Q6_K | dflash n=5 | 53.4 | 2.29 | 52% | 54 | 56 | 50 |
| 27B-IQ4_XS-rq | plain | 31.2 | 1.00 |  | 32 | 31 | 30 |
| 27B-IQ4_XS-rq | mtp n=3 | 65.5 | 2.10 | 73% | 75 | 61 | 62 |
| 27B-IQ4_XS-rq | dflash n=3 | 65.2 | 2.09 | 69% | 75 | 63 | 60 |
| 27B-IQ4_XS-rq | dflash n=5 | 67.7 | 2.17 | 52% | 71 | 68 | 64 |
| 9B-Q6_K | plain | 65.8 | 1.00 |  | 69 | 66 | 63 |
| 9B-Q6_K | mtp n=3 | 110.8 | 1.68 | 63% | 119 | 110 | 104 |
| 9B-Q6_K | dflash n=3 | 116.0 | 1.76 | 60% | 123 | 115 | 111 |
| 9B-Q6_K | dflash n=5 | 115.3 | 1.75 | 42% | 129 | 107 | 112 |

## Sampling, R9700 (six domains, 512 tokens; t=0.7 over seeds 0-2, t=1.0 seed 0)

| model | drafter | temperature | runs | tok/s | accept |
|---|---|---|---|---|---|
| 27B-IQ4_XS-rq | dflash n=5 | 0.7 | 18 | 75.6 | 53% |
| 27B-IQ4_XS-rq | dflash n=5 prob | 0.7 | 18 | 75.9 | 53% |
| 27B-IQ4_XS-rq | mtp n=3 | 0.7 | 18 | 66.3 | 66% |
| 27B-IQ4_XS-rq | dflash n=5 | 1.0 | 6 | 75.5 | 53% |
| 27B-IQ4_XS-rq | dflash n=5 prob | 1.0 | 6 | 75.5 | 53% |
| 27B-IQ4_XS-rq | mtp n=3 | 1.0 | 6 | 66.5 | 67% |
| 27B-UD-Q6_K | dflash n=5 | 0.7 | 18 | 60.3 | 55% |
| 27B-UD-Q6_K | dflash n=5 prob | 0.7 | 18 | 60.5 | 55% |
| 27B-UD-Q6_K | mtp n=3 | 0.7 | 18 | 51.6 | 67% |
| 27B-UD-Q6_K | dflash n=5 | 1.0 | 6 | 60.4 | 55% |
| 27B-UD-Q6_K | dflash n=5 prob | 1.0 | 6 | 60.4 | 55% |
| 27B-UD-Q6_K | mtp n=3 | 1.0 | 6 | 51.2 | 66% |
| 9B-Q6_K | dflash n=5 | 0.7 | 18 | 131.5 | 47% |
| 9B-Q6_K | dflash n=5 prob | 0.7 | 18 | 131.5 | 47% |
| 9B-Q6_K | mtp n=3 | 0.7 | 18 | 114.9 | 62% |
| 9B-Q6_K | dflash n=5 | 1.0 | 6 | 126.0 | 45% |
| 9B-Q6_K | dflash n=5 prob | 1.0 | 6 | 126.0 | 45% |
| 9B-Q6_K | mtp n=3 | 1.0 | 6 | 110.1 | 58% |

## Repetitive editing, R9700 (return a file with small additions, 1024 tokens)

| model | drafter | tok/s | x plain | accept | edit_cpp | edit_python |
|---|---|---|---|---|---|---|
| 27B-UD-Q6_K | plain | 24.2 | 1.00 |  | 24 | 24 |
| 27B-UD-Q6_K | mtp n=3 | 65.4 | 2.70 | 94% | 65 | 65 |
| 27B-UD-Q6_K | dflash n=5 | 75.5 | 3.12 | 75% | 80 | 71 |
| 27B-UD-Q6_K | ngram-mod n=3 | 66.3 | 2.74 | 72% | 60 | 74 |
| 27B-UD-Q6_K | mtp,ngram-mod n=3 | 128.6 | 5.32 | 78% | 115 | 146 |
| 27B-UD-Q6_K | dflash,ngram-mod n=5 | 137.3 | 5.68 | 73% | 127 | 149 |
| 27B-UD-Q6_K | dflash,ngram-map-k n=5 | 110.5 | 4.57 | 78% | 80 | 178 |
| 27B-IQ4_XS-rq | plain | 32.6 | 1.00 |  | 33 | 33 |
| 27B-IQ4_XS-rq | mtp n=3 | 84.6 | 2.60 | 95% | 84 | 85 |
| 27B-IQ4_XS-rq | dflash n=5 | 96.7 | 2.97 | 75% | 102 | 92 |
| 27B-IQ4_XS-rq | ngram-mod n=3 | 92.1 | 2.83 | 75% | 81 | 106 |
| 27B-IQ4_XS-rq | mtp,ngram-mod n=3 | 173.6 | 5.33 | 78% | 152 | 203 |
| 27B-IQ4_XS-rq | dflash,ngram-mod n=5 | 177.0 | 5.43 | 72% | 163 | 194 |
| 27B-IQ4_XS-rq | dflash,ngram-map-k n=5 | 142.2 | 4.37 | 78% | 102 | 236 |

## Four concurrent requests, R9700 (aggregate tok/s, six domains in groups of 4)

| model | drafter | tok/s | x plain | accept | group0 | group1 | group2 | group3 | group4 | group5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 27B-UD-Q6_K | plain | 74.7 | 1.00 |  | 75 | 74 | 75 | 75 | 75 | 75 |
| 27B-UD-Q6_K | mtp n=3 | 109.8 | 1.47 | 70% | 120 | 104 | 104 | 125 | 89 | 127 |
| 27B-UD-Q6_K | dflash n=5 | 116.1 | 1.55 | 58% | 135 | 114 | 103 | 131 | 111 | 108 |
| 27B-IQ4_XS-rq | plain | 96.8 | 1.00 |  | 97 | 96 | 97 | 97 | 97 | 97 |
| 27B-IQ4_XS-rq | mtp n=3 | 148.0 | 1.53 | 70% | 161 | 148 | 136 | 160 | 143 | 144 |
| 27B-IQ4_XS-rq | dflash n=5 | 151.8 | 1.57 | 58% | 167 | 139 | 128 | 165 | 158 | 163 |

## Partly offloaded, RX 9070 XT at a VRAM budget (code, prose, JSON; 256 tokens)

Plain and stock-build MTP rows are from the offload milestone (offload_mtp_gpu0.jsonl). All spec rows fit the same budget rule.

| model | VRAM | plain | MTP n=3 (stock) | MTP n=3 | DFlash n=5 | DFlash n=7 | n=5, drafter in RAM | n=7, drafter in RAM |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B-IQ4_XS | 4 GB | 24.4 (ngl 20) | 41.5 (ngl 18) | 41.2 (ngl 18) | 18.8 (ngl 0) | 23.6 (ngl 0) | 22.8 (ngl 11) | 29.4 (ngl 11) |
| Qwen3.5-9B-IQ4_XS | 6 GB | 88.3 (ngl 34) | 136.1 (ngl 34) | 138.7 (ngl 34) | 36.0 (ngl 16) | 45.6 (ngl 16) | 37.2 (ngl 26) | 48.6 (ngl 26) |
| Qwen3.5-9B-Q6_K | 4 GB | 13.8 (ngl 14) | 31.6 (ngl 12) | 30.9 (ngl 12) | 22.4 (ngl 0) | 17.2 (ngl 0) | 22.4 (ngl 8) | 21.5 (ngl 8) |
| Qwen3.5-9B-Q6_K | 6 GB | 24.4 (ngl 25) | 48.2 (ngl 23) | 48.4 (ngl 23) | 32.3 (ngl 11) | 26.6 (ngl 11) | 31.7 (ngl 19) | 31.0 (ngl 19) |
| Qwen3.5-9B-Q6_K | 8 GB | 69.3 (ngl 34) | 111.1 (ngl 34) | 118.0 (ngl 34) | 53.6 (ngl 23) | 47.1 (ngl 23) | 47.3 (ngl 30) | 50.2 (ngl 30) |
| Qwen3.5-9B-Q8_0 | 8 GB | 24.7 (ngl 28) | 46.9 (ngl 26) | 46.4 (ngl 26) | 30.4 (ngl 17) | 38.0 (ngl 17) | 28.4 (ngl 23) | 38.8 (ngl 23) |
| Qwen3.8-27B-IQ4_XS-rq | 8 GB | 6.5 (ngl 31) | 12.5 (ngl 28) | 12.7 (ngl 28) | 9.7 (ngl 17) | 15.2 (ngl 17) | 10.5 (ngl 27) | 16.2 (ngl 27) |
| Qwen3.8-27B-IQ4_XS-rq | 12 GB | 11.5 (ngl 50) | 21.6 (ngl 47) | 21.2 (ngl 47) | 15.0 (ngl 36) | 22.0 (ngl 36) | 15.8 (ngl 45) | 23.1 (ngl 45) |
| Qwen3.8-27B-IQ4_XS-rq | 16 GB | 32.3 (ngl 66) | 69.0 (ngl 66) | 70.6 (ngl 66) | 33.2 (ngl 56) | 43.2 (ngl 56) | 34.8 (ngl 64) | 46.8 (ngl 64) |
| Qwen3.8-27B-Q3_K_M-rq | 8 GB | 7.9 (ngl 35) | 14.7 (ngl 32) | 14.7 (ngl 32) | 11.6 (ngl 20) | 12.4 (ngl 20) | 11.9 (ngl 30) | 13.5 (ngl 30) |
| Qwen3.8-27B-Q3_K_M-rq | 12 GB | 16.9 (ngl 57) | 27.9 (ngl 54) | 28.6 (ngl 54) | 17.9 (ngl 40) | 19.4 (ngl 40) | 19.0 (ngl 51) | 22.0 (ngl 51) |
| Qwen3.8-27B-UD-Q6_K | 8 GB | 3.8 (ngl 20) | 9.4 (ngl 18) | 9.6 (ngl 18) | 9.3 (ngl 11) | 10.1 (ngl 11) | 9.2 (ngl 18) | 10.4 (ngl 18) |
| Qwen3.8-27B-UD-Q6_K | 12 GB | 5.2 (ngl 34) | 12.7 (ngl 31) | 12.9 (ngl 31) | 11.6 (ngl 24) | 12.8 (ngl 24) | 11.6 (ngl 31) | 12.7 (ngl 31) |
| Qwen3.8-27B-UD-Q6_K | 16 GB | 7.6 (ngl 48) | 18.2 (ngl 45) | 18.4 (ngl 45) | 15.4 (ngl 37) | 17.6 (ngl 37) | 15.3 (ngl 44) | 17.9 (ngl 44) |

