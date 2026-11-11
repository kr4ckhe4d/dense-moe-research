# Depth profile and MMVQ re-check on llama.cpp b12164

## MMVQ patch on b12164 (4b77dee05): ms per batch of n tokens

llama-bench -ngl 99 -fa 1 -r 5. Ratio = patched / stock time (< 1 is faster).

| GPU | model | build | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 512 | t(6)/t(1) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RX 9070 XT | 27B-Q3_K_M | stock | 30.0 | 32.2 | 33.8 | 38.8 | 44.0 | 54.4 | 63.1 | 398 | 1.82 |
| RX 9070 XT | 27B-Q3_K_M | patched | 29.9 | 31.9 | 33.0 | 35.4 | 39.5 | 44.0 | 54.3 | 401 | 1.47 |
| | | ratio | 1.00 | 0.99 | 0.98 | 0.91 | 0.90 | 0.81 | 0.86 | 1.01 | |
| R9700 | 27B-Q8_0 | stock | 53.5 | 54.4 | 56.1 | 56.4 | 57.6 | 59.0 | 64.7 | 423 | 1.10 |
| R9700 | 27B-Q8_0 | patched | 53.8 | 53.6 | 54.6 | 54.9 | 55.8 | 56.7 | 58.6 | 425 | 1.06 |
| | | ratio | 1.00 | 0.98 | 0.97 | 0.97 | 0.97 | 0.96 | 0.90 | 1.01 | |
| R9700 | 27B-UD-Q6_K | stock | 47.9 | 43.9 | 45.2 | 48.3 | 52.9 | 56.7 | 70.8 | 533 | 1.37 |
| R9700 | 27B-UD-Q6_K | patched | 41.4 | 43.4 | 44.3 | 45.5 | 46.4 | 48.8 | 54.3 | 543 | 1.18 |
| | | ratio | 0.87 | 0.99 | 0.98 | 0.94 | 0.88 | 0.86 | 0.77 | 1.02 | |

## Decode against context depth: Qwen3.8-27B UD-Q6_K on the R9700, b12164

Mean of the two tasks (explain, code), 256 tokens each, temperature 0. Depth is the prompt length in tokens (actual: 2010, 32867, 64108, 122601).

### Decode tok/s

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 23.9 | 20.3 | 17.7 | 14.3 |
| mtp3-q8kv | 55.5 | 43.7 | 37.3 | 29.1 |
| mtp3-q8kv-patched | 58.8 | 45.0 | 40.2 | 30.0 |
| mtp3-f16kv-64k | 55.8 | 49.4 | 38.8 |  |
| plain-f16kv | 24.4 | 22.6 | 21.0 | 18.7 |
| mtp3-f16kv | 55.8 | 49.4 | 38.8 | 35.5 |
| mtp3-f16kv-patched | 59.1 | 49.9 | 44.3 | 36.9 |
| plain-q8kv-fattn | 24.3 | 22.3 | 20.5 | 17.8 |
| mtp3-q8kv-fattn | 59.0 | 45.9 | 43.2 | 34.1 |

### ms per target evaluation (plain: per token; MTP: per verify step)

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 41.7 | 49.2 | 56.2 | 69.4 |
| mtp3-q8kv | 60.1 | 71.6 | 84.0 | 106.8 |
| mtp3-q8kv-patched | 57.0 | 68.4 | 80.3 | 104.7 |
| mtp3-f16kv-64k | 59.4 | 66.8 | 74.4 |  |
| plain-f16kv | 40.9 | 44.1 | 47.4 | 53.4 |
| mtp3-f16kv | 59.4 | 66.9 | 74.4 | 88.2 |
| mtp3-f16kv-patched | 56.4 | 63.5 | 70.8 | 84.3 |
| plain-q8kv-fattn | 41.0 | 44.8 | 48.6 | 56.1 |
| mtp3-q8kv-fattn | 56.8 | 67.1 | 74.7 | 92.3 |

### Tokens per target evaluation

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| mtp3-q8kv | 3.34 | 3.14 | 3.14 | 3.12 |
| mtp3-q8kv-patched | 3.36 | 3.09 | 3.24 | 3.16 |
| mtp3-f16kv-64k | 3.33 | 3.31 | 2.90 |  |
| mtp3-f16kv | 3.33 | 3.31 | 2.90 | 3.14 |
| mtp3-f16kv-patched | 3.34 | 3.18 | 3.14 | 3.12 |
| mtp3-q8kv-fattn | 3.36 | 3.09 | 3.24 | 3.16 |

### Cost of depth: ms added per target evaluation, relative to 2K

`ideal` is the time to read the KV cache once at the card's peak bandwidth (640 GB/s).

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 0.0 | 7.4 | 14.5 | 27.7 |
| mtp3-q8kv | 0.0 | 11.6 | 24.0 | 46.7 |
| mtp3-q8kv-patched | 0.0 | 11.4 | 23.4 | 47.7 |
| mtp3-f16kv-64k | 0.0 | 7.4 | 14.9 |  |
| plain-f16kv | 0.0 | 3.2 | 6.5 | 12.5 |
| mtp3-f16kv | 0.0 | 7.4 | 15.0 | 28.7 |
| mtp3-f16kv-patched | 0.0 | 7.1 | 14.3 | 27.8 |
| plain-q8kv-fattn | 0.0 | 3.7 | 7.6 | 15.0 |
| mtp3-q8kv-fattn | 0.0 | 10.3 | 17.9 | 35.5 |
| ideal, f16 KV | 0.0 | 3.2 | 6.4 | 12.3 |
| ideal, q8_0 KV | 0.0 | 1.7 | 3.4 | 6.6 |

### Prefill tok/s over each depth range (explain task; the prompt cache covers the rest)

| config | to 2K | to 32K | to 64K | to 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 828 | 751 | 609 | 482 |
| mtp3-q8kv | 858 | 774 | 618 | 480 |
| mtp3-q8kv-patched | 863 | 775 | 618 | 480 |
| mtp3-f16kv-64k | 863 | 779 | 622 |  |
| plain-f16kv | 822 | 750 | 615 | 489 |
| mtp3-f16kv | 869 | 778 | 624 | 486 |
| mtp3-f16kv-patched | 865 | 780 | 624 | 487 |
| plain-q8kv-fattn | 840 | 750 | 610 | 482 |
| mtp3-q8kv-fattn | 863 | 774 | 618 | 480 |

### Peak VRAM (MiB) and per-task rows

| config | peak VRAM | task | depth | tok/s | accepted / drafted |
|---|---:|---|---:|---:|---|
| plain-q8kv | 25323 | explain | 2010 | 23.7 | 0 / 0 |
| plain-q8kv | 25323 | code | 2009 | 24.1 | 0 / 0 |
| plain-q8kv | 25323 | explain | 32867 | 20.2 | 0 / 0 |
| plain-q8kv | 25323 | code | 32866 | 20.3 | 0 / 0 |
| plain-q8kv | 25323 | explain | 64108 | 17.7 | 0 / 0 |
| plain-q8kv | 25323 | code | 64107 | 17.7 | 0 / 0 |
| plain-q8kv | 25323 | explain | 122601 | 14.3 | 0 / 0 |
| plain-q8kv | 25323 | code | 122600 | 14.4 | 0 / 0 |
| mtp3-q8kv | 26946 | explain | 2010 | 51.1 | 173 / 245 |
| mtp3-q8kv | 26946 | code | 2009 | 59.8 | 185 / 209 |
| mtp3-q8kv | 26946 | explain | 32867 | 43.5 | 174 / 241 |
| mtp3-q8kv | 26946 | code | 32866 | 43.9 | 175 / 240 |
| mtp3-q8kv | 26946 | explain | 64108 | 32.9 | 164 / 272 |
| mtp3-q8kv | 26946 | code | 64107 | 41.7 | 183 / 214 |
| mtp3-q8kv | 26946 | explain | 122601 | 27.1 | 168 / 259 |
| mtp3-q8kv | 26946 | code | 122600 | 31.0 | 179 / 226 |
| mtp3-q8kv-patched | 26947 | explain | 2010 | 54.6 | 174 / 242 |
| mtp3-q8kv-patched | 26947 | code | 2009 | 63.1 | 185 / 209 |
| mtp3-q8kv-patched | 26947 | explain | 32867 | 43.8 | 171 / 251 |
| mtp3-q8kv-patched | 26947 | code | 32866 | 46.1 | 175 / 237 |
| mtp3-q8kv-patched | 26947 | explain | 64108 | 36.9 | 170 / 250 |
| mtp3-q8kv-patched | 26947 | code | 64107 | 43.5 | 183 / 214 |
| mtp3-q8kv-patched | 26947 | explain | 122601 | 28.4 | 169 / 256 |
| mtp3-q8kv-patched | 26947 | code | 122600 | 31.6 | 180 / 223 |
| mtp3-f16kv-64k | 26284 | explain | 2010 | 51.0 | 172 / 249 |
| mtp3-f16kv-64k | 26284 | code | 2009 | 60.5 | 185 / 209 |
| mtp3-f16kv-64k | 26284 | explain | 32867 | 46.5 | 174 / 241 |
| mtp3-f16kv-64k | 26284 | code | 32866 | 52.3 | 183 / 215 |
| mtp3-f16kv-64k | 26284 | explain | 64108 | 37.2 | 164 / 272 |
| mtp3-f16kv-64k | 26284 | code | 64107 | 40.4 | 171 / 250 |
| plain-f16kv | 28684 | explain | 2010 | 24.4 | 0 / 0 |
| plain-f16kv | 28684 | code | 2009 | 24.4 | 0 / 0 |
| plain-f16kv | 28684 | explain | 32867 | 22.6 | 0 / 0 |
| plain-f16kv | 28684 | code | 32866 | 22.6 | 0 / 0 |
| plain-f16kv | 28684 | explain | 64108 | 21.0 | 0 / 0 |
| plain-f16kv | 28684 | code | 64107 | 21.0 | 0 / 0 |
| plain-f16kv | 28684 | explain | 122601 | 18.6 | 0 / 0 |
| plain-f16kv | 28684 | code | 122600 | 18.7 | 0 / 0 |
| mtp3-f16kv | 30305 | explain | 2010 | 51.1 | 172 / 249 |
| mtp3-f16kv | 30305 | code | 2009 | 60.4 | 185 / 209 |
| mtp3-f16kv | 30305 | explain | 32867 | 46.5 | 174 / 241 |
| mtp3-f16kv | 30305 | code | 32866 | 52.3 | 183 / 215 |
| mtp3-f16kv | 30305 | explain | 64108 | 37.2 | 164 / 272 |
| mtp3-f16kv | 30305 | code | 64107 | 40.3 | 171 / 250 |
| mtp3-f16kv | 30305 | explain | 122601 | 32.4 | 167 / 262 |
| mtp3-f16kv | 30305 | code | 122600 | 38.7 | 181 / 219 |
| mtp3-f16kv-patched | 30306 | explain | 2010 | 54.4 | 173 / 245 |
| mtp3-f16kv-patched | 30306 | code | 2009 | 63.7 | 185 / 209 |
| mtp3-f16kv-patched | 30306 | explain | 32867 | 48.9 | 174 / 241 |
| mtp3-f16kv-patched | 30306 | code | 32866 | 50.9 | 177 / 231 |
| mtp3-f16kv-patched | 30306 | explain | 64108 | 39.1 | 164 / 272 |
| mtp3-f16kv-patched | 30306 | code | 64107 | 49.5 | 183 / 214 |
| mtp3-f16kv-patched | 30306 | explain | 122601 | 33.9 | 167 / 262 |
| mtp3-f16kv-patched | 30306 | code | 122600 | 39.9 | 180 / 223 |
| plain-q8kv-fattn | 25325 | explain | 2010 | 24.3 | 0 / 0 |
| plain-q8kv-fattn | 25325 | code | 2009 | 24.3 | 0 / 0 |
| plain-q8kv-fattn | 25325 | explain | 32867 | 22.3 | 0 / 0 |
| plain-q8kv-fattn | 25325 | code | 32866 | 22.3 | 0 / 0 |
| plain-q8kv-fattn | 25325 | explain | 64108 | 20.5 | 0 / 0 |
| plain-q8kv-fattn | 25325 | code | 64107 | 20.5 | 0 / 0 |
| plain-q8kv-fattn | 25325 | explain | 122601 | 17.8 | 0 / 0 |
| plain-q8kv-fattn | 25325 | code | 122600 | 17.8 | 0 / 0 |
| mtp3-q8kv-fattn | 26948 | explain | 2010 | 54.7 | 174 / 242 |
| mtp3-q8kv-fattn | 26948 | code | 2009 | 63.3 | 185 / 209 |
| mtp3-q8kv-fattn | 26948 | explain | 32867 | 43.8 | 171 / 251 |
| mtp3-q8kv-fattn | 26948 | code | 32866 | 48.0 | 175 / 237 |
| mtp3-q8kv-fattn | 26948 | explain | 64108 | 39.7 | 170 / 250 |
| mtp3-q8kv-fattn | 26948 | code | 64107 | 46.7 | 183 / 214 |
| mtp3-q8kv-fattn | 26948 | explain | 122601 | 31.7 | 169 / 256 |
| mtp3-q8kv-fattn | 26948 | code | 122600 | 36.4 | 180 / 223 |
