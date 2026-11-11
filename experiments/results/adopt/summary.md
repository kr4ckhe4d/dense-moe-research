# Stock vs patched build on the daily two-card preset

## Code quality, 50 checks over 7 tasks, thinking off

| run | build | scores | mean | sd | answers that did not run |
|---|---|---|---:|---:|---:|
| temperature 0, 3 repeats | stock | 44 44 44 | 44.0 | 0 | 0 |
| temperature 0, 3 repeats | patched | 41 41 41 | 41.0 | 0 | 3 |
| temperature 1.0, seeds 1-12 | stock | 41 43 42 35 39 40 31 39 36 32 29 41 | 37.3 | 4.7 | 3 |
| temperature 1.0, seeds 1-12 | patched | 36 42 34 41 35 35 41 39 35 40 35 43 | 38.0 | 3.3 | 3 |

### Mean passes per task at temperature 1.0 (12 seeds)

| task | stock | patched |
|---|---:|---:|
| glob_match | 6.08 | 6.25 |
| shlex_split | 6.25 | 6.17 |
| urljoin | 4.08 | 4.42 |
| textwrap | 1.17 | 1.67 |
| csv_rows | 7.00 | 7.00 |
| add_months | 6.00 | 6.00 |
| parse_qsl | 6.75 | 6.50 |

## Depth profile on both cards (model split by llama-server), 27B UD-Q6_K

### Decode tok/s

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 23.5 | 19.9 | 17.1 | 13.8 |
| plain-q8kv-patched | 23.7 | 21.7 | 20.1 | 17.1 |
| mtp3-q8kv | 55.0 | 43.3 | 36.9 | 28.5 |
| mtp3-q8kv-patched | 57.7 | 45.6 | 41.5 | 32.7 |
| mtp3-f16kv | 55.1 | 48.9 | 38.5 | 34.9 |
| mtp3-f16kv-patched | 57.7 | 48.8 | 43.3 | 35.6 |

### ms per target evaluation

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| plain-q8kv | 42.3 | 50.0 | 58.4 | 72.0 |
| plain-q8kv-patched | 42.1 | 45.8 | 49.6 | 58.3 |
| mtp3-q8kv | 60.6 | 72.3 | 84.9 | 109.1 |
| mtp3-q8kv-patched | 58.1 | 67.5 | 77.8 | 96.2 |
| mtp3-f16kv | 60.2 | 67.5 | 74.9 | 89.9 |
| mtp3-f16kv-patched | 57.7 | 64.9 | 72.3 | 87.3 |

### Tokens per target evaluation

| config | 2K | 32K | 64K | 120K |
|---|---:|---:|---:|---:|
| mtp3-q8kv | 3.34 | 3.14 | 3.14 | 3.12 |
| mtp3-q8kv-patched | 3.36 | 3.09 | 3.24 | 3.16 |
| mtp3-f16kv | 3.33 | 3.31 | 2.90 | 3.14 |
| mtp3-f16kv-patched | 3.34 | 3.18 | 3.14 | 3.12 |

### Peak VRAM over both cards (MiB)

| config | peak |
|---|---:|
| plain-q8kv | 27882 |
| plain-q8kv-patched | 27902 |
| mtp3-q8kv | 30069 |
| mtp3-q8kv-patched | 30070 |
| mtp3-f16kv | 32956 |
| mtp3-f16kv-patched | 32960 |
