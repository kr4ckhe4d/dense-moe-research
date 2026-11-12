# Stock, this repo's patches and the RDNA fork on the two-card preset

## UD-Q6_K (pulled 2026-10-06)

Decode tok/s explain / code, ms per target evaluation, accepted / drafted tokens over both tasks, prefill tok/s over the first 120K, peak VRAM on both cards, and how many of the four outputs match the first row's text.

| build | 2K tok/s | 2K ms | 2K drafts | 120K tok/s | 120K ms | 120K drafts | prefill | VRAM MiB | same text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock-b12164 | 51.1 / 59.5 | 60.6 | 359 / 450 | 28.1 / 32.0 | 109.1 | 356 / 458 | 761 | 30400 | 4 of 4 |
| stock-a691 | 51.3 / 59.3 | 60.6 | 359 / 450 | 28.1 / 32.0 | 109.2 | 356 / 458 | 755 | 30403 | 4 of 4 |
| ours | 53.5 / 62.0 | 58.1 | 359 / 450 | 40.3 / 50.1 | 69.7 | 348 / 481 | 748 | 30656 | 3 of 4 |
| ours-sl65k-n4 | 59.2 / 75.6 | 57.3 | 378 / 523 | 43.6 / 56.6 | 72.9 | 370 / 556 | 716 | 31026 | 3 of 4 |
| fork-n3 | 61.4 / 68.2 | 52.6 | 362 / 440 | 45.6 / 55.7 | 62.7 | 350 / 477 | 667 | 29581 | 1 of 4 |
| fork-adaptive | 62.4 / 81.2 | 59.0 | 389 / 508 | 41.7 / 51.9 | 77.8 | 370 / 565 | 658 | 30218 | 1 of 4 |

## UD-Q4_K_XL v3.1

Decode tok/s explain / code, ms per target evaluation, accepted / drafted tokens over both tasks, prefill tok/s over the first 120K, peak VRAM on both cards, and how many of the four outputs match the first row's text.

| build | 2K tok/s | 2K ms | 2K drafts | 120K tok/s | 120K ms | 120K drafts | prefill | VRAM MiB | same text |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock-b12164 | 55.5 / 67.0 | 53.5 | 355 / 461 | 28.9 / 33.8 | 102.2 | 352 / 469 | 936 | 26621 | 4 of 4 |
| stock-a691 | 55.6 / 67.3 | 53.4 | 355 / 461 | 28.9 / 33.8 | 102.2 | 352 / 469 | 936 | 26619 | 4 of 4 |
| ours | 57.3 / 70.2 | 51.2 | 354 / 464 | 50.7 / 54.1 | 62.9 | 357 / 455 | 936 | 26668 | 2 of 4 |
| ours-sl65k-n4 | 69.4 / 85.1 | 51.7 | 383 / 500 | 52.4 / 58.9 | 67.6 | 376 / 531 | 888 | 27050 | 1 of 4 |
| fork-n3 | 68.7 / 78.4 | 45.2 | 358 / 454 | 56.9 / 59.9 | 55.3 | 354 / 463 | 719 | 25780 | 1 of 4 |
| fork-adaptive | 74.4 / 92.3 | 50.9 | 390 / 503 | 51.0 / 54.2 | 69.3 | 372 / 546 | 718 | 26416 | 1 of 4 |

## Fork settings on UD-Q4_K_XL v3.1 (benchmarks/grid_fork_tune.sh)

Decode tok/s explain / code. `ubN` = `-ub N`; `tensor` = `-sm tensor`; `vocabNk` = the fork's `--spec-draft-mtp-vocab`.

| config | 2K | 120K | ms at deepest | drafts at deepest | VRAM MiB |
|---|---:|---:|---:|---:|---:|
| ft-stock-ub2048 | 56.0 / 67.7 | 29.0 / 33.9 | 102.0 | 352 / 469 | 32962 |
| ft-fork-ub2048 | 63.3 / 79.5 | 57.0 / 60.7 | 55.2 | 355 / 460 | 27294 |
| ft-stock-ub4096 | 55.4 / 67.0 | 29.0 / 33.8 | 102.2 | 352 / 469 | 41631 |
| ft-fork-ub4096 | 63.5 / 79.5 | 57.0 / 63.1 | 55.3 | 358 / 453 | 29191 |
| ft-fork-adaptive-n3 | 68.8 / 78.5 | 57.0 / 59.9 | 55.3 | 354 / 463 | 25771 |
| ft-fork-adaptive-n4 | 67.2 / 80.8 | 53.7 / 58.7 | 61.5 | 364 / 494 | 25965 |
| ft-fork-adaptive-n5 | 67.2 / 81.6 | 53.4 / 58.4 | 61.7 | 364 / 494 | 26064 |
| ft-fork-n4 | 75.9 / 89.0 | 51.7 / 54.9 | 68.4 | 372 / 546 | 25937 |
| ft-fork-n4-vocab98k | 82.2 / 96.0 | 54.7 / 58.1 | 64.6 | 372 / 546 | 26334 |
| ft-fork-n4-vocab65k | 78.0 / 91.6 | 53.9 / 58.8 | 63.9 | 370 / 554 | 26194 |
| ft-fork-adaptive-n7-vocab98k | 80.5 / 96.0 | 54.0 / 57.3 | 65.5 | 372 / 546 | 26776 |
| ft-fork-ub512-tensor | 92.5 / 123.3 | 88.3 / 97.5 | 34.6 | 353 / 468 | 25363 |
| ft-fork-ub4096-tensor | 101.2 / 120.4 | 91.3 / 94.1 | 34.8 | 354 / 462 | 28038 |

## Layer split at 16K-64K, UD-Q4_K_XL v3.1

Decode tok/s explain / code.

| config | 2K | 16K | 32K | 64K | 120K | ms at deepest | drafts at deepest | VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mid-stock-n3 |  | 53.6 / 62.0 | 44.2 / 52.8 | 39.4 / 41.1 |  | 77.8 | 349 / 477 | 26883 |
| mid-fork-n3 |  | 65.2 / 77.3 | 64.0 / 70.9 | 57.5 / 68.4 |  | 50.4 | 350 / 477 | 25744 |
| mid-fork-n3-vocab98k | 72.4 / 82.4 | 71.1 / 81.2 | 68.0 / 76.5 | 64.7 / 71.7 | 60.8 / 63.2 | 52.4 | 355 / 461 | 26229 |
| mid-fork-n4-vocab98k |  | 65.2 / 92.9 | 64.5 / 78.6 | 56.8 / 75.4 |  | 54.6 | 368 / 561 | 26376 |
| mid-fork-adaptive-n7 |  | 60.3 / 87.7 | 59.0 / 72.8 | 52.4 / 70.5 |  | 59.8 | 370 / 567 | 26431 |
| mid-fork-adaptive-n7-vocab98k |  | 65.0 / 95.2 | 63.4 / 78.4 | 56.0 / 75.6 |  | 55.9 | 370 / 567 | 26841 |

## -sm tensor on every build (benchmarks/grid_tensor.sh)

Decode tok/s explain / code, draft 3 unless named.

| config | 2K | 16K | 32K | 64K | 120K | ms at deepest | drafts at deepest | VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q4xl-stock-tensor-n3 | 83.4 / 95.8 | 78.7 / 91.0 | 71.7 / 78.5 | 61.1 / 71.2 | 51.3 / 57.7 | 60.6 | 357 / 455 | 25724 |
| q4xl-fork-tensor-n3 | 93.0 / 123.7 | 110.7 / 120.4 | 100.1 / 111.0 | 94.7 / 108.2 | 90.9 / 99.0 | 34.5 | 356 / 456 | 25362 |
| q4xl-ours-tensor-n3 | 82.5 / 97.6 | 88.1 / 96.1 | 80.5 / 89.9 | 78.3 / 88.7 | 75.7 / 77.3 | 42.7 | 356 / 457 | 25867 |
| q6k-stock-tensor-n3 | 73.8 / 87.1 | 68.9 / 83.8 | 63.9 / 71.7 | 58.6 / 54.7 | 44.9 / 53.5 | 64.8 | 351 / 473 | 29721 |
| q6k-fork-tensor-n3 | 87.4 / 108.5 | 84.4 / 108.6 | 84.1 / 97.5 | 85.0 / 77.2 | 73.1 / 90.4 | 38.2 | 347 / 484 | 29167 |
| q4xl-fork-tensor-n3-vocab98k | 91.0 / 122.5 | 111.5 / 121.4 | 101.1 / 112.1 | 95.6 / 110.1 | 92.0 / 100.1 | 34.1 | 356 / 456 | 26156 |
| q4xl-fork-tensor-n4 | 96.4 / 139.6 | 113.4 / 135.0 | 96.5 / 119.5 | 87.9 / 113.5 | 85.6 / 95.9 | 41.1 | 375 / 535 | 25619 |
| q4xl-fork-tensor-adaptive-n7 | 94.7 / 143.7 | 111.6 / 136.8 | 94.6 / 111.8 | 86.6 / 116.3 | 84.5 / 93.3 | 42.0 | 375 / 535 | 26063 |
| q4xl-ours-tensor-sl65k-n4 | 25.8 / 26.4 | 25.0 / 24.9 | 24.5 / 24.6 | 23.6 / 23.5 | 21.4 / 21.6 | 46.5 | 2 / 2012 | 26565 |

