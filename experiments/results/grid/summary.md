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

