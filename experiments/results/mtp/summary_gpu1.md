# MTP tuning: mtp_tune_gpu1.jsonl

660 generations, 512 tokens each, GPU 1. Speed is the harmonic mean over prompts; acceptance = accepted / drafted.

## Temperature 0.0

### BF16 (KLD 0.0000), no speculation 34.3 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=1 | 50.0 | 1.46x | 86% | 1.85 | 50 | 48 | 49 | 52 | 49 | 53 | 6/6 |
| draft-mtp n_max=2 | 51.1 | 1.49x | 74% | 2.47 | 54 | 47 | 47 | 56 | 46 | 59 | 6/6 |
| draft-mtp n_max=3 | 49.9 | 1.46x | 65% | 2.92 | 54 | 46 | 42 | 58 | 43 | 62 | 2/6 |
| draft-mtp n_max=4 | 50.3 | 1.47x | 56% | 3.22 | 55 | 43 | 42 | 63 | 43 | 67 | 4/6 |
| draft-mtp n_max=4 p_min=0.3 | 51.2 | 1.50x | 62% | 3.15 | 54 | 44 | 45 | 62 | 45 | 67 | 3/6 |
| draft-mtp n_max=4 p_min=0.6 | 51.6 | 1.51x | 86% | 2.62 | 54 | 45 | 44 | 62 | 46 | 66 | 4/6 |
| draft-mtp n_max=4 p_min=0.8 | 48.6 | 1.42x | 95% | 2.26 | 53 | 42 | 41 | 59 | 44 | 62 | 4/6 |
| draft-mtp n_max=5 | 49.0 | 1.43x | 49% | 3.41 | 54 | 41 | 40 | 64 | 41 | 69 | 4/6 |
| draft-mtp n_max=6 | 47.4 | 1.39x | 43% | 3.54 | 55 | 38 | 38 | 64 | 38 | 72 | 4/6 |
| draft-mtp n_max=6 p_min=0.3 | 50.2 | 1.47x | 53% | 3.46 | 55 | 42 | 41 | 63 | 43 | 72 | 3/6 |
| draft-mtp n_max=6 p_min=0.6 | 52.6 | 1.54x | 82% | 2.79 | 55 | 45 | 44 | 66 | 46 | 71 | 4/6 |
| draft-mtp n_max=6 p_min=0.8 | 49.7 | 1.45x | 94% | 2.36 | 53 | 42 | 41 | 60 | 44 | 71 | 4/6 |
| draft-mtp n_max=8 | 43.1 | 1.26x | 34% | 3.71 | 50 | 35 | 32 | 60 | 34 | 71 | 6/6 |
| draft-mtp n_max=8 p_min=0.3 | 47.8 | 1.40x | 46% | 3.60 | 50 | 41 | 39 | 59 | 40 | 71 | 5/6 |
| draft-mtp n_max=8 p_min=0.6 | 52.4 | 1.53x | 79% | 2.85 | 54 | 45 | 44 | 65 | 46 | 73 | 6/6 |
| draft-mtp n_max=8 p_min=0.8 | 49.8 | 1.45x | 93% | 2.40 | 53 | 42 | 40 | 61 | 44 | 73 | 6/6 |
| none | 34.3 | 1.00x | - | 1.00 | 35 | 34 | 34 | 34 | 34 | 34 | 6/6 |

### Q8_0 (KLD 0.0016), no speculation 58.8 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=1 | 84.4 | 1.43x | 87% | 1.86 | 87 | 81 | 81 | 87 | 82 | 89 | 1/6 |
| draft-mtp n_max=2 | 99.4 | 1.69x | 76% | 2.51 | 107 | 94 | 89 | 108 | 92 | 112 | 1/6 |
| draft-mtp n_max=3 | 104.8 | 1.78x | 66% | 2.98 | 114 | 95 | 88 | 120 | 94 | 129 | 1/6 |
| draft-mtp n_max=3 n_min=2 | 105.1 | 1.79x | 66% | 2.98 | 115 | 96 | 89 | 120 | 94 | 130 | 1/6 |
| draft-mtp n_max=4 | 104.6 | 1.78x | 59% | 3.33 | 116 | 87 | 88 | 129 | 92 | 136 | 1/6 |
| draft-mtp n_max=4 n_min=3 | 104.6 | 1.78x | 59% | 3.33 | 116 | 87 | 88 | 130 | 92 | 136 | 1/6 |
| draft-mtp n_max=4 p_min=0.3 | 102.8 | 1.75x | 65% | 3.21 | 115 | 85 | 83 | 128 | 93 | 135 | 1/6 |
| draft-mtp n_max=4 p_min=0.6 | 95.6 | 1.63x | 88% | 2.69 | 109 | 86 | 77 | 117 | 79 | 128 | 1/6 |
| draft-mtp n_max=4 p_min=0.8 | 85.7 | 1.46x | 96% | 2.26 | 97 | 72 | 68 | 110 | 74 | 118 | 1/6 |
| draft-mtp n_max=5 | 100.1 | 1.70x | 51% | 3.51 | 110 | 83 | 81 | 128 | 88 | 136 | 2/6 |
| draft-mtp n_max=6 | 94.6 | 1.61x | 45% | 3.68 | 109 | 75 | 75 | 124 | 81 | 138 | 2/6 |
| draft-mtp n_max=6 p_min=0.3 | 95.9 | 1.63x | 54% | 3.48 | 106 | 87 | 73 | 122 | 80 | 138 | 1/6 |
| draft-mtp n_max=6 p_min=0.6 | 98.5 | 1.67x | 85% | 2.94 | 110 | 89 | 75 | 125 | 85 | 134 | 1/6 |
| draft-mtp n_max=6 p_min=0.8 | 85.7 | 1.46x | 94% | 2.31 | 90 | 72 | 67 | 117 | 72 | 132 | 0/6 |
| draft-mtp n_max=8 | 84.8 | 1.44x | 35% | 3.75 | 98 | 67 | 61 | 119 | 74 | 136 | 0/6 |
| draft-mtp n_max=8 p_min=0.3 | 91.1 | 1.55x | 47% | 3.58 | 104 | 71 | 72 | 119 | 79 | 137 | 0/6 |
| draft-mtp n_max=8 p_min=0.6 | 96.6 | 1.64x | 82% | 2.95 | 108 | 78 | 76 | 128 | 83 | 141 | 0/6 |
| draft-mtp n_max=8 p_min=0.8 | 85.8 | 1.46x | 94% | 2.36 | 90 | 72 | 69 | 114 | 71 | 133 | 0/6 |
| draft-mtp,ngram-map-k n_max=3 | 104.5 | 1.78x | 63% | 2.97 | 114 | 95 | 89 | 120 | 94 | 126 | 1/6 |
| draft-mtp,ngram-mod n_max=3 | 104.8 | 1.78x | 66% | 2.98 | 114 | 95 | 88 | 119 | 94 | 130 | 1/6 |
| ngram-mod n_max=3 | 58.9 | 1.00x | 5% | 1.00 | 59 | 59 | 59 | 59 | 59 | 59 | 6/6 |
| none | 58.8 | 1.00x | - | 1.00 | 59 | 59 | 59 | 59 | 59 | 59 | 6/6 |

### Q6_K (KLD 0.0028), no speculation 70.9 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=1 | 96.0 | 1.35x | 86% | 1.86 | 95 | 92 | 92 | 101 | 95 | 102 | 1/6 |
| draft-mtp n_max=2 | 112.5 | 1.59x | 76% | 2.51 | 113 | 109 | 100 | 124 | 106 | 129 | 1/6 |
| draft-mtp n_max=3 | 113.7 | 1.60x | 66% | 2.95 | 114 | 105 | 96 | 133 | 105 | 140 | 1/6 |
| draft-mtp n_max=4 | 107.8 | 1.52x | 58% | 3.31 | 118 | 96 | 89 | 134 | 94 | 136 | 1/6 |
| draft-mtp n_max=4 p_min=0.3 | 108.3 | 1.53x | 64% | 3.19 | 122 | 94 | 84 | 132 | 98 | 143 | 0/6 |
| draft-mtp n_max=4 p_min=0.6 | 104.7 | 1.48x | 87% | 2.63 | 116 | 89 | 85 | 127 | 94 | 138 | 1/6 |
| draft-mtp n_max=4 p_min=0.8 | 95.3 | 1.34x | 95% | 2.22 | 103 | 79 | 80 | 119 | 85 | 125 | 1/6 |
| draft-mtp n_max=5 | 98.9 | 1.39x | 50% | 3.48 | 110 | 87 | 79 | 125 | 83 | 133 | 1/6 |
| draft-mtp n_max=6 | 91.4 | 1.29x | 45% | 3.64 | 101 | 80 | 71 | 119 | 75 | 136 | 1/6 |
| draft-mtp n_max=6 p_min=0.3 | 97.0 | 1.37x | 53% | 3.50 | 101 | 84 | 79 | 121 | 86 | 136 | 0/6 |
| draft-mtp n_max=6 p_min=0.6 | 101.6 | 1.43x | 83% | 2.71 | 105 | 91 | 84 | 125 | 89 | 137 | 1/6 |
| draft-mtp n_max=6 p_min=0.8 | 95.3 | 1.34x | 94% | 2.28 | 98 | 77 | 80 | 121 | 87 | 135 | 0/6 |
| draft-mtp n_max=8 | 90.0 | 1.27x | 35% | 3.75 | 103 | 75 | 64 | 128 | 75 | 148 | 0/6 |
| draft-mtp n_max=8 p_min=0.3 | 97.5 | 1.37x | 48% | 3.64 | 94 | 91 | 77 | 120 | 86 | 144 | 0/6 |
| draft-mtp n_max=8 p_min=0.6 | 103.0 | 1.45x | 81% | 2.80 | 104 | 86 | 88 | 127 | 91 | 146 | 0/6 |
| draft-mtp n_max=8 p_min=0.8 | 95.8 | 1.35x | 94% | 2.33 | 97 | 77 | 80 | 123 | 86 | 139 | 1/6 |
| none | 70.9 | 1.00x | - | 1.00 | 71 | 71 | 71 | 71 | 71 | 71 | 6/6 |

### Q4_K_M (KLD 0.0228), no speculation 84.8 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=1 | 109.7 | 1.29x | 86% | 1.86 | 113 | 107 | 104 | 114 | 104 | 117 | 2/6 |
| draft-mtp n_max=2 | 126.6 | 1.49x | 76% | 2.51 | 134 | 125 | 111 | 138 | 115 | 143 | 2/6 |
| draft-mtp n_max=3 | 124.2 | 1.46x | 66% | 2.97 | 140 | 118 | 103 | 143 | 105 | 152 | 2/6 |
| draft-mtp n_max=4 | 114.1 | 1.35x | 57% | 3.25 | 127 | 100 | 91 | 138 | 103 | 150 | 1/6 |
| draft-mtp n_max=4 p_min=0.3 | 116.2 | 1.37x | 64% | 3.16 | 124 | 110 | 91 | 139 | 103 | 151 | 1/6 |
| draft-mtp n_max=4 p_min=0.6 | 114.6 | 1.35x | 87% | 2.59 | 128 | 100 | 98 | 138 | 97 | 148 | 1/6 |
| draft-mtp n_max=4 p_min=0.8 | 107.9 | 1.27x | 95% | 2.25 | 119 | 97 | 87 | 133 | 94 | 139 | 2/6 |
| draft-mtp n_max=5 | 103.9 | 1.23x | 49% | 3.42 | 116 | 90 | 80 | 128 | 92 | 145 | 1/6 |
| draft-mtp n_max=6 | 95.2 | 1.12x | 44% | 3.60 | 110 | 80 | 71 | 122 | 82 | 141 | 1/6 |
| draft-mtp n_max=6 p_min=0.3 | 105.0 | 1.24x | 55% | 3.55 | 112 | 96 | 83 | 124 | 97 | 138 | 1/6 |
| draft-mtp n_max=6 p_min=0.6 | 114.7 | 1.35x | 85% | 2.82 | 125 | 100 | 96 | 138 | 100 | 150 | 1/6 |
| draft-mtp n_max=6 p_min=0.8 | 105.8 | 1.25x | 94% | 2.29 | 115 | 93 | 85 | 132 | 92 | 143 | 2/6 |
| draft-mtp n_max=8 | 94.0 | 1.11x | 34% | 3.66 | 106 | 79 | 72 | 133 | 73 | 152 | 0/6 |
| draft-mtp n_max=8 p_min=0.3 | 105.4 | 1.24x | 49% | 3.62 | 119 | 89 | 80 | 134 | 94 | 150 | 1/6 |
| draft-mtp n_max=8 p_min=0.6 | 116.2 | 1.37x | 83% | 2.90 | 130 | 103 | 95 | 140 | 99 | 155 | 0/6 |
| draft-mtp n_max=8 p_min=0.8 | 106.7 | 1.26x | 93% | 2.35 | 114 | 92 | 85 | 133 | 94 | 151 | 1/6 |
| none | 84.8 | 1.00x | - | 1.00 | 85 | 85 | 85 | 85 | 85 | 85 | 6/6 |

### IQ4_XS (KLD 0.0184), no speculation 91.0 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=1 | 116.4 | 1.28x | 85% | 1.85 | 118 | 112 | 110 | 122 | 113 | 124 | 0/6 |
| draft-mtp n_max=2 | 134.2 | 1.47x | 75% | 2.48 | 142 | 125 | 118 | 147 | 126 | 154 | 0/6 |
| draft-mtp n_max=3 | 135.9 | 1.49x | 64% | 2.90 | 148 | 123 | 115 | 158 | 119 | 172 | 0/6 |
| draft-mtp n_max=3 n_min=2 | 136.0 | 1.49x | 64% | 2.90 | 148 | 123 | 115 | 157 | 119 | 172 | 0/6 |
| draft-mtp n_max=4 | 130.4 | 1.43x | 55% | 3.20 | 152 | 114 | 100 | 162 | 114 | 175 | 0/6 |
| draft-mtp n_max=4 n_min=3 | 130.5 | 1.43x | 55% | 3.20 | 153 | 114 | 100 | 161 | 113 | 175 | 0/6 |
| draft-mtp n_max=4 p_min=0.3 | 131.0 | 1.44x | 63% | 3.15 | 145 | 114 | 108 | 160 | 113 | 175 | 0/6 |
| draft-mtp n_max=4 p_min=0.6 | 127.0 | 1.40x | 88% | 2.65 | 149 | 109 | 101 | 156 | 112 | 166 | 1/6 |
| draft-mtp n_max=4 p_min=0.8 | 116.9 | 1.28x | 96% | 2.27 | 123 | 102 | 96 | 150 | 101 | 158 | 0/6 |
| draft-mtp n_max=5 | 120.0 | 1.32x | 48% | 3.39 | 141 | 104 | 88 | 159 | 104 | 168 | 0/6 |
| draft-mtp n_max=6 | 111.8 | 1.23x | 43% | 3.54 | 135 | 95 | 81 | 149 | 94 | 168 | 0/6 |
| draft-mtp n_max=6 p_min=0.3 | 116.8 | 1.28x | 52% | 3.36 | 129 | 101 | 98 | 149 | 94 | 165 | 0/6 |
| draft-mtp n_max=6 p_min=0.6 | 125.3 | 1.38x | 85% | 2.80 | 140 | 108 | 103 | 154 | 106 | 170 | 0/6 |
| draft-mtp n_max=6 p_min=0.8 | 116.1 | 1.28x | 94% | 2.34 | 125 | 100 | 94 | 143 | 100 | 165 | 1/6 |
| draft-mtp n_max=8 | 106.3 | 1.17x | 35% | 3.74 | 121 | 95 | 74 | 151 | 85 | 171 | 1/6 |
| draft-mtp n_max=8 p_min=0.3 | 112.4 | 1.24x | 46% | 3.46 | 122 | 94 | 88 | 149 | 94 | 172 | 0/6 |
| draft-mtp n_max=8 p_min=0.6 | 124.9 | 1.37x | 82% | 2.88 | 131 | 111 | 101 | 157 | 108 | 173 | 0/6 |
| draft-mtp n_max=8 p_min=0.8 | 115.0 | 1.26x | 94% | 2.35 | 122 | 99 | 94 | 147 | 96 | 165 | 0/6 |
| draft-mtp,ngram-map-k n_max=3 | 136.1 | 1.49x | 59% | 2.91 | 148 | 123 | 115 | 158 | 120 | 171 | 0/6 |
| draft-mtp,ngram-mod n_max=3 | 135.9 | 1.49x | 64% | 2.90 | 148 | 123 | 115 | 157 | 119 | 172 | 0/6 |
| ngram-mod n_max=3 | 90.8 | 1.00x | 0% | 1.00 | 91 | 91 | 91 | 91 | 91 | 90 | 5/6 |
| none | 91.0 | 1.00x | - | 1.00 | 91 | 91 | 91 | 91 | 91 | 91 | 6/6 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| BF16 | 0.0000 | 34.3 | draft-mtp n_max=6 p_min=0.6 | 52.6 | 1.54x | prose 44 | json 71 |
| Q8_0 | 0.0016 | 58.8 | draft-mtp n_max=3 n_min=2 | 105.1 | 1.79x | prose 89 | json 130 |
| Q6_K | 0.0028 | 70.9 | draft-mtp n_max=3 | 113.7 | 1.60x | prose 96 | json 140 |
| Q4_K_M | 0.0228 | 84.8 | draft-mtp n_max=2 | 126.6 | 1.49x | prose 111 | json 143 |
| IQ4_XS | 0.0184 | 91.0 | draft-mtp,ngram-map-k n_max=3 | 136.1 | 1.49x | prose 115 | json 171 |

## Temperature 0.7

### Q8_0 (KLD 0.0016), no speculation 58.9 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 96.4 | 1.64x | 72% | 2.43 | 105 | 88 | 82 | 110 | 89 | 114 | 0/6 |
| draft-mtp n_max=2 --spec-draft-sampling probabilistic | 98.9 | 1.68x | 75% | 2.49 | 109 | 89 | 89 | 112 | 89 | 113 | 0/6 |
| draft-mtp n_max=3 | 102.3 | 1.74x | 64% | 2.91 | 116 | 90 | 83 | 121 | 91 | 130 | 0/6 |
| draft-mtp n_max=3 --spec-draft-sampling probabilistic | 103.7 | 1.76x | 65% | 2.95 | 118 | 97 | 84 | 123 | 89 | 127 | 0/6 |
| none | 58.9 | 1.00x | - | 1.00 | 59 | 59 | 59 | 59 | 59 | 59 | 6/6 |

### Q6_K (KLD 0.0028), no speculation 71.0 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 109.2 | 1.54x | 72% | 2.44 | 120 | 100 | 97 | 123 | 96 | 129 | 0/6 |
| draft-mtp n_max=2 --spec-draft-sampling probabilistic | 111.1 | 1.57x | 74% | 2.48 | 119 | 104 | 99 | 122 | 103 | 125 | 0/6 |
| draft-mtp n_max=3 | 108.7 | 1.53x | 62% | 2.83 | 124 | 95 | 92 | 131 | 92 | 141 | 0/6 |
| draft-mtp n_max=3 --spec-draft-sampling probabilistic | 111.7 | 1.57x | 64% | 2.91 | 116 | 103 | 88 | 135 | 107 | 137 | 0/6 |
| none | 71.0 | 1.00x | - | 1.00 | 71 | 71 | 71 | 71 | 71 | 71 | 6/6 |

### IQ4_XS (KLD 0.0184), no speculation 90.8 tok/s

| config | tok/s | vs none | accept | tokens/step | code_python | code_cpp | prose | math | instruction | json | same text as none |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-mtp n_max=2 | 130.7 | 1.44x | 71% | 2.42 | 141 | 125 | 107 | 145 | 124 | 153 | 0/6 |
| draft-mtp n_max=2 --spec-draft-sampling probabilistic | 133.0 | 1.46x | 74% | 2.46 | 146 | 125 | 117 | 148 | 118 | 153 | 0/6 |
| draft-mtp n_max=3 | 131.5 | 1.45x | 61% | 2.81 | 151 | 126 | 98 | 155 | 121 | 164 | 0/6 |
| draft-mtp n_max=3 --spec-draft-sampling probabilistic | 136.5 | 1.50x | 64% | 2.91 | 155 | 126 | 112 | 161 | 117 | 169 | 0/6 |
| none | 90.8 | 1.00x | - | 1.00 | 91 | 91 | 91 | 91 | 91 | 91 | 6/6 |

### Best config per quant

| quant | KLD | none tok/s | best config | tok/s | vs none | worst domain | best domain |
|---|---|---|---|---|---|---|---|
| Q8_0 | 0.0016 | 58.9 | draft-mtp n_max=3 --spec-draft-sampling probabilistic | 103.7 | 1.76x | prose 84 | json 127 |
| Q6_K | 0.0028 | 71.0 | draft-mtp n_max=3 --spec-draft-sampling probabilistic | 111.7 | 1.57x | prose 88 | json 137 |
| IQ4_XS | 0.0184 | 90.8 | draft-mtp n_max=3 --spec-draft-sampling probabilistic | 136.5 | 1.50x | prose 112 | json 169 |

