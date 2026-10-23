Gate 3 (single token) and stretch (MTP): sparse variant on base quant q vs the fastest dense quant with KLD <=
estimated combined KLD (KLD_q + KLD_sparse, additivity assumed, untested). Ceilings: cross-quant fit (optimistic).

| variant on quant | est. KLD | ceiling tok/s | best dense at <= KLD | its tok/s | gain | MTP ceiling | best dense+MTP at <= KLD | gain |
|---|---|---|---|---|---|---|---|---|
| gate B=1 keep=0.5 on Q8_0 | 0.0931 | 67.6 | IQ4_XS | 89.6 | -24.5% | 122.2 | IQ4_XS | -17.5% |
| gate B=1 keep=0.5 on Q6_K | 0.0943 | 81.9 | IQ4_XS | 89.6 | -8.6% | 123.4 | IQ4_XS | -16.6% |
| gate B=1 keep=0.5 on IQ4_XS | 0.1099 | 101.4 | IQ4_XS | 89.6 | +13.2% | 154.2 | IQ4_XS | +4.2% |
| gate B=16 keep=0.5 on Q8_0 | 0.2729 | 67.6 | IQ4_XS | 89.6 | -24.5% | 122.1 | IQ4_XS | -17.5% |
| gate B=16 keep=0.5 on Q6_K | 0.2741 | 81.9 | IQ4_XS | 89.6 | -8.6% | 123.4 | IQ4_XS | -16.6% |
| gate B=16 keep=0.5 on IQ4_XS | 0.2897 | 101.4 | IQ4_XS | 89.6 | +13.2% | 154.2 | IQ4_XS | +4.2% |
| teal B=1 keep=0.5 x_keep=0.5 xB=1 on Q8_0 | 0.0929 | 74.2 | IQ4_XS | 89.6 | -17.2% | 121.0 | IQ4_XS | -18.2% |
| teal B=1 keep=0.5 x_keep=0.5 xB=1 on Q6_K | 0.0941 | 89.2 | IQ4_XS | 89.6 | -0.4% | 122.3 | IQ4_XS | -17.3% |
| teal B=1 keep=0.5 x_keep=0.5 xB=1 on IQ4_XS | 0.1097 | 108.6 | IQ4_XS | 89.6 | +21.3% | 153.1 | IQ4_XS | +3.4% |
| wina B=1 keep=0.5 x_keep=0.5 xB=1 on Q8_0 | 0.0930 | 74.2 | IQ4_XS | 89.6 | -17.2% | 121.0 | IQ4_XS | -18.3% |
| wina B=1 keep=0.5 x_keep=0.5 xB=1 on Q6_K | 0.0942 | 89.2 | IQ4_XS | 89.6 | -0.4% | 122.3 | IQ4_XS | -17.4% |
| wina B=1 keep=0.5 x_keep=0.5 xB=1 on IQ4_XS | 0.1098 | 108.6 | IQ4_XS | 89.6 | +21.3% | 153.1 | IQ4_XS | +3.4% |
| wina B=1 keep=0.75 x_keep=0.75 xB=1 on Q8_0 | 0.0109 | 64.8 | Q6_K | 70.3 | -7.8% | 116.3 | Q6_K | -1.0% |
| wina B=1 keep=0.75 x_keep=0.75 xB=1 on Q6_K | 0.0121 | 78.6 | Q6_K | 70.3 | +11.9% | 117.8 | Q6_K | +0.3% |
| wina B=1 keep=0.75 x_keep=0.75 xB=1 on IQ4_XS | 0.0277 | 98.2 | IQ4_XS | 89.6 | +9.6% | 148.4 | IQ4_XS | +0.3% |
| teal B=1 keep=0.75 x_keep=0.75 xB=1 on Q8_0 | 0.0109 | 64.8 | Q6_K | 70.3 | -7.8% | 116.3 | Q6_K | -1.0% |
| teal B=1 keep=0.75 x_keep=0.75 xB=1 on Q6_K | 0.0121 | 78.6 | Q6_K | 70.3 | +11.9% | 117.8 | Q6_K | +0.3% |
| teal B=1 keep=0.75 x_keep=0.75 xB=1 on IQ4_XS | 0.0277 | 98.2 | IQ4_XS | 89.6 | +9.6% | 148.4 | IQ4_XS | +0.3% |
| wina B=1 keep=0.625 x_keep=0.625 xB=1 on Q8_0 | 0.0332 | 69.2 | IQ4_XS | 89.6 | -22.8% | 117.6 | IQ4_XS | -20.5% |
| wina B=1 keep=0.625 x_keep=0.625 xB=1 on Q6_K | 0.0344 | 83.6 | IQ4_XS | 89.6 | -6.7% | 119.1 | IQ4_XS | -19.5% |
| wina B=1 keep=0.625 x_keep=0.625 xB=1 on IQ4_XS | 0.0500 | 103.2 | IQ4_XS | 89.6 | +15.1% | 149.8 | IQ4_XS | +1.2% |
| wina B=1 keep=0.9 x_keep=0.9 xB=1 on Q8_0 | 0.0027 | 60.2 | Q8_0 | 57.5 | +4.7% | 115.9 | Q8_0 | +0.0% |
| wina B=1 keep=0.9 x_keep=0.9 xB=1 on Q6_K | 0.0039 | 73.4 | Q6_K | 70.3 | +4.4% | 117.4 | Q6_K | +0.0% |
| wina B=1 keep=0.9 x_keep=0.9 xB=1 on IQ4_XS | 0.0195 | 92.8 | IQ4_XS | 89.6 | +3.6% | 148.0 | IQ4_XS | +0.0% |
| wina B=1 keep=1.0 x_keep=0.5 xB=1 on Q8_0 | 0.0893 | 67.6 | IQ4_XS | 89.6 | -24.5% | 119.0 | IQ4_XS | -19.6% |
| wina B=1 keep=1.0 x_keep=0.5 xB=1 on Q6_K | 0.0905 | 81.9 | IQ4_XS | 89.6 | -8.6% | 120.4 | IQ4_XS | -18.6% |
| wina B=1 keep=1.0 x_keep=0.5 xB=1 on IQ4_XS | 0.1061 | 101.4 | IQ4_XS | 89.6 | +13.2% | 151.1 | IQ4_XS | +2.1% |
| wina B=32 keep=0.75 x_keep=0.75 xB=32 on Q8_0 | 0.4021 | 64.8 | IQ4_XS | 89.6 | -27.7% | 116.7 | IQ4_XS | -21.1% |
| wina B=32 keep=0.75 x_keep=0.75 xB=32 on Q6_K | 0.4033 | 78.6 | IQ4_XS | 89.6 | -12.2% | 118.2 | IQ4_XS | -20.1% |
| wina B=32 keep=0.75 x_keep=0.75 xB=32 on IQ4_XS | 0.4189 | 98.2 | IQ4_XS | 89.6 | +9.6% | 148.9 | IQ4_XS | +0.6% |
