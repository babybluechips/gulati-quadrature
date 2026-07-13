# Nested scale-phase Cauchy benchmark

The production path stores one scale tree and sparse block endpoint records. Fixed-size Chebyshev transfer and interaction tiles are compiled once and reused; no `N x N` table is formed.

| n-scale | nested ms | direct ms | rel. error | compressed pairs | cluster records | block records |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 7.150e+00 | 5.002e+00 | 6.513e-16 | 0.476190 | 15 | 32 |
| 128 | 1.735e+01 | 2.018e+01 | 7.735e-16 | 0.716535 | 31 | 81 |
| 256 | 3.893e+01 | 8.104e+01 | 9.392e-16 | 0.852941 | 63 | 186 |
| 512 | 8.322e+01 | 3.242e+02 | 2.317e-15 | 0.925147 | 127 | 404 |
| 1024 | 1.727e+02 | not run | not run | 0.962243 | 255 | 847 |

The fitted nested-mode exponent is 1.145; the direct reference exponent over its measured range is 2.006. Persistent pointwise pair factors remain zero. The retained interaction tiles have fixed `32 x 32` size, so their total storage is linear in the scale-node count.

| total nodes | full QJet ms | rel. error | constant residual |
|---:|---:|---:|---:|
| 128 | 6.757e+00 | 3.731e-16 | 0.000e+00 |
| 256 | 2.040e+01 | 4.917e-16 | 0.000e+00 |
| 512 | 5.805e+01 | 1.536e-15 | 0.000e+00 |
| 1024 | 1.402e+02 | 3.925e-15 | 0.000e+00 |
| 2048 | 3.164e+02 | not run | 0.000e+00 |
| 4096 | 6.716e+02 | not run | 0.000e+00 |
| 8192 | 1.410e+03 | not run | 0.000e+00 |

The full QJet exponent is 1.273. These tests audit the exact continuum angular Fourier operator. A finite angular pair sum has additional alias rungs and is deliberately not used as the reference.
