# Discriminant three-jet benchmark

The generated root-of-unity path uses the foundational QJet FFT and the fixed-width weighted discriminant contraction. It stores no dense matrix and selects no numerical rank.

## Scaling and accuracy

| N | build + first ms | generated ms | contraction ms | direct ms | rel. error | direct / generated |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5.030e-01 | 2.508e-01 | 1.196e-01 | 1.236e-01 | 2.626e-15 | 0.49x |
| 64 | 9.525e-01 | 5.515e-01 | 2.557e-01 | 4.968e-01 | 6.052e-15 | 0.90x |
| 128 | 1.717e+00 | 1.016e+00 | 4.658e-01 | 1.988e+00 | 1.172e-14 | 1.96x |
| 256 | 3.492e+00 | 1.957e+00 | 9.507e-01 | 7.790e+00 | 2.235e-14 | 3.98x |
| 512 | 7.400e+00 | 4.171e+00 | 1.861e+00 | 3.224e+01 | 5.257e-14 | 7.73x |
| 1024 | 1.498e+01 | 8.785e+00 | 3.632e+00 | 1.277e+02 | 1.103e-13 | 14.54x |
| 2048 | 3.036e+01 | 1.822e+01 | 7.489e+00 | - | - | - |
| 4096 | 6.147e+01 | 3.748e+01 | 1.497e+01 | - | - | - |

The fitted exponents are 1.026 for the full generator plus contraction, 0.988 for the pre-generated contraction, and 2.003 for the quadratic streamed reference.

## Non-radix-two audit

| N | transform | build + first ms | generated ms | direct ms | relative error |
|---:|---|---:|---:|---:|---:|
| 150 | mixed_radix | 4.788e+00 | 2.583e+00 | 2.635e+00 | 1.900e-14 |
| 151 | bluestein | 5.098e+00 | 4.137e+00 | 2.702e+00 | 1.499e-13 |
| 300 | mixed_radix | 8.180e+00 | 5.324e+00 | 1.079e+01 | 3.877e-14 |
| 600 | mixed_radix | 1.688e+01 | 1.193e+01 | 4.340e+01 | 8.009e-14 |

## Metric closure audit

| geometry | best scalar | relative pair-kernel residual | univariate closure |
|---|---:|---:|:---:|
| common_circle | 1-1.46e-17i | 1.002e-15 | yes |
| ellipse | 0.215544+4.54e-16i | 7.988e-01 | no |
| scale_phase_spiral | 0.771809-1.63e-15i | 7.193e-01 | no |

The common-circle Euclidean kernel closes exactly through the holomorphic three-jet. A scalar copy of that closure fails on the ellipse and varying-radius curve. Those geometries require their Schwarz/Joukowski or bivariate resultant generator; the three-jet contraction by itself does not remove that requirement.
