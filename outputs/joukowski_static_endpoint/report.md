# Static Joukowski endpoint benchmark

The inverse-square and normalized inverse-cube physical chords are compiled from the exact Joukowski factorization. No target/source pair table is retained. The singular difference coordinate is handled by the foundational QJet FFT; the smooth sum-coordinate quotient is a finite geometric channel list whose tail is checked before application.

## Headline

- golden ellipse maximum static/direct error: `6.824e-14`
- maximum Cartesian-reference cancellation diagnostic: `4.003e-14`
- exterior annulus Q2 maximum static/direct error: `5.022e-15`
- normalized exterior annulus Q3 maximum static/direct error: `1.423e-14`
- conic-surface same-slice error: `3.229e-15`
- fitted static apply exponent: `1.100`
- fitted direct exponent: `2.023`
- golden modulation channels: `39`
- golden quotient tail bound: `6.667e-18`
- dense matrices stored: `no`

## Golden scaling

| nodes | compile ms | static apply ms | direct ms | stable error | Cartesian reference loss | dense entries avoided |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.440 | 1.212 | 1.374 | `1.583e-15` | `1.025e-15` | 1024 |
| 64 | 8.879 | 2.435 | 5.497 | `3.993e-15` | `2.830e-15` | 4096 |
| 128 | 18.355 | 5.394 | 22.600 | `7.437e-15` | `4.017e-15` | 16384 |
| 256 | 37.257 | 11.468 | 98.523 | `1.285e-14` | `1.176e-14` | 65536 |
| 512 | 78.757 | 25.553 | 381.204 | `3.348e-14` | `1.993e-14` | 262144 |
| 1024 | 165.072 | 53.184 | 1473.002 | `6.824e-14` | `4.003e-14` | 1048576 |
| 2048 | 320.295 | 114.189 | 0.000 | `0.000e+00` | `0.000e+00` | 4194304 |
| 4096 | 658.175 | 244.527 | 0.000 | `0.000e+00` | `0.000e+00` | 16777216 |

## Eccentricity audit

| chart | mu | eccentricity | channels | tail | error |
|---|---:|---:|---:|---:|---:|
| nearly_circular | 2.00000 | 0.26580 | 19 | `1.586e-19` | `7.607e-15` |
| golden | 0.96242 | 0.66667 | 39 | `6.667e-18` | `7.437e-15` |
| moderately_slender | 0.55000 | 0.86572 | 67 | `6.419e-17` | `7.200e-15` |
| near_cusp_but_regular | 0.30000 | 0.95663 | 125 | `1.335e-16` | `6.943e-15` |

## Two-dimensional exterior chart

| power | grid | nodes | static ms | direct ms | error | channels |
|---:|---|---:|---:|---:|---:|---:|
| 2 | 2x8 | 16 | 8.074 | 0.087 | `5.107e-16` | 289 |
| 2 | 4x16 | 64 | 28.806 | 1.091 | `2.379e-15` | 324 |
| 2 | 8x32 | 256 | 99.364 | 19.807 | `5.022e-15` | 324 |
| 3 | 2x8 | 16 | 9.282 | 0.088 | `3.948e-15` | 324 |
| 3 | 4x16 | 64 | 33.332 | 1.296 | `5.805e-15` | 361 |
| 3 | 8x32 | 256 | 112.968 | 18.125 | `1.423e-14` | 361 |

## Scope

This implementation removes the `1e-3` Barnes-Hut error on the compiled Joukowski channels. The 3D conic integration currently replaces same-slice singular interactions only. A complete arbitrary curved-surface operator still needs statically compiled cross-slice chart residuals, tangent-cell repayment, and the lower-order DtN geometry channel. Near `mu=0`, the geometric series is rejected and the map must hand off to the Mellin cusp channels rather than increasing the Fourier rank without bound.
