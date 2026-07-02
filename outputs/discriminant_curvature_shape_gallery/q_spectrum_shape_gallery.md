# Q Spectrum Shape Gallery

This is the spectrum companion to the discriminant-curvature gallery.  It applies the pullback principal Q matrix-free from complex boundary samples and reports Ritz eigenvalue levels for each shape.

![Q spectrum gallery](/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_spectrum_shape_gallery.png)

## Operator

```text
a_i = 1 / |dz/dtheta|_i
(Q_pull v)_i = a_i sum_{j != i} c_ij a_j (v_i - v_j)
c_ij = |exp(i theta_i) - exp(i theta_j)|^(-2)
```

No dense Q matrix is stored.  The dense object is only the small Lanczos tridiagonal used for the spectral readout.  Q application is the fast pullback-metric path: circle FFT kernel, metric-speed borrow, local repay.

The units are inverse normalized-radius squared.  All shapes are centered and scaled before Q is applied.

## Circle Calibration

For the unit circle with `n=256`, the low Q levels are `lambda_m=m(n-m)/2`, with the usual sine/cosine multiplicity away from the Nyquist mode.  The relative L2 error of the computed low Ritz levels against the first 12 exact multiplicity-aware levels is `1.067e-14`.

## Cost Model

The previous diagnostic applied Q by direct pairs, costing `n(n-1)/2` pair interactions per apply.  The new path stores only the pullback speed jet and uses the circle FFT kernel, so each apply costs about `O(n log n)` work units for radix-2 sample counts.

| shape | fast apply units | pairwise units | pairwise/fast | speed anisotropy |
|---|---:|---:|---:|---:|
| circle | `5632` | `32640` | `5.795e+00` | `1.000e+00` |
| golden_ellipse | `5632` | `32640` | `5.795e+00` | `1.342e+00` |
| rounded_square | `5632` | `32640` | `5.795e+00` | `1.036e+01` |
| flower_nonconvex | `5632` | `32640` | `5.795e+00` | `2.206e+00` |
| cardioid_one_cusp | `5632` | `32640` | `5.795e+00` | `1.629e+02` |
| nephroid_two_cusps | `5632` | `32640` | `5.795e+00` | `8.144e+01` |
| astroid_four_cusps | `5632` | `32640` | `5.795e+00` | `4.073e+01` |
| joukowski_airfoil | `5632` | `32640` | `5.795e+00` | `7.309e+01` |
| square_polygon | `5632` | `32640` | `5.795e+00` | `1.414e+00` |
| star_polygon | `5632` | `32640` | `5.795e+00` | `2.855e+00` |
| double_concave_stealth | `5632` | `32640` | `5.795e+00` | `5.050e+00` |
| rational_asymptote_chart | `5632` | `32640` | `5.795e+00` | `1.017e+00` |

For larger radix-2 runs the ratio grows quickly:

| n | fast apply units | pairwise units | pairwise/fast |
|---:|---:|---:|---:|
| `256` | `5632` | `32640` | `5.795e+00` |
| `1024` | `26624` | `523776` | `1.967e+01` |
| `4096` | `122880` | `8386560` | `6.825e+01` |
| `16384` | `557056` | `134209536` | `2.409e+02` |

## Eigenvalue Summary

| shape | lambda1 | lambda2 | lambda3 | lambda4 | lambda8 | lambda12 | largest Ritz | condition proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| circle | `1.275000e+02` | `1.275000e+02` | `2.540000e+02` | `2.540000e+02` | `5.040000e+02` | `7.500000e+02` | `8.191727e+03` | `6.424884e+01` |
| golden_ellipse | `1.672298e+02` | `1.676710e+02` | `3.284590e+02` | `3.316955e+02` | `6.515680e+02` | `9.672783e+02` | `1.457378e+04` | `8.714820e+01` |
| rounded_square | `2.270874e+02` | `3.270223e+02` | `4.038393e+02` | `4.211299e+02` | `5.995901e+02` | `8.718438e+02` | `3.176627e+04` | `1.398857e+02` |
| flower_nonconvex | `1.465923e+02` | `1.468913e+02` | `2.882551e+02` | `2.889387e+02` | `5.619520e+02` | `8.239596e+02` | `2.233399e+04` | `1.523545e+02` |
| cardioid_one_cusp | `3.014179e+02` | `3.550489e+02` | `4.673283e+02` | `6.217673e+02` | `1.236531e+03` | `1.788818e+03` | `6.389845e+07` | `2.119929e+05` |
| nephroid_two_cusps | `3.053946e+02` | `4.156202e+02` | `4.414383e+02` | `6.483326e+02` | `1.055178e+03` | `1.497661e+03` | `1.262839e+07` | `4.135106e+04` |
| astroid_four_cusps | `4.407150e+02` | `5.503562e+02` | `7.414820e+02` | `7.791726e+02` | `1.195530e+03` | `1.668986e+03` | `3.163953e+06` | `7.179138e+03` |
| joukowski_airfoil | `5.397307e+02` | `6.809141e+02` | `8.392435e+02` | `9.190423e+02` | `1.997018e+03` | `3.035983e+03` | `2.120802e+07` | `3.929370e+04` |
| square_polygon | `1.593170e+02` | `1.593170e+02` | `3.173192e+02` | `3.173612e+02` | `6.295677e+02` | `9.366979e+02` | `1.264113e+04` | `7.934576e+01` |
| star_polygon | `1.087484e+02` | `1.087504e+02` | `2.162847e+02` | `2.162927e+02` | `4.278540e+02` | `6.350278e+02` | `1.555415e+04` | `1.430287e+02` |
| double_concave_stealth | `2.606724e+02` | `2.606929e+02` | `5.182557e+02` | `5.183589e+02` | `1.025709e+03` | `1.523589e+03` | `6.703709e+04` | `2.571699e+02` |
| rational_asymptote_chart | `1.248684e+03` | `1.248707e+03` | `2.487537e+03` | `2.487623e+03` | `4.936010e+03` | `7.345162e+03` | `8.067754e+04` | `6.461006e+01` |

## Artifacts

- figure: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_spectrum_shape_gallery.png`
- Ritz CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_spectrum_shape_gallery_ritz.csv`
- summary CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_spectrum_shape_gallery_summary.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_spectrum_shape_gallery.json`
