# Corner Correction Benchmark

This run tests corner corrections on projected Q spectra, not on a stored dense boundary matrix.

![Corner correction benchmark](/Users/rick/Documents/New project 2/outputs/corner_correction_benchmark/corner_correction_benchmark.png)

## Protocol

- Raw Q: blockwise chord-kernel application with weighted boundary samples.
- Kondrat'ev/Mellin repayment: each vertex contributes a stored low-rank profile with exponent `lambda = pi / omega`, where `omega` is the interior angle.
- Hurwitz/zeta scaling: the corner amplitude uses an Euler-Maclaurin regularized `zeta(1-lambda, 1/2)` factor.
- Joukowsky prewarp: each polygon edge is sampled with `t_k = (1 - cos(pi k / m)) / 2`, clustering near panel endpoints.
- Reference: the same projected Q spectrum on a fine uniform boundary sample.

The correction is therefore judged by convergence of a small generalized Ritz spectrum.

## Latest Refinement

### `global_fourier`

| Shape | Probe dim | Raw err | KM default err | Joukowsky err | Combined default err | Best method | Best err | Raw/best | Corner lambdas |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| square_convex_corners | 12 | `1.142e-03` | `1.142e-03` | `1.498e-03` | `1.498e-03` | `raw` | `1.142e-03` | `1.00x` | `2.000, 2.000, 2.000, 2.000` |
| l_notch_reentrant | 12 | `1.017e-03` | `1.018e-03` | `1.222e-03` | `1.223e-03` | `raw` | `1.017e-03` | `1.00x` | `2.000, 2.000, 2.000, 0.667, 2.000, 2.000` |
| star_alternating_corners | 12 | `7.362e-04` | `7.367e-04` | `1.342e-03` | `1.342e-03` | `raw` | `7.362e-04` | `1.00x` | `4.588, 0.724, 4.588, 0.724, 4.588, 0.724, 4.588, 0.724, 4.588, 0.724` |
| stealth_double_concave | 12 | `8.386e-04` | `8.384e-04` | `1.189e-03` | `1.189e-03` | `kondratiev_mellin_best_grid` | `8.365e-04` | `1.00x` | `4.128, 1.080, 7.768, 0.755, 0.755, 7.768, 1.080` |

### `corner_augmented`

| Shape | Probe dim | Raw err | KM default err | Joukowsky err | Combined default err | Best method | Best err | Raw/best | Corner lambdas |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| square_convex_corners | 16 | `4.322e-03` | `4.322e-03` | `1.227e-03` | `1.227e-03` | `joukowsky_prewarp` | `1.227e-03` | `3.52x` | `2.000, 2.000, 2.000, 2.000` |
| l_notch_reentrant | 22 | `3.577e-01` | `3.577e-01` | `3.322e-01` | `3.322e-01` | `joukowsky_prewarp` | `3.322e-01` | `1.08x` | `2.000, 2.000, 2.000, 0.667, 2.000, 2.000` |
| star_alternating_corners | 32 | `2.122e-01` | `2.122e-01` | `1.684e-01` | `1.684e-01` | `joukowsky_prewarp` | `1.684e-01` | `1.26x` | `4.588, 0.724, 4.588, 0.724, 4.588, 0.724, 4.588, 0.724, 4.588, 0.724` |
| stealth_double_concave | 26 | `2.845e-01` | `2.845e-01` | `2.718e-01` | `2.718e-01` | `joukowsky_prewarp` | `2.718e-01` | `1.05x` | `4.128, 1.080, 7.768, 0.755, 0.755, 7.768, 1.080` |


## Reference Spectra

| Shape | Basis | Probe dim | Reference n | Normalized Ritz values, first modes | Raw Ritz values, first modes |
|---|---|---:|---:|---|---|
| square_convex_corners | `global_fourier` | 12 | 3072 | `0.14618789, 0.13246900, 0.11805558, 0.11805558, 0.10186663, 0.08860063` | `5.48031172, 4.96601607, 4.42568354, 4.42568354, 3.81879016, 3.32147254` |
| square_convex_corners | `corner_augmented` | 16 | 3072 | `0.14647319, 0.11374394, 0.10788628, 0.10788628, 0.07662748, 0.06943523` | `10.47559949, 8.13484004, 7.71590673, 7.71590673, 5.48031172, 4.96592987` |
| l_notch_reentrant | `global_fourier` | 12 | 3072 | `0.14387108, 0.13192800, 0.11953300, 0.11319382, 0.09826635, 0.09077701` | `5.31291846, 4.87188038, 4.41415379, 4.18005832, 3.62881174, 3.35224316` |
| l_notch_reentrant | `corner_augmented` | 22 | 3072 | `0.44018269, 0.05588433, 0.05364464, 0.04583163, 0.04460268, 0.04084775` | `89.96242133, 11.42137156, 10.96363382, 9.36684810, 9.11568130, 8.34826597` |
| star_alternating_corners | `global_fourier` | 12 | 3072 | `0.14217226, 0.14217222, 0.14061923, 0.09315409, 0.09315407, 0.08219855` | `9.32275682, 9.32275444, 9.22091891, 6.10845581, 6.10845426, 5.39006047` |
| star_alternating_corners | `corner_augmented` | 32 | 3072 | `0.14815570, 0.14279042, 0.14278671, 0.13003953, 0.13003306, 0.02358588` | `133.16400268, 128.34163313, 128.33830142, 116.88098468, 116.87517703, 21.19925618` |
| stealth_double_concave | `global_fourier` | 12 | 3072 | `0.20453493, 0.16995042, 0.10988593, 0.09244097, 0.08777831, 0.07480524` | `15.94007255, 13.24478896, 8.56376815, 7.20422553, 6.84084913, 5.82981570` |
| stealth_double_concave | `corner_augmented` | 26 | 3072 | `0.22708564, 0.15346186, 0.08836033, 0.08424097, 0.04810660, 0.04169436` | `120.76528921, 81.61179272, 46.99047208, 44.79977306, 25.58333371, 22.17327085` |

## Interpretation

Verdict from the latest refinement:

- Smooth global probes: raw weighted Q is already the best or tied-best result; corner corrections do not improve the smooth projected spectrum.
- Corner-augmented probes: the Joukowsky/Chebyshev endpoint prewarp is the useful correction in this run.
- The current positive low-rank Kondrat'ev/Mellin/Hurwitz repayment is effectively neutral. The best positive weight grid usually selects zero, so this layer should not yet be called load-bearing for corner spectra.

The default Kondrat'ev/Mellin repayment is deliberately fixed rather than fitted. The `*_best_grid` columns in the JSON/CSV show how much calibration headroom exists if the corner amplitude is tuned against a reference.

When `lambda < 1`, the domain has a reentrant corner and the singular corner channel is genuinely strong. Convex right-angle corners have `lambda > 1`; in those cases the low-rank repayment is expected to be weaker and endpoint placement usually matters more than the zeta amplitude.

The Joukowsky row here is a sampling/preconditioning test, not a full exterior Riemann-map solve. It answers whether endpoint clustering helps the same Q spectrum before installing a heavier conformal pullback.

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/corner_correction_benchmark/corner_correction_benchmark.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/corner_correction_benchmark/corner_correction_benchmark.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/corner_correction_benchmark/corner_correction_benchmark.png`
