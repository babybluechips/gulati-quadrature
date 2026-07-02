# Complex Fourier Q Spectrum Benchmark

This tests the paper's de Moivre route: project the pullback Q operator onto complex characters `exp(i k theta)`, apply Q by the FFT QJet, and diagonalize only a small Hermitian Fourier block.

![complex benchmark](/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_complex_fourier_spectrum_benchmark.png)

This is faster because it replaces two 124-step real Lanczos runs with direct probes of selected complex Fourier bands. It is exact on the circle when the requested modes are inside the block; on strongly singular shapes the high branch is a band probe and can differ from full Lanczos if the top mode is localized outside the selected Fourier window.

| shape | complex ms | real Lanczos ms | speedup | low rel L2 | high rel L2 |
|---|---:|---:|---:|---:|---:|
| circle | `104.361` | `7681.256` | `7.360e+01` | `1.373e-14` | `7.517e-03` |
| golden_ellipse | `1333.455` | `7362.088` | `5.521e+00` | `4.619e-14` | `1.613e-02` |
| rounded_square | `796.272` | `5732.578` | `7.199e+00` | `1.605e-01` | `2.129e-02` |
| flower_nonconvex | `3416.271` | `7632.619` | `2.234e+00` | `3.444e-05` | `1.868e-01` |
| cardioid_one_cusp | `3462.572` | `6465.576` | `1.867e+00` | `5.193e-01` | `1.174e+00` |
| nephroid_two_cusps | `1808.357` | `6417.043` | `3.549e+00` | `3.911e-01` | `1.170e+00` |
| astroid_four_cusps | `919.834` | `6260.718` | `6.806e+00` | `3.355e-01` | `1.146e+00` |
| joukowski_airfoil | `3874.421` | `6608.141` | `1.706e+00` | `3.645e-01` | `1.168e+00` |
| square_polygon | `430.223` | `7764.240` | `1.805e+01` | `2.424e-04` | `6.801e-02` |
| star_polygon | `1448.987` | `7519.686` | `5.190e+00` | `1.852e-03` | `3.337e-01` |
| double_concave_stealth | `1685.209` | `7592.555` | `4.505e+00` | `6.469e-04` | `4.512e-01` |
| rational_asymptote_chart | `2601.827` | `8080.472` | `3.106e+00` | `6.466e-10` | `6.398e-03` |

## Artifacts

- figure: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_complex_fourier_spectrum_benchmark.png`
- CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_complex_fourier_spectrum_benchmark.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_shape_gallery/q_complex_fourier_spectrum_benchmark.json`
