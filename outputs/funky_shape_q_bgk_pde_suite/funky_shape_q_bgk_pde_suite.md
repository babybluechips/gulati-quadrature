# Funky Shape Q+BGK/PDE Suite

This tests the combined slick path on a wider exact-Laurent shape suite, then sends the same conformal-coordinate data through the boundary-only PDE pipeline.

![funky Q BGK PDE suite](/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_q_bgk_pde_suite.png)

## Shape Diagnostics

| Shape | Family | Anisotropy | Intersections |
|---|---|---:|---:|
| circle | baseline | `1.000` | `0` |
| golden_ellipse | closed_form_conic | `1.342` | `0` |
| eccentric_ellipse | closed_form_conic | `2.500` | `0` |
| three_petal | smooth_nonconvex | `3.167` | `0` |
| four_lobed_wiggle | smooth_multiscale | `4.302` | `0` |
| asymmetric_limaçon | asymmetric | `2.253` | `0` |
| crescent_teardrop | near_cusp_smooth | `2.644` | `0` |
| peanut | pinched_smooth | `2.416` | `0` |
| high_frequency_gear | smooth_high_frequency | `3.460` | `0` |
| rotated_mixed | complex_coefficients | `2.200` | `0` |

## BGK Ladder

| Shape | Raw | BGK-1 | BGK-2 | BGK-4 | BGK-8 |
|---|---:|---:|---:|---:|---:|
| circle | `1.975e-03` | `6.889e-08` | `2.335e-12` | `2.148e-15` | `2.148e-15` |
| golden_ellipse | `8.066e-05` | `2.975e-09` | `1.243e-13` | `1.144e-16` | `1.144e-16` |
| eccentric_ellipse | `1.428e-04` | `5.965e-09` | `3.429e-13` | `4.488e-16` | `4.488e-16` |
| three_petal | `1.000e-03` | `5.226e-08` | `3.768e-12` | `2.241e-15` | `2.241e-15` |
| four_lobed_wiggle | `1.007e-03` | `6.342e-08` | `6.043e-12` | `2.630e-15` | `2.630e-15` |
| asymmetric_limaçon | `4.397e-02` | `2.037e-06` | `1.309e-10` | `1.160e-13` | `1.160e-13` |
| crescent_teardrop | `8.082e-04` | `1.773e-08` | `6.630e-13` | `4.238e-15` | `4.238e-15` |
| peanut | `5.063e-02` | `2.561e-06` | `1.840e-10` | `2.146e-13` | `2.146e-13` |
| high_frequency_gear | `1.359e-03` | `8.534e-08` | `1.069e-11` | `1.594e-14` | `1.594e-14` |
| rotated_mixed | `8.227e-04` | `2.279e-09` | `4.457e-12` | `1.305e-14` | `1.305e-14` |

Max BGK-4 relative error across the shape suite: `2.146e-13`.

## PDE Pipeline

| Problem | Max continuum error | Max generated-Q residual |
|---|---:|---:|
| laplace_dtn | `3.418e-03` | `5.820e-13` |
| heat | `4.076e-03` | `4.361e-15` |
| poisson | `3.266e-03` | `4.473e-15` |
| helmholtz | `9.561e-03` | `4.369e-15` |
| wave | `5.967e-03` | `4.303e-15` |

Max PDE continuum relative error across all shapes/problems: `9.561e-03`.
Max generated-Q implementation residual across all shapes/problems: `5.820e-13`.

For Laplace DtN the continuum comparison is against physical conformal-map flux `m cos(m theta)/|psi'|`. For heat, Poisson, Helmholtz, and wave the continuum comparison is against the exact unit-circle modal multiplier after borrowing the conformal coordinate. The generated-Q residual compares against the finite Q spectrum actually used by the engine.

## Artifacts

- Shape diagnostics CSV: `/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_diagnostics.csv`
- BGK rows CSV: `/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_bgk_ladder_rows.csv`
- PDE rows CSV: `/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_pde_rows.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_q_bgk_pde_suite.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/funky_shape_q_bgk_pde_suite/funky_shape_q_bgk_pde_suite.png`
