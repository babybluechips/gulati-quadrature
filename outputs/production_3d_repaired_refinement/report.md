# Repaired curved-panel 3D QJet refinement

## Protocol

All final traces are degree-four or degree-five solid harmonics. The adaptive compiler retains at most degree two and may inspect only degree three. Therefore every reported final mode is independent of both fitting and model selection.

The exact-sphere rows isolate singular quadrature because the radial chart lies exactly on the unit sphere and the continuum Q3 operator has DtN eigenvalue `l`. Ellipsoid and funky PN rows test the full bounded-remainder approximation and must not be read as singular-quadrature rates alone.

## Geometry and adaptivity

| Shape | Level | Nodes | Geometry | Boundary E | Nonmanifold E | Nonmanifold V | Seam gap | Selected d | Certified |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| exact_sphere | 0 | 72 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 0 | True |
| exact_sphere | 1 | 288 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 0 | True |
| exact_sphere | 2 | 1152 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 0 | True |
| exact_ellipsoid | 0 | 72 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 2 | False |
| exact_ellipsoid | 1 | 288 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 2 | False |
| exact_ellipsoid | 2 | 1152 | RadialQuadricTriangle | 0 | 0 | 0 | 0.000e+00 | 2 | False |
| funky_pn | 0 | 72 | CubicPNTriangle | 0 | 0 | 0 | 1.241e-16 | 2 | False |
| funky_pn | 1 | 288 | CubicPNTriangle | 0 | 0 | 0 | 2.238e-16 | 2 | False |
| funky_pn | 2 | 1152 | CubicPNTriangle | 0 | 0 | 0 | 2.483e-16 | 2 | False |

## Independent held-out refinement

| Shape | Level | Nodes | Max raw error | Max repaid error | Median repaid |
|---|---:|---:|---:|---:|---:|
| exact_sphere | 0 | 72 | 5.597e-01 | 1.087e+00 | 1.075e+00 |
| exact_sphere | 1 | 288 | 4.188e-01 | 4.946e-01 | 4.344e-01 |
| exact_sphere | 2 | 1152 | 4.825e-01 | 3.496e-01 | 3.281e-01 |
| exact_ellipsoid | 0 | 72 | 5.042e-01 | 2.526e+00 | 1.846e+00 |
| exact_ellipsoid | 1 | 288 | 4.169e-01 | 6.042e-01 | 5.158e-01 |
| exact_ellipsoid | 2 | 1152 | 4.763e-01 | 3.842e-01 | 3.463e-01 |
| funky_pn | 0 | 72 | 4.265e-01 | 1.242e+00 | 1.055e+00 |
| funky_pn | 1 | 288 | 5.579e-01 | 6.490e-01 | 5.156e-01 |
| funky_pn | 2 | 1152 | 5.253e-01 | 3.699e-01 | 3.168e-01 |

## Independently checked Mellin/Kondratiev feature moments

All edge and vertex channels are compiled with an order-twelve curved-panel rule and checked against a separate order-sixteen rule.

| Kind | Channel | Rung | Exponent | Raw abs. error | Repaid abs. error | Reference gap |
|---|---|---:|---:|---:|---:|---:|
| edge | edge_0 | 0 | 2.000000 | 4.293e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_0 | 1 | 2.000000 | 2.214e-03 | 5.551e-17 | 5.551e-17 |
| edge | edge_0 | 2 | 2.000000 | 3.847e-03 | 3.053e-16 | 3.053e-16 |
| edge | edge_0 | 3 | 2.000000 | 8.841e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_2 | 0 | 2.000000 | 1.935e-03 | 3.331e-16 | 3.331e-16 |
| edge | edge_2 | 1 | 2.000000 | 9.127e-03 | 3.886e-16 | 3.886e-16 |
| edge | edge_2 | 2 | 2.000000 | 1.747e-02 | 5.551e-17 | 5.551e-17 |
| edge | edge_2 | 3 | 2.000000 | 2.112e-02 | 1.110e-16 | 1.110e-16 |
| edge | edge_3 | 0 | 2.000000 | 4.293e-03 | 4.441e-16 | 4.441e-16 |
| edge | edge_3 | 1 | 2.000000 | 2.214e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_3 | 2 | 2.000000 | 3.847e-03 | 2.498e-16 | 2.498e-16 |
| edge | edge_3 | 3 | 2.000000 | 8.841e-03 | 9.714e-17 | 9.714e-17 |
| edge | edge_5 | 0 | 2.000000 | 1.548e-02 | 3.331e-16 | 3.331e-16 |
| edge | edge_5 | 1 | 2.000000 | 2.300e-02 | 2.776e-16 | 2.776e-16 |
| edge | edge_5 | 2 | 2.000000 | 2.545e-02 | 1.665e-16 | 1.665e-16 |
| edge | edge_5 | 3 | 2.000000 | 2.329e-02 | 0.000e+00 | 0.000e+00 |
| edge | edge_6 | 0 | 2.000000 | 4.293e-03 | 2.220e-16 | 2.220e-16 |
| edge | edge_6 | 1 | 2.000000 | 2.214e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_6 | 2 | 2.000000 | 3.847e-03 | 0.000e+00 | 0.000e+00 |
| edge | edge_6 | 3 | 2.000000 | 8.841e-03 | 6.939e-17 | 6.939e-17 |
| edge | edge_8 | 0 | 2.000000 | 1.935e-03 | 3.331e-16 | 3.331e-16 |
| edge | edge_8 | 1 | 2.000000 | 9.127e-03 | 3.331e-16 | 3.331e-16 |
| edge | edge_8 | 2 | 2.000000 | 1.747e-02 | 5.551e-17 | 5.551e-17 |
| edge | edge_8 | 3 | 2.000000 | 2.112e-02 | 0.000e+00 | 0.000e+00 |
| edge | edge_9 | 0 | 2.000000 | 4.293e-03 | 2.220e-16 | 2.220e-16 |
| edge | edge_9 | 1 | 2.000000 | 2.214e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_9 | 2 | 2.000000 | 3.847e-03 | 0.000e+00 | 0.000e+00 |
| edge | edge_9 | 3 | 2.000000 | 8.841e-03 | 6.939e-17 | 6.939e-17 |
| edge | edge_12 | 0 | 2.000000 | 4.293e-03 | 2.220e-16 | 2.220e-16 |
| edge | edge_12 | 1 | 2.000000 | 2.214e-03 | 1.110e-16 | 1.110e-16 |
| edge | edge_12 | 2 | 2.000000 | 3.847e-03 | 0.000e+00 | 0.000e+00 |
| edge | edge_12 | 3 | 2.000000 | 8.841e-03 | 6.939e-17 | 6.939e-17 |
| edge | edge_13 | 0 | 2.000000 | 1.548e-02 | 6.661e-16 | 6.661e-16 |
| edge | edge_13 | 1 | 2.000000 | 2.300e-02 | 1.665e-16 | 1.665e-16 |
| edge | edge_13 | 2 | 2.000000 | 2.545e-02 | 4.441e-16 | 4.441e-16 |
| edge | edge_13 | 3 | 2.000000 | 2.329e-02 | 1.665e-16 | 1.665e-16 |
| edge | edge_15 | 0 | 2.000000 | 7.728e-03 | 6.661e-16 | 6.661e-16 |
| edge | edge_15 | 1 | 2.000000 | 8.710e-03 | 5.551e-17 | 5.551e-17 |
| edge | edge_15 | 2 | 2.000000 | 9.425e-03 | 2.220e-16 | 2.220e-16 |
| edge | edge_15 | 3 | 2.000000 | 1.018e-02 | 0.000e+00 | 0.000e+00 |
| edge | edge_16 | 0 | 2.000000 | 1.548e-02 | 4.441e-16 | 4.441e-16 |
| edge | edge_16 | 1 | 2.000000 | 2.300e-02 | 1.110e-16 | 1.110e-16 |
| edge | edge_16 | 2 | 2.000000 | 2.545e-02 | 6.106e-16 | 6.106e-16 |
| edge | edge_16 | 3 | 2.000000 | 2.329e-02 | 2.220e-16 | 2.220e-16 |
| edge | edge_17 | 0 | 2.000000 | 2.903e-02 | 5.551e-16 | 5.551e-16 |
| edge | edge_17 | 1 | 2.000000 | 3.688e-02 | 5.551e-17 | 5.551e-17 |
| edge | edge_17 | 2 | 2.000000 | 3.344e-02 | 4.441e-16 | 4.441e-16 |
| edge | edge_17 | 3 | 2.000000 | 2.545e-02 | 2.220e-16 | 2.220e-16 |
| vertex | vertex_0 | 0 | 3.036737 | 1.233e-04 | 1.757e-07 | 1.757e-07 |
| vertex | vertex_0 | 1 | 3.036737 | 1.363e-03 | 1.837e-07 | 1.837e-07 |
| vertex | vertex_0 | 2 | 3.036737 | 8.618e-04 | 1.908e-07 | 1.908e-07 |
| vertex | vertex_0 | 3 | 3.036737 | 4.908e-05 | 1.980e-07 | 1.980e-07 |
| vertex | vertex_1 | 0 | 3.036737 | 2.611e-04 | 8.773e-08 | 8.773e-08 |
| vertex | vertex_1 | 1 | 3.036737 | 6.826e-04 | 9.187e-08 | 9.187e-08 |
| vertex | vertex_1 | 2 | 3.036737 | 3.142e-04 | 9.538e-08 | 9.538e-08 |
| vertex | vertex_1 | 3 | 3.036737 | 5.077e-05 | 9.900e-08 | 9.900e-08 |
| vertex | vertex_2 | 0 | 3.036737 | 3.155e-05 | 2.892e-07 | 2.892e-07 |
| vertex | vertex_2 | 1 | 3.036737 | 1.617e-04 | 2.686e-07 | 2.686e-07 |
| vertex | vertex_2 | 2 | 3.036737 | 7.974e-04 | 2.474e-07 | 2.474e-07 |
| vertex | vertex_2 | 3 | 3.036737 | 1.132e-03 | 2.250e-07 | 2.250e-07 |
| vertex | vertex_3 | 0 | 3.036737 | 2.555e-04 | 8.793e-08 | 8.793e-08 |
| vertex | vertex_3 | 1 | 3.036737 | 4.185e-04 | 9.185e-08 | 9.185e-08 |
| vertex | vertex_3 | 2 | 3.036737 | 4.360e-04 | 9.538e-08 | 9.538e-08 |
| vertex | vertex_3 | 3 | 3.036737 | 1.884e-04 | 9.900e-08 | 9.900e-08 |
| vertex | vertex_4 | 0 | 3.036737 | 4.851e-04 | 2.894e-07 | 2.894e-07 |
| vertex | vertex_4 | 1 | 3.036737 | 1.023e-04 | 2.686e-07 | 2.686e-07 |
| vertex | vertex_4 | 2 | 3.036737 | 9.192e-04 | 2.474e-07 | 2.474e-07 |
| vertex | vertex_4 | 3 | 3.036737 | 1.371e-03 | 2.250e-07 | 2.250e-07 |
| vertex | vertex_5 | 0 | 3.036737 | 1.472e-03 | 3.138e-06 | 3.138e-06 |
| vertex | vertex_5 | 1 | 3.036737 | 5.675e-05 | 3.198e-06 | 3.198e-06 |
| vertex | vertex_5 | 2 | 3.036737 | 1.202e-03 | 3.245e-06 | 3.245e-06 |
| vertex | vertex_5 | 3 | 3.036737 | 1.679e-03 | 3.278e-06 | 3.278e-06 |
| vertex | vertex_6 | 0 | 3.036737 | 3.108e-03 | 6.276e-06 | 6.276e-06 |
| vertex | vertex_6 | 1 | 3.036737 | 5.774e-04 | 6.396e-06 | 6.396e-06 |
| vertex | vertex_6 | 2 | 3.036737 | 2.073e-03 | 6.490e-06 | 6.490e-06 |
| vertex | vertex_6 | 3 | 3.036737 | 3.354e-03 | 6.556e-06 | 6.556e-06 |
| vertex | vertex_7 | 0 | 3.036737 | 2.186e-03 | 3.138e-06 | 3.138e-06 |
| vertex | vertex_7 | 1 | 3.036737 | 5.908e-04 | 3.198e-06 | 3.198e-06 |
| vertex | vertex_7 | 2 | 3.036737 | 1.231e-03 | 3.245e-06 | 3.245e-06 |
| vertex | vertex_7 | 3 | 3.036737 | 1.995e-03 | 3.278e-06 | 3.278e-06 |

The eight cube vertex pencils have exponent spread `8.882e-16`. Their maximum error against the exact octant exponent three is `3.674e-02`.

## Conclusion

The exact-sphere endpoint fit for the repaid maximum error is `0.818` in the node-spacing variable. This is evidence of refinement, not a machine-precision certificate. Feature basis moments are repaid to the error shown above. No dense Q matrix or global pair table is stored.

The arbitrary-domain held-out rows remain the controlling limitation. A failed adaptive validation flag means the configured degree-two bounded remainder did not meet its next-degree target; it is not converted into a pass by the final benchmark.
