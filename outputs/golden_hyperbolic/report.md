# Golden hyperbolic normalization benchmark

The golden coordinate is the scaled Cayley map

```text
tau = sqrt(5) (g(w)-i)/(g(w)+i).
```

It maps the whole hyperbolic upper half-plane to `|tau| < sqrt(5)` and sends signed distance `+/-2 log(phi)` to `+/-1`.

## Integer-trace optimality

| trace | half length | disk radius | Bernstein radius | order at 2e-16 |
|---:|---:|---:|---:|---:|
| 3 | 0.962424 | 2.236068 | 4.236068 | 26 |
| 4 | 1.316958 | 1.732051 | 3.146264 | 32 |
| 5 | 1.566799 | 1.527525 | 2.682226 | 37 |
| 6 | 1.762747 | 1.414214 | 2.414214 | 42 |
| 7 | 1.924847 | 1.341641 | 2.236068 | 45 |
| 8 | 2.063437 | 1.290994 | 2.107491 | 49 |
| 9 | 2.184644 | 1.253566 | 2.009495 | 52 |
| 10 | 2.292432 | 1.224745 | 1.931852 | 55 |
| 11 | 2.389526 | 1.201850 | 1.868517 | 58 |
| 12 | 2.477889 | 1.183216 | 1.815671 | 61 |

Trace three uniquely maximizes both radii. Its Bernstein factor is `sqrt(5)+2=phi^3`; order 24 has geometric factor `8.972e-16`.

## Frame invariance

| shape | inverse error | distance error | strip error | disk fraction |
|---|---:|---:|---:|---:|
| cylinder | 4.965e-16 | 2.217e-16 | 4.434e-16 | 0.410365 |
| cone | 4.175e-16 | 0.000e+00 | 4.434e-16 | 0.460977 |
| sphere_cap | 5.237e-16 | 2.219e-16 | 2.219e-16 | 0.300827 |
| corrugated | 5.467e-16 | 2.217e-16 | 4.437e-16 | 0.373545 |
| double_neck | 4.578e-16 | 0.000e+00 | 4.434e-16 | 0.475768 |
| cusp_right_chart | 3.511e-16 | 0.000e+00 | 4.439e-16 | 0.375289 |
| airfoil_body | 3.832e-16 | 0.000e+00 | 2.218e-16 | 0.459502 |

## Three-jet atlas

| shape | source nodes | patches | total length | max patch | Q error |
|---|---:|---:|---:|---:|---:|
| long_geodesic | 17 | 2 | 3.849695 | 1.924847 | 4.058e-16 |
| polynomial_meridian | 17 | 1 | 1.356912 | 1.356912 | 9.917e-16 |
| corrugated_meridian | 25 | 2 | 1.958881 | 1.881444 | 1.646e-14 |

## Interpretation

Uniform hyperbolic sampling removes parameter-induced node clustering. Golden compactification keeps that property while fixing a canonical algebraic interval and analytic collar. A coordinate change on the same badly clustered nodes does not repair the sampling; the atlas must generate the nodes from the normalized jets.

The tetration input is only the fixed-point multiplier `1/(2 phi)`. It fixes `phi^-2 = 4 mu_T^2`; the geometric map itself is the branch-free Cayley/PSL(2,R) normalization, not raw complex tetration.

## Production execution contract

Production objects expose only fixed-rank QJet applies. Static compilation is capped at `64N` pair visits, `16N` block records, and `64N` exact local pairs per mode. A cap violation raises; there is no quadratic fallback. Streamed pairwise oracles live only in `inverse_shape.testing`.
