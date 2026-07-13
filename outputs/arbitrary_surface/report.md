# Certified arbitrary-surface QJet benchmark

All rows use the topology-free weighted-node API. Separated blocks use fixed-order analytic Gegenbauer moments; only adjacent leaf blocks are evaluated directly. No rejected far block, adaptive rank, pair table, or dense matrix is present.

| shape | N | compile ms | hierarchy ms | direct ms | rel. error | compressed pairs | exact pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| sphere | 96 | 5.794e+02 | 4.761e+02 | 3.600e+00 | 5.215e-16 | 0.268 | 0.732 |
| ellipsoid | 96 | 6.692e+02 | 5.299e+02 | 3.075e+00 | 1.020e-15 | 0.310 | 0.690 |
| torus | 96 | 7.551e+02 | 6.075e+02 | 3.106e+00 | 1.184e-15 | 0.403 | 0.597 |
| folded_sheet | 96 | 7.686e+02 | 5.957e+02 | 3.218e+00 | 8.689e-16 | 0.462 | 0.538 |
| mobius | 96 | 8.225e+02 | 6.434e+02 | 3.398e+00 | 1.607e-15 | 0.821 | 0.179 |
| star_surface | 96 | 5.359e+02 | 4.618e+02 | 3.398e+00 | 7.767e-16 | 0.095 | 0.905 |
| spherical_spiral | 256 | 2.330e+03 | 1.740e+03 | 2.140e+01 | 4.824e-15 | 0.927 | 0.073 |

## Two-dimensional surface scaling

| N | hierarchy ms | direct ms | rel. error | exact pairs | analytic units | far blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 2.042e+02 | 8.097e-01 | 3.092e-16 | 1.000 | 0 | 0 |
| 96 | 4.878e+02 | 3.131e+00 | 5.215e-16 | 0.732 | 382884 | 120 |
| 192 | 1.498e+03 | 1.254e+01 | 1.001e-15 | 0.558 | 3255024 | 612 |
| 384 | 6.074e+03 | 5.033e+01 | 8.798e-16 | 0.250 | 19317156 | 2942 |

The fitted apply exponent is 1.630; the streamed reference exponent is 1.988. The compiled analytic-work and far-block slopes over the same finite range are 2.828 and 2.308. Finite-range slopes are measurements, not asymptotic proofs. The compiled backend enforces the asymptotic contract directly through fixed expansion order, linear persistent moments, a symmetric WSPD, exact terminal leaves, and explicit near-field and analytic-work budgets.

Production gate: PASS; maximum relative error 4.824e-15, measured apply slope 1.630 < 1.9, no dense matrix, and no quadratic fallback.

The streamed direct reference is isolated in the testing module and is not callable through the production QJet API.
