# 3D Mellin--Kondratiev corner convergence

This campaign separates the continuum corner error from the discrete surface-graph compression error. The references are independent adaptive integrals after a radial power substitution that removes the known singular exponent.

## Pencil audit

| link h-level | spherical nodes | interior dofs | lambda_h | |lambda_h-lambda_*| | coupled error at 512 | residual | solve ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 63 | 51 | 0.483632884544 | 2.946e-02 | 1.223e-06 | 5.673e-10 | 6.067 |
| 6 | 136 | 118 | 0.468830048973 | 1.466e-02 | 6.325e-07 | 4.386e-10 | 19.320 |
| 8 | 237 | 213 | 0.463132323303 | 8.959e-03 | 3.925e-07 | 1.077e-09 | 46.856 |
| 12 | 523 | 487 | 0.458691380321 | 4.518e-03 | 2.003e-07 | 4.781e-10 | 157.923 |
| 16 | 921 | 873 | 0.456976429333 | 2.803e-03 | 1.248e-07 | 4.743e-10 | 386.283 |

The reference exponent is `0.45417371533061`. The local angular problem is assembled as sparse P1 stiffness and mass rows; no dense eigenmatrix is formed.
The raw level-16 exponent leaves a `1.248e-07` coupled radial error. A three-level power extrapolation gives `lambda=0.454309671992` and `6.097e-09` error. The machine-scale vertex row below uses the held-out published exponent, not the finite P1 estimate.

The two continuum reference values are each recomputed after two different radial substitutions. Their absolute discrepancies are `8.327e-17` (edge) and `1.318e-16` (vertex).

## Layer-potential convergence

| case | expected raw power | fitted raw | expected 3-jet power | fitted corrected | finest raw error | finest corrected error |
|---|---:|---:|---:|---:|---:|---:|
| re-entrant edge | 0.666667 | 0.674 | 4.666667 | 4.698 | 3.470e-04 | 8.500e-16 |
| Fichera vertex | 1.454174 | 1.444 | 5.454174 | 5.431 | 7.374e-06 | 4.452e-14 |

The edge test is a 3D Laplace single-layer contribution on one face of a 270-degree prism; its tangential integral is evaluated in closed form. The vertex test is a face-sector contribution at the Fichera exponent, including the required surface-measure shift from `lambda` to `lambda+1`. Both use a localized eighth-order cutoff so the displayed slope isolates the corner endpoint.

Each correction stores four amplitude coefficients and has `O(1)` apply cost per retained edge or vertex mode. The global smooth surface backend has separate complexity accounting.
