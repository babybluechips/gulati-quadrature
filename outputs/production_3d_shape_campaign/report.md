# Production 3D shape QJet campaign

All rows apply weighted inverse-distance graph operators without forming a global distance or operator matrix. Audit references are isolated streamed pair sums and are not production code paths.

## Golden axisymmetric surfaces

| shape | p | nodes | compile ms | apply ms | error | constant | hard cap |
|---|---:|---:|---:|---:|---:|---:|---|
| cylinder | 2 | 384 | 1.083e+01 | 2.009e+01 | 2.302e-15 | 0.000e+00 | yes |
| cone | 2 | 384 | 9.886e+00 | 2.030e+01 | 2.204e-15 | 0.000e+00 | yes |
| sphere_cap | 2 | 384 | 9.905e+00 | 2.019e+01 | 3.302e-15 | 0.000e+00 | yes |
| corrugated | 2 | 384 | 9.891e+00 | 2.044e+01 | 6.222e-14 | 0.000e+00 | yes |
| hourglass | 2 | 384 | 9.752e+00 | 2.052e+01 | 2.086e-15 | 0.000e+00 | yes |
| airfoil_meridian | 2 | 384 | 9.844e+00 | 2.048e+01 | 1.749e-15 | 0.000e+00 | yes |

## Curved conic and aircraft surfaces

| shape | p | nodes | compile ms | apply ms | error | constant | hard cap |
|---|---:|---:|---:|---:|---:|---:|---|
| circular_cylinder | 2 | 192 | 2.130e+03 | 1.554e+03 | 6.423e-16 | 0.000e+00 | yes |
| circular_cylinder | 3 | 192 | 1.867e+03 | 1.433e+03 | 1.198e-15 | 0.000e+00 | yes |
| elliptic_taper | 2 | 192 | 2.052e+03 | 1.514e+03 | 8.035e-16 | 0.000e+00 | yes |
| elliptic_taper | 3 | 192 | 1.938e+03 | 1.480e+03 | 2.054e-15 | 0.000e+00 | yes |
| bent_tube | 2 | 192 | 2.022e+03 | 1.522e+03 | 5.532e-16 | 0.000e+00 | yes |
| bent_tube | 3 | 192 | 2.020e+03 | 1.491e+03 | 1.314e-15 | 0.000e+00 | yes |
| twisted_ellipse | 2 | 192 | 2.190e+03 | 1.619e+03 | 5.891e-16 | 0.000e+00 | yes |
| twisted_ellipse | 3 | 192 | 2.109e+03 | 1.566e+03 | 1.642e-15 | 0.000e+00 | yes |
| toroidal_bundle | 2 | 192 | 2.116e+03 | 1.575e+03 | 1.773e-15 | 0.000e+00 | yes |
| toroidal_bundle | 3 | 192 | 2.257e+03 | 1.683e+03 | 5.277e-15 | 0.000e+00 | yes |
| smooth_aircraft_body | 2 | 192 | 1.913e+03 | 1.460e+03 | 7.867e-16 | 0.000e+00 | yes |
| smooth_aircraft_body | 3 | 192 | 1.827e+03 | 1.408e+03 | 3.609e-15 | 0.000e+00 | yes |

## Refined polyhedral surfaces

| shape | p | nodes | compile ms | apply ms | error | constant | hard cap |
|---|---:|---:|---:|---:|---:|---:|---|
| tetrahedron | 2 | 34 | 1.614e+02 | 1.134e+02 | 3.782e-16 | 0.000e+00 | yes |
| tetrahedron | 3 | 34 | 1.569e+02 | 1.116e+02 | 6.171e-16 | 0.000e+00 | yes |
| cube | 2 | 98 | 4.510e+02 | 3.956e+02 | 1.072e-15 | 0.000e+00 | yes |
| cube | 3 | 98 | 4.433e+02 | 3.933e+02 | 1.081e-15 | 0.000e+00 | yes |
| concave_dented_cube | 2 | 114 | 4.498e+02 | 4.008e+02 | 8.253e-16 | 0.000e+00 | yes |
| concave_dented_cube | 3 | 114 | 4.416e+02 | 3.948e+02 | 1.365e-15 | 0.000e+00 | yes |
| octahedron | 2 | 66 | 2.655e+02 | 2.126e+02 | 6.152e-16 | 0.000e+00 | yes |
| octahedron | 3 | 66 | 2.598e+02 | 2.166e+02 | 1.009e-15 | 0.000e+00 | yes |
| wing_body_aircraft | 2 | 514 | 8.292e+03 | 5.808e+03 | 1.971e-15 | 0.000e+00 | yes |
| wing_body_aircraft | 3 | 514 | 8.183e+03 | 5.669e+03 | 3.084e-14 | 0.000e+00 | yes |

## Unstructured curved surfaces

| shape | p | nodes | compile ms | apply ms | error | constant | hard cap |
|---|---:|---:|---:|---:|---:|---:|---|
| folded_sheet | 2 | 96 | 7.261e+02 | 5.636e+02 | 7.439e-16 | 0.000e+00 | yes |
| folded_sheet | 3 | 96 | 6.997e+02 | 5.602e+02 | 1.979e-15 | 0.000e+00 | yes |
| mobius_strip | 2 | 96 | 8.122e+02 | 6.224e+02 | 1.326e-15 | 0.000e+00 | yes |
| mobius_strip | 3 | 96 | 7.935e+02 | 6.107e+02 | 3.427e-15 | 0.000e+00 | yes |

## Scaling fits

| family | compile | apply | tail compile | tail apply | reference |
|---|---:|---:|---:|---:|---:|
| axisymmetric_golden | 1.892e+00 | 1.416e+00 | 1.275e+00 | 1.180e+00 | 2.038e+00 |
| curved_conic_production_wspd | 1.739e+00 | 1.664e+00 | 1.882e+00 | 1.794e+00 | not run |
| polyhedral_hierarchy | 1.399e+00 | 1.452e+00 | 2.051e+00 | 1.847e+00 | 1.979e+00 |

## Curved-atlas tolerance sweep

| tolerance | admissibility | error | exact pairs | low-rank pairs |
|---:|---:|---:|---:|---:|
| 1.000e-09 | 0.30 | 1.224e-13 | 0.851449 | 0.065217 |
| 1.000e-10 | 0.30 | 1.038e-14 | 0.851449 | 0.065217 |
| 1.000e-10 | 0.60 | 2.362e-14 | 0.786232 | 0.130435 |
| 1.000e-12 | 0.60 | 3.238e-16 | 0.916667 | 0.000000 |
| 2.000e-13 | 0.60 | 3.238e-16 | 0.916667 | 0.000000 |
| 1.000e-10 | 0.90 | 2.362e-14 | 0.786232 | 0.130435 |

## Interpretation

The golden axisymmetric backend and the arbitrary-node Riesz backend both enforce no-quadratic compilation and application contracts. The general Riesz backend is used for the conic, polyhedral, folded, non-orientable, and aircraft rows. It uses fixed-order source moments, a symmetric WSPD, exact terminal leaves, and explicit work guards. The older cross-slice atlas appears only in the tolerance sweep as diagnostic context.

Production gate: PASS; every production backend has a hard no-quadratic contract, every measured apply fit is below 1.9, maximum error is below 1e-13, and no dense matrix is stored.

The polyhedral rows certify the discrete weighted graph on a refined triangulation. They do not certify continuum layer potential convergence at edges and vertices. The separate sparse edge/vertex Mellin-Kondratiev channel and continuum refinement campaign are implemented in polyhedral_kondratiev.py and polyhedral_corner_convergence.py.
