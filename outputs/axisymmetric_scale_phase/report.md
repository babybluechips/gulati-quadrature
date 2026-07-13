# Axisymmetric scale-phase benchmark

Every cross-ring mode uses the hyperbolic meridian distance. The closed alias quotient includes every finite-angle Fourier rung, so the independent reference is the physical all-pairs graph.

| shape | nodes | nested ms | direct ms | relative error | constant |
|---|---:|---:|---:|---:|---:|
| cylinder | 304 | 1.907e+01 | 7.723e+01 | 1.830e-15 | 0.000e+00 |
| cone | 304 | 1.897e+01 | 7.667e+01 | 1.936e-15 | 0.000e+00 |
| sphere_cap | 304 | 1.896e+01 | 7.734e+01 | 2.003e-15 | 0.000e+00 |
| corrugated | 304 | 1.919e+01 | 7.683e+01 | 1.978e-15 | 0.000e+00 |
| double_neck | 304 | 1.919e+01 | 7.745e+01 | 1.781e-15 | 0.000e+00 |
| cusp_meridian | 304 | 1.920e+01 | 7.760e+01 | 1.334e-15 | 0.000e+00 |
| airfoil_body | 304 | 1.924e+01 | 7.693e+01 | 1.394e-15 | 0.000e+00 |

| nodes | nested ms | direct ms | relative error | compressed pairs |
|---:|---:|---:|---:|---:|
| 128 | 3.564e+00 | 1.479e+01 | 1.096e-15 | 0.000000 |
| 256 | 1.157e+01 | 5.469e+01 | 2.323e-15 | 0.129032 |
| 512 | 3.682e+01 | 2.220e+02 | 5.803e-15 | 0.476190 |
| 1024 | 9.577e+01 | 8.957e+02 | 1.153e-14 | 0.716535 |
| 2048 | 2.257e+02 | not run | not run | 0.852941 |
| 4096 | 4.940e+02 | not run | not run | 0.925147 |

The nested apply exponent is 1.423; its tail fit is 1.247. The physical pair stream fits exponent 1.978. All retained mode tiles have fixed size and total linear storage in the surface-node count.
