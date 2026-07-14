# Production 3D CAD boundary-PDE validation

## Error classes

The previous `1.779e-8` maximum was a discrete implementation/PDE solve error against the same finite operator. It was not a continuum surface-discretization bound. This report separates compiled-mode accuracy, algebraic residual, and held-out continuum error.

## CAD coverage

Every source triangle in every QCAD3J archive is scanned. The PDE atlas is nondimensional and topology-bearing but intentionally coarse. Its lumped measure is inherited from all source faces; the geometric coarse-triangle area is reported separately.

| Shape | Source V | Source F | PDE nodes | PDE faces | Measure ratio | Local cells repaid |
|---|---:|---:|---:|---:|---:|---:|
| NASA SOFIA aircraft | 45838 | 92974 | 103 | 504 | 1.000000 | 6 |
| FreeCAD cement mixer | 10661 | 22256 | 48 | 484 | 1.000000 | 0 |
| NASA Curiosity manufacturing plates | 374470 | 749346 | 155 | 959 | 1.000000 | 1 |
| NASA Curiosity assembled | 192301 | 384942 | 120 | 744 | 1.000000 | 3 |
| buildingSMART IFC bridge | 6436 | 12468 | 130 | 206 | 1.000000 | 0 |

## PDE results

| Shape | Problem | Reference | Error | Residual | Q applies | Pass |
|---|---|---|---:|---:|---:|---|
| NASA SOFIA aircraft | laplace_dtn | compiled exact solid-harmonic degree one | 9.904e-13 | 0.000e+00 | 1 | True |
| NASA SOFIA aircraft | poisson_boundary_inverse | manufactured exact harmonic flux with mean-zero gauge | 3.326e-16 | 2.908e-16 | 0 | True |
| NASA SOFIA aircraft | screened_poisson_boundary_inverse | manufactured exact harmonic flux | 3.958e-16 | 4.163e-16 | 0 | True |
| NASA SOFIA aircraft | helmholtz_dtn | compiled exact whole-space plane wave | 5.414e-13 | 0.000e+00 | 1 | True |
| NASA SOFIA aircraft | heat_boundary_semigroup | Pade denominator residual at t=1.202e-04; no bulk heat claim | n/a | 2.917e-08 | 24 | True |
| NASA SOFIA aircraft | wave_boundary_functional_calculus | Newmark denominator residual at t=1.202e-04; no bulk wave claim | n/a | 3.295e-09 | 12 | True |
| FreeCAD cement mixer | laplace_dtn | compiled exact solid-harmonic degree one | 2.621e-14 | 0.000e+00 | 1 | True |
| FreeCAD cement mixer | poisson_boundary_inverse | manufactured exact harmonic flux with mean-zero gauge | 1.744e-16 | 2.580e-16 | 0 | True |
| FreeCAD cement mixer | screened_poisson_boundary_inverse | manufactured exact harmonic flux | 1.581e-16 | 1.885e-16 | 0 | True |
| FreeCAD cement mixer | helmholtz_dtn | compiled exact whole-space plane wave | 2.081e-13 | 0.000e+00 | 1 | True |
| FreeCAD cement mixer | heat_boundary_semigroup | Pade denominator residual at t=1.938e-03; no bulk heat claim | n/a | 9.400e-08 | 30 | True |
| FreeCAD cement mixer | wave_boundary_functional_calculus | Newmark denominator residual at t=1.938e-03; no bulk wave claim | n/a | 1.103e-08 | 20 | True |
| NASA Curiosity manufacturing plates | laplace_dtn | compiled exact solid-harmonic degree one | 9.326e-12 | 0.000e+00 | 1 | True |
| NASA Curiosity manufacturing plates | poisson_boundary_inverse | manufactured exact harmonic flux with mean-zero gauge | 2.226e-16 | 2.318e-16 | 0 | True |
| NASA Curiosity manufacturing plates | screened_poisson_boundary_inverse | manufactured exact harmonic flux | 3.626e-16 | 3.119e-16 | 0 | True |
| NASA Curiosity manufacturing plates | helmholtz_dtn | compiled exact whole-space plane wave | 8.274e-12 | 0.000e+00 | 1 | True |
| NASA Curiosity manufacturing plates | heat_boundary_semigroup | Pade denominator residual at t=2.026e-04; no bulk heat claim | n/a | 8.634e-08 | 150 | True |
| NASA Curiosity manufacturing plates | wave_boundary_functional_calculus | Newmark denominator residual at t=2.026e-04; no bulk wave claim | n/a | 9.240e-08 | 82 | True |
| NASA Curiosity assembled | laplace_dtn | compiled exact solid-harmonic degree one | 5.304e-13 | 0.000e+00 | 1 | True |
| NASA Curiosity assembled | poisson_boundary_inverse | manufactured exact harmonic flux with mean-zero gauge | 2.101e-16 | 2.594e-16 | 0 | True |
| NASA Curiosity assembled | screened_poisson_boundary_inverse | manufactured exact harmonic flux | 1.713e-16 | 1.968e-16 | 0 | True |
| NASA Curiosity assembled | helmholtz_dtn | compiled exact whole-space plane wave | 1.601e-13 | 0.000e+00 | 1 | True |
| NASA Curiosity assembled | heat_boundary_semigroup | Pade denominator residual at t=1.652e-04; no bulk heat claim | n/a | 1.704e-08 | 33 | True |
| NASA Curiosity assembled | wave_boundary_functional_calculus | Newmark denominator residual at t=1.652e-04; no bulk wave claim | n/a | 4.773e-08 | 20 | True |
| buildingSMART IFC bridge | laplace_dtn | compiled exact solid-harmonic degree one | 1.790e-12 | 0.000e+00 | 1 | True |
| buildingSMART IFC bridge | poisson_boundary_inverse | manufactured exact harmonic flux with mean-zero gauge | 2.816e-16 | 2.415e-16 | 0 | True |
| buildingSMART IFC bridge | screened_poisson_boundary_inverse | manufactured exact harmonic flux | 2.384e-16 | 3.064e-16 | 0 | True |
| buildingSMART IFC bridge | helmholtz_dtn | compiled exact whole-space plane wave | 3.206e-13 | 0.000e+00 | 1 | True |
| buildingSMART IFC bridge | heat_boundary_semigroup | Pade denominator residual at t=7.920e-05; no bulk heat claim | n/a | 5.310e-08 | 57 | True |
| buildingSMART IFC bridge | wave_boundary_functional_calculus | Newmark denominator residual at t=7.920e-05; no bulk wave claim | n/a | 8.353e-08 | 38 | True |

## Held-out continuum audit

The following degree-four harmonic was not included in the degree-three moment compiler. These values are the relevant continuum generalization test and are not expected to equal the compiled-mode residual.

| Shape | Held-out degree | Relative continuum error |
|---|---:|---:|
| NASA SOFIA aircraft | 4 | 1.105e+01 |
| FreeCAD cement mixer | 4 | 1.991e+01 |
| NASA Curiosity manufacturing plates | 4 | 2.037e+02 |
| NASA Curiosity assembled | 4 | 5.594e+01 |
| buildingSMART IFC bridge | 4 | 2.617e+01 |

## Singular-cell and curvature repayment

The sphere table isolates the topology-aware tangent-cell series from the fixed-rank harmonic channel on a held-out degree-four mode.

| N | Method | Held-out continuum error |
|---:|---|---:|
| 18 | raw_Q3 | 6.667e-01 |
| 18 | singular_cell_curvature | 6.667e-01 |
| 18 | full_degree3_repayment | 6.379e-01 |
| 66 | raw_Q3 | 3.189e-01 |
| 66 | singular_cell_curvature | 1.095e-01 |
| 66 | full_degree3_repayment | 1.083e-01 |
| 258 | raw_Q3 | 1.700e-01 |
| 258 | singular_cell_curvature | 6.174e-02 |
| 258 | full_degree3_repayment | 6.182e-02 |

## Scope

`poisson_boundary_inverse`, heat, and wave are boundary functional-calculus problems generated by the repaid DtN discretization. They are not arbitrary volumetric-source solves. `helmholtz_dtn` has an exact plane-wave continuum reference.

No result in this report justifies a universal 3D machine-precision claim. Machine-level rows certify retained moment channels. The maximum held-out error remains the honest continuum limitation.
