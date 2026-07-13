# Axisymmetric 3D scale-phase QJet benchmark

## Scope

This campaign tests the matrix-free 3D principal boundary operator and its exact
axisymmetric scale-phase geometry. It does **not** treat the old dense EFIE
eigendecomposition in `paper_em.tex` as an independent reference, and it does
not claim a validated Maxwell/RCS solver.

For a surface of revolution,

```text
|X-X'|^2 = (z-z')^2
         + 2 exp(rho+rho') [cosh(rho-rho') - cos(theta-theta')].
```

On the full two-dimensional surface, the DtN principal kernel is
`(2*pi)^-1 |X-X'|^-3`. Integrating out azimuth gives

```text
r' integral |X-X'|^-3 dtheta' ~ 2 / |s-s'|^2,
```

so the original inverse-square Q kernel reappears in the reduced meridional
problem. Applying `|X-X'|^-2` directly on the full surface instead produces an
inverse-first reduced singularity and the wrong sphere spectrum.

## Headline results

| check | result |
|---|---:|
| shapes | 14 |
| exact distance identity, max relative error | `6.546e-15` |
| FFT stream vs independent pair stream | `1.113e-15` |
| constant nullspace, max residual | `0.000e+00` |
| weighted self-adjointness, max relative defect | `3.087e-15` |
| phase-shift equivariance, max relative defect | `3.765e-14` |
| finest sphere max error, inverse-cube raw | `5.460e-02` |
| finest sphere max error, tangent-cell repaid | `1.398e-02` |
| fitted raw sphere convergence exponent | `-1.011` |
| fitted repaid sphere convergence exponent | `-1.414` |
| exact normal-form 2D FFT vs direct, max error | `3.169e-15` |
| exact normal-form 2D FFT vs streamed, max error | `1.561e-13` |
| 27-case Koenigs/tetration chart sweep, max defect | `7.533e-15` |
| measured fast runtime exponent vs total nodes | `0.920` |
| measured streamed runtime exponent vs total nodes | `1.358` |
| finest measured normal-form speedup | `3.82x` |

The exact identities and discrete structural invariants are at floating-point
roundoff. The sphere continuum comparison is convergent but not at machine
precision: the local tangent-cell repayment reduces the leading error constant,
while higher-order product integration is still required. This distinction is
intentional and visible in the tables.

## Complexity

For cylinder, log-cone, stereographic-sphere, and Koenigs/tetration-cone normal
forms, the operator is diagonal--2D-convolution--diagonal. Zero-padded scale
FFT plus circular phase FFT therefore gives **`O(N log N)` work and `O(N)`
storage** for `N=n_s n_theta`. The stored data are one generated convolution
symbol and one sparse meridional three-jet per scale line.

The general axisymmetric stream remains
`O(n_s^2 n_theta log n_theta) = O(N^(3/2) log N)` on a balanced grid. To claim
`O(N log N)` for arbitrary surfaces, a finite normal-form atlas must be paired
with a certified fixed-rank far correction (for example, a 3D multipole tree)
and bounded local three-jet remainders. Tetration does not remove those
requirements: it is useful exactly where a single-valued Koenigs coordinate
makes scale and phase affine in height.

## Interpretation

The test supports the cylinder/conic pullback and the inverse-square
meridional reduction. It rejects the naive statement that the planar
inverse-square graph can be placed unchanged on a two-dimensional surface and
remain the 3D DtN operator. On the sphere, that naive operator tends to
`H_ell`; the inverse-cube surface operator tends to the exact interior DtN
eigenvalue `ell/R`.

## Files

- `geometry_identity.csv`: stable, hyperbolic, and Cartesian chord audit.
- `cylinder_reduction.csv`: inverse-cube to inverse-square asymptotic reduction.
- `sphere_spectrum.csv`: every tested `(ell,m)` Rayleigh value and residual.
- `sphere_convergence.csv`: refinement envelopes.
- `scaling_covariance.csv`: exact `R^-1` scaling audit.
- `shape_invariants.csv`: nullspace, self-adjointness, positivity, and phase covariance.
- `shape_ritz.csv`: explicitly labeled discrete probe Ritz diagnostics.
- `performance_scaling.csv`: timings, work model, and dense storage avoided.
- `fast_normal_form_parity.csv`: exact cylinder/cone/sphere/tetration FFT checks.
- `fast_normal_form_scaling.csv`: `O(N log N)` path against the ring-pair stream.
- `tetration_chart_sweep.csv`: multiplier, phase-shear, and cone-slope stress test.
