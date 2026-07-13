# Static scale-phase Cauchy and hyperbolic-meridian calculus

## Purpose

This path removes the pairwise Barnes-Hut approximation from scale-phase
charts. It stores a one-dimensional cluster tree, fixed-size interpolation
transforms, sparse three-jet geometry, and exact terminal blocks. It does not
store an `N x N` distance or operator matrix.

Two cases are implemented:

1. the exact exponential chord chart;
2. an arbitrary nonperiodic surface of revolution with positive radius.

The second case includes cylinders, cones, sphere caps, corrugated bodies,
necks, cusp meridians, and airfoil-like bodies.

## Exponential chord chart

Let

\[
 V_i=e^{\rho_i+i\theta_i}.
\]

Then

\[
 |V_i-V_j|^2
 =2e^{\rho_i+\rho_j}
 \{\cosh(\rho_i-\rho_j)-\cos(\theta_i-\theta_j)\}.
\]

The angular Fourier coefficient of the inverse-square kernel is

\[
 \widehat K_m(i,j)
 =\frac{e^{-|m||\rho_i-\rho_j|}}
 {|e^{2\rho_i}-e^{2\rho_j}|}.
\]

`ScalePhaseCauchyQJet` applies these coefficients by a fixed-rank nested
Chebyshev pass in `rho`. The pass has four stages:

1. compute cluster moments upward;
2. apply fixed-size cross-cluster transforms;
3. propagate local expansions downward;
4. evaluate unresolved neighboring blocks directly.

The positive and negative exponential factors are anchored at cluster
endpoints. Every generated exponential is therefore a decay. This prevents
overflow even for large scale ranges and high angular modes.

The same-ring singular channel is not interpolated. It uses the exact finite
cycle eigenvalue

\[
 \lambda_m=\frac{m(n_\theta-m)}2.
\]

The persistent storage is `O(N)` for fixed interpolation order. The apply
work is

\[
 O(p^2N+N\log n_\theta),
\]

where `p=32` in the checked benchmark. The implementation has no adaptive
rank.

## Geometry gate

The exponential chord identity is not a universal three-dimensional metric
identity. `certify_scale_phase_chord` checks the physical distances before the
fast path is accepted. The exhaustive benchmark gives the following maximum
relative metric residuals:

| geometry | residual |
|---|---:|
| rigidly embedded flat annulus | `4.32e-15` |
| straight cylinder | `1.00e+00` |
| tapered circular cone | `9.48e-01` |
| bent circular tube | `1.00e+00` |
| twisted ellipse tube | `1.00e+00` |
| toroidal bundle | `1.00e+00` |
| aircraft bundle | `1.00e+00` |

The point-cloud wrapper defaults to an exhaustive streamed certificate and
fails closed when it does not pass. The sampled `O(N log n_scale)` check is a
diagnostic, not a proof. Structurally generated exponential charts use the
identity directly and require no pair audit. Applying the flat Cauchy kernel
to one of the rejected geometries would change the physical operator.

## General surface of revolution

For a meridian point `(r,z)` with `r>0`, define the upper-half-plane distance
`a` by

\[
 \cosh a
 =\frac{r_i^2+r_j^2+(z_i-z_j)^2}{2r_ir_j}.
\]

The inverse-square cross-ring Fourier coefficient is

\[
 c_m(i,j)=\frac{e^{-|m|a}}{2r_ir_j\sinh a}.
\]

On a finite angular grid, the sampled convolution contains every alias
`m+k n_theta`. These aliases have the closed sum

\[
 c_m^{(n_\theta)}(i,j)
 =\frac{q^m+q^{n_\theta-m}}
 {2r_ir_j\sinh(a)(1-q^{n_\theta})},
 \qquad 0<m<n_\theta,
\]

and

\[
 c_0^{(n_\theta)}(i,j)
 =\frac{1+q^{n_\theta}}
 {2r_ir_j\sinh(a)(1-q^{n_\theta})},
 \qquad q=e^{-a}.
\]

The code does not evaluate `q` through `log` and `exp`. With

\[
 s^2=\frac{(r_i-r_j)^2+(z_i-z_j)^2}{4r_ir_j},
\]

it uses

\[
 q=(\sqrt{1+s^2}+s)^{-2},
 \qquad
 1-q=\frac{2s}{\sqrt{1+s^2}+s}.
\]

The denominator `1-q^n` is evaluated as `(1-q)(1+q+...+q^(n-1))`
with a binary geometric sum. This avoids cancellation for adjacent rings.

`AxisymmetricScalePhaseQJet` combines this exact alias repayment with the
nested meridian transforms and the exact same-ring cycle spectrum. Its output
agrees with a stable physical all-pairs graph, not with a self-generated mode
reference.

## Sparse three-jet geometry

`MeridianThreeJetSpline` removes the need to retain a general geometry
callback. On each meridian interval it reconstructs radius and height as
degree-seven Hermite polynomials from endpoint data

\[
 (f,f',f'',f''').
\]

Each interval stores sixteen scalars: eight for radius and eight for height.
The interpolation reproduces every polynomial meridian of degree at most
seven to roundoff. `axisymmetric_qjet_from_three_jets` lowers these generators
directly into the fast operator.

## Measured results

The nonuniform exponential-Cauchy campaign reports a fitted mode-apply
exponent `1.137`, compared with `2.010` for the direct stream. Errors through
the checked 512 scale nodes are between `6.5e-16` and `2.4e-15`. At 1,024
scale nodes one mode takes about `173 ms`.

The surface-of-revolution campaign reports:

| case | nodes | nested | direct | relative error |
|---|---:|---:|---:|---:|
| cylinder | 304 | `18.9 ms` | `74.5 ms` | `1.83e-15` |
| cone | 304 | `18.9 ms` | `75.5 ms` | `1.94e-15` |
| sphere cap | 304 | `18.9 ms` | `75.7 ms` | `2.00e-15` |
| cusp meridian | 304 | `19.2 ms` | `76.0 ms` | `1.33e-15` |
| airfoil body | 304 | `18.9 ms` | `75.1 ms` | `1.39e-15` |

At 1,024 nodes the nested apply takes `97.8 ms`, the direct reference takes
`895.5 ms`, and the relative error is `1.15e-14`. At 4,096 nodes the nested
apply takes `494 ms`; the quadratic reference was not run. The last doubling
has measured exponent about `1.13`.

Reproduce the two campaigns with:

```sh
PYTHONPATH=src python3 scripts/scale_phase_cauchy_benchmark.py
PYTHONPATH=src python3 scripts/axisymmetric_scale_phase_benchmark.py
PYTHONPATH=src python3 scripts/scale_phase_geometry_audit.py
```

## Remaining scope

This closes arbitrary nonperiodic surfaces of revolution with positive radius.
An axis crossing is an endpoint channel and must be handled by the existing
Mellin/Joukowski repayment. A non-axisymmetric moving conic has angular mode
coupling rather than a diagonal angular symbol. Its cross-slice residual still
requires the conic atlas. A single global scale-phase coordinate cannot be an
exact chord coordinate for every embedded surface; the geometry audit above is
a direct counterexample to that claim.
