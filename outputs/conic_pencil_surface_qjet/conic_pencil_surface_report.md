# Conic-pencil surface QJet benchmark

## Construction

The retained geometry is a bundle of moving conics

```text
X(u,theta) = c(u) + a(u) cos(theta) e1(u) + b(u) sin(theta) e2(u).
```

Each slice stores value/three-jets of the center, SU(2) frame rotor, and two log axes: 36 scalars. Surface nodes and area weights are generated only during an apply; meridional tangents are differentiated directly from those jets. No dense distance matrix, surface operator, or reduced shape Hessian is retained.

The inverse-square operator supplies the shape metric. Shape parameters are lowered by

```text
delta p -> J delta p -> Q_inverse_square(J delta p) -> J* Q J delta p.
```

The normalized `(2*pi)^-1 |X-Y|^-3` action is reported separately as the discretized off-diagonal three-dimensional DtN principal channel.

## Headline checks

- shapes: `6`; refinement cases: `18`
- maximum inverse-square tree/direct error: `1.181e-03`
- maximum inverse-cube tree/direct error: `2.230e-03`
- maximum constant residual: `0.000e+00`
- fitted tree runtime exponent: `1.554`
- fitted streamed-direct runtime exponent: `1.937`
- minimum conic-pencil determinant magnitude: `2.441e+00`
- dense matrices stored: `no`

## Finest tested grids

| shape | nodes | Q2 rel. err. | Q3/(2pi) rel. err. | Q2 speedup | Q3 speedup | dense entries avoided |
|---|---:|---:|---:|---:|---:|---:|
| circular_cylinder | 1024 | `3.850e-04` | `1.330e-03` | `1.04x` | `1.07x` | `1048576` |
| elliptic_taper | 1024 | `5.399e-04` | `1.396e-03` | `1.03x` | `1.02x` | `1048576` |
| bent_tube | 1024 | `9.620e-04` | `1.680e-03` | `1.02x` | `0.97x` | `1048576` |
| twisted_ellipse | 1024 | `7.354e-04` | `1.408e-03` | `1.01x` | `0.95x` | `1048576` |
| toroidal_bundle | 1024 | `1.106e-03` | `2.209e-03` | `1.51x` | `1.46x` | `1048576` |
| aircraft_body | 1024 | `9.834e-04` | `1.561e-03` | `1.06x` | `1.07x` | `1048576` |

## Shape response

A bend/pinch/twist load on `128` generated surface nodes and `64` conic parameters converged in `28` matrix-free CG iterations.
The relative residual was `3.101e-10` and maximum generated node motion at step 0.3 was `3.498e-02`.

The deformation is not a free unconstrained repulsion. It is the response of the conic parameter bundle under the positive reduced metric `J*QJ`; the ridge stabilizes its translation nullspace and shell smoothness controls slice-to-slice oscillation.

## What the three source papers contribute

- The cone paper supplies shell ordering and the warning that nearest-shell block tridiagonality belongs to a local stencil; its direct Schur sweep is not a fast all-pairs quadrature.
- The discriminant paper supplies `-d_lambda^2 log det(A0+lambda A1)`, an O(1) inverse-square certificate for each 3x3 conic pencil. It detects chart degeneration but is not the surface distance matrix.
- The SU(2) paper supplies frame transport and exact group convolution on genuine subgroup orbits. Here rotors transport local frames; generic bent conic bundles are not falsely declared group convolutions.

## Complexity and limitation

The quadrupole tree uses exact leaf interactions and generated far moments. At fixed accuracy and bounded reach it targets `O(N log N)` work and uses `O(N)` storage. The measured exponent in this Python campaign is reported above and is not relabeled as `O(N log N)`. Close sheets, cusps, or collapsing conic pencils grow the near list and can approach the streamed `O(N^2)` reference cost.

The Q3 comparison audits only tree compression against the independent direct discretization. It is not a continuum DtN accuracy result: tangent-cell singular repayment and the lower-order geometry operator have not yet been added to this 3D path. Open tube examples are surface patches; the toroidal bundle is the closed-surface case.
