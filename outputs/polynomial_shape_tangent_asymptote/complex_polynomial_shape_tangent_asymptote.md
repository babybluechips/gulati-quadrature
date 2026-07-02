# Complex Polynomial Shape Tangent/Asymptote Atlas

The pasted rational-function rule becomes a shape atlas rule once the boundary is represented as a loop of complex numbers.

![tangent and asymptote atlas](/Users/rick/Documents/New project 2/outputs/polynomial_shape_tangent_asymptote/complex_polynomial_shape_tangent_asymptote.png)

## Shape Encoding

A planar boundary is stored as a closed complex curve

```text
Gamma = { z(theta) = x(theta) + i y(theta) : 0 <= theta < 2 pi }.
```

After sampling, the shape is the ordered vector

```text
z = (z_0, ..., z_{n-1}) in C^n.
```

The Q operator is built from complex chords only:

```text
Q_ij = -1 / |z_i - z_j|^2,  i != j
Q_ii = sum_{j != i} 1 / |z_i - z_j|^2.
```

So the complex loop is the primary shape object. Translation and rotation do not change Q, while scaling by `r` sends `Q -> r^(-2) Q`.

For analytic boundaries, the circle is the normal form:

```text
z(theta) = Phi^(-1)(exp(i theta)),
w = Phi(z) = exp(rho + i theta).
```

That separates phase `theta`, scale/standoff `rho`, and shape deformation. The tangent/asymptote atlas below is the local algebra used when this complex loop has hard pieces.

## Rule

For a rational chart, divide once:

```text
z(t) = x(t) + i N(t)/D(t)
N(t)/D(t) = Q(t) + R(t)/D(t).
```

`Q` is the carrier geometry. If it is constant it is a horizontal asymptote; if it is linear it is a slant asymptote; if it has higher degree it is the polynomial asymptote. The hard piece is the residual `R/D`.

For a finite polynomial chart, expand at the hard point:

```text
z(t0+u) = z0 + a_m u^m + a_{m+1} u^{m+1} + ...
```

The first nonzero coefficient `a_m` gives the tangent line or tangent cone. If `m=1` the boundary is smooth. If `m>1` the chart has a cusp or singular contact, but the tangent carrier is still explicit.

## Examples

- `cardioid_single_cusp`: `z(w)=w-(w^2+1)/2`, one derivative-zero cusp. The hard part is represented by the cusp tangent cone plus higher polynomial jets.
- `nephroid_two_cusps`: `z(w)=3w-w^3`, two derivative-zero cusps. The two local tangent cones separate the hard endpoints.
- rational chart: the quotient `Q` is the asymptote and `R/D` is the encoded defect. Roots of `R` are contacts with the asymptote. Even multiplicity touches/bounces; odd multiplicity crosses.

## Recovered Division

- `deg Q = 1`
- `deg R = 5`
- `deg D = 6`
- carrier: `Q(t) = -0.02 + 0.18 t`
- residual roots: `(t-0.45)^2 (t+0.85)^3` up to the stored scale

## Use In Q

This is the representation layer for hard geometry. Store the generating QJets for the carrier line/asymptote and the residual polynomial factors. The Q kernel still uses physical complex chords, but the singular/cusp/asymptotic behavior is classified by the tangent/asymptote atlas before the Mellin/zeta or BGK repayment is selected.

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/polynomial_shape_tangent_asymptote/rational_asymptote_samples.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/polynomial_shape_tangent_asymptote/complex_polynomial_shape_tangent_asymptote.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/polynomial_shape_tangent_asymptote/complex_polynomial_shape_tangent_asymptote.png`
