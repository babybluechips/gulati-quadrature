# Discriminant Curvature Through `exp(i theta)`

The complex-plane shape story is controlled by the log-discriminant.

![discriminant curvature](/Users/rick/Documents/New project 2/outputs/discriminant_curvature_complex_shape/discriminant_curvature_complex_shape.png)

## Chain

Start on the circle:

```text
z_j = exp(i theta_j) = cos(theta_j) + sqrt(-1) sin(theta_j)
D(z) = prod_{i<j} (z_i - z_j)
L(theta) = log |D(z(theta))|.
```

Differentiating through `exp(i theta)` gives

```text
partial_i partial_j L = 1 / |z_i - z_j|^2,  i != j.
```

With the positive graph-Laplacian sign convention used by Q,

```text
Q = - Hess_theta L
Q_ij = -1 / |z_i - z_j|^2
Q_ii = sum_{j != i} 1 / |z_i - z_j|^2.
```

So Q is discriminant curvature: it is the boundary stiffness of the log-discriminant after the shape is encoded as complex samples.

## Why This Captures The Hard Parts

For an analytic shape, write the boundary as a deformation of the circle:

```text
z(theta) = psi(exp(i theta)).
```

The same chain rule sends

```text
d/dtheta psi(exp(i theta)) = psi'(exp(i theta)) i exp(i theta).
```

When this tangent is nonzero, nearby chords scale like `|z_theta| |theta_i-theta_j|`. When the tangent vanishes, the first nonzero jet gives a cusp or tangent cone and the inverse-square chord curvature spikes. For rational charts, polynomial division `N/D = Q + R/D` separates the asymptotic carrier `Q` from the residual; the chord curvature still sees both because it is computed from the physical complex points.

## Numeric Checks

- circle curvature row-sum variation: `1.000000`; it is constant up to roundoff/sampling symmetry
- cardioid maximum curvature index: `383` near its single cusp
- nephroid cusp-neighbor index pairs: `[383, 0]` at `z=2`, `[191, 192]` at `z=-2`

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_complex_shape/discriminant_curvature_samples.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_complex_shape/discriminant_curvature_complex_shape.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/discriminant_curvature_complex_shape/discriminant_curvature_complex_shape.png`
