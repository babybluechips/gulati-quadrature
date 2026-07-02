# Algorithm Notes

This package implements the stable finite-dimensional pieces of the inverse
shape reconstruction program.

## 1. Boundary Geometry

Boundaries are represented as counterclockwise point samples. The geometry layer
provides area, perimeter, centroid, arclength resampling, curvature estimates,
Hausdorff diagnostics, and low-mode star-shaped Fourier models.

## 2. Gulati Operator

For sampled points `x_i`, the inverse-square Gulati Laplacian is

```text
G_ij = -|x_i-x_j|^-2
G_ii = sum_{j != i} |x_i-x_j|^-2.
```

Given a Gulati matrix, pairwise distances are recovered from off-diagonal
entries and classical MDS reconstructs the point set up to Euclidean isometry.
For sampled data, `gu = gulati_laplacian(points)` is the finite matrix used in
forms written `<u, Gu>`.

## 3. Regular-Cycle Gulati Calculus

On equispaced unit-circle nodes, the finite Gulati matrix is circulant with
closed-form eigenvalues

```text
lambda_m = m(n-m)/2,  m = 0, ..., n-1.
```

The `inverse_shape.quadrature` module applies `phi(G_n)` by FFT, including the
forward Gulati action, the mean-zero pseudoinverse, fractional powers, heat
kernels, wave kernels, and resolvents. It also implements the stable Fourier
log-layer evaluation on the unit circle and a trapezoid comparison for the
near-singular regime.

The pressure runner
`scripts/pressure_test_gulati_quadrature.py` checks dense-spectrum agreement,
Cauchy-Gram conservation, boundary-solve residuals, the `pi/delta` coercivity
scaling, and near-singular log-layer errors.

For non-circular closed curves,
`scripts/pressure_test_offcircle_gulati.py` tests the dense Gulati matrix on an
ellipse, a smooth star-shaped curve, and a piecewise-curved boundary. These
off-circle diagnostics do not use FFT diagonalization. They check row-sum
conservation, positive semidefiniteness, Cauchy-Gram factorization, the local
`delta * G_abs / pi -> 1` coercivity law, and the principal Weyl slope of paired
low modes of `h G_n`.

## 4. Hadamard Hessian Flux Extraction

The corrected boundary microlocal model is order `+1`, not logarithmic. The
double-normal reduced resolvent has the same principal symbol as the Steklov
operator and the Gulati Laplacian. Near the diagonal,

```text
G_R(s,s+eps) = -1/pi f.p. |gamma(s+eps)-gamma(s)|^-2 + lower order.
```

The Hessian residual is flux-dressed:

```text
H_res(s,t) = -2 p(s)p(t) G_R(s,t)
           ~= 2/pi p(s)p(t) |gamma(s)-gamma(t)|^-2.
```

The leading finite-part Laurent coefficient gives the ground-state boundary
flux:

```text
p(s)^2 = (pi/2) Coef_{eps^-2}[H_res(s,s+eps)].
```

For sampled curves, this becomes a product equation for near-neighbor pairs:

```text
(pi/2) H_ij |x_i-x_j|^2 ~= p_i p_j.
```

Equivalently, once the Gulati matrix has been built, no distances need to be
recomputed. Define `D_p = diag(p)` and the positive zero-diagonal adjacency

```text
W_ij = -G_ij,  i != j
W_ii = 0.
```

The sampled zero-diagonal pressure Hessian is

```text
H_p = (2/pi) D_p W D_p,
H_ij = -(2/pi) p_i G_ij p_j,  i != j.
```

This gives the fast application formula

```text
(H_p u)_i = (2/pi) p_i sum_{j != i} W_ij p_j u_j
```

and the Gulati-only flux product identity

```text
p_i p_j = -(pi/2) H_ij / G_ij.
```

The conservative energy representation is the dressed Gulati Laplacian

```text
K_p = (2/pi) D_p G D_p = B_p.T B_p,
```

whose off-diagonal entries are `-H_ij`. The implementation exposes these as
`pressure_hessian_from_gulati`, `apply_pressure_hessian_from_gulati`,
`extract_flux_from_gulati_hessian`, and `pressure_gulati_energy_factor`. The
flux recovery solves the product equations as an overdetermined log-linear
system.

## 5. Dirichlet Spectrum Only Path

The spectrum-only layer uses a positive five-point finite-difference Dirichlet
Laplacian on a rasterized domain:

```text
(-Delta_h u)_ij = (4u_ij - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}) / h^2,
```

with exterior and boundary-adjacent nodes treated by homogeneous Dirichlet
conditions. The sparse matrix is diagonalized with `eigsh` for the smallest
eigenvalues.

The inverse demo solves a constrained problem:

```text
r(theta) = 1 + a cos(2theta) + b sin(2theta),
area(gamma) = 1,
min_{a,b} ||lambda_1..lambda_k - lambda_1(a,b)..lambda_k(a,b)||_rel.
```

The optimizer sees only the target eigenvalues. Rigid rotation is searched after
the solve because the Dirichlet spectrum cannot determine absolute orientation.
This path is a finite-dimensional numerical sanity check, not a universal
finite-spectrum reconstruction theorem.

## 6. Heat-Trace Fitting

The package includes a finite-data least-squares fit for the two-dimensional
Dirichlet heat trace:

```text
Z(t) ~= A/(4*pi*t) - L/(8*sqrt(pi*t)) + b_0 + b_1 t^1/2 + ...
```

This is a numerical estimator for experiments and diagnostics. Exact recovery of
all heat invariants from a finite spectrum is outside the scope of the package.

## 7. Production Boundaries

The package is designed around explicit failure modes:

- duplicate boundary points are rejected;
- non-positive Hopf flux samples are rejected;
- Gulati reconstruction reports relative distance residual;
- regular-cycle Gulati pressure tests report conservation and solve residuals;
- off-circle Gulati pressure tests report coercivity and Weyl diagnostics;
- finite-difference Dirichlet eigenvalues are checked for positivity;
- spectrum-only reconstruction reports both initial and final residuals;
- finite heat-trace fits report least-squares residual;
- all public array inputs are shape-checked.

## 8. Visual Regression Gallery

The checked-in gallery below is generated by
`examples/reconstruction/visual_reconstruction_gallery.py`.

![Reconstruction gallery](assets/reconstruction_gallery.png)

It covers four production checks:

- a complicated polygon reconstructed from its Gulati matrix;
- a sampled piecewise-cubic curved boundary reconstructed from its Gulati matrix;
- Hadamard finite-part flux extraction from the dressed residual kernel;
- a low-mode star-shaped approximation of the same curved boundary.

The spectrum-only figure is generated by
`examples/reconstruction/spectrum_only_reconstruction.py`.

![Spectrum-only reconstruction](assets/spectrum_only_reconstruction.png)
