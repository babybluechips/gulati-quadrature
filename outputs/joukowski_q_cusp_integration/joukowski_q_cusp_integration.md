# Joukowski Q Cusp Integration

This demo shows the cusp case where Q naturally uses a Joukowski pullback rather than an external dense correction.

![Joukowski Q cusp integration](/Users/rick/Documents/New project 2/outputs/joukowski_q_cusp_integration/joukowski_q_cusp_integration.png)

## Figure 3 Check

I checked the current Cauchy transport PDF, page 11. Figure 3 says the exterior Riemann map carries arclength samples to equispaced circle samples. Read literally, that is not true for a generic analytic boundary.

The correct statement is:

```text
theta = arg Phi(z)
dtheta = |Phi'(gamma(s))| ds
ds = |(Phi^{-1})'(e^{i theta}) i e^{i theta}| dtheta
```

So arclength-uniform samples map to equispaced circle angles only if `|Phi'(gamma(s))|` is constant along the boundary. In general, equispaced circle samples are harmonic-measure samples on Γ, not arclength-uniform samples. The regularization is the metric weight `ds/dtheta`.

In this Joukowski cusp run, the measured nonuniformity is explicit:

- `max |s(theta)/L - theta/(2pi)| = 7.081e-02`
- sampled `max(ds/dtheta) / min(ds/dtheta) = 1.165e+03`
- if 512 arclength-uniform samples are mapped back to circle angles, the largest angle gap is `2.197e+01x` the smallest

## Closed-Form Ellipse Sanity Check

For the ellipse in your note, with semi-axes `a=3` and `b=sqrt(5)`, the exterior map is the Joukowski-type Laurent map

```text
z(w) = alpha w + beta w^(-1)
alpha = (3 + sqrt(5))/2
beta  = (3 - sqrt(5))/2
```

On `|w|=1`, `w=exp(i theta)`, this gives exactly

```text
z(exp(i theta)) = 3 cos(theta) + i sqrt(5) sin(theta).
```

So the conic/ellipse pullback is closed-form. No Schwarz-Christoffel or Beltrami solve is needed. But it is still not arclength-preserving unless `a=b`:

```text
ds/dtheta = sqrt(9 sin(theta)^2 + 5 cos(theta)^2).
```

For this ellipse, `max(ds/dtheta)/min(ds/dtheta) = 1.341640` and `max |s(theta)/L - theta/(2pi)| = 1.152e-02`. This is the benign smooth version of the same regularization: equispaced circle samples are valid when carried with the metric weight.

## Intrinsic Pullback

The generator is

```text
zeta(theta) = c + R exp(i(theta_c + theta))
J(zeta) = zeta + a^2 / zeta
z(theta) = J(zeta(theta))
```

The circle is chosen so it passes through the critical point `zeta_c = a`. Therefore `J'(zeta_c)=0` and the trailing edge is a cusp.

The Q ingredients are generated, not stored:

```text
w_i = |J'(zeta_i) zeta_theta_i| Delta theta
Q[u]_i = (1/pi) sum_j w_j (u_i - u_j) / |J(zeta_i)-J(zeta_j)|^2
I[target] = sum_i w_i sigma_i log |target - J(zeta_i)|
```

This is the regularized version of the Figure 3 diagram: choose equispaced `theta_j` on the circle, evaluate physical points through the inverse chart, and carry the metric weights. If instead one starts from arclength samples on Γ, the corresponding `theta_j` are nonuniform and need resampling, NUFFT, FMM, or an H-matrix path.

Near the cusp, the local expansion is quadratic:

```text
z - z_c = 0.5 J''(zeta_c) (zeta-zeta_c)^2 + ...
ds/dtheta ~ C |theta-theta_c|
```

Measured local slopes: `|z-z_c| ~ |theta|^2.002` and `ds/dtheta ~ |theta|^1.000`.

## Latest Numerical Result

| n | Joukowski log abs err | Equal-arclength log abs err | Joukowski Q energy rel err | Equal-arclength Q energy rel err |
|---:|---:|---:|---:|---:|
| 128 | `6.718e-04` | `8.355e-03` | `6.542e-02` | `7.129e-01` |
| 256 | `1.659e-04` | `3.004e-03` | `3.766e-02` | `6.655e-01` |
| 512 | `4.135e-05` | `8.380e-04` | `2.151e-02` | `6.332e-01` |
| 1024 | `1.033e-05` | `2.157e-04` | `1.135e-02` | `6.117e-01` |
| 2048 | `2.582e-06` | `5.431e-05` | `4.618e-03` | `5.975e-01` |

At the finest tested level, the intrinsic Joukowski pullback gives:

- log-layer error `2.582e-06` vs equal-arclength `5.431e-05`
- Q-energy relative error `4.618e-03` vs equal-arclength `5.975e-01`

## Interpretation

For cusps, the natural Q protocol is exactly the Joukowski-style pullback: borrow the nonsingular circle coordinate, compute physical chord distances and pullback weights, then repay the cusp through the vanishing Jacobian. This is different from polygon corners, where the relevant local model is a wedge with Kondrat'ev/Mellin exponents.

The equal-arclength baseline is not wrong; it is just blind to the square-root inverse chart. The Joukowski generator puts resolution where the cusp chart says the physical boundary compresses.

## Run Parameters

- dense matrix stored: `False`
- reference log integral nodes: `262144`
- reference Q energy nodes: `4096`
- normalized cusp: `[1.0, 0.0]`
- near-cusp target: `[0.97, 0.065]`

## Artifacts

- CSV: `/Users/rick/Documents/New project 2/outputs/joukowski_q_cusp_integration/joukowski_q_cusp_integration.csv`
- JSON: `/Users/rick/Documents/New project 2/outputs/joukowski_q_cusp_integration/joukowski_q_cusp_integration.json`
- Figure: `/Users/rick/Documents/New project 2/outputs/joukowski_q_cusp_integration/joukowski_q_cusp_integration.png`
