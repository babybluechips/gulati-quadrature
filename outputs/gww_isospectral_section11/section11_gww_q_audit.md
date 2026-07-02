# Section 11 Audit: GWW Dirichlet Isospectral Pair Versus Q

Generated: deterministic float64 CPU run; no dense Q matrix stored

## Claim Boundary

This section uses the following careful claim:

> At the discretized chord-operator level, this standard isospectral pair is separated robustly under refinement.

It does **not** claim that this experiment proves existence of a new continuum invariant unless a separate convergence theorem for the continuum Q spectrum is supplied. The exact continuum statement used here is only the classical Gordon-Webb-Wolpert Dirichlet isospectrality theorem.

## Exact GWW Polygon Coordinates

The raw coordinate lists are the eight-vertex GWW/Driscoll polygon tables. In the raw order below both polygons have signed area `-14`, so they are clockwise. The implementation reverses each list before normalization so the working boundary orientation is counterclockwise.

Left raw clockwise vertices:

```text
[(-3, -3), (-3, -1), (1, 3), (1, 1), (3, 1), (1, -1), (-1, -1), (-1, -3)]
```

Right raw clockwise vertices:

```text
[(-3, 1), (1, 1), (1, 3), (3, 1), (1, -1), (-1, -1), (-1, -3), (-3, -1)]
```

Working normalized counterclockwise vertices have unit area. The normalization is

```text
x'_i = (x_i - mean_vertex_x) / sqrt(14),   y'_i = (y_i - mean_vertex_y) / sqrt(14)
```

after orientation reversal.

Left normalized CCW vertices:

```text
[(-0.200445931434, -0.668153104781), (-0.200445931434, -0.133630620956), (0.334076552391, -0.133630620956), (0.868599036215, 0.400891862869), (0.334076552391, 0.400891862869), (0.334076552391, 0.935414346693), (-0.734968415259, -0.133630620956), (-0.734968415259, -0.668153104781)]
```

Right normalized CCW vertices:

```text
[(-0.734968415259, -0.267261241912), (-0.200445931434, -0.801783725737), (-0.200445931434, -0.267261241912), (0.334076552391, -0.267261241912), (0.868599036215, 0.267261241912), (0.334076552391, 0.801783725737), (0.334076552391, 0.267261241912), (-0.734968415259, 0.267261241912)]
```

Both normalized domains have area `1` and perimeter `5.474921741004456`.

## Boundary Sampling Protocol

For a polygon with vertices `v_0,...,v_7` and perimeter `L`, nodes are placed at equal arclength positions

```text
s_k = ((k + alpha) mod n) L / n,   k = 0,...,n-1.
```

The default phase is `alpha = 1/2`, i.e. midpoint sampling of arclength cells. This avoids placing a node exactly on a corner. The polygon is not smoothed. Corners remain exact vertices of the piecewise-linear boundary. If `alpha = 0` is used in the corner-placement control, the first node is exactly on the first corner, but corners are still not duplicated and no two boundary nodes coincide.

## Q Normalization

For sampled boundary points `x_i`, perimeter `L`, and `h = L/n`, the discrete generated chord operator is

```text
(Lambda_Q,n f)_i = (h/pi) sum_{j != i} (f_i - f_j) / |x_i - x_j|^2.
```

The implementation applies this operator blockwise. It stores boundary samples and generated QJet weights only; it does not store the dense `n x n` Q matrix. The Ritz values below are exact float64 eigenvalues of the small generalized projected problem

```text
G c = lambda M c,
G_ab = Phi_a^T Lambda_Q,n Phi_b,
M_ab = Phi_a^T Phi_b,
Phi = [cos(theta), sin(theta), ..., cos(6 theta), sin(6 theta)].
```

This is a fixed 12-dimensional trace subspace, so refinement tests whether the same audited Q spectral witness stabilizes as the boundary sampling is refined.

## Dirichlet Spectrum

The exact continuum Dirichlet statement is

```text
lambda_k(D_left) = lambda_k(D_right) for every k >= 1.
```

The exact values are not known in closed form from this construction. The finite-difference values below are a reproducibility check only; they are not the mathematical spectrum and their mismatch is a grid artifact.

Finite-difference Dirichlet check at grid `96`:

| k | left FD eigenvalue | right FD eigenvalue | relative artifact |
| --- | --- | --- | --- |
| 1 | 34.063772 | 33.163845 | 0.026418888 |
| 2 | 48.152291 | 48.685393 | 0.011071161 |
| 3 | 69.538192 | 69.62006 | 0.001177308 |
| 4 | 89.126441 | 88.652629 | 0.0053161861 |
| 5 | 97.105822 | 96.117235 | 0.010180519 |
| 6 | 124.21696 | 125.11417 | 0.0072229644 |
| 7 | 141.67281 | 138.50791 | 0.022339515 |
| 8 | 150.97417 | 154.23271 | 0.021583413 |

## Q Projected Ritz Convergence

| n | relative split | left delta prev | right delta prev | left Ritz values 1-6 | right Ritz values 1-6 |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.074181035 |  |  | 12.347247, 9.5412852, 8.6005181, 7.8825603, 7.0918629, 6.5684585 | 11.416318, 10.828102, 8.3495053, 7.9712542, 7.0880968, 6.4704386 |
| 512 | 0.073772484 | 0.0063586254 | 0.0060875854 | 12.420724, 9.6076452, 8.6562296, 7.9275681, 7.1470724, 6.6057406 | 11.474881, 10.885889, 8.4163066, 8.0316317, 7.1284556, 6.505546 |
| 1024 | 0.07351271 | 0.003162863 | 0.0030713891 | 12.456167, 9.6408128, 8.6852017, 7.9499079, 7.17486, 6.6246046 | 11.505483, 10.914669, 8.4500666, 8.0618351, 7.1489962, 6.5229859 |
| 2048 | 0.07336365 | 0.0015528486 | 0.0015588463 | 12.47275, 9.6571525, 8.7000251, 7.9609033, 7.1887671, 6.6340961 | 11.521617, 10.929578, 8.4670223, 8.0766841, 7.1593762, 6.5316741 |
| 4096 | 0.073287398 | 0.00078516484 | 0.00077973538 | 12.48123, 9.6654167, 8.7075386, 7.9664274, 7.1957452, 6.6388618 | 11.529682, 10.936921, 8.4755427, 8.084203, 7.1645971, 6.536008 |

## Symmetry And Sampling Controls

The controls below compare the first six projected Q Ritz values for the left GWW domain at `n = 1024`. Rotation and translation are invariant to roundoff. Orientation reversal and node phase/corner placement move only at the expected discretization level. Uniform scaling is reported after multiplying eigenvalues by the scale factor, because this first-order boundary operator scales as inverse length.

| control | n | phase | scale post-factor | relative deviation | Ritz values 1-6 |
| --- | --- | --- | --- | --- | --- |
| base_midpoint_nodes | 1024 | 0.5 | 1 | 0 | 12.456167, 9.6408128, 8.6852017, 7.9499079, 7.17486, 6.6246046 |
| rotate_37_translate | 1024 | 0.5 | 1 | 1.0039738e-15 | 12.456167, 9.6408128, 8.6852017, 7.9499079, 7.17486, 6.6246046 |
| reverse_orientation | 1024 | 0.5 | 1 | 9.6671592e-06 | 12.456368, 9.6408746, 8.6851958, 7.9499363, 7.1748645, 6.6246067 |
| node_phase_0p125 | 1024 | 0.125 | 1 | 5.5742582e-05 | 12.455971, 9.640471, 8.6840839, 7.9498377, 7.174649, 6.6243962 |
| corner_node_phase_0 | 1024 | 0 | 1 | 9.0673053e-05 | 12.457014, 9.6406973, 8.6834498, 7.9499463, 7.1745902, 6.624306 |
| node_phase_golden | 1024 | 0.618034 | 1 | 5.0905949e-05 | 12.457222, 9.6411352, 8.6850688, 7.9500285, 7.1748634, 6.6245951 |
| scale_area_2p25_rescaled | 1024 | 0.5 | 1.5 | 1.3800936e-15 | 12.456167, 9.6408128, 8.6852017, 7.9499079, 7.17486, 6.6246046 |

## Meshless Shape-Optimization Loss

The shape optimization demo uses a boundary map `gamma_p(theta_i)` and probe traces `u_a(theta_i)`. Its reduced Q Gram matrix is

```text
G_ab(p) = (perimeter(gamma_p)/(pi n))
         sum_{i<j} ((u_a(i)-u_a(j))(u_b(i)-u_b(j)))
                  / (|gamma_p(i)-gamma_p(j)|^2 + epsilon^2).
```

For polygonal corner-fixed runs, a low-rank corner repayment Gram `C(p)` is added:

```text
G_corr(p) = G(p) + corner_q_weight C(p).
```

The scalar loss minimized in `examples/q_autograd_meshless_shape_optimization.py` is exactly

```text
L(p) =
  mean((G_corr(p)/tr G_corr(p) - G_target/tr G_target)^2)
  + q_trace_weight * [log(tr G_corr(p) / tr G_target)]^2
  + moment_weight * mean((m(p)-m_target)^2) / mean(m_target^2)
  + corner_weight * mean((c(p)-c_target)^2) / mean(c_target^2)
  + area_weight * ((A(p)-A_target)/|A_target|)^2
  + centroid_weight * |mean_i gamma_p(theta_i)|^2
  + roughness_weight * R(p).
```

Current default weights are:

```text
q_trace_weight = 0.5
moment_weight = 4.0
corner_weight = 1.5
area_weight = 50.0
centroid_weight = 2.0
roughness_weight = 2.0e-4
epsilon = 1.0e-8
```

For smooth Fourier boundaries `R(p)` is spectral coefficient roughness. For polygonal boundaries it is vertex second-difference roughness plus `0.02 * var(edge_lengths)`.

## Reproducibility

Run:

```bash
python3 examples/isospectral/gww_section11_audit.py
```

Output files are written under:

```text
/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11
```

References:

- https://arxiv.org/abs/math/9207215
- https://www.comsol.com/model/download/1166951/models.mph.isospectral_drums.pdf
- https://eudml.org/doc/144038
