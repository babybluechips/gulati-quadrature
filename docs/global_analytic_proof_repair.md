# Global Analytic Proof Repair

This note repairs the global analytic claims in `optimal_quadrature_v41` by
separating the finite algebra, local geometry, analytic trapezoid estimates, and
Hardy-class minimax statements. The finite regular-cycle Gulati calculus is
sound. The original global proof is not airtight unless the statements below
replace the overstrong versions.

## 1. Gulati-DtN Identification

### Repaired Statement

Let `Gamma` be a smooth embedded closed planar curve parametrized by arclength
`gamma(s)`, and define the Gulati operator on mean-zero boundary data by

```text
(G_Gamma f)(s) = p.v. int_Gamma (f(s) - f(t)) / |gamma(s)-gamma(t)|^2 dt.
```

Then `G_Gamma` is an elliptic first-order boundary operator with principal
symbol `pi |xi|`. If `Lambda_Gamma` is the interior Dirichlet-to-Neumann
operator, then

```text
G_Gamma = pi Lambda_Gamma + A_0,
```

where `A_0` is a bounded order-zero operator. For `C^\infty` curves this means
`A_0 in Psi^0(Gamma)`. For `C^3` curves the safe version is the corresponding
bounded form statement, not a full classical pseudodifferential expansion. On
the unit circle, `A_0 = 0` exactly.

The word "compact" must be removed from the theorem unless the full order-zero
part has also been subtracted. A nonzero order-zero pseudodifferential operator
is generally bounded but not compact on `H^{1/2}(Gamma)`.

### Proof Skeleton

In a local arclength chart centered at `s=0`, the Frenet expansion gives

```text
|gamma(s)-gamma(t)|^-2
  = |s-t|^-2 + kappa(s)^2/12 + O(|s-t|)
```

after symmetrizing the kernel near the diagonal. Therefore the singular part of
the bilinear form is

```text
p.v. int (f(s)-f(t)) / (s-t)^2 dt,
```

whose Fourier symbol is `pi |xi|`. The curvature and chart-change terms are
lower order. Gluing these local forms by a partition of unity gives a global
first-order elliptic operator with principal symbol `pi |xi|`.

The Calderon projector gives

```text
Lambda_Gamma = |D_s| + B_0,
```

with `B_0` order zero. Hence `G_Gamma - pi Lambda_Gamma` is order zero. The
circle is the special case where the chord kernel is exactly translation
invariant:

```text
|e^{is}-e^{it}|^-2 = (4 sin^2((s-t)/2))^-1,
```

and the Fourier symbol is exactly `pi |m|`.

## 2. Trapezoid Error: Corrected Quadrature-Curvature Law

### Problem In The Old Statement

The old theorem tried to prove

```text
|E_n(x)| <= C G(x)^(-1/2) exp(-2 pi n delta / L).
```

This cannot be right in the stated range. Since `G(x) ~ pi/delta`, the right
side is `O(sqrt(delta))` for fixed `n`, while the logarithmic branch-cut
contribution from the trapezoid rule is `O(h exp(-2 pi delta/h))`, with
`h = L/n`. For `delta << h`, that contribution is `O(h)`, not
`O(sqrt(delta))`.

The proof also used Donaldson-Elliott contour estimates, which require analytic
continuation. They cannot prove an exponential bound from only `C^3` regularity.

### Repaired Statement

Assume `Gamma` and `sigma` extend analytically to a complex arclength strip of
half-width `a`, and assume the target `x` produces one nearest logarithmic
branch point

```text
t_x = s_x + i delta_x + O(delta_x^2)
```

with nonzero local amplitude `A_x = sigma(gamma(t_x)) gamma'(t_x)`. Let
`h = L/n`. Then the trapezoid error for the logarithmic layer potential satisfies

```text
|E_n(x)| <= C h exp(-2 pi delta_x / h) + C_a exp(-2 pi a / h).
```

If `|A_x| >= a_0 > 0` and the nearest branch contribution does not cancel with
another singularity, then

```text
|E_n(x)| >= c h exp(-2 pi delta_x / h) - C_a exp(-2 pi a / h).
```

In particular, in the near-singular regime `0 < delta_x <= h/4` and for `n`
large enough,

```text
|E_n(x)| asymp h.
```

The geometric Gulati coercivity still enters through

```text
G_abs(x) := int_Gamma |sigma(y)| / |x-y|^2 ds(y) asymp pi |sigma(y_*)| / delta,
```

but the correct trapezoid scaling is

```text
h exp(-2 pi n delta / L),
```

not `G_abs(x)^(-1/2) exp(-2 pi n delta/L)`.

### Proof Skeleton

Donaldson-Elliott gives the trapezoid remainder as a contour integral against
`k_n(t) = (exp(2 pi i t/h)-1)^-1`. Deforming the contour to the closest log
branch cut contributes

```text
A_x int_{t_x}^{t_x+infty} k_n(t) dt,
```

which is bounded above and below by constants times

```text
h exp(-2 pi delta_x/h).
```

All other singularities sit at height at least `a` or at a larger branch height
and are absorbed into `C_a exp(-2 pi a/h)` or into the non-cancellation
hypothesis.

## 3. Circle Hardy Minimax Theorem

### Repaired Statement

Let `A_rho = {rho^-1 < |z| < rho}`, let `H^2(A_rho)` be the Hardy space with

```text
||sigma||^2 = sum_{k in Z} |sigma_hat_k|^2 rho^(2|k|),
```

and let `x = r exp(i theta_0)`, `r > 1`. For the unit ball of `H^2(A_rho)`,
the minimax error of recovering the exterior circle log-layer functional from
`n` samples satisfies

```text
kappa_n(S^1, x; H^2(A_rho)) asymp n^-1 (rho r)^(-n/2).
```

Equispaced nodes and Fourier truncation attain this rate. Since point samples
are a restricted form of linear information, the lower bound follows from the
Hilbert n-width for arbitrary `n` linear measurements; the upper bound is the
explicit FFT reconstruction with aliasing controlled by the Hardy tail.

### Proof Skeleton

For `r > 1`,

```text
log |r e^{i theta_0} - e^{i theta}|
  = log r - sum_{k>=1} r^-k cos(k(theta-theta_0))/k.
```

Thus the functional coefficients have magnitude

```text
|ell_k| asymp |k|^-1 r^(-|k|),  k != 0.
```

The tail norm after retaining `M ~ n/2` positive and negative Fourier modes is

```text
(sum_{|k|>M} |ell_k|^2 rho^(-2|k|))^(1/2)
  asymp M^-1 (rho r)^(-M).
```

Taking `M ~ n/2` gives the stated rate. The finite regular-cycle Gulati
functional calculus is the fast way to apply the retained Fourier multipliers.
The point-evaluation algorithm must still carry the signed Fourier phases and
the target angle; it is not just a scalar function of the eigenvalues.

## 4. Analytic-Curve Minimax Theorem

### Repaired Statement

Let `Gamma` be analytic and let `Phi` be the exterior conformal map from the
exterior of `Gamma` to `|w| > 1`. Put `psi = Phi^{-1}` and `r = |Phi(x)| > 1`.
Define the analytic density class by pulling the measure density to conformal
angle:

```text
tau(e^{i theta}) = sigma(psi(e^{i theta})) |psi'(e^{i theta})|.
```

Assume `tau` belongs to the unit ball of `H^2(A_rho)` and that no singularity of
`theta -> log |x - psi(e^{i theta})|` lies closer to the unit circle than
`w = Phi(x)`. Then

```text
kappa_n(Gamma, x; F_rho) asymp n^-1 (rho |Phi(x)|)^(-n/2).
```

Equispaced nodes in conformal angle attain the upper bound:

```text
y_j = psi(exp(2 pi i j/n)).
```

These are equilibrium-distributed, hence asymptotically Fekete-distributed, but
the theorem should not claim that the raw pairwise-distance Gulati matrix on a
general curve is diagonalized by FFT. The FFT operator is the pullback cycle
operator in conformal coordinates.

### Proof Skeleton

The pulled-back functional is

```text
L_x(tau) = int_0^{2pi} tau(e^{i theta})
          log |x - psi(e^{i theta})| dtheta.
```

The kernel is analytic in the annulus until its nearest logarithmic singularity
at `w = Phi(x)`, so its Fourier coefficients have the same leading decay

```text
|ell_k| asymp |k|^-1 |Phi(x)|^(-|k|).
```

The Hardy-tail argument from the circle gives the minimax rate. Constants depend
on upper and lower bounds for `psi'` and on separation from other singularities
in the conformal collar.

## 5. Operator-Optimality: Safe Form

The safe synthesis is:

1. On `S^1`, the finite Gulati matrices are exactly diagonalized by the DFT and
   supply the FFT functional calculus.
2. On analytic curves, the minimax algorithm is the conformal pullback of the
   circle FFT algorithm.
3. The continuum Gulati operator supplies the elliptic first-order boundary
   structure because `G_Gamma = pi Lambda_Gamma + Psi^0`.
4. The pairwise Gulati matrix on nonuniform Fekete nodes is not yet proved to be
   the exact diagonal FFT primitive. It should be presented as a geometric
   discretization/preconditioner unless a separate spectral equivalence theorem
   is proved.

## 6. Certified Enclosures

### What Cannot Be Repaired As Stated

The sign-monotonicity lemma is false in general. Even on the unit circle, take
`x = r > 1` and a nonnegative density concentrated near `theta = 0`. Moving that
source point outward from radius `1` to radius `1+epsilon < r` decreases
`log |r - a|`, so the outer level potential can decrease rather than increase.
Therefore the claimed monotone sandwich

```text
I_n^-(x; epsilon) <= I(x) <= I_n^+(x; epsilon)
```

does not follow from nonnegativity of the density.

### Repaired A Posteriori Certificate

The usable theorem is non-monotone:

```text
|I(x) - I_n^\pm(x; epsilon)|
  <= C_geom epsilon
     + C_quad n^-1 exp(-(n/2)(log rho - epsilon + delta_Phi(x))).
```

Consequently each side gives a certified interval

```text
I(x) in [I_n^\pm - eta_\pm, I_n^\pm + eta_\pm],
```

and the intersection of the two intervals is also certified when nonempty. A
true Archimedean inner/outer sandwich requires an additional monotonicity
hypothesis that must be verified for the specific density, target, and
deformation.

## 7. BEM Conditioning Claim

The `O(n)` conditioning is theorem-grade on the regular circle because

```text
lambda_m = m(n-m)/2,
cond(G_n | 1^\perp) = lambda_floor(n/2) / lambda_1 = O(n).
```

For general nonuniform nodes or general curves, the paper needs a separate
spectral-equivalence theorem before claiming `O(n)` conditioning. A safe
replacement is:

```text
On quasi-uniform arclength nodes of a smooth curve, the Gulati matrix is a
first-order elliptic discretization and is expected to be spectrally equivalent
to a DtN Galerkin matrix after mass normalization.
```

That is a program statement unless proved with mesh-dependent norm estimates.

## 8. Replacement Map

Use this replacement map in the paper:

```text
Old:  G = pi Lambda + compact K0.
New:  G = pi Lambda + A0, A0 order zero; compact only after subtracting A0.

Old:  C1 G(x)^-1/2 <= |E_n| <= C2 G(x)^-1/2 exp(-2 pi n delta/L).
New:  |E_n| asymp h exp(-2 pi delta/h) under analytic continuation and
      non-cancellation; G_abs(x) asymp pi/delta is a geometric indicator.

Old:  Global analytic minimax by raw Gulati quadrature on Gamma.
New:  Global analytic minimax by conformal-angle pullback of the circle FFT
      Gulati calculus.

Old:  Nonnegative densities give inner/outer monotone certified enclosures.
New:  Nonmonotone intervals with explicit geometric-plus-quadrature radii;
      monotone sandwich only under extra verified hypotheses.

Old:  General BEM O(n) conditioning.
New:  Circle O(n) conditioning theorem; general-curve conditioning is open
      until spectral equivalence is proved.
```

## 9. Status

With these repairs, the paper can honestly claim:

- exact finite Gulati algebra on regular cycles;
- exact FFT functional calculus on the circle;
- stable unit-circle near-singular spectral evaluation;
- a corrected analytic trapezoid branch-cut estimate;
- a theorem-grade circle Hardy minimax result;
- a plausible analytic-curve minimax theorem after conformal pullback, with
  carefully stated density norm and singularity-separation assumptions;
- a continuum elliptic identification `G = pi Lambda + Psi^0`.

It should not claim, without additional proof:

- compactness of the unrenormalized `G - pi Lambda`;
- the old `G(x)^(-1/2)` trapezoid law;
- monotone Archimedean enclosures for arbitrary nonnegative densities;
- FFT diagonalization of raw nonuniform Gulati matrices;
- `O(n)` BEM conditioning on general curves.
