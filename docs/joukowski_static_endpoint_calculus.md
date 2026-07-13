# Static Joukowski endpoint calculus

## Purpose

The original conic-surface prototype used an order-two Barnes-Hut tree. Its
approximately `1e-3` discrepancy was a compression error, not an unavoidable
property of the inverse-square operator. This module removes that tree on
Joukowski charts. It compiles the operator into fixed FFT channels and static
Mellin endpoint values.

The implementation has no external numerical dependency. It uses the
project's radix-two QJet FFT, stores `O(N)` symbols and work vectors, and never
forms a pair-distance or operator matrix.

## Joukowski map

Let

```text
V = exp(mu + i theta),
J(V) = s (V + V^-1),
mu > 0.
```

The image of a fixed `mu` is the ellipse

```text
a = s (exp(mu) + exp(-mu)),
b = s (exp(mu) - exp(-mu)).
```

Conversely,

```text
s  = 0.5 sqrt(a^2-b^2),
mu = 0.5 log((a+b)/(a-b)).
```

The exact two-point identity is

```text
J(V_i)-J(V_j)
  = s (V_i-V_j) (1-(V_i V_j)^-1).
```

It separates a difference-coordinate singularity from a smooth
sum-coordinate quotient.

At the golden slice,

```text
mu_* = 2 log(phi),
R_*  = exp(mu_*) = phi^2,
a    = 3,
b    = sqrt(5),
q_*  = exp(-2 mu_*) = phi^-4.
```

The factor `q_*` is approximately `0.145898`, so the Joukowski quotient has a
short geometric Fourier expansion. The implementation needs 39 signed
channels at tolerance `2e-16`; its coefficient-tail bound is `6.67e-18`.

## Fixed ellipse

For `z(theta)=a cos(theta)+i b sin(theta)`, define

```text
C(delta) = 1 / (4 sin^2(delta/2)),
q        = (a-b)/(a+b).
```

Then

```text
1 / |z(theta)-z(phi)|^2
  = C(theta-phi)
    sum_(ell in Z) q^|ell| exp(i ell(theta+phi)) / (a b).
```

Each signed mode is one operation of the form

```text
diagonal -> cycle FFT -> diagonal.
```

The graph difference is evaluated without subtracting a large row sum from a
large potential. If `G` is the exact cycle graph with eigenvalues

```text
lambda_m = m(n-m)/2,
```

and `g` is one modulated source-weight channel, then

```text
f C(g) - C(f g) = G(f g) - f G(g).
```

This commutator cancels the constant symbol algebraically before floating
point evaluation.

The precision FFT uses a reusable `QJetFFTPlan`. Bit reversal and all stage
twiddles are generated once. The plan stores exactly `n-1` twiddles and then
uses only butterfly arithmetic; multiplicative twiddle drift does not
accumulate between channels.

## Exterior elliptic annulus

For uniform cell centers in `(rho,theta)`, the base chord is

```text
|V_i-V_j|^2
  = 4 exp(rho_i+rho_j)
    {sinh^2((rho_i-rho_j)/2)+sin^2((theta_i-theta_j)/2)}.
```

After diagonal radius factors, this is Toeplitz in `rho` and circulant in
`theta`. A zero-padded two-dimensional QJet FFT therefore applies it without
radial or angular pair traversal.

For kernel power `p`, put `nu=p/2` and

```text
(1-z)^-nu = sum_(r>=0) c_r z^r,
c_r = (nu)_r / r!,
z = (V_i V_j)^-1.
```

Consequently,

```text
|J(V_i)-J(V_j)|^-p
  = s^-p |V_i-V_j|^-p
    sum_(r,t>=0) c_r c_t
      (V_i^-r conjugate(V_i)^-t)
      (V_j^-r conjugate(V_j)^-t).
```

Every `(r,t)` term is another diagonal, base convolution, and diagonal. The
compiler chooses the channel radius from an explicit first-omitted-tail bound.

The implemented powers are:

```text
p=2: inverse-square area-weighted surface graph;
p=3: normalized (2*pi)^-1 inverse-cube principal DtN channel.
```

Channels with the same `r+t` share their radial modulation. Their angular
modulations become index shifts in Fourier space, so one forward/inverse pair
serves each grouped radial power. The resulting annulus work is
`O(L N log N + L^2 N)` with `O(N)` storage at fixed distance from the critical
circle. The fixed ellipse uses the sharper `O(L N log N)` one-index expansion.

## Static Mellin endpoints

A local corner or cusp channel with exponent `lambda`, coefficient `c`, grid
phase `beta`, and spacing `h` contributes

```text
c h^lambda zeta(1-lambda,beta).
```

`MellinEndpointChannel` evaluates this quantity once through a branch-free
Euler-Maclaurin Hurwitz evaluator. No physical-grid refinement is performed.
The implementation records the first omitted Bernoulli term as a numerical
diagnostic. Simple Mellin poles are implemented; logarithmic channels from
repeated pencil roots remain to be added.

If `mu` approaches zero, the Joukowski Fourier compiler refuses to increase
its rank without bound and requests this Mellin cusp chart instead.

## Conic-surface integration

`ConicPencilSurfaceQJet.apply_same_slice_joukowski` compiles each elliptical
slice from its two axes and applies the exact local channel using the actual
generated surface-cell weights. Translation and SU(2) frame rotation do not
change same-slice distances.

This replaces the most singular same-ring interactions. It does not yet
include interactions between different moving conic slices. Completing the
arbitrary-surface path requires a statically compiled atlas residual for
centerline bending, frame variation, and changing conic parameters.

## Measured results

The checked campaign reports:

| check | result |
|---|---:|
| golden fixed-ellipse static/direct error, through 1024 nodes | `6.82e-14` |
| exterior annulus Q2 static/direct error | `5.02e-15` |
| normalized exterior annulus Q3 static/direct error | `1.42e-14` |
| twisted 3D conic same-slice error | `3.23e-15` |
| constant residual | `0` |
| fitted fixed-ellipse static exponent | `1.11` |
| fitted direct-pair exponent | `2.02` |
| 1024-node fixed-ellipse static apply | `53.2 ms` |
| 1024-node compensated direct stream | `1473 ms` |

The fixed-ellipse error grows mildly with resolution because the discrete
inverse-square graph is increasingly cancellation sensitive. The stable
commutator and planned twiddles reduce this to tens or hundreds of machine
epsilon rather than claiming exact real arithmetic.

The Q3 result is a discrete principal-channel comparison. Continuum DtN
accuracy still requires tangent-cell repayment and the lower-order geometric
operator.

## API

```python
from inverse_shape import (
    GOLDEN_MU,
    MellinEndpointChannel,
    StaticJoukowskiAnnulusQJet,
    StaticMellinEndpointRepayment,
    golden_joukowski_ellipse_qjet,
)

ellipse = golden_joukowski_ellipse_qjet(1024)
q2_boundary = ellipse.apply(boundary_values)

annulus = StaticJoukowskiAnnulusQJet(
    scale=1.0,
    rho_start=GOLDEN_MU,
    rho_stop=GOLDEN_MU + 0.4,
    n_scale=8,
    n_theta=32,
    kernel_power=3.0,
)
q3_principal = annulus.apply(surface_values)

endpoint = StaticMellinEndpointRepayment((
    MellinEndpointChannel(0.5, amplitude, phase=0.5),
))
certificate = endpoint.evaluate(step=h)
```

Run the campaign with

```sh
PYTHONPATH=src python3 scripts/joukowski_static_endpoint_benchmark.py
```

Machine-readable results and the scaling figure are under
`outputs/joukowski_static_endpoint/`.
