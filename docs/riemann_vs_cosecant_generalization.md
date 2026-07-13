# Riemann and cosecant pullbacks for the 3D QJet

## Scope of the source audit

The supplied `quadrature_riemann (27).tex` and `cosecant_pullback_3.txt` were
iCloud placeholders during this run. The audit used already-local extracted
copies:

- `quadrature_riemann.txt`, containing *Optimal Quadrature on Riemann
  Surfaces*;
- `tmp/pdfs/combined_full/cosecant.txt`, especially its unit-circle pullback;
- the current axisymmetric and scale-phase implementation.

The local sources agree on the circle formulas. Version identity with the two
unhydrated files was not assumed.

## Riemann route

The Riemann manuscript studies a one-dimensional loop inside a
two-real-dimensional Riemann surface. It does not construct a Riemann map from
an arbitrary boundary surface in three dimensions to a convolution domain.
This is a dimensional distinction: the volume DtN operator in three dimensions
acts on a two-dimensional boundary.

For an axisymmetric surface

\[
X(s,\theta)=(r(s)\cos\theta,r(s)\sin\theta,z(s)),
\]

the exact isothermal coordinate is nevertheless useful:

\[
\eta(s)=\int^s\frac{d\sigma}{r(\sigma)},\qquad
w=e^{\eta+i\theta}.
\]

It gives

\[
ds_S^2=r(\eta)^2(d\eta^2+d\theta^2),\qquad
dS=r(\eta)^2d\eta d\theta.
\]

This makes the intrinsic metric conformally flat. It does not make extrinsic
chords translation invariant. A generic chart still has

\[
|X(u)-X(v)|^{-3}\sqrt{\det G(v)},
\]

which depends separately on both chart points. Close sheets are the simplest
counterexample: their interaction is extrinsic and is not determined by the
conformal factor.

The manuscript's stronger assertion that a generic equal-arclength prime-form
kernel is a function only of `s-t` is false. For a unit-speed planar curve,

\[
|\gamma(s+h)-\gamma(s)|^{-2}
=h^{-2}+\frac{\kappa(s)^2}{12}+O(h).
\]

Varying curvature already breaks translation invariance. The best
lag-averaged circulant projection implemented in the comparison is therefore
an optimistic control, not the exact Riemann operator. It retains only `O(n)`
data but requires `O(n^2)` streamed setup.

## Cosecant route

For a period-`L` coordinate, Mittag-Leffler periodization gives the exact
identity

\[
K_L(h)=\left(\frac{\pi}{L}\right)^2
\csc^2\!\left(\frac{\pi h}{L}\right)
=\sum_{k\in\mathbb Z}\frac{1}{(h+kL)^2}.
\]

On a circle of length `L`, this is exactly the physical inverse-square chord
kernel. On a general unit-speed planar curve,

\[
|\gamma(s)-\gamma(t)|^{-2}=K_L(s-t)+C_\Gamma(s,t),
\]

where

\[
C_\Gamma(s,s)=\frac{\kappa(s)^2}{12}-\frac{\pi^2}{3L^2}.
\]

Thus the cosecant term is the exact universal principal channel. The residual
is bounded near a regular diagonal, but is not a convolution and can become
large at nonlocal near-self-contact.

For the three-dimensional inverse-cube surface kernel, azimuthal reduction
gives

\[
r'\int_0^{2\pi}|X-X'|^{-3}d\phi\sim\frac{2}{|s-s'|^2}.
\]

This is why the one-dimensional cosecant operator is useful on a periodic
meridian. It is not the whole reduced kernel. For a cylinder of radius `R`,

\[
A_R(h)=\frac{2}{h^2}
+\frac{\log(8R/|h|)}{4R^2}
-\frac{3}{8R^2}
+O(h^2\log|h|).
\]

The benchmark measures the logarithmic coefficient as `0.24996685` for
`R=1`, against the predicted `0.25`. Consequently, repayment must include a
logarithmic product-integration channel; subtracting cosecant alone does not
leave a smooth remainder.

## Full surface normal form

For a non-axisymmetric two-dimensional surface chart, the correct flat model
is the periodized Riesz kernel

\[
\mathcal R_\Lambda f(x)=\frac1{2\pi}\operatorname{p.v.}
\int(f(x)-f(y))
\sum_{\ell\in\Lambda}|x-y+\ell|^{-3}dy,
\]

with Fourier symbol `|xi|`. A tensor product of one-dimensional cosecant
operators has symbol `|xi_1|+|xi_2|`, not
`sqrt(xi_1^2+xi_2^2)`. At `(xi_1,xi_2)=(1,1)`, its relative symbol error is
`sqrt(2)-1`, about `41.4%`.

## Numerical comparison

The reproducible campaign is

```sh
PYTHONPATH=src python3 scripts/pullback_generalization_duel.py
```

It covers 54 planar actions and 9 axisymmetric torus cases without retaining a
dense matrix.

- Circle action defects are `5.94e-14` for cosecant and `1.39e-13` for the
  best lag-only control.
- The largest measured off-circle prime-form/chord far defect is `0.375`.
- On a cardioid cusp at `n=256`, three exact edge bands reduce the high-mode
  error from `0.637` to `0.0186`; the lag-only proxy remains at `0.636`.
- On a square, the same repayment reduces `0.0828` to `0.0163`.
- On the finest torus cases, the meridional preconditioner has errors from
  `0.20%` to `0.72%` and is about `10.4x` faster including setup. Repeated
  applications are more than `4,100x` faster than the streamed reference.

These are preconditioner/principal-channel results, not complete DtN accuracy
claims.

## Decision

The Riemann route is useful for constructing isothermal coordinates and for
the exact planar DtN covariance after a genuine conformal map. It does not
make a generic physical chord kernel circulant and does not generalize the 3D
volume DtN operator by itself.

The cosecant route works better as the matrix-free principal channel. The
production extension should be:

1. borrow the one-dimensional cosecant channel after axisymmetric reduction,
   or the two-dimensional periodic Riesz channel on a general surface patch;
2. repay inverse-square, logarithmic, corner, and cusp channels from local
   jets and product integration;
3. evaluate nonlocal geometry with a certified FMM or `H^2` remainder;
4. retain the exact diagonal-convolution-diagonal implementation on cylinder,
   cone, sphere-strip, and valid Koenigs charts.

Under bounded reach, curvature, patch overlap, and fixed accuracy, this is a
credible `O(N log N)` and `O(N)` design. It is not a uniform worst-case theorem
for unrestricted surfaces.
