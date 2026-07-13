# Axisymmetric 3D Q in scale-phase and tetration coordinates

## Result

For an axisymmetric surface

\[
X(s,\theta)=(r(s)\cos\theta,r(s)\sin\theta,z(s)),\qquad \rho=\log r,
\]

the exact chord identity is

\[
|X-X'|^2=(z-z')^2
+2e^{\rho+\rho'}\{\cosh(\rho-\rho')-\cos(\theta-\theta')\}.
\]

The production implementation evaluates the cancellation-safe equivalent

\[
(z-z')^2+(r-r')^2+4rr'\sin^2\frac{\theta-\theta'}2.
\]

This identity is valid for every positive radius. The dimensional operator
order must nevertheless change from the planar boundary case. A two-dimensional
surface in three dimensions has the DtN principal kernel

\[
(Q_3 f)(X)=\frac1{2\pi}\operatorname{p.v.}
\int_\Gamma\frac{f(X)-f(Y)}{|X-Y|^3}\,dS_Y.
\]

On the unit sphere, its continuum eigenvalues are exactly \(\ell\) on
spherical harmonics of degree \(\ell\). Applying \(|X-Y|^{-2}\) directly on
the full surface instead gives the harmonic-number spectrum \(H_\ell\), not
the DtN spectrum.

The original inverse-square kernel returns after azimuthal reduction. On a
unit-radius cylinder with meridional separation \(s\),

\[
\int_0^{2\pi}
\frac{d\vartheta}{(s^2+4\sin^2(\vartheta/2))^{3/2}}
\sim \frac{2}{s^2}.
\]

Thus the three-dimensional inverse-cube surface kernel reduces to the
one-dimensional inverse-square Q singularity.

## Exact O(N log N) normal forms

Let \(N=n_s n_\theta\). The following cases are implemented as one generated
two-dimensional convolution symbol, diagonal scale factors, and sparse
meridional three-jets. A zero-padded FFT handles the nonperiodic scale axis and
a circular FFT handles phase. Storage is \(O(N)\); no \(N\)-by-\(N\) matrix or
ring-pair kernel table is formed.

### Cylinder

For \(r=R\),

\[
|X-X'|^2=(z-z')^2+4R^2\sin^2\frac{\Delta\theta}{2}.
\]

The kernel is translation invariant in \((z,\theta)\), so the graph action is
a two-dimensional Toeplitz-circulant convolution.

### Cone

For \(r=e^\rho\) and \(z=ar+b\),

\[
|X-X'|^2=4e^{\rho+\rho'}
\left[(1+a^2)\sinh^2\frac{\Delta\rho}{2}
+\sin^2\frac{\Delta\theta}{2}\right].
\]

Since

\[
dS'=\sqrt{1+a^2}\,e^{2\rho'}d\rho' d\theta',
\]

the inverse-cube graph action has the exact form

\[
Q_3f(\rho,\theta)=
\frac{\sqrt{1+a^2}}{16\pi}e^{-3\rho/2}
\left[
f\,K*e^{\rho'/2}-K*(e^{\rho'/2}f)
\right],
\]

where

\[
K(\Delta\rho,\Delta\theta)=
\left[(1+a^2)\sinh^2(\Delta\rho/2)
+\sin^2(\Delta\theta/2)\right]^{-3/2}.
\]

This is diagonal-convolution-diagonal and is therefore exactly
\(O(N\log N)\).

### Stereographic sphere strip

With \(\eta=\log\tan(u/2)\),

\[
r=R\operatorname{sech}\eta,\qquad
z=-R\tanh\eta,
\]

and

\[
|X-X'|^2=2rr'
\{\cosh(\eta-\eta')-\cos(\theta-\theta')\}.
\]

The chord factor separates into one target diagonal, one source diagonal, and
a convolution in \((\eta,\theta)\). The finite computational strip is handled
by zero padding in \(\eta\).

### Koenigs/tetration cone

Suppose a holomorphic iteration admits a single-valued Koenigs coordinate
\(\xi\) on the active basin and

\[
\xi(T_h(z))=\omega^h\xi(z),\qquad
\log\omega=\alpha+i\beta.
\]

Along a cone embedded linearly in \(|\xi|\),

\[
\rho(h)=\rho_0+\alpha h,qquad
\theta_{\rm physical}=\theta+\beta h.
\]

The conic kernel depends only on

\[
(h-h',\ \theta-\theta'+\beta(h-h')),
\]

so it is a sheared two-dimensional convolution. The implementation unshears
the local correction, applies a positive sparse edge form in physical
coordinates, and shears it back. This preserves weighted self-adjointness.

## What tetration contributes

Tetration is useful here as a coordinate generator, not as a replacement for
the FFT or the multipole method.

The useful statement is local and precise:

\[
T_h=\epsilon^{-1}(\omega^h\epsilon),
\]

where \(\epsilon\) is a Koenigs or Schroeder coordinate on a specified
single-valued linearization domain. Uniform height then means uniform
translation in \(\log|\epsilon|\) and uniform phase shear in
\(\arg\epsilon\).

The supplied sources also show why this cannot be asserted globally without
qualification:

- `exp_intertwiner-tetration7 (1).pdf`, Theorem 9.15 and Remark 9.16, restricts
  the fractional iterate to the domain where \(\omega^h\epsilon(z)\) remains
  inside the linearization image.
- The same paper's Remark 9.17 states that the logarithm branch is an external
  normalization.
- `z_i_tetration_rabbit_hole.md` explicitly labels the Abel-cylinder synthesis
  speculative and demonstrates that periodic Abel ghosts are invisible at
  integer height but exponentially amplified at imaginary height.
- `golden_flow_paper.pdf`, Remark 20, explicitly distinguishes the branch-free
  positive-real Joukowski matrix flow from tetration.

Accordingly, raw tetration values do not create translation invariance. A
validated Koenigs coordinate does.

### Golden hyperbolic gauge

The golden tetration multiplier can nevertheless fix the remaining scale
gauge without evaluating tetration. Put

\[
\mu_T=\frac1{2\phi},\qquad
r_*=4\mu_T^2=\phi^{-2},\qquad
\mu_*= -\log r_*=2\log\phi.
\]

For an oriented hyperbolic frame, the branch-free Cayley coordinate

\[
\tau=\sqrt5\frac{g(w)-i}{g(w)+i}
\]

maps the whole upper half-plane to \(|\tau|<\sqrt5\) and sends signed distance
\(\pm\mu_*\) to \(\pm1\). Among integer-trace hyperbolic gauges, trace three
uniquely maximizes the disk radius and the associated Bernstein ellipse. This
is the rigorous numerical role of the golden point. It is a normalization of
the Koenigs/scale-phase chart, not an identification of matrix powering with
tetration.

The implementation and audit are in
[`golden_hyperbolic_normalization.md`](golden_hyperbolic_normalization.md).

## Sparse three-jet representation

Each scale line stores 13 scalars:

\[
(h,r,z,\phi; r',r'',r''';\ z',z'',z''';\ \phi',\phi'',\phi''').
\]

The exact normal-form symbol has \(O(N)\) entries. The omitted principal-value
cell is added as a positive nearest-neighbor edge form generated from the local
metric. This correction is \(O(N)\), annihilates constants, is nonnegative, and
is weighted self-adjoint by construction.

For a surface outside an exact normal form, the intended extension is:

1. Build an adaptive finite atlas of cylinder, cone, sphere, or Koenigs patches.
2. Borrow the nearest exact normal-form symbol on each patch.
3. Compute patch principal actions by two-dimensional FFT.
4. Repay near interactions with sparse three-jet edge corrections.
5. Evaluate the smooth far remainder with an \(O(N)\) or \(O(N\log N)\)
   hierarchical multipole tree.
6. Certify every patch by a held-out chord residual and a fourth-derivative or
   subdivision remainder bound.

The last two items matter. A three-jet alone does not determine arbitrary
far-field geometry. Either a certified remainder bound or higher multipole
moments are required.

## Complexity statement

| Geometry class | Apply | Stored working data | Current status |
|---|---:|---:|---|
| cylinder / exact cone | \(O(N\log N)\) | \(O(N)\) | implemented, direct-pair verified |
| stereographic sphere strip | \(O(N\log N)\) | \(O(N)\) | implemented, direct-pair verified |
| Koenigs cone in one basin | \(O(N\log N)\) | \(O(N)\) | implemented, complex shear verified |
| arbitrary axisymmetric stream | \(O(N^{3/2}\log N)\) balanced | \(O(N)\) | implemented baseline |
| bounded atlas + fixed-rank far field | expected \(O(N\log N)\) | \(O(N)\) | far-field certificate not yet implemented |
| unrestricted worst-case geometry | no uniform claim | no uniform claim | close sheets/cusps can force refinement |
| high-frequency Helmholtz/Maxwell | frequency dependent | frequency dependent | directional compression required |

Therefore, "all cases are worst-case \(O(N\log N)\)" is currently proved by
construction only for the exact normal forms. It is a reasonable fixed-accuracy
target for a bounded geometric atlas, but it is not yet a theorem for arbitrary
surfaces or growing frequency.

## Numerical campaign

The reproducible campaign is generated by
`scripts/axisymmetric_3d_qjet_benchmark.py`.

Current results:

- 14 axisymmetric surfaces, including spheroids, oscillatory and necked radial
  graphs, cusp tips, superquadrics, two tori, cylinder, frustum, near-tip cone,
  and a double-concave hourglass;
- maximum scale-phase identity error: `6.55e-15`;
- angular FFT stream versus independent physical pair stream: `1.11e-15`;
- exact normal-form 2D FFT versus direct physical pair stream: `3.17e-15`;
- exact normal-form 2D FFT versus general streamed Q: `1.56e-13`;
- maximum constant residual: `0`;
- maximum weighted self-adjoint defect after local repayment: `3.09e-15`;
- measured fast-path exponent versus total nodes: `0.920`;
- measured general-stream exponent versus total nodes: `1.36`;
- finest tested production-path speedup: `3.82x`;
- finest sphere maximum spectral error: `5.46e-2` raw and `1.40e-2`
  after sparse tangent-cell repayment.

The exact discrete identities pass at floating-point roundoff. The continuum
sphere result is convergent but is not yet machine precision; higher-order
product integration is still required.

## Electromagnetic status

The new code validates the geometric principal operator. It does not validate
the RCS claims in `paper_em.tex`.

The old `benchmark_em.py` builds and stores a dense EFIE matrix, uses
NumPy/SciPy eigendecomposition, and compares two algebraic treatments of the
same discretized matrix. Its so-called ISL solve does not apply the
inverse-square Q operator. That comparison is not an independent Maxwell
benchmark.

The patent's Proposition 5.3 states a slender-body relation
\(K^{(0)}=L^{-1/2}+O(R_{\max}^2/L^2)\), but the supplied document does not
provide a proof sufficient to extend it to finite-frequency Maxwell scattering.
The next independent validation layer should use Mie scattering for a PEC
sphere and a published body-of-revolution CFIE/Mueller reference before any RCS
or stealth claim is retained.
