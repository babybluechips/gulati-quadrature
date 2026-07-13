# Curved cross-slice atlas residual

## Scope

`StaticCrossSliceAtlasQJet` completes the conic-pencil surface operator by adding interactions between distinct curved slices. The supported geometry is

\[
X(u_s,\theta_j)
=c_s+a_s\cos\theta_j\,e_{1,s}
+b_s\sin\theta_j\,e_{2,s},
\]

where the centers, axes, and frames are generated from the retained value/three-jets. The slices may bend, twist, taper, or close periodically. This is an arbitrary curved conic-bundle atlas, not yet a general atlas for surfaces with unrestricted topology.

The implementation stores no `N x N` distance or operator matrix and imports no external numerical package. FFT work uses the project's radix-two QJet kernel.

## Operator split

For weights \(w_j\), kernel power \(p\), and nodal field \(f\),

\[
(Q_p f)_i=\sum_{j\ne i}w_j\frac{f_i-f_j}{|X_i-X_j|^p}.
\]

The compiled action is

\[
Q_p=Q_p^{\mathrm{slice}}+Q_p^{\mathrm{local\ cross}}
   +Q_p^{\mathrm{far\ atlas}}.
\]

- For \(p=2\), `Q_slice` is the exact finite-cycle/Joukowski commutator on each complete conic. Circular slices use the eigenvalues \(m(n-m)/2\) directly.
- For \(p=3\), `Q_slice` is an exact streamed local repayment. `apply_dtn_principal` adds the factor \((2\pi)^{-1}\).
- Adjacent slices use the phase-difference chart below.
- Remaining cross-slice pairs use product-patch low-rank blocks or exact terminal repayment.

## Discriminant three-jet contraction

The three-jet notebook supplies an exact fixed-width contraction for the holomorphic kernel. Let

\[
P(z)=\prod_j(z-z_j),\qquad
A_a(z)=\sum_j a_j\frac{P(z)}{z-z_j}.
\]

At a simple root \(z_i\), define

\[
n_0=A_a'(z_i)-\frac{a_iP''(z_i)}2,
\qquad
n_1=\frac{A_a''(z_i)}2-\frac{a_iP'''(z_i)}6.
\]

Then

\[
\sum_{j\ne i}\frac{a_j}{(z_i-z_j)^2}
=\frac{n_0P''(z_i)/2-n_1P'(z_i)}{P'(z_i)^2}.
\]

Taking \(a=w\) and \(a=wf\) gives the complete signed graph action in `O(N)` once the generating jets of \(P,A_w,A_{wf}\) are available. `GeneratedDiscriminantThreeJetQJet` implements exactly this contraction with zero adaptive rank and `O(N)` persistent storage.

On the root-of-unity cycle this closes for every sampled density, not only a sparse one. For

\[
P(z)=z^N-1,
\quad
P'(z_i)=Nz_i^{-1},
\quad
P''(z_i)=N(N-1)z_i^{-2},
\quad
P'''(z_i)=N(N-1)(N-2)z_i^{-3},
\]

the interpolation data are simply \(A_a(z_i)=a_iP'(z_i)\). The foundational QJet FFT recovers the coefficients of \(A_a\), and two spectral differentiations give \(A_a'(z_i)\) and \(A_a''(z_i)\). `RootOfUnityDiscriminantQJet` therefore supplies the full generator plus contraction in `O(N log N)` time and `O(N)` storage, with no numerical rank. Non-radix-two lengths use bounded-factor mixed-radix stages and a QJet Bluestein reduction for residual large factors. The Bluestein convolution length is smaller than \(4N\), and the implementation never falls back to a quadratic full-length DFT.

This separates contraction cost from generator cost without hiding either one. On arbitrary root sets, a sparse rational density can supply numerator jets directly, as in the notebook. An arbitrary vector of \(N\) values on arbitrary roots still requires a product/remainder generator; the root-of-unity FFT argument does not provide it.

On a common circle of radius \(R\), the Euclidean metric closes holomorphically:

\[
\frac1{|z_i-z_j|^2}
=-\frac{z_i z_j}{R^2(z_i-z_j)^2}.
\]

Hence the discriminant contraction gives Euclidean Q directly there, although the cycle FFT is cheaper. Away from a common Schwarz relation, the exact identity is

\[
|z_i-z_j|^{-2}
=\bigl[(z_i-z_j)(\bar z_i-\bar z_j)\bigr]^{-1},
\]

which is not a single univariate three-jet. Joukowski conics close this second factor through their algebraic Schwarz map. The arbitrary curved cross-slice residual still needs a certified bivariate/resultant generator before an unconditional rank-free complexity claim is valid.

A direct metric-closure audit makes the boundary visible. After fitting the best complex scalar multiplying `-z_i z_j/(z_i-z_j)^2` on 16 nodes, the relative pair-kernel residual is `1.00e-15` on a circle, `7.99e-1` on an ellipse, and `7.19e-1` on a varying-radius scale-phase curve. The circle identity is exact; a scalar univariate closure is structurally wrong in the other two cases. The ellipse is recovered by its Joukowski/Schwarz correction, while the varying cross-slice case requires the corresponding dual generating jet.

The reproducible cycle benchmark shows near-linear scaling for the complete generator plus contraction and for contraction from pre-generated jets, versus quadratic scaling for a streamed direct reference. At `N=1024`, the relative error is `1.10e-13`, and the constant channel is annihilated exactly. Non-radix-two audits at `N=150,300,600` verify the mixed-radix path; `N=151` exercises the Bluestein path. The generated report records the measured timings and fitted exponents.

```sh
PYTHONPATH=src python3 scripts/discriminant_three_jet_benchmark.py
```

See the [generated discriminant benchmark report](../outputs/discriminant_three_jet/report.md).

## Phase-difference chart

For two adjacent slices, let \(i\) and \(j\) be phase indices and \(d=i-j\pmod n\). The sampled kernel is transformed only in the source phase:

\[
K(i,j)=\sum_{\ell}c_\ell(d)e^{2\pi i\ell j/n}.
\]

Each retained modulation mode is one circular convolution:

\[
\sum_j K(i,j)x_j
=\sum_\ell
\bigl(c_\ell * (e^{2\pi i\ell(\cdot)/n}x)\bigr)_i.
\]

The reverse action uses the reflected difference kernel \(c_\ell(-d)\). Therefore the same chart supplies both directions of the cross-slice graph block. Modes are retained in signed-frequency order until the sum of omitted coefficient sup norms is below the requested chart tolerance. When this ladder is not sparse, the compiler discards the temporary coefficients and retains only two slice indices for exact streamed repayment.

Compilation makes one \(n_\theta^2\) norm-only pass per local chart. A sparse chart makes a second pass to retain only its selected \(L\times n_\theta\) channels. There are only a bounded number of local charts per slice. No local pair table is retained, even temporarily.

## Product-patch atlas

The remaining domain is recursively divided into rectangular chart patches

\[
[s_0,s_1)\times[k_0,k_1).
\]

A patch stores four integer endpoints and a bounding sphere. Two patches are eligible for compression when their bounding spheres have positive gap \(g\) and

\[
\frac{\max(r_A,r_B)}{g}\le\eta.
\]

Adaptive cross approximation samples pivot rows, pivot columns, and deterministic slice/phase audit points. It accepts

\[
K_{AB}\approx\sum_{q=1}^{r}u_qv_q^T
\]

only if the sampled residual is below tolerance and the economic-rank condition

\[
r(m+n)<mn
\]

holds. Otherwise the larger geometric patch is split. A terminal failure stores only the two four-endpoint patch descriptors and is evaluated exactly during apply.

## Symmetric graph application

For one retained factor \(uv^T\), the contribution from patch \(B\) to patch \(A\) is

\[
(Q_{B\to A}f)_i
=u_i\left(
f_i\sum_{j\in B}v_jw_j
-\sum_{j\in B}v_jw_jf_j
\right).
\]

The reverse direction uses the transpose factor:

\[
(Q_{A\to B}f)_j
=v_j\left(
f_j\sum_{i\in A}u_iw_i
-\sum_{i\in A}u_iw_if_i
\right).
\]

This form gives `Q 1 = 0` block by block and preserves weighted self-adjointness. No post-hoc symmetrization is needed.

## Pair-partition certificate

Every unordered pair of distinct slices contributes exactly \(n_\theta^2\) node pairs. At compile time the implementation checks

\[
\sum_{C\in\text{phase charts}}|C|
+\sum_{B\in\text{low rank}}|B|
+\sum_{E\in\text{exact}}|E|
=\binom{n_s}{2}n_\theta^2.
\]

A nonzero residual aborts construction. This is a combinatorial coverage certificate; it is separate from the sampled ACA accuracy diagnostic.

## Cost

The retained storage is

\[
O\!\left(N+L_{\rm local}N
+\sum_b r_b(m_b+n_b)\right).
\]

The apply work is

\[
O\!\left(
L_{\rm local}N\log n_\theta
+\sum_b r_b(m_b+n_b)
+P_{\rm near}
+Nn_\theta
\right).
\]

For bounded local modulation count, bounded rank, and bounded near neighbors, this is `O(r N log N + N n_theta)`. The exact same-slice term is linear in `N` when `n_theta` is fixed. Arbitrary folded geometry can force rank growth or many exact terminal pairs, so there is no unconditional subquadratic worst-case claim.

## Numerical audit

The benchmark in `outputs/cross_slice_atlas` compares against the independent streamed all-pairs action. On six curved surfaces at 192 nodes, the maximum relative errors were approximately `7.8e-14` for the atlas and `2.3e-3` for the old quadrupole tree. The constant residual was exactly zero in every reported case. On the 128 to 1024 node twisted-ellipse scaling run, the fitted exponents were approximately `1.52` for the warm atlas apply and `1.91` for the direct stream.

The atlas was not faster than the direct Python stream through 1024 nodes in that run. Its purpose at this stage is to replace the `1e-3` far-field error with a static, auditable representation while retaining nonquadratic structure as ranks stabilize. The sampled block residual is not a continuum error theorem; direct-reference comparison remains part of the production acceptance test.

## Usage

```python
from inverse_shape import (
    StaticCrossSliceAtlasQJet,
    twisted_ellipse_tube_qjet,
)

surface = twisted_ellipse_tube_qjet(
    length=5.0,
    axis_a=0.5,
    axis_b=0.2,
    total_twist=1.8,
    n_slices=32,
    n_theta=16,
)
atlas = StaticCrossSliceAtlasQJet(surface, kernel_power=2.0)
result = atlas.apply(values)
audit = atlas.evaluate(values)
print(audit.stats)
```

Geometry changes invalidate the static factors. Shape differentiation still acts through the original sparse conic QJets; a changed accepted geometry must compile a new atlas.
