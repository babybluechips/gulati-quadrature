# Certified arbitrary-surface QJet

## Scope

`CertifiedArbitrarySurfaceQJet` applies

\[
(Q_pf)_i=\sum_{j\ne i}w_j
\frac{f_i-f_j}{\lVert X_i-X_j\rVert^p}
\]

to arbitrary weighted nodes \(X_i\in\mathbb R^3\). The input need not be axisymmetric, orientable, closed, or globally parameterized. The implementation stores points, weights, a fair-split tree, fixed-order source moments, and far-block endpoints. It never stores an \(N\times N\) distance or operator matrix, a target interaction list, or a persistent pair table.

`from_triangle_mesh(vertices, triangles)` accepts an arbitrary open or closed triangle mesh. It computes one-third triangle-area weights at each incident vertex and then discards connectivity; the operator remains boundary-node based. Use `CertifiedPolyhedralSurfaceQJet` when signed dihedral edges, spherical vertex links, and continuum Mellin--Kondratiev repayment are required. That wrapper retains `O(V+E+F)` topology and keeps the graph and corner certificates separate.

## Euclidean log-discriminant identity

The three-jet construction does extend to three dimensions, but through the Euclidean Laplacian rather than one holomorphic derivative. For \(d>2\),

\[
\Delta_x\log\lVert x-y\rVert^2
=\frac{2(d-2)}{\lVert x-y\rVert^2}.
\]

For residues \(a_j\), define the peeled log-discriminant at node \(i\),

\[
L_{a,i}(x)=\sum_{j\ne i}a_j
\log\lVert x-X_j\rVert^2.
\]

Then in three dimensions

\[
S_a(i):=\sum_{j\ne i}\frac{a_j}{\lVert X_i-X_j\rVert^2}
=\frac12\Delta L_{a,i}(X_i),
\qquad
(Q_2f)_i=f_iS_w(i)-S_{wf}(i).
\]

`GeneratedEuclideanDiscriminantQJet` implements this fixed-width contraction in `O(N)` after the peeled Laplacian jets of \(L_w\) and \(L_{wf}\) have been generated. This is the exact higher-dimensional analogue supplied by the three-jet invariant.

The identity does not make jet generation free. A local surface three-jet determines the singular channel near one point, but it does not determine the distances from that point to every remote sheet of an arbitrary embedding. Two surfaces can have identical local jets and different remote folds. Generating all peeled log-discriminant jets is therefore the global interaction problem.

`MultivariateResultantPeeledJetQJet` implements the exact global common-denominator construction

\[
D=\prod_j\lVert x-X_j\rVert^2,
\qquad
N_a=\sum_j a_j\prod_{k\ne j}\lVert x-X_k\rVert^2,
\]

and extracts the peeled sums from \(\Delta D\), \(\Delta^2D\), and \(\Delta N_a\). It is useful when multivariate support remains sparse. Generic support grows as `Theta(N^3)`, so a support budget and full numerical audit return larger or ill-conditioned cases to the certified hierarchy. See the [resultant derivation and benchmark](multivariate_resultant_generator.md).

## Certified hierarchy

The production implementation uses a symmetric well-separated pair
decomposition and a fixed-order analytic Riesz expansion.

1. For a source node \(B(c_A,a_A)\) and target node \(B(c_B,a_B)\), set \(R_{\min}=\lVert c_A-c_B\rVert-a_B\) and \(\rho=a_A/R_{\min}\).
2. If \(\rho<1\), choose the smallest bounded order whose Gegenbauer tail is below tolerance. Both block orientations are tested and the lower-work certified orientation is retained.
3. If neither orientation passes, split a node. Only unresolved terminal leaf pairs remain exact.
4. Build source moments upward, apply the retained source expansion to its target block, and apply the same low-rank map by its exact transpose in the reverse direction.
5. Form the graph action as \(f_i S_w(i)-S_{wf}(i)\), so constants are annihilated exactly and weighted self-adjointness is preserved.

For the resulting symmetric approximation \(\widetilde K\), compilation accumulates

\[
e_i=\sum_j w_j\lvert K_{ij}-\widetilde K_{ij}\rvert.
\]

For any supplied field,

\[
\lvert (Q-\widetilde Q)f_i\rvert
\le (\lvert f_i\rvert+\lVert f\rVert_\infty)e_i.
\]

`compression_inf_bound` returns the maximum of this discrete bound. The tail
bound follows from \(C_n^{p/2}(1)=(p)_n/n!\). The certificate concerns
compression error in floating-point arithmetic; it is not an
interval-arithmetic continuum theorem.

## Complexity contract

For fixed kernel tolerance, maximum expansion order, leaf size, dimension,
and floating-point coordinate depth:

- fair-split construction is `O(N log^2 N)` with the current repeated sorts;
- the symmetric WSPD has `O(N)` far blocks and terminal pairs;
- upward moments and persistent block/tree state are `O(N)`;
- symmetric block application is `O(N log N)`;
- persistent storage is `O(N)`;
- expansion rank is capped independently of `N`.

Compilation counts far blocks, exact terminal pairs, source-moment
translations, and symmetric analytic block work. Exceeding any configured
near-linear budget raises `NearLinearContractError`. There is no quadratic
production fallback. Thus a
pathological point set may fail closed rather than silently violate the cost
contract.

The exact discriminant contraction remains `O(N)` after its peeled jets have
been generated. The tree generator is the general route for arbitrary sampled
data; the sparse multivariate resultant is a separate algebraic fast path, not
a fallback.

## Numerical audit

The same API is tested on a sphere, ellipsoid, torus, folded sheet, Möbius
strip, star-shaped surface, two close sheets, and a logarithmically clustered
point set. Q2 and Q3 are compared with the isolated exact pair stream; the
tests also audit the analytic error bound, weighted self-adjointness, constant
nullspace, scale covariance, pair partition, and all three work guards. The
generated report records strict-tolerance accuracy and finite-size timings.
The current sphere refinement fits apply exponent `1.630`, versus `1.988` for
the isolated pair stream, and the maximum relative error in the shape table is
`4.824e-15`. The production gate requires exponent below `1.9`, error below
`5e-14`, no dense matrix, and no quadratic fallback.

```sh
PYTHONPATH=src python3 scripts/arbitrary_surface_benchmark.py
```

See the [generated benchmark report](../outputs/arbitrary_surface/report.md).

## Polyhedral continuum corners

The graph benchmark above does not measure convergence to a continuum layer
potential at an edge or vertex. The separate
[3D Mellin--Kondratiev channel](polyhedral_mellin_kondratiev.md) now provides:

- signed material-side dihedral angles and `lambda=m*pi/omega` edge pencils;
- sparse spherical-link P1 pencils with `lambda*(lambda+1)=mu`;
- the vertex surface-measure shift from `lambda` to `lambda+1`;
- fixed four-coefficient power/log jets evaluated through differentiated
  Hurwitz endpoint defects.

The Fichera continuum campaign improves the localized Laplace single-layer
edge error from order `0.674` to `4.698` and the vertex error from order
`1.444` to `5.431`. The finest errors are `8.57e-16` and `4.45e-14`,
respectively. These are local corner-channel results, not a revised complexity
claim for the global arbitrary-surface hierarchy. The machine-scale vertex row
uses the held-out published Fichera exponent; the finite sparse P1 link solve
is reported separately and converges algebraically.
