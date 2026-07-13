# Sparse 3D Mellin--Kondratiev corner channels

## Purpose

`CertifiedArbitrarySurfaceQJet` certifies a finite weighted distance graph. A
refinement sequence of those graphs is not by itself a continuum corner
quadrature theorem. `polyhedral_kondratiev.py` supplies the separate local
analysis needed at three-dimensional edges and vertices while retaining the
matrix-free smooth backend.

The implementation is for scalar Laplace Dirichlet asymptotics. Other elliptic
systems require their corresponding local operator pencils; they cannot be
obtained by changing one exponent.

## Edge pencil

Let a polyhedral edge have material-side opening angle \(\omega_e\). In a
normal cross-section, the homogeneous Dirichlet modes are

\[
u_m(r,\theta)=r^{\lambda_{e,m}}
\sin(\lambda_{e,m}\theta),\qquad
\lambda_{e,m}=\frac{m\pi}{\omega_e},\quad m\geq 1.
\]

The boundary flux is proportional to \(r^{\lambda_{e,m}-1}\). Since surface
measure near an edge is \(dr\,ds\), its radial quadrature exponent is
\(\nu_{e,m}=\lambda_{e,m}\). `PolyhedralMeshTopology` keeps triangle
connectivity, normalizes a closed mesh to outward orientation, and evaluates
the signed interior angle. A cube gives \(\omega=\pi/2\); the three re-entrant
Fichera edges give \(\omega=3\pi/2\) and \(\lambda=2/3\).
For an open sheet, the material side is ambiguous; the API requires an
explicit `EdgeMellinPencil(opening_angle)` instead of inferring one.

## Vertex pencil

At a conical vertex write \(u(r,\vartheta)=r^\lambda\Phi(\vartheta)\). The
spherical link \(G\subset S^2\) satisfies

\[
-\Delta_{S^2}\Phi=\mu\Phi,\qquad
\lambda(\lambda+1)=\mu.
\]

The flux scales as \(r^{\lambda-1}\), but boundary surface measure contributes
one additional radial power. The vertex quadrature exponent is therefore

\[
\nu_v=\lambda+1.
\]

`SparseSphericalDirichletPencil` assembles P1 stiffness and consistent mass
rows on a triangulated spherical link. It uses sparse conjugate-gradient
solves inside inverse iteration and never forms a dense eigenmatrix.
`VertexMellinPencil` can also consume a previously certified angular
eigenvalue or exponent.

This is the standard polyhedral singularity structure described by
[Costabel, Dauge, and Nicaise](https://arxiv.org/abs/1002.1772) and the
[Dauge corner--edge notes](https://perso.univ-rennes1.fr/monique.dauge/publis/corneredge.html).

## Three-jet repayment

After a partition of unity localizes a corner, one retained power-log family
has the form

\[
r^{\nu-1}(\log r)^q
\left(a_0+a_1r+a_2r^2+a_3r^3+O(r^4)\right).
\]

For midpoint phase \(\beta\), the singular endpoint part of the sampled sum is

\[
h\sum_{k\geq0} f((k+\beta)h)-\int_0^\infty f(r)\,dr
\sim
\partial_\nu^q\!\left[h^\nu
\zeta(1-\nu,\beta)\right].
\]

The correction has the opposite sign. `MellinThreeJetChannel` evaluates all
four rungs with a third-order Taylor algebra in \(\nu\), so logarithmic terms
are analytic exponent derivatives rather than finite differences. Hurwitz
zeta is evaluated by a fixed Euler--Maclaurin ladder. This is related to the
zeta-correction construction of
[Wu and Martinsson](https://arxiv.org/abs/2007.13898), but here the exponent is
supplied by a three-dimensional edge or spherical-link pencil.

For \(C\) retained power-log families, persistent corner storage and apply work
are both \(O(C)\). A fixed scalar Laplace model with a bounded number of modes
per geometric corner therefore has fixed work per corner. The local spherical
pencil compile is sparse and is performed once.

## API

```python
from inverse_shape import (
    CertifiedPolyhedralSurfaceQJet,
    MellinThreeJetChannel,
    VertexMellinPencil,
)

surface = CertifiedPolyhedralSurfaceQJet(
    vertices,
    triangles,
    kernel_power=3.0,
)

edge_pencil = surface.edge_pencil(reentrant_edge_index)
edge_channel = MellinThreeJetChannel(
    edge_pencil,
    edge_amplitude_jet,
    label="edge-0",
)

vertex_pencil = VertexMellinPencil.from_exponent(vertex_lambda)
vertex_channel = MellinThreeJetChannel(
    vertex_pencil,
    vertex_amplitude_jet,
    label="vertex-0",
)
```

Channels are problem-specific because their amplitude jets depend on the
layer-potential density, kernel, and partition of unity. Geometry determines
the pencil exponent; geometry alone does not determine those amplitudes.

## Continuum benchmark

The reproducible campaign uses the closed Fichera cube-minus-octant mesh. The
edge case integrates a 3D Laplace single-layer contribution over one face of a
270-degree prism and evaluates the tangential coordinate in closed form. The
vertex case integrates a face-sector contribution with the published Fichera
Dirichlet exponent \(0.45417371533061\). Each continuum reference is recomputed
after two different radial substitutions.

| channel | raw fitted order | corrected fitted order | finest raw error | finest corrected error |
|---|---:|---:|---:|---:|
| Fichera edge | 0.674 | 4.698 | 3.470e-4 | 8.500e-16 |
| Fichera vertex | 1.444 | 5.431 | 7.374e-6 | 4.452e-14 |

The sparse Fichera spherical-link solve converges monotonically from
\(0.48363\) at refinement 4 to \(0.45698\) at refinement 16. The remaining
\(2.80\times10^{-3}\) difference is angular P1 discretization error, not the
radial Mellin repayment error.

At 512 radial nodes, the finite level-16 exponent leaves `1.248e-7` coupled
error. Three-level power extrapolation reduces this to `6.097e-9`. The
machine-scale vertex row uses the held-out published exponent. Thus the table
separates radial-channel convergence from angular-pencil convergence.

```sh
PYTHONPATH=src python3 scripts/polyhedral_corner_convergence.py
```

See the [generated report](../outputs/polyhedral_corner_convergence/report.md)
and [machine-readable summary](../outputs/polyhedral_corner_convergence/summary.json).

## Claim boundary

This campaign establishes the local continuum convergence of the retained
edge and vertex channels for the manufactured Laplace layer contributions.
It does not prove that every pathological point distribution satisfies the
separate far-field work budget, and it does not constitute a complete Fichera
boundary-value solve. The production far-field backend fails closed instead
of entering quadratic work. The angular pencil, tangential quadrature, smooth
global remainder, and corner repayment remain separately auditable.
The Euler--Maclaurin next-term value in the runtime ledger diagnoses Hurwitz
evaluation only; it is not presented as a bound for the omitted amplitude jet.
