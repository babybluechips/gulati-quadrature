# Gulati Quadrature

Standalone repository for the boundary-only Q quadrature, DtN/PDE, QBX
comparison, BGK/corner-correction, and shape-optimization work. The code is
curated from the larger research workspace so this repo contains the
quadrature/PDE package, tests, examples, benchmark artifacts, reports, and the
interactive drawing UI without unrelated nested projects or scratch caches.

## Production 3D Surface Pipeline

The public three-dimensional path evaluates the weighted surface graph

```text
(Q_p f)_i = sum_{j != i} w_j (f_i - f_j) / |X_i - X_j|^p
```

from surface nodes and weights without constructing a dense distance or
operator matrix. For a two-dimensional surface in `R^3`, `Q_3/(2*pi)` has the
local principal normalization `|xi|_g` of the Laplace Dirichlet-to-Neumann
operator. `Q_2` is retained as the scale-invariant inverse-square discriminant
operator.

The production backend uses a fair-split pair partition, fixed-order Cartesian
source jets, a symmetric forward/transpose action, exact near-field repayment,
and an analytic Gegenbauer tail bound. Its fixed-parameter contract is:

| Stage | Time | Storage |
|---|---:|---:|
| Compile | `O(N log^2 N)` | `O(N)` |
| Apply | `O(N log N)` | `O(N)` |
| Triangle-soup manifold repair | `O(V+F)` expected | `O(V+F)` |
| Cubic panel quadrature (`q` fixed) | `O(F q^2)` | `O(F q^2)` |
| Smooth-panel singular repayment | `O(N q)` | `O(N+q^2)` |
| Tangent-cell/curvature repayment | `O(N+E)` | `O(N+E)` |
| Fixed-degree harmonic repayment | `O(N d^2)` | `O(N d^2)` |
| Mellin/Kondratiev feature repayment | `O(s_c)` | `O(s_c)` |
| QCAD3J encode/decode | `O(V+F)` | `O(V+F)` |

Here `s_c` is the total fixed local support of the retained edge and vertex
channels. At fixed panel order `q`, harmonic degree `d`, and four-rung feature
rank, every displayed storage cost is linear in the surface size.
The operator fails closed with `NearLinearContractError` if a compiled work
budget is exceeded. There is no dense or quadratic production fallback.

### Exact transparent shell tails

Autonomous cylindrical or conic ends do not need to be represented by a deep
stack of shells. After the angular QJet FFT, each mode has the Riccati map

```text
Phi(sigma) = d - u^2/sigma,
w + w^-1 = d/u,
Sigma_star = u/w,  |w| < 1.
```

`Sigma_star` is the exact first-tail Schur pivot, `u*w` is the self-energy
inserted into the retained system, `u*(1-w)` is the interface flux DtN symbol,
and `-log(w)` is the discrete half-Laplacian generator. These normalizations
are separate in the API.

```python
from gulati_quadrature import CylindricalTransparentDtN

cap = CylindricalTransparentDtN.for_problem(
    128,
    "helmholtz",
    parameter=0.7,
    damping=0.2,  # limiting absorption selects the decaying branch
)
trace = tuple(complex(index == 0) for index in range(128))
flux = cap.apply_boundary_dtn(trace)

assert flux.ledger.status == "balanced"
assert cap.stats()["dense_shell_matrix_stored"] is False
assert cap.stats()["tail_depth_dependence"] == "none"
```

For `N_theta=128`, eight right-hand sides, and explicit depth `L=512`, the
audited cap is 110–124 times faster than the streamed direct shell solve across
Laplace, screened Poisson, heat-resolvent, damped Helmholtz, and causal-wave
resolvent cases. The direct and finite-tail implementations agree within
`1.827e-14`; the maximum fixed-point residual is `2.036e-15`. The cap has zero
truncation error for the autonomous tail by the Schur fixed-point identity.

| Tail method | Setup | Repeated apply | Storage |
|---|---:|---:|---:|
| Direct streamed shells | none | `O(K N_theta L + K N_theta log N_theta)` | `O(N_theta+L)` |
| Compiled finite tail | `O(N_theta L)` | `O(K N_theta log N_theta)` | `O(N_theta)` |
| Exact fixed-point cap | `O(N_theta)` | `O(K N_theta log N_theta)` | `O(N_theta)` |

Run `PYTHONPATH=src python3 scripts/transparent_tail_benchmark.py`. The formal
cross-ratio proof, cylinder symbol audit, golden Fibonacci checksum, and branch
conventions are in
[`docs/transparent_tail_dtn.md`](docs/transparent_tail_dtn.md). This closure
removes a structured exterior tail; it does not repair continuum information
discarded by an aggressively compressed arbitrary CAD atlas.

```python
from gulati_quadrature import SurfaceQConfig, build_mesh_engine

vertices = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)
faces = (
    (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
    (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
)
values = tuple(x + 0.2*y - 0.1*z for x, y, z in vertices)

engine = build_mesh_engine(
    vertices,
    faces,
    config=SurfaceQConfig(kernel_power=3.0),
)
result = engine.apply_dtn_principal(values)  # Q_3 / (2*pi)

assert result.ledger.status == "borrowed_repaid"
assert engine.stats()["quadratic_fallback"] is False
assert engine.stats()["dense_q_matrix_stored"] is False
```

For imperfect triangle soups, use the repaired high-order path:

```python
from gulati_quadrature import (
    CurvedPanelConfig,
    ManifoldRepairConfig,
    SurfaceQConfig,
    build_repaired_mesh_engine,
)

engine = build_repaired_mesh_engine(
    vertices,
    faces,
    repair_config=ManifoldRepairConfig(sharp_angle_degrees=32.0),
    panel_config=CurvedPanelConfig(quadrature_order=4),
    config=SurfaceQConfig(
        harmonic_moment_degree=5,
        adaptive_moment_degree=True,
        moment_validation_tolerance=1e-4,
    ),
)

certificate = engine.repair_certificate
assert certificate.watertight
assert certificate.manifold
assert certificate.nonmanifold_vertices == 0
assert certificate.consistently_oriented
```

This path welds duplicate vertices, removes degenerate and duplicate faces,
splits disconnected face fans at nonmanifold vertices and edges, propagates a
consistent orientation, fills closed boundary loops, and orients each
nonzero-volume component outward. It certifies incidence topology; global
self-intersection remains a separate, explicitly unaudited property.

Each repaired face becomes a cubic PN triangle with analytic first and second
jets. Sharp edges use one shared straight Bezier edge, and every shared curved
edge is audited for physical seam closure. The production smooth-panel
correction stores the discrete odd tangent
moment that must vanish in the principal value and applies a
measure-symmetric weak Duffy Laplace--Beltrami jet for the stable first even
rung. It is applied at every smooth panel node. Sharp dihedral edges use
`pi/omega` Mellin exponents; feature vertices use a sparse spherical-link
Kondratiev pencil. Their four moment defects are precomputed with a higher
order rule on the same curved geometry. The generated campaign compiles all
80 cube feature rungs at order 12 and checks them against a separate order-16
rule. `engine.repay_feature_integral(...)`
applies these channels to a target-local scalar layer integral.

Adaptive harmonic repayment never validates on its fitted space. A degree
`d` candidate is tested on all degree `d+1` solid harmonics, and only the
selected candidate is retained. Final scientific benchmarks must still hold
out degrees beyond every degree inspected during selection.

The same API accepts arbitrary weighted nodes with `build_surface_engine`.
Triangle meshes retain their topology. By default the DtN path adds the
curvature-adjusted omitted-cell series
`-a*Delta/4 - a^3*Delta^2/192 - a^5*Delta^3/11520` through the sparse
cotangent Laplace-Beltrami operator, then compiles solid-harmonic normal-flux
moments through degree three. The moment rank is `d(d+2)` and does not grow
with `N`.
Axisymmetric surfaces use `build_spheroid_engine`, `build_torus_engine`,
`build_radial_profile_engine`, and `build_axisymmetric_conic_engine`; each
geometry is lowered into the same hard-no-quadratic production backend before
application. Moving-conic atlases use `build_conic_pencil_engine`.
Consistently oriented polyhedral meshes use `build_polyhedral_engine`, which
adds explicit Mellin-Hurwitz edge and vertex channels without mesh refinement.

### Matrix-Free 3D Boundary PDEs

`SurfacePDESolver` provides a functional calculus over
`A = Q_3/(2*pi)` without assembling a surface operator:

| Public problem | Equation |
|---|---|
| `laplace_dtn` | `g = A f` |
| `poisson` | `A u = f`, with a weighted mean-zero gauge |
| `screened_poisson` / `yukawa` | `(A + mu I)u = f` |
| `helmholtz_dtn` | `g = Lambda_k f`, with fixed plane-wave repayment |
| `helmholtz` | `(A^2 - k^2 I + i eta I)u = f` |
| `heat` | `partial_t u + A u = 0` |
| `wave` | `partial_tt u + A^2 u = 0` |

```python
from gulati_quadrature import (
    SurfacePDEConfig,
    build_surface_pde_solver,
)

solver = build_surface_pde_solver(
    engine,
    config=SurfacePDEConfig(tolerance=1e-10),
)
flux = solver.solve("laplace_dtn", values)
screened = solver.solve("screened_poisson", values, mass=0.4)
helmholtz_flux = solver.solve("helmholtz_dtn", values, wavenumber=0.7)
heat = solver.solve("heat", values, time=0.05, steps=2)
```

Right-hand sides in the retained harmonic subspace use a fixed-rank direct
modal inverse. Other non-self-adjoint systems use weighted BiCGSTAB; an
optional self-adjoint weak-moment projection permits CG. Heat uses diagonal
`[3/3]` Pade propagation and wave uses implicit midpoint/Newmark. For `k`
QJet applications, the iterative path costs `O(k N log N)` time and `O(N)`
auxiliary storage. Every result records the equation residual, QJet
compression bound, application count, and no-dense/no-fallback status.

These names refer to boundary equations generated by the discrete DtN
principal operator. They do not claim a general volumetric heat or Poisson
solve with arbitrary interior sources. `helmholtz_dtn` reproduces the retained
whole-space plane-wave channels exactly; that finite Herglotz audit is not a
universal full-frequency DtN error bound. `helmholtz` remains the separate
damped boundary resolvent.
The independent discrete and exact-sphere continuum audits are generated by:

```sh
PYTHONPATH=src python3 scripts/production_3d_pde_validation.py
```

See
[`outputs/production_3d_pde_validation/report.md`](outputs/production_3d_pde_validation/report.md)
for the separate algebraic and continuum error tables.

The continuum-repaid CAD campaign is generated by:

```sh
PYTHONPATH=src python3 scripts/production_3d_cad_pde_validation.py
```

It scans all source faces, compiles a nondimensional topology-bearing audit
surface, and runs Laplace DtN, Poisson, screened Poisson, Helmholtz DtN, heat,
and wave on every CAD object. See
[`outputs/production_3d_cad_pde_validation/report.md`](outputs/production_3d_cad_pde_validation/report.md).

The lossless `QCAD3J` channel stores ordered IEEE coordinate keys through a
unit-triangular third-difference transform plus exact triangle connectivity.
`encode_mesh` and `decode_mesh` are bitwise invertible and store exactly
`3V + 3F + O(parts)` integers before compression. This geometry archive is
separate from the non-injective moment hierarchy used for operator application.

### Audited 3D Results

The repaired curved-panel refinement campaign is generated by:

```sh
PYTHONPATH=src python3 scripts/production_3d_repaired_refinement.py
```

It reserves degree-four and degree-five solid harmonics while compilation uses
at most degree two and adaptive selection sees at most degree three. Across
exact-sphere, exact-ellipsoid, and funky PN surfaces, all nine meshes are
watertight and both edge- and vertex-manifold. The largest curved-panel seam
gap is `2.483e-16`. The maximum held-out repaid errors from 72 to 1,152 nodes
are:

| Surface | 72 nodes | 288 nodes | 1,152 nodes |
|---|---:|---:|---:|
| exact sphere | `1.087e0` | `4.946e-1` | `3.496e-1` |
| exact ellipsoid | `2.526e0` | `6.042e-1` | `3.842e-1` |
| funky cubic PN | `1.242e0` | `6.490e-1` | `3.699e-1` |

The exact-sphere endpoint rate is `0.818` in node spacing. On a cube, all 80
edge/vertex rungs improve when channels compiled with an order-12 panel rule
are scored against a separate order-16 rule; the largest corrected discrepancy
is `6.56e-6`. All eight vertex-link pencils return the same exponent to
roundoff, within `3.68e-2` of the exact octant exponent three at three spherical
refinements. These are feature-basis checks, not universal continuum accuracy.
The complete protocol and per-mode rows are in
[`outputs/production_3d_repaired_refinement/report.md`](outputs/production_3d_repaired_refinement/report.md).

The checked extended campaign contains 16 geometry classes, 32 `p=2/p=3`
operator cases, and 96 independent field comparisons. It includes smooth
genus-zero and genus-one surfaces, polyhedra, an open nonorientable strip,
twisted conic atlases, near-collision and dynamic-range stress cases, and
airplane, car, and bridge assemblies. Against an isolated streamed discrete
pair sum, the recorded gates are below `1e-12` on standard cases and below
`2e-12` on stress cases. These numbers audit the discrete graph application;
they are not presented as a universal continuum discretization theorem.

The public CAD audit covers five independently sourced models:

- NASA SOFIA aircraft;
- FreeCAD cement-mixer truck;
- two NASA Curiosity manufacturing layouts;
- buildingSMART IFC bridge.

Across 629,706 vertices and 1,261,986 faces, every coordinate bit pattern,
triangle index, part range, and SHA-256 checksum is recovered exactly. The
largest tested model avoids a hypothetical 1.12 TB dense pair table.

On the compiled CAD PDE campaign, the maximum retained harmonic/plane-wave
reference error is `9.326e-12`, and the maximum algebraic heat/wave residual is
`9.400e-8`. These are not continuum error bounds. A degree-four harmonic that
was not used by the degree-three compiler has relative errors from `1.105e1`
to `2.037e2` on the aggressively coarse CAD atlases. The controlled sphere
audit at `N=258` improves from `1.700e-1` for raw `Q_3` to `6.174e-2` after the
singular-cell/curvature series. The repository therefore does not claim
universal machine-precision 3D continuum accuracy.

The separate NASA field gallery solves Laplace DtN, Poisson, screened Poisson,
Helmholtz DtN, heat, and wave on 24 to 42 compressed nodes, then lifts the
computed values onto all 1,227,262 triangles decoded from the SOFIA and two
Curiosity QCAD3J archives. Across 18 displayed rows, the maximum retained
manufactured-reference error is `3.431e-16`, the maximum self-adjoint heat/wave
algebraic residual is `7.638e-15`, and the median warm solve is `44.2 ms`; the
strictest implicit heat solve is `3.68 s`. Those numbers audit retained channels
or finite-operator equations, not the independent held-out continuum modes.
The generated color maps are numerical fields, not illustrative textures.

```sh
PYTHONPATH=src python3 -m gulati_quadrature.cli surface-demo
PYTHONPATH=src python3 scripts/production_3d_qjet_extended_validation.py
PYTHONPATH=src python3 scripts/cad_qjet_invertibility_campaign.py
PYTHONPATH=src python3 scripts/production_nasa_cad_pde_visualization.py
```

The formal derivation and executable examples are in
[`notebooks/production_3d_qjet_method.ipynb`](notebooks/production_3d_qjet_method.ipynb).
The standalone audited report is
[`outputs/production_3d_qjet_html/production_3d_qjet_method.html`](outputs/production_3d_qjet_html/production_3d_qjet_method.html).
The CAD source manifest, licenses, archives, reconstructions, and exactness
report are under `benchmarks/cad_invertibility/` and
`outputs/cad_qjet_invertibility/`.

## Interactive UI

Run the production Q UI backend:

```sh
PYTHONPATH=src python3 scripts/q_engine_ui_backend.py --host 127.0.0.1 --port 8790
```

Then open `http://127.0.0.1:8790/`. The UI starts blank, keeps drawn shapes as
boundary-only curves, and only runs the production Q solve after `Solve`.

## Symbol of Observation Certificate

The repository includes an executable audit for `symbol_of_observation (1).pdf`,
covering complex Spitzer algebra, strobe/zeta transfer, first-zero blindness,
FFT-Spitzer fluctuation constants, total-variation grading, and the unitary
sawtooth law. The pedagogical explanation, proof sketches, diagrams, and
reproduction command are in
[`outputs/symbol_of_observation/README.md`](outputs/symbol_of_observation/README.md).

## Beta Counterterm Certificate

The beta counterterm bridge is now executable as its own audit. It tests the
finite-cycle sum

```text
S_n(s) = sum_{k=1}^{n-1} [k (1-k/n)]^{-s}
```

against the ledger

```text
S_n(s) = n^(1-s) B(1-s,1-s)
       + 2 sum_j (s)_j zeta(s-j) n^(-j)/j!
       + residual.
```

The generated README explains why the beta term is the bulk continuum channel,
why the endpoint rungs are zeta/BGK repayments, and how this is the same
bookkeeping pattern as subtracting `pi R^2` before studying Gauss-circle error.
Current runs certify `O(n^-1)`, `O(n^-2)`, and `O(n^-3)` residual decay after
successive repayments on real and complex test cases.

```sh
PYTHONPATH=src python3 scripts/beta_counterterm_certificate.py \
  --out-dir outputs/beta_counterterm_certificate
```

Read the proof sketches, diagrams, and audit tables in
[`outputs/beta_counterterm_certificate/README.md`](outputs/beta_counterterm_certificate/README.md).

## Hardy-Voronoi Flux Certificate

The Gauss-circle flux audit now includes a dedicated Hardy-Voronoi obstruction
and alias-tower certificate. It verifies that a dyadic block already has
diagonal RMS size

```text
RMS(E_Q) ~ X^(1/2) Q^(-1/2) (log Q)^(1/2),
```

so a full-collar pointwise target of order `X^(1/2)/Q` is below the block's own
average size. The same script also checks the useful positive structure:
Gaussian-unit symmetry forces the angular alias coefficient `W(nu,m)` and the
alias rung `A(m)` to vanish unless `4` divides `m`, so odd angular sample counts
kill the first three alias levels.

```sh
PYTHONPATH=src python3 scripts/hardy_voronoi_flux_certificate.py \
  --out-dir outputs/hardy_voronoi_flux_certificate
```

Read the derivation, Li-Yang context, exact-count correlation, and alias tables
in
[the Hardy-Voronoi certificate](outputs/hardy_voronoi_flux_certificate/README.md).

## Package Core

Production-oriented numerical primitives for inverse spectral and Hadamard shape
reconstruction in planar domains.

This repository accompanies the complete audibility manuscript in
[`audibility_complete_combined.tex`](audibility_complete_combined.tex). The code
does not claim to solve the full infinite-dimensional inverse spectral problem
from finitely many eigenvalues. It provides the tested computational layers that
the paper uses:

- inverse-square Gulati operators for sampled boundaries and polygons;
- reconstruction of labelled polygons from Gulati matrices or directed moments;
- Hadamard Hessian flux extraction via the finite-part Laurent coefficient;
- star-shaped low-mode Fourier fitting;
- finite-difference Dirichlet eigenvalue computation;
- constrained low-mode reconstruction from Dirichlet eigenvalues only;
- heat-trace coefficient fitting from finite spectral samples;
- a CLI and synthetic end-to-end demo.

## Install

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,plot]"
```

For local development without a virtualenv, the tests can also be run with
`PYTHONPATH=src`.

## Quick Start

Run the synthetic reconstruction demo:

```sh
PYTHONPATH=src python -m inverse_shape.cli demo --out artifacts/demo --samples 160
```

This writes:

- `boundary.csv`: normalized sampled boundary;
- `gulati.npy`: serialized Gulati matrix;
- `gulati_reconstruction.csv`: MDS reconstruction from the Gulati matrix;
- `hadamard_residual.npy`: flux-dressed Hadamard residual model;
- `flux_true.csv` and `flux_recovered.csv`;
- `summary.json`: reconstruction diagnostics.

CLI entry points after installation:

```sh
inverse-shape summarize-boundary artifacts/demo/boundary.csv
inverse-shape polygon-from-gulati artifacts/demo/gulati.npy --out artifacts/demo/recovered.csv
inverse-shape flux-from-hessian artifacts/demo/boundary.csv artifacts/demo/hadamard_residual.npy --out artifacts/demo/flux.csv
inverse-shape dirichlet-spectrum artifacts/demo/boundary.csv --out artifacts/demo/eigenvalues.csv --count 8
```

## Visual Reconstruction Gallery

The repository includes a generated visual regression gallery:

![Reconstruction gallery](docs/assets/reconstruction_gallery.png)

Regenerate it with:

```sh
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src \
  python3 examples/reconstruction/visual_reconstruction_gallery.py --out-dir docs/assets
```

Current gallery diagnostics:

- complicated polygon from G: RMS `2.0e-16`;
- piecewise curved sampled boundary from G: RMS `3.3e-16`;
- Hadamard finite-part flux extraction: relative error `2.9e-16`;
- 8-mode star-shaped approximation of the piecewise curved boundary: Hausdorff `2.8e-2`.

## Dirichlet Spectrum Only Demo

The repo now includes a direct test of reconstruction using only Dirichlet
eigenvalues. This is intentionally a constrained finite-dimensional inverse
problem: the target is an area-normalized low-mode star-shaped domain, the data
are only the first finite-difference Dirichlet eigenvalues, and the optimizer is
not given the boundary, G matrix, flux, or Hadamard Hessian.

![Spectrum-only reconstruction](docs/assets/spectrum_only_reconstruction.png)

Regenerate it with:

```sh
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src \
  python3 examples/reconstruction/spectrum_only_reconstruction.py --out-dir docs/assets
```

Current spectrum-only diagnostics:

- first Dirichlet eigenvalues used: `7`;
- initial relative spectral residual: `1.74e-1`;
- recovered relative spectral residual: `3.64e-3`;
- boundary Hausdorff error after rigid-motion alignment: `7.12e-3`.

This is a positive sanity check for a low-dimensional spectral reconstruction
pipeline. It is not a claim that finitely many eigenvalues reconstruct arbitrary
domains; non-congruent isospectral examples and finite-data instability remain
real obstructions outside the constrained model.

## Library API

```python
import numpy as np
from inverse_shape.geometry import BoundaryCurve, StarShapeModel
from inverse_shape.operators import (
    apply_pressure_hessian_from_gulati,
    dressed_gulati_hessian,
    extract_flux_from_gulati_hessian,
    gulati_laplacian,
    pressure_gulati_energy_factor,
)
from inverse_shape.reconstruction import reconstruct_polygon_from_gulati
from inverse_shape.spectrum_inverse import reconstruct_star_shape_from_spectrum

model = StarShapeModel(
    center=np.array([0.0, 0.0]),
    base_radius=1.0,
    cos=np.array([0.12, -0.04, 0.03]),
    sin=np.array([0.0, 0.07, -0.02]),
)
curve = BoundaryCurve(model.boundary_points(160)).normalized()
gu = gulati_laplacian(curve.points)
recovered = reconstruct_polygon_from_gulati(gu)

theta = np.linspace(0, 2 * np.pi, curve.n, endpoint=False)
u = np.cos(theta)
energy = float(u @ (gu @ u))  # sampled <u, Gu> form
flux = 1.0 + 0.18 * np.cos(2 * theta)
h_res = dressed_gulati_hessian(curve.points, flux)
h_action = apply_pressure_hessian_from_gulati(gu, flux, u)
energy_factor = pressure_gulati_energy_factor(curve.points, flux)
flux_hat = extract_flux_from_gulati_hessian(gu, h_res)
```

## Mathematical Conventions

The inverse square boundary operator introduced by Gulati 2026 is denoted `G`.
Its finite Gulati matrices use

```text
G_ij = -|x_i - x_j|^-2,  i != j
G_ii = -sum_{j != i} G_ij
```

For sampled boundary data, `gu = gulati_laplacian(points)` is the matrix used in
quadratic forms written `<u, Gu>`.

## Gulati Cycle Quadrature

The regular-circle implementation from the optimal-quadrature note is available
as `inverse_shape.quadrature`. It includes the closed-form spectrum
`lambda_m = m(n-m)/2`, FFT application of `phi(G_n)`, pseudoinverse boundary
solves on the mean-zero subspace, heat/resolvent/wave/fractional propagators,
Cauchy-Gram factorization, and near-singular logarithmic layer evaluation on the
unit circle.

```python
from inverse_shape.quadrature import (
    apply_cycle_gulati,
    circle_log_layer_spectral,
    near_singular_circle_table,
    solve_cycle_gulati,
)
```

Run the numerical pressure suite with:

```sh
PYTHONPATH=src python3 scripts/pressure_test_gulati_quadrature.py \
  --max-power 16 --near-n 4096 --repeats 3 \
  --json docs/assets/gulati_quadrature_pressure.json
```

The checked pressure artifact reports exact constant-mode conservation through
`n = 65536`, boundary-solve residual `3.4e-12` at `n = 65536`, and at
`delta = 1e-6` the classical trapezoid relative error `2.6e-4` versus Gulati
spectral relative error `1.8e-16`.

For non-circular closed curves, run:

```sh
PYTHONPATH=src python3 scripts/pressure_test_offcircle_gulati.py \
  --json docs/assets/offcircle_gulati_pressure.json
```

This checks an ellipse, a smooth star-shaped curve, and a piecewise-curved
boundary for row-sum conservation, positive semidefiniteness, Cauchy-Gram
factorization, the local `pi/delta` coercivity law, and the principal Weyl slope
of paired low modes of `h G_n`.

To compare the local bridge correction against a point-QBX baseline on hard
non-circular shapes, run:

```sh
PYTHONPATH=src python3 scripts/pressure_test_qbx_comparison.py \
  --json docs/assets/qbx_comparison_pressure.json
```

The default suite uses the peanut, teardrop, wavy-star, asymmetric-gear, and
three-lobed finite-Fourier boundaries. With `n = 1024` source samples and
targets down to `delta/h = 0.05`, the checked artifact reports bridge
improvement over plain trapezoid in all 20 cases and a point-QBX baseline below
`5.2e-7` relative error against a high-resolution reference.

The global analytic claims from the optimal-quadrature note are separated into
repaired theorem statements in
[`docs/global_analytic_proof_repair.md`](docs/global_analytic_proof_repair.md).

For the Hadamard residual kernel,

```text
H_res(s,t) ~= (2/pi) p(s) p(t) / |gamma(s)-gamma(t)|^2
```

near the diagonal, where `p = partial_nu u_1`. Thus

```text
p(s)^2 = (pi/2) Coef_{epsilon^-2}[H_res(s, s + epsilon)]
```

in the continuum finite-part Laurent expansion. The sampled extractor solves the
corresponding log-linear product system from near-neighbor pairs.

Finite Gulati matrices make the pressure Hessian cheap to represent. Let
`D_p = diag(p)` and let `W` be the zero-diagonal positive adjacency

```text
W_ij = -G_ij = |x_i - x_j|^-2,  i != j
W_ii = 0.
```

Then the sampled zero-diagonal pressure Hessian is the diagonal dressing

```text
H_p = (2/pi) D_p W D_p,
(H_p u)_i = (2/pi) p_i sum_{j != i} W_ij p_j u_j.
```

So the geometry is stored once in `G`, pressure updates are diagonal scalings,
and the product identity becomes

```text
p_i p_j = -(pi/2) H_ij / G_ij,  i != j.
```

The conservative positive semidefinite companion is the dressed Gulati
Laplacian

```text
K_p = (2/pi) D_p G D_p = B_p.T B_p,
```

whose off-diagonal entries are `-H_ij`. In code this is
`pressure_hessian_from_gulati`, `apply_pressure_hessian_from_gulati`,
`extract_flux_from_gulati_hessian`, and `pressure_gulati_energy_factor`.

## Odd Dirichlet Trace Regularization

The correct finite-to-continuum regularization for odd Dirichlet signs is not a
raw signed cutoff.  The sign has to be carried by an odd theta characteristic,
equivalently by the prime-form/theta-Vandermonde geometry, so that the twist is
noncentral.

For the primitive odd character modulo `4`,

```text
chi_4(n) = 0,  n even
chi_4(n) = 1,  n = 1 mod 4
chi_4(n) = -1, n = 3 mod 4,
```

the Jacobi theta convention

```text
theta_1(z | tau)
  = 2 sum_{m>=0} (-1)^m q^{(m+1/2)^2} sin((2m+1)z),
q = exp(pi i tau),
```

gives, after differentiating at `z = 0` and setting `tau = i t`,

```text
(1/2) theta_1'(0 | i t)
  = sum_{m>=0} (-1)^m (2m+1)
      exp(-pi (2m+1)^2 t / 4)
  = sum_{n odd >= 1} chi_4(n) n exp(-pi n^2 t / 4).
```

Thus the odd character is encoded by the theta characteristic, not by multiplying
an otherwise positive canonical system by scalar signs.  Taking the Mellin
transform gives the completed odd Dirichlet trace.  For `Re(s) > 1`,

```text
int_0^infty t^{(s+1)/2 - 1} (1/2) theta_1'(0 | i t) dt
  = sum_{n odd >= 1} chi_4(n) n
      int_0^infty t^{(s+1)/2 - 1} exp(-pi n^2 t / 4) dt
  = Gamma((s+1)/2) (4/pi)^((s+1)/2)
      sum_{n odd >= 1} chi_4(n) n^{-s}
  = Gamma((s+1)/2) (4/pi)^((s+1)/2) L(s, chi_4).
```

The interchange of sum and integral is justified in the initial half-plane by
absolute convergence.  The resulting identity continues meromorphically, and in
this odd primitive case the completed function is entire.

The same Mellin calculation works for any primitive odd Dirichlet character
`chi` modulo `q` once its odd theta kernel is written as

```text
Theta_chi(t) = sum_{n>=1} chi(n) n exp(-pi n^2 t / q).
```

Then, initially for `Re(s) > 1`,

```text
int_0^infty t^{(s+1)/2 - 1} Theta_chi(t) dt
  = Gamma((s+1)/2) (q/pi)^((s+1)/2) L(s, chi).
```

For `chi_4`, `Theta_chi` is exactly `(1/2) theta_1'(0 | i t)`.  For general
primitive odd `chi`, it is the corresponding finite linear combination of
shifted odd theta characteristics, with Gauss-sum phases.  This is the
theta-Vandermonde/prime-form lift of the signed trace.

## Tests

```sh
PYTHONPATH=src pytest
```

The tests cover:

- geometry normalization and Fourier star-shape fitting;
- Gulati matrix construction and distance reconstruction;
- regular-cycle Gulati spectral calculus and near-singular quadrature;
- off-circle Gulati conservation, coercivity, and Weyl-slope diagnostics;
- Hadamard flux extraction from a dressed Gulati residual;
- finite-difference Dirichlet eigenvalue computation;
- constrained low-mode reconstruction from Dirichlet spectrum only;
- heat-trace coefficient fitting on synthetic data.

## Repository Layout

```text
src/inverse_shape/      package source
tests/                  unit tests
examples/reconstruction runnable examples
docs/                   algorithm notes
.github/workflows/ci.yml GitHub Actions CI
```

The older Fast Zeta Metal prototype files are left in place for reference, but
the production package in this repo is `inverse-shape`.
# drum
