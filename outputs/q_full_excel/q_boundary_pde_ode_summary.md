# Q Boundary PDE/ODE Benchmark Summary Report

Generated: 2026-07-01  
Workbook: `/Users/rick/Documents/New project 2/outputs/q_full_excel/q_boundary_pde_ode_benchmark.xlsx`  
Workbook SHA-256: `bd8e9285ec06b13e44444f1174fb7942f7c4335682c1daa78f024c9e5c226e48`

![Dashboard preview](</Users/rick/Documents/New project 2/outputs/q_full_excel/preview_Dashboard.png>)

## Executive Summary

The Excel workbook consolidates the current Q engine benchmark corpus into one auditable artifact:

| Block | Rows / Cases | Purpose |
|---|---:|---|
| PDE/DtN analytic reference with FEM baseline | 35 | Boundary-only Q/DtN solves for Laplace, heat, Poisson, Helmholtz, and wave on disk modes; Q and volumetric P1 FEM are both compared against exact formulas |
| Arbitrary planar Q/DtN | 25 | Matrix-free planar chord-QJet solves on ellipse, nonconvex flower, square, star polygon, and cardioid/cusp domains |
| Planar corrected quadrature | 5 | Multipole/zeta near-singular layer-potential correction on the same arbitrary planar domain suite |
| Structural quadrature | 30 | Trapezoid, singularity subtraction, adaptive panel, Gulati Q bridge, multipole/zeta Q, and QBX refined across hard planar shapes |
| Q spectrum | 232 | Per-mode Q symbol diagnostics, spectral pair splitting, symbol power, and error-type classification |
| QBX failures | 11 | Consolidated cusp cases where QBX violates the source-free expansion disk condition |
| QBX scaling fits | 8 shapes | Fitted QBX and multipole/zeta scaling exponents |
| Hard domain audit | 17 | Funky planar domains plus explicit status for currently unsupported 3D surface domains |
| ODE stress | 24 | Stiff decay, Riccati near blow-up, damped oscillator, Van der Pol, and Robertson-style kinetics |

The central result is that the boundary-only Q/DtN path is load-bearing for the cited modal disk PDE benchmark: after the finite-cycle endpoint defect is repaid to the standard unit-disk DtN spectrum, median Q-formula relative error is `0`, median explicit Q-operator relative error is `2.201e-14`, and median volumetric FEM relative error is `3.506e-3`. Median Q-formula runtime is `0.0060 ms` versus FEM median runtime `44.50 ms`, a median `7042.3x` timing speedup for the formula path. The explicit Q-operator application path is slower than the closed-form modal formula but still median `2.38x` faster than the FEM baseline.

For hard shape quadrature, the multipole/zeta Q variant is the best overall method in this benchmark set: median relative error `1.435e-6`, median improvement `5073.8x` over trapezoid, and zero recorded failures. QBX refined is accurate when it applies, but has cusp failures where the target falls outside the source-free QBX expansion disk.

The arbitrary planar extension now uses the paper's method directly: arclength/chord QJets implement `Q = pi Lambda_Gamma + K0`, while the borrow-compute-repay ledger records the bounded geometry correction and the active Q spectral error channel. Smooth domains, nonconvex domains, polygons, and cusp-like domains run through the same matrix-free operator path. Near-boundary layer potentials use the multipole/zeta correction channel rather than QBX when the spectrum indicates corner/cusp risk.

## PDE/DtN Results

PDE benchmark parameters:

| Parameter | Value |
|---|---:|
| Boundary samples | 8192 |
| FEM radial levels | 40 |
| FEM angular segments | 160 |
| FEM nodes | 6401 |
| FEM triangles | 12640 |
| Modes | 1, 2, 4, 8, 12, 16, 24 |
| Q/DtN normalization | raw cycle `Lambda_Q = (h/pi) Q` has `m - m^2/n` dispersion; production Q repays the endpoint defect to `|m|` before exact continuum claims |

Aggregate PDE results:

| Method | Median Relative Error | Max Relative Error | Median Runtime ms |
|---|---:|---:|---:|
| Q formula, repaid production symbol | `0` | `0` | `0.0060` |
| Q operator, repaid production symbol | `2.201e-14` | `5.285e-14` | `18.551` |
| Raw finite-cycle Q diagnostic | `1.329e-3` | `1.337e-2` | tracked in JSON only |
| Volumetric P1 FEM baseline | `3.506e-3` | `3.945e-2` | `44.495` |

Problem families solved from the boundary modal representation:

| PDE / Operator | Parameters |
|---|---|
| Laplace DtN | Modal DtN eigenvalue test |
| Heat | `time = 0.17` |
| Poisson | `mass = 0.35` |
| Helmholtz | `wavenumber = 3.7`, `damping = 0.02` |
| Wave | `time = 0.8` |

Interpretation: the Q/DtN path is not just a quadrature post-process in this run. It directly drives boundary-only amplitudes for elliptic, parabolic, Helmholtz, and wave-type operators. The production Q columns use the repaid continuum symbol and are compared against cited standard disk/Steklov and Bessel references; FEM is not used as truth. The old `1e-3` discrepancy is still present only as the raw finite-cycle diagnostic.

## Arbitrary Planar Domain Extension

The disk modal benchmark is no longer the only PDE path. A new planar-domain QJet path accepts arbitrary closed planar samples and applies the paper's correction structure:

- Borrow: arclength/chord samples of the physical boundary and their generated inverse-square QJet.
- Compute: matrix-free applications of the planar chord QJet, scaled by `h/pi`, with iterative heat, Poisson, Helmholtz, and wave solves.
- Repay: retain the bounded `K0` geometry correction through physical chord distances, attach the Q spectral error type, and use multipole/zeta quadrature for near-singular field evaluation.

Arbitrary planar PDE benchmark:

| Shape | Family | PDE Rows | Active Error Type | Max Runtime ms |
|---|---|---:|---|---:|
| `ellipse_3_to_1` | smooth | 5 | `mixed_geometry_spectral_tail` | `47.17` |
| `funky_flower_curve` | smooth nonconvex | 5 | `cusp_endpoint_channel` | `46.98` |
| `square_polygon` | corner | 5 | `cusp_endpoint_channel` | `46.84` |
| `star_polygon` | corner / vertex scattering | 5 | `mixed_geometry_spectral_tail` | `47.13` |
| `cardioid_cusp` | cusp endpoint | 5 | `cusp_endpoint_channel` | `101.04` |

Near-singular corrected quadrature on the same arbitrary planar suite:

| Shape | Method | Relative Error vs 2048-point Reference | Zeta Exponent | Dense Matrix Stored |
|---|---|---:|---:|---|
| `ellipse_3_to_1` | multipole/zeta Q | `1.345e-4` | `0.837` | false |
| `funky_flower_curve` | multipole/zeta Q | `1.802e-3` | `1.430` | false |
| `square_polygon` | multipole/zeta Q | `2.508e-6` | `2.011` | false |
| `star_polygon` | multipole/zeta Q | `2.024e-2` | `1.000` | false |
| `cardioid_cusp` | multipole/zeta Q | `4.357e-3` | `1.502` | false |

The implementation does not store a dense matrix. For smooth analytic maps, an autodiff QJet pullback path is also available and repays physical normal derivatives using `|dz/dtheta|^-1`. For general sampled planar domains, the chord QJet path is the load-bearing one because it carries the bounded correction term `K0` directly in the generated pairwise chord weights while still avoiding dense storage.

## Structural Quadrature Results

Overall method summary:

| Method | Median Rel Error | Median ms | Median Work Units | Failures | Median Improvement vs Trap |
|---|---:|---:|---:|---:|---:|
| Trapezoid | `1.125e-2` | `0.628` | `512` | 0 | `1.0x` |
| Singularity subtraction | `4.998e-4` | `0.916` | `533` | 0 | `11.2x` |
| Adaptive panel | `2.961e-4` | `1.098` | `926` | 0 | `36.9x` |
| Gulati Q bridge | `4.534e-4` | `0.890` | `513` | 0 | `12.8x` |
| Multipole/zeta Q | `1.435e-6` | `18.445` | `66619` | 0 | `5073.8x` |
| QBX refined | `9.068e-5` | `27.641` | `245760` | 3 | `159.6x` |

The multipole/zeta Q variant is the new preferred Q for these tests. It spends more work per target than the bridge methods, but it is far more accurate and remained robust across the structural benchmark. The workbook tracks both single-target and cached-target work so the cost model can distinguish one-off evaluations from reused multipole moments.

## Q Spectrum Determines Error Type

The workbook records spectral signatures that classify the dominant error mechanism:

| Shape Class | Recorded Error Type | Practical Meaning |
|---|---|---|
| Smooth ellipse / conformal-hard / rounded square | `smooth_spectral_tail` | Error behaves mainly like a smooth high-mode tail |
| Square polygon | `mixed_geometry_spectral_tail` | Smooth spectral behavior mixed with vertex effects |
| Star polygon | `corner_vertex_scattering_channel` | Corners open a distinct scattering channel |
| NACA0012 airfoil | `cusp_endpoint_channel` | Endpoint/cusp-like geometry controls the error |
| Cardioid and astroid cusps | `cusp_endpoint_channel` | Tip behavior dominates; QBX can violate its expansion geometry |

This supports the working principle: the spectrum of Q is not just a convergence diagnostic; it identifies which error channel is active and therefore which quadrature strategy should be trusted.

## Section 11 GWW Isospectral Audit

A dedicated Section 11 audit now covers the Gordon-Webb-Wolpert isospectral polygon pair. It records the exact raw polygon coordinates, the clockwise-to-counterclockwise orientation convention, the equal-arclength node distribution rule, corner treatment, Q normalization, finite-difference Dirichlet reproducibility values, projected Q Ritz eigenvalue convergence, and symmetry controls.

The defensible claim used there is:

> At the discretized chord-operator level, this standard isospectral pair is separated robustly under refinement.

It does not claim that the numerical experiment alone proves a new continuum Q-spectrum invariant. The exact continuum fact used is the classical GWW Dirichlet isospectrality theorem.

Primary Section 11 artifacts:

| Artifact | Path |
|---|---|
| Audit Markdown | `/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11/section11_gww_q_audit.md` |
| Q convergence CSV | `/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11/section11_gww_q_convergence.csv` |
| Symmetry controls CSV | `/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11/section11_gww_q_symmetry_controls.csv` |
| Audit JSON | `/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11/section11_gww_q_audit.json` |
| Convergence figure | `/Users/rick/Documents/New project 2/outputs/gww_isospectral_section11/section11_gww_q_convergence.png` |

At `n = 4096`, the first six projected Q Ritz values are:

| Domain | Ritz values 1-6 |
|---|---|
| GWW left | `12.481230`, `9.665417`, `8.707539`, `7.966427`, `7.195745`, `6.638862` |
| GWW right | `11.529682`, `10.936921`, `8.475543`, `8.084203`, `7.164597`, `6.536008` |

The first-six relative Q split stabilizes to `7.329e-2` at `n = 4096`. Rotation/translation changes are invariant to roundoff; reverse orientation, node phase shifts, and corner-node placement change the first-six spectrum only at the `1e-5` to `1e-4` discretization level in the `n = 1024` control table.

## QBX Failure Cases

![QBX failure rows](</Users/rick/Documents/New project 2/outputs/q_full_excel/preview_QBX_Failures.png>)

The consolidated workbook contains 11 QBX failure rows. The cusp-specific benchmark reports 8 direct QBX failures. The failure mode is consistent:

`ValueError: target is outside the source-free QBX expansion disk`

The failures are concentrated at cusp tips, especially cardioid and nephroid-like two-cusp geometries. This is the regime where QBX's local expansion geometry breaks down. Gulati Q bridge and singularity subtraction do not throw geometric expansion failures there, but their errors can still be large at exact tips. Multipole/zeta Q is the more reliable replacement in this set: it has zero failures and materially lower cusp median error than the bridge methods.

## Scaling Exponents

Median fitted exponents from the QBX scaling suite:

| Quantity | Median Fit |
|---|---:|
| QBX order error alpha | `0.00214` |
| QBX order effective ratio | `0.99786` |
| QBX order time power | `0.472` |
| QBX sample error power | `3.820` |
| QBX sample time power | `1.021` |
| QBX sample work power | `1.000` |
| Zeta error power | `3.015` |
| Zeta time power | `0.993` |
| Zeta cached work power | `0.931` |
| Zeta single-target work power | `0.998` |

Interpretation: increasing QBX samples gives a clear error decrease with roughly linear time/work growth, but the order sweep is nearly flat in median error for these cases. Zeta refinement shows near-linear time and single-target work scaling, while cached-target work grows slightly sublinearly in the fitted median.

## Hard Domains

Planar cases completed successfully:

| Case | Status | Bridge Improvement vs Trap |
|---|---|---:|
| `ellipse_3p3_to_1` | ok | `285.8x` |
| `funky_flower_curve` | ok | `355.2x` |
| `kidney_nonconvex_curve` | ok | `366.2x` |
| `rounded_square_superellipse` | ok | `481.4x` |
| `square_polygon` | ok | `262.1x` |
| `star_polygon` | ok | `6.7x` |
| `naca0012_airfoil` | ok | `5.9x` |
| `airplane_planar_silhouette` | ok | `77.2x` |

3D surface cases remain explicitly marked unsupported by the current planar logarithmic layer engine:

| Case | Status | Reason |
|---|---|---|
| `finite_cylinder_surface` | unsupported | Requires a 3D surface kernel, surface QJets, and surface quadrature reference |
| `cube_polyhedron` | unsupported | Requires a 3D surface kernel, surface QJets, and surface quadrature reference |
| `tetrahedron_polyhedron` | unsupported | Requires a 3D surface kernel, surface QJets, and surface quadrature reference |
| `airplane_3d_mesh` | unsupported | Requires a 3D surface kernel, surface QJets, and surface quadrature reference |
| `torus_higher_genus_surface` | unsupported | Requires a 3D surface kernel, surface QJets, and surface quadrature reference |

This is an important boundary of the current implementation: arbitrary planar domains are now supported by the planar chord-QJet path, while cylinders, polyhedra, airplanes as 3D meshes, and higher-genus surfaces still require a separate surface kernel and surface QJet stack.

## ODE Stress Suite

The ODE sheet covers:

- Stiff scalar decay: `lambda = 50, 500, 2000`
- Riccati near finite-time blow-up
- High-frequency damped oscillator
- Van der Pol relaxation oscillator with `mu = 25`
- Short Robertson-style stiff kinetics burst

ODE method summary:

| Method | Rows | Median Abs Error | Median Rel Error | Median ms |
|---|---:|---:|---:|---:|
| Explicit Euler | 7 | `5.276e-10` | `1.652e-7` | `0.122` |
| Implicit Euler scalar | 3 | `7.889e-31` | `7.889e-1` | `0.0039` |
| Q spectral exact oscillator | 1 | `0` | `0` | `0.0045` |
| Q spectral exact scalar map | 1 | `0` | `0` | `0.0101` |
| Q spectral exact semigroup | 3 | `0` | `0` | `0.0017` |
| RK4 fixed | 7 | `9.992e-16` | `2.158e-13` | `0.193` |
| Reference RK4 fine | 2 | `0` | `0` | `0.0425` |

The ODE results are primarily a stress harness, not a claim that every nonlinear ODE has a Q spectral closed form. Where exact semigroups or scalar maps are available, the Q-style propagator records zero numerical error. For nonlinear stiff systems, the workbook uses fine RK4 references and records method error and stability behavior.

## Implementation and Audit Notes

- The workbook stores benchmark outputs, not a dense Q matrix.
- The current quadrature engine path is built around generating QJets and applying Q through the borrow-compute-repay evaluation protocol.
- The disk/cycle modal PDE benchmark remains the exact-reference accuracy test.
- Arbitrary planar domains now use `PlanarDomainQJet`: a matrix-free chord-QJet representation of `Q = pi Lambda_Gamma + K0`, with Q spectral error-channel reporting and iterative PDE solves.
- Near-singular planar layer-potential evaluation uses the existing multipole/zeta error-correction method on nested point levels.
- The 3D surface cases require a separate surface kernel and surface QJet stack.

## Source Artifacts

| Artifact | Path |
|---|---|
| Workbook | `/Users/rick/Documents/New project 2/outputs/q_full_excel/q_boundary_pde_ode_benchmark.xlsx` |
| Workbook builder | `/Users/rick/Documents/New project 2/outputs/q_full_excel/build_full_workbook.mjs` |
| PDE/DtN analytic-reference benchmark JSON | `/Users/rick/Documents/New project 2/docs/assets/q_dtn_vs_fem_benchmark.json` |
| Structural methods JSON | `/Users/rick/Documents/New project 2/docs/assets/structural_quadrature_methods_benchmark.json` |
| QBX cusp JSON | `/Users/rick/Documents/New project 2/docs/assets/qbx_gulati_cusp_benchmark.json` |
| QBX scaling JSON | `/Users/rick/Documents/New project 2/docs/assets/qbx_scaling_fit.json` |
| Hard domain JSON | `/Users/rick/Documents/New project 2/docs/assets/qjet_quadrature_hard_benchmark.json` |
| Arbitrary planar Q/DtN JSON | `/Users/rick/Documents/New project 2/docs/assets/q_dtn_arbitrary_planar_benchmark.json` |
| Dashboard preview | `/Users/rick/Documents/New project 2/outputs/q_full_excel/preview_Dashboard.png` |
| QBX failure PNG | `/Users/rick/Documents/New project 2/docs/assets/qbx_failure_examples.png` |
