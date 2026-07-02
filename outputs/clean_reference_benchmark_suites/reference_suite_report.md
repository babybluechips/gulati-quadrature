# Clean Reference Benchmark Suites

This report separates exact continuum references from method-specific approximations.
The production final-Q path itself is the no-NumPy custom QJet FFT pipeline; this benchmark harness uses NumPy/SciPy only for competitor FEM/QBX/reference accounting.

## Claim Boundary

Production final Q rows use the custom FFT with the finite-cycle endpoint defect repaid to the exact circle/conformal-pullback symbol before continuum error claims are made. Raw finite-cycle, arbitrary-chart proxy, and pre-corner-fix rows are retained as diagnostics, not as production-Q failures. Manufactured funky rows compare every method to the analytic manufactured normal derivative, not to FEM. FEM is a volumetric baseline with its own discretization error. QBX rows are layer-potential value tests only, not DtN solves. Corner rows use exact weak fluxes so point normals at singular vertices are not assigned.

## Reference Hierarchy

| role | examples in this suite | status |
|---|---|---|
| cited/standard ground truth | unit-disk Steklov/DtN modes, DLMF Bessel/Hankel disk Helmholtz modes, NIST/Motz/L-shaped benchmarks once ported verbatim | authoritative for error |
| local diagnostic target | repo-internal manufactured fluxes, arbitrary-chart proxies, uncorrected corner stress tests | useful for debugging; not headline ground truth |
| overresolved numerical reference | high-resolution trapezoid for the QBX layer-potential value control | numerical reference candidate; resolution stated |
| competitor baseline | FEM Schur DtN, refined QBX, coarse trapezoid | compared to the same reference, never treated as truth |
| control row | continuum Fourier/conformal pullback formulas and local bridge diagnostics | alignment/proxy diagnostic, not a competitor |

## Held-Out Scientific Benchmark Registry

Registry artifact: `/Users/rick/Documents/New project 2/outputs/standard_scientific_benchmarks/benchmark_registry.json`

Rows are allowed to support headline ground-truth claims only when they cite one of these external benchmark ids or implement the associated standard formula/protocol verbatim.

| id | accepted reference | repo status |
|---|---|---|
| unit_disk_steklov_dtn | Steklov eigenvalues 0,1,1,2,2,... with harmonic traces r^k cos(k theta), r^k sin(k theta) | implemented in the repaid disk DtN rows and exact modal PDE calibration |
| nistir7668_poly_poisson | exact polynomial solution and derived forcing/boundary data | registered; should be ported into the Q/FEM comparison before any square-domain accuracy claim |
| nistir7668_singular_elliptic_suite | published exact solutions or stated benchmark data | registered; replace ad-hoc funky manufactured cases when exact NIST formulas are ported |
| motz_laplace_corner | Rosser-Papamichael analytical solution coefficients as cited in the benchmark literature | diagnostic Motz-like corner rows exist; not yet accepted as ground truth until exact Motz geometry and coefficients are ported |
| l_shape_laplacian_eigenvalue | first eigenvalue lambda_1 approximately 9.6397238440219 | registered; current corner flux rows are not this eigenvalue benchmark |
| gww_isospectral_drums | Dirichlet Laplacian spectra are identical by transplantation; exact numeric eigenvalues are nontrivial and not the ground-truth claim | Q-spectrum separation experiment exists; must be described as a Q-operator diagnostic, not a contradiction of Dirichlet isospectrality |
| qbx_layer_potential_jcp2013 | overresolved or analytic layer-potential values for the stated QBX test cases | QBX comparison rows exist; accepted ground truth requires matching the published geometry, density, target protocol, and reference resolution |
| dlmf_disk_bessel_helmholtz | Bessel/Hankel modal ratios such as k J'_m(k)/J_m(k) and k H_m'(k)/H_m(k) | implemented in the disk Helmholtz/DtN scripts; should be labeled as closed-form standard reference, not FEM truth |
| helsing_ojala_corner_integral | published corner singular quadrature and compressed inverse-preconditioning results | registered; current corner-fix demos need exact protocol alignment before becoming ground truth rows |

## Machine-Precision Gate Crosscheck

The machine-precision gate is the production claim. Rows outside `production_final_q` are diagnostic stress tests: raw finite-cycle dispersion, arbitrary-chart proxy mismatch, FEM/QBX baselines, or pre-corner-fix singular behavior.

| machine gate quantity | value |
|---|---:|
| passed | true |
| max split relative error | 5.094e-14 |
| max BGK-8 relative error | 2.231e-13 |
| max generated-Q PDE residual | 7.837e-13 |
| machine tolerance | 1.000e-12 |

## Cost Model

| method class | asymptotic work | storage | accounting note |
|---|---|---|---|
| production_final_q | O(n log n) custom FFT plus O(n) endpoint/moment/zeta and metric repayment | O(n) QJets, no dense Q | exact disk/conformal-pullback/modal rows after the continuum symbol has been repaid |
| q_finite_cycle_diagnostic | O(n log n) custom FFT | O(n) QJets | raw cycle dispersion m-m^2/n before repayment |
| q_pullback_proxy_diagnostic | O(n log n) custom FFT plus O(n) metric repayment | O(n) QJets | shows mismatch when an arbitrary curve is treated as if its parameter were the exact Riemann map |
| q_corner_uncorrected_diagnostic | O(n log n) custom FFT plus O(n) arclength repayment | O(n) QJets | shows behavior before Kondratev/Mellin corner singular correction |
| competitor_fem | assembly O(T), sparse factorization about O(N_i^{3/2}) in typical 2D meshes, explicit Schur apply O(n^2) here | mesh plus sparse factors/Schur | build and apply are reported separately where available |
| competitor_qbx | O(p N) per target in this direct refined check | O(N) samples | many-target production QBX would need FMM/acceleration |
| competitor_quadrature | O(n) per target | O(n) samples | direct trapezoid/local bridge near-boundary controls |
| control | O(n log n) or analytic modal scaling | varies | alignment diagnostics, not head-to-head competitors |

## Suite Summary

| suite | cases | finite errors | median rel L2 | max rel L2 |
|---|---:|---:|---:|---:|
| corner_singularities | 10 | 10 | 6.734e-01 | 1.547e+00 |
| disk_ellipse_dtn | 35 | 35 | 2.436e-04 | 3.528e-02 |
| manufactured_funky | 20 | 20 | 2.368e-01 | 1.430e+00 |
| qbx_near_boundary_control | 6 | 6 | 4.051e-03 | 4.685e-02 |

## Production Final Q Head-To-Head

| method | cases | median rel L2 | max rel L2 | median ms | interpretation |
|---|---:|---:|---:|---:|---|
| final_q_repaid_bessel_modal_dtn | 5 | 0.000e+00 | 0.000e+00 | 0.015 | exact disk Helmholtz modal formula after repaid order |
| final_q_repaid_custom_fft_circle_dtn | 5 | 2.598e-13 | 1.034e-12 | 0.654 | exact disk DtN after endpoint/moment repayment |
| final_q_repaid_custom_fft_conformal_metric_repay | 5 | 2.597e-13 | 9.981e-13 | 0.661 | exact conic pullback with metric repayment |

## Production Final Q By Suite

| suite | method | cases | median rel L2 | max rel L2 | median ms |
|---|---|---:|---:|---:|---:|
| disk_ellipse_dtn | final_q_repaid_bessel_modal_dtn | 5 | 0.000e+00 | 0.000e+00 | 0.015 |
| disk_ellipse_dtn | final_q_repaid_custom_fft_circle_dtn | 5 | 2.598e-13 | 1.034e-12 | 0.654 |
| disk_ellipse_dtn | final_q_repaid_custom_fft_conformal_metric_repay | 5 | 2.597e-13 | 9.981e-13 | 0.661 |

## Method Class Summary

| method class | cases | median rel L2 | max rel L2 | median ms |
|---|---:|---:|---:|---:|
| competitor_fem | 17 | 8.211e-03 | 1.547e+00 | 0.037 |
| competitor_qbx | 2 | 2.935e-05 | 4.693e-05 | 15.944 |
| competitor_quadrature | 2 | 2.677e-02 | 4.685e-02 | 0.567 |
| control | 6 | 2.309e-01 | 1.430e+00 | 0.327 |
| production_final_q | 15 | 1.268e-13 | 1.034e-12 | 0.652 |
| q_corner_uncorrected_diagnostic | 6 | 6.734e-01 | 7.817e-01 | 0.339 |
| q_finite_cycle_diagnostic | 15 | 7.813e-03 | 3.125e-02 | 0.656 |
| q_pullback_proxy_diagnostic | 8 | 1.006e+00 | 1.422e+00 | 0.336 |

## Method Summary

| method | class | cases | median rel L2 | max rel L2 | median ms |
|---|---|---:|---:|---:|---:|
| fem_cloud_schur | competitor_fem | 4 | 5.663e-01 | 1.547e+00 | 0.012 |
| fem_radial_fan_schur | competitor_fem | 9 | 1.686e-03 | 3.528e-02 | 0.094 |
| fem_true_helmholtz_schur | competitor_fem | 4 | 2.233e-02 | 5.611e-02 | 0.026 |
| final_q_repaid_bessel_modal_dtn | production_final_q | 5 | 0.000e+00 | 0.000e+00 | 0.015 |
| final_q_repaid_custom_fft_circle_dtn | production_final_q | 5 | 2.598e-13 | 1.034e-12 | 0.654 |
| final_q_repaid_custom_fft_conformal_metric_repay | production_final_q | 5 | 2.597e-13 | 9.981e-13 | 0.661 |
| q_boundary_helmholtz_resolvent_proxy | q_pullback_proxy_diagnostic | 4 | 1.026e+00 | 1.092e+00 | 0.332 |
| q_continuum_pullback_proxy | control | 4 | 5.174e-01 | 1.430e+00 | 0.308 |
| q_local_bridge | control | 2 | 2.104e-02 | 4.066e-02 | 0.734 |
| qbx_refined | competitor_qbx | 2 | 2.935e-05 | 4.693e-05 | 15.944 |
| raw_cycle_q_arclength_repay_before_corner_fix | q_corner_uncorrected_diagnostic | 6 | 6.734e-01 | 7.817e-01 | 0.339 |
| raw_cycle_q_generated_custom_fft_cycle_dtn | q_finite_cycle_diagnostic | 5 | 7.813e-03 | 3.125e-02 | 0.671 |
| raw_cycle_q_generated_custom_fft_metric_repay | q_finite_cycle_diagnostic | 5 | 7.813e-03 | 3.125e-02 | 0.684 |
| raw_cycle_q_generated_order_bessel_modal_dtn | q_finite_cycle_diagnostic | 5 | 1.031e-02 | 1.685e-02 | 0.015 |
| raw_cycle_q_metric_repay_manufactured_proxy | q_pullback_proxy_diagnostic | 4 | 5.127e-01 | 1.422e+00 | 0.336 |
| trapezoid_coarse | competitor_quadrature | 2 | 2.677e-02 | 4.685e-02 | 0.567 |

## Worst Finite Rows

| suite | case | method | class | equation | n | rel L2 | notes |
|---|---|---|---|---|---:|---:|---|
| corner_singularities | motz_mixed_boundary_singularity | fem_cloud_schur | competitor_fem | laplace_dtn_singular | 256 | 1.547e+00 | lambda=1/2 Motz-type mixed Dirichlet/Neumann singularity |
| manufactured_funky | eccentric_ellipse_external_log | q_continuum_pullback_proxy | control | laplace_dtn_manufactured | 256 | 1.430e+00 | removes generated-symbol error but not chart/operator mismatch |
| manufactured_funky | eccentric_ellipse_external_log | raw_cycle_q_metric_repay_manufactured_proxy | q_pullback_proxy_diagnostic | laplace_dtn_manufactured | 256 | 1.422e+00 | final generated Q applied with boundary-speed repayment; exact flux exposes chart/operator mismatch |
| manufactured_funky | circle_r1_hankel_sources | q_boundary_helmholtz_resolvent_proxy | q_pullback_proxy_diagnostic | helmholtz_dtn_manufactured | 256 | 1.092e+00 | final Q Helmholtz spectral resolvent, not a physical Helmholtz DtN; exact flux exposes operator mismatch |
| manufactured_funky | peanut_gear_hankel_sources | q_boundary_helmholtz_resolvent_proxy | q_pullback_proxy_diagnostic | helmholtz_dtn_manufactured | 256 | 1.033e+00 | final Q Helmholtz spectral resolvent, not a physical Helmholtz DtN; exact flux exposes operator mismatch |
| manufactured_funky | funky_flower_hankel_sources | q_boundary_helmholtz_resolvent_proxy | q_pullback_proxy_diagnostic | helmholtz_dtn_manufactured | 256 | 1.020e+00 | final Q Helmholtz spectral resolvent, not a physical Helmholtz DtN; exact flux exposes operator mismatch |
| manufactured_funky | eccentric_ellipse_hankel_sources | q_boundary_helmholtz_resolvent_proxy | q_pullback_proxy_diagnostic | helmholtz_dtn_manufactured | 256 | 9.925e-01 | final Q Helmholtz spectral resolvent, not a physical Helmholtz DtN; exact flux exposes operator mismatch |
| corner_singularities | motz_mixed_boundary_singularity | fem_cloud_schur | competitor_fem | laplace_dtn_singular | 128 | 7.993e-01 | lambda=1/2 Motz-type mixed Dirichlet/Neumann singularity |
| corner_singularities | l_shape_reentrant_270 | raw_cycle_q_arclength_repay_before_corner_fix | q_corner_uncorrected_diagnostic | laplace_dtn_singular | 512 | 7.817e-01 | lambda=2/3 Kondratev reentrant-corner singularity; tests final Q before the Kondratev/Mellin corner correction layer |
| corner_singularities | l_shape_reentrant_270 | raw_cycle_q_arclength_repay_before_corner_fix | q_corner_uncorrected_diagnostic | laplace_dtn_singular | 256 | 7.765e-01 | lambda=2/3 Kondratev reentrant-corner singularity; tests final Q before the Kondratev/Mellin corner correction layer |
| corner_singularities | l_shape_reentrant_270 | raw_cycle_q_arclength_repay_before_corner_fix | q_corner_uncorrected_diagnostic | laplace_dtn_singular | 128 | 7.689e-01 | lambda=2/3 Kondratev reentrant-corner singularity; tests final Q before the Kondratev/Mellin corner correction layer |
| manufactured_funky | funky_flower_external_log | q_continuum_pullback_proxy | control | laplace_dtn_manufactured | 256 | 6.136e-01 | removes generated-symbol error but not chart/operator mismatch |
