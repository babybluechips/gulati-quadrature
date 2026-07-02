import ast

import numpy as np

import inverse_shape.quadrature as quadrature_module
from inverse_shape.operators import gulati_laplacian
from inverse_shape.quadrature import (
    LocalQuadratureEvaluation,
    MultipoleLeafQJet,
    MultipoleZetaQEvaluation,
    apply_cycle_gulati,
    build_boundary_qjet,
    circle_log_layer_borrow_compute_repay,
    circle_gulati_coercivity,
    circle_log_layer_spectral,
    circle_log_layer_trapezoid,
    cycle_gulati_condition_number,
    cycle_gulati_eigenvalues,
    cycle_gulati_energy,
    cycle_gulati_fractional_power,
    cycle_gulati_heat,
    cycle_gulati_resolvent,
    gulati_incidence_factor,
    build_pullback_metric_qjet,
    log_layer_local_bridge,
    log_layer_adaptive_panel_borrow_compute_repay,
    log_layer_multipole_bridge,
    log_layer_multipole_zeta_q_borrow_compute_repay,
    log_layer_qbx,
    log_layer_qbx_auto,
    log_layer_singularity_subtraction,
    log_layer_singularity_subtraction_borrow_compute_repay,
    log_layer_trapezoid,
    near_singular_circle_table,
    outward_unit_normals,
    q_spectral_error_signature,
    regular_polygon_points,
    solve_cycle_gulati,
)


def test_quadrature_engine_has_no_import_dependencies() -> None:
    with open(quadrature_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    assert not any(isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(tree))


def test_cycle_gulati_spectrum_matches_dense_regular_polygon() -> None:
    for n in (8, 9, 16):
        gu = gulati_laplacian(regular_polygon_points(n))
        dense_eigenvalues = np.sort(np.linalg.eigvalsh(gu))
        closed_eigenvalues = np.sort(cycle_gulati_eigenvalues(n))
        assert np.allclose(dense_eigenvalues, closed_eigenvalues, atol=4e-13)
        assert np.linalg.norm(gu @ np.ones(n), ord=np.inf) < 1e-13


def test_cauchy_gram_factorization_and_energy() -> None:
    points = regular_polygon_points(10, phase=0.13)
    gu = gulati_laplacian(points)
    factor = gulati_incidence_factor(points)
    assert np.allclose(factor.T @ factor, gu, atol=1e-12)

    theta = 2.0 * np.pi * np.arange(10) / 10
    values = np.cos(2.0 * theta) + 0.25 * np.sin(3.0 * theta)
    assert np.isclose(cycle_gulati_energy(values), values @ (gu @ values), atol=1e-12)


def test_fft_gulati_application_matches_dense_matrix() -> None:
    n = 32
    theta = 2.0 * np.pi * np.arange(n) / n
    values = np.cos(3.0 * theta) - 0.4 * np.sin(7.0 * theta)
    dense = gulati_laplacian(regular_polygon_points(n)) @ values
    spectral = apply_cycle_gulati(values)
    assert np.allclose(spectral, dense, atol=2e-12)


def test_boundary_qjet_applies_without_dense_storage() -> None:
    n = 24
    theta = 2.0 * np.pi * np.arange(n) / n
    points = regular_polygon_points(n)
    values = np.cos(2.0 * theta) + 0.1 * np.sin(5.0 * theta)
    qjet = build_boundary_qjet(points)
    assert "matrix" not in vars(qjet)
    assert np.allclose(qjet.apply(values), gulati_laplacian(points) @ values, atol=2e-12)


def test_pullback_metric_fft_qjet_matches_scaled_circle_dense_matrix() -> None:
    n = 64
    radius = 2.75
    theta = 2.0 * np.pi * np.arange(n) / n
    values = np.cos(4.0 * theta) - 0.2 * np.sin(9.0 * theta)
    qjet = build_pullback_metric_qjet([radius] * n)
    dense = gulati_laplacian(regular_polygon_points(n, radius=radius)) @ values

    assert "matrix" not in vars(qjet)
    assert qjet.uses_radix2_fft
    assert np.allclose(qjet.apply(values), dense, atol=3e-12)


def test_pullback_metric_fft_qjet_matches_direct_weighted_edge_form() -> None:
    n = 32
    theta = 2.0 * np.pi * np.arange(n) / n
    speeds = 1.0 + 0.18 * np.cos(2.0 * theta) + 0.07 * np.sin(5.0 * theta)
    values = np.cos(3.0 * theta + 0.2) + 0.3 * np.sin(6.0 * theta)
    qjet = build_pullback_metric_qjet(speeds)
    direct = np.zeros(n)
    circle = np.exp(1j * theta)
    inv_speeds = 1.0 / speeds
    for i in range(n):
        for j in range(i + 1, n):
            weight = inv_speeds[i] * inv_speeds[j] / abs(circle[i] - circle[j]) ** 2
            update = weight * (values[i] - values[j])
            direct[i] += update
            direct[j] -= update

    assert qjet.stats()["protocol"] == "pullback_metric_fft_qjet"
    assert qjet.stats()["stored_dense_q_matrix"] is False
    assert np.allclose(qjet.apply(values), direct, atol=2e-11)
    assert np.linalg.norm(qjet.apply(np.ones(n)), ord=np.inf) < 2e-11


def test_boundary_solve_is_mean_zero_pseudoinverse() -> None:
    n = 128
    theta = 2.0 * np.pi * np.arange(n) / n
    rhs = np.cos(3.0 * theta) + 0.25 * np.sin(5.0 * theta)
    solution = solve_cycle_gulati(rhs)
    residual = apply_cycle_gulati(solution) - rhs
    assert abs(np.mean(solution)) < 1e-14
    assert np.linalg.norm(residual, ord=np.inf) < 1e-12


def test_functional_calculus_specializations_on_single_mode() -> None:
    n = 64
    mode = 5
    theta = 2.0 * np.pi * np.arange(n) / n
    values = np.exp(1j * mode * theta)
    eigenvalue = cycle_gulati_eigenvalues(n)[mode]

    assert np.allclose(
        cycle_gulati_heat(values, time=0.02),
        np.exp(-0.02 * eigenvalue) * values,
        atol=1e-13,
    )
    assert np.allclose(
        cycle_gulati_fractional_power(values, exponent=-0.5),
        eigenvalue**-0.5 * values,
        atol=1e-13,
    )
    assert np.allclose(
        cycle_gulati_resolvent(values, spectral_parameter=-3.0),
        (-3.0 - eigenvalue) ** -1 * values,
        atol=1e-13,
    )


def test_condition_number_has_linear_growth() -> None:
    assert np.isclose(cycle_gulati_condition_number(64), 16.253968253968253)
    assert cycle_gulati_condition_number(512) < 129.0


def test_circle_gulati_coercivity_scales_like_pi_over_delta() -> None:
    for delta in (1e-2, 1e-4, 1e-6):
        scaled = circle_gulati_coercivity(1.0 + delta) * delta / np.pi
        assert abs(scaled - 1.0) < 6e-3


def test_spectral_log_layer_is_stable_in_near_singular_regime() -> None:
    n = 1024
    phase = 0.7
    theta = 2.0 * np.pi * np.arange(n) / n
    density = np.cos(theta)
    point = (1.0 + 1e-6) * np.exp(1j * phase)
    exact = -np.pi * np.cos(phase) / abs(point)

    trapezoid = circle_log_layer_trapezoid(density, point)
    spectral = circle_log_layer_spectral(density, point)
    relative_trapezoid = abs(trapezoid - exact) / abs(exact)
    relative_spectral = abs(spectral - exact) / abs(exact)

    assert relative_trapezoid > 1e-4
    assert relative_spectral < 1e-13


def test_spectral_log_layer_emits_borrow_compute_repay_ledger() -> None:
    n = 1024
    phase = 0.7
    theta = 2.0 * np.pi * np.arange(n) / n
    density = np.cos(theta)
    point = (1.0 + 1e-6) * np.exp(1j * phase)
    exact = -np.pi * np.cos(phase) / abs(point)

    evaluation = circle_log_layer_borrow_compute_repay(density, point)
    assert evaluation.ledger.status == "borrowed_repaid"
    assert any("eigenvalues" in item for item in evaluation.ledger.repaid)
    assert len(evaluation.rational_denominators) == n - 1
    assert abs(float(evaluation.value) - exact) / abs(exact) < 1e-13


def test_near_singular_pressure_table_reports_expected_crossover() -> None:
    rows = near_singular_circle_table(n=1024, deltas=(1e-2, 1e-4, 1e-6))
    assert rows[0]["trapezoid_relative_error"] < 1e-6
    assert rows[-1]["trapezoid_relative_error"] > 1e-4
    assert max(row["spectral_relative_error"] for row in rows) < 1e-13


def test_point_qbx_matches_exact_circle_log_layer() -> None:
    n = 16384
    phase = 0.7
    delta = 1e-3
    theta = 2.0 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    density = np.cos(theta)
    target_complex = (1.0 + delta) * np.exp(1j * phase)
    center_complex = (1.0 + 4.0 * delta) * np.exp(1j * phase)
    target = np.array([target_complex.real, target_complex.imag])
    center = np.array([center_complex.real, center_complex.imag])

    qbx = log_layer_qbx(points, density, target, center, order=60)
    exact = -np.pi * np.cos(phase) / abs(target_complex)
    assert abs(qbx - exact) / abs(exact) < 1e-8


def test_auto_qbx_matches_exact_circle_log_layer() -> None:
    n = 16384
    phase = 0.0
    delta = 1e-3
    theta = 2.0 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    density = np.cos(theta)
    target_complex = 1.0 + delta
    target = np.array([target_complex, 0.0])

    qbx = log_layer_qbx_auto(points, density, target, sample_index=0, order=60)
    exact = -np.pi / target_complex
    assert abs(qbx - exact) / abs(exact) < 1e-8


def test_local_bridge_improves_hard_shape_trapezoid() -> None:
    from inverse_shape.geometry import BoundaryCurve

    n = 512
    ref_n = 32768
    theta = np.linspace(0.0, 2.0 * np.pi, 8 * ref_n, endpoint=False)
    radius = 1.0 + 0.32 * np.cos(2.0 * theta) - 0.10 * np.cos(5.0 * theta)
    dense = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    points = BoundaryCurve(dense).resample(n).points
    reference_points = BoundaryCurve(dense).resample(ref_n).points
    tau = 2.0 * np.pi * np.arange(n) / n
    ref_tau = 2.0 * np.pi * np.arange(ref_n) / ref_n
    density = 1.0 + 0.25 * np.cos(3.0 * tau + 0.2)
    reference_density = 1.0 + 0.25 * np.cos(3.0 * ref_tau + 0.2)
    sample_index = n // 7
    h = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean()
    target = points[sample_index] + 0.2 * h * outward_unit_normals(points)[sample_index]

    reference = log_layer_trapezoid(reference_points, reference_density, target)
    trapezoid = log_layer_trapezoid(points, density, target)
    bridge = log_layer_local_bridge(points, density, target, sample_index=sample_index)
    assert abs(bridge - reference) < abs(trapezoid - reference)


def test_singularity_subtraction_improves_underresolved_circle() -> None:
    n = 64
    points = regular_polygon_points(n)
    density = [1.0] * n
    target = (1.02, 0.0)
    exact = 2.0 * np.pi * np.log(1.02)

    trapezoid = log_layer_trapezoid(points, density, target)
    subtraction = log_layer_singularity_subtraction(points, density, target, sample_index=0, window=8)
    evaluation = log_layer_singularity_subtraction_borrow_compute_repay(
        points,
        density,
        target,
        sample_index=0,
        window=8,
    )

    assert isinstance(evaluation, LocalQuadratureEvaluation)
    assert evaluation.ledger.status == "borrowed_repaid"
    assert evaluation.work_units == n + 17
    assert abs(subtraction - exact) < 0.02 * abs(trapezoid - exact)


def test_adaptive_panel_refinement_reports_local_work_and_improves_circle() -> None:
    n = 64
    points = regular_polygon_points(n)
    density = [1.0] * n
    target = (1.02, 0.0)
    exact = 2.0 * np.pi * np.log(1.02)
    trapezoid = log_layer_trapezoid(points, density, target)

    evaluation = log_layer_adaptive_panel_borrow_compute_repay(
        points,
        density,
        target,
        sample_index=0,
        panel_radius=6,
        subdivisions=16,
    )

    assert isinstance(evaluation, LocalQuadratureEvaluation)
    assert evaluation.method == "adaptive_panel"
    assert evaluation.ledger.status == "borrowed_repaid"
    assert evaluation.stats["near_panels"] == 14
    assert evaluation.work_units == 14 * 16 + (n - 14)
    assert abs(evaluation.value - exact) < 0.02 * abs(trapezoid - exact)


def test_q_spectral_error_signature_classifies_smooth_cycle() -> None:
    signature = q_spectral_error_signature(regular_polygon_points(128), mode_start=4, mode_stop=16)
    assert signature.error_type == "smooth_spectral_tail"
    assert signature.recommended_q == "multipole_zeta_q"
    assert signature.median_pair_split < 1.0e-12
    assert 0.85 < signature.symbol_power < 1.05


def test_q_spectral_error_signature_detects_cusp_channel() -> None:
    n = 256
    points = []
    for index in range(n):
        theta = 2.0 * np.pi * index / n
        c = np.cos(theta)
        s = np.sin(theta)
        points.append((c * c * c, s * s * s))

    signature = q_spectral_error_signature(points, mode_start=4, mode_stop=16)

    assert signature.error_type == "cusp_endpoint_channel"
    assert signature.median_pair_split > 0.5
    assert signature.recommended_q == "multipole_zeta_q"


def test_multipole_qjet_bridge_matches_direct_bridge_when_forced_direct() -> None:
    n = 64
    points = regular_polygon_points(n)
    theta = 2.0 * np.pi * np.arange(n) / n
    density = 1.0 + 0.1 * np.cos(3.0 * theta)
    target = (1.0 + 0.01, 0.0)

    qjet = MultipoleLeafQJet(points, density, order=8, leaf_size=8, theta=0.5)
    assert "matrix" not in vars(qjet)
    assert qjet.moment_build_units == n * 8
    assert len(qjet.leaves) == 8

    direct_multipole = log_layer_multipole_bridge(
        points,
        density,
        target,
        sample_index=0,
        order=8,
        leaf_size=8,
        theta=1.0e-30,
    )
    bridge = log_layer_local_bridge(points, density, target, sample_index=0)
    assert abs(direct_multipole - bridge) < 1.0e-11


def test_multipole_zeta_q_reports_ledger_and_work_units() -> None:
    levels = (32, 64, 128)
    point_sets = [regular_polygon_points(n) for n in levels]
    density_sets = [
        1.0 + 0.2 * np.cos(2.0 * np.pi * np.arange(n) / n)
        for n in levels
    ]
    target = (1.02, 0.0)
    evaluation = log_layer_multipole_zeta_q_borrow_compute_repay(
        point_sets,
        density_sets,
        target,
        sample_indices=(0, 0, 0),
        order=8,
        leaf_size=8,
    )

    assert isinstance(evaluation, MultipoleZetaQEvaluation)
    assert evaluation.ledger.status == "borrowed_repaid"
    assert len(evaluation.levels) == 3
    assert evaluation.moment_build_units == sum(levels) * 8
    assert evaluation.single_target_work_units >= evaluation.cached_target_work_units
    assert np.isfinite(abs(evaluation.value))
