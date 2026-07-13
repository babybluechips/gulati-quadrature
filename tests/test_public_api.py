import math

from gulati_quadrature import (
    build_engine,
    cosine_trace,
    cycle_certificate,
    inverse_square_chord_from_scale_phase,
    integrate_log_layer,
    prepare_boundary,
    scale_phase_chord_squared,
    scale_phase_point,
    solve_pde,
    star_boundary,
)


def test_public_engine_uses_production_q_path_without_dense_matrix() -> None:
    points = star_boundary(64)
    values = cosine_trace(64, 4)
    engine = build_engine(points)

    dtn = engine.apply_dtn(values)
    heat = engine.solve("heat", values, time=0.02, max_steps=8)
    stats = engine.stats()

    assert dtn.ledger.status == "borrowed_repaid"
    assert heat.ledger.status == "borrowed_repaid"
    assert stats["protocol"] == "planar_chord_qjet_harmonic_zeta_repaid"
    assert stats["dense_q_matrix_stored"] is False
    assert "matrix" not in vars(engine._qjet.qjet)
    assert len(dtn.values) == 64


def test_public_one_shot_solve_and_log_layer() -> None:
    points = star_boundary(64)
    values = cosine_trace(64, 3)
    solved = solve_pde(points, values, "poisson", mass=0.4, iterations=16)
    density = tuple(1.0 + 0.1 * math.cos(2.0 * math.pi * i / 64) for i in range(64))
    layer = integrate_log_layer(points, density, (1.5, 0.1))

    assert solved.ledger.status == "borrowed_repaid"
    assert layer.ledger.status == "borrowed_repaid"
    assert layer.method == "multipole_zeta_refined_q"
    assert layer.stats["moment_build_units"] > 0


def test_prepare_boundary_removes_duplicate_endpoint_and_orients_ccw() -> None:
    points = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    prepared = prepare_boundary(points)
    assert prepared.n == 3
    assert prepared.removed_duplicate_endpoint is True
    assert prepared.orientation == "ccw"
    assert prepared.signed_area > 0.0


def test_cycle_certificate_exact_scaled_determinant() -> None:
    cert = cycle_certificate(6)
    assert cert["dense_q_matrix_stored"] is False
    assert cert["pseudo_determinant_scaled"] == math.factorial(5) ** 2
    assert cert["cofactor_scaled_by_n"] == math.factorial(5) ** 2
    assert cert["trace"] == 6 * (6 * 6 - 1) / 12
    assert cert["nullity"] == 1


def test_scale_phase_chord_identity_generalizes_unit_circle_kernel() -> None:
    theta_i = 0.25
    theta_j = 1.1
    rho_i = 0.35
    rho_j = -0.2
    vi = scale_phase_point(theta_i, rho_i)
    vj = scale_phase_point(theta_j, rho_j)
    direct = abs(vi - vj) ** 2
    stable = scale_phase_chord_squared(theta_i, rho_i, theta_j, rho_j)
    kernel = inverse_square_chord_from_scale_phase(theta_i, rho_i, theta_j, rho_j)

    assert abs(stable - direct) / direct < 1.0e-14
    assert abs(kernel - 1.0 / direct) / (1.0 / direct) < 1.0e-14

    circle = scale_phase_chord_squared(theta_i, 0.0, theta_j, 0.0)
    expected_circle = 4.0 * math.sin(0.5 * (theta_i - theta_j)) ** 2
    assert abs(circle - expected_circle) < 1.0e-14
