import ast

import inverse_shape.joukowski_endpoint as joukowski_module
from inverse_shape.joukowski_endpoint import (
    GOLDEN_MU,
    PHI,
    JoukowskiMapQJet,
    MellinEndpointChannel,
    StaticJoukowskiAnnulusQJet,
    StaticJoukowskiEllipseQJet,
    StaticMellinEndpointRepayment,
    golden_joukowski_ellipse_qjet,
    hurwitz_zeta_euler_maclaurin,
)
from inverse_shape.quadrature import PI, TAU, QJetFFTPlan, _cos, _sin, _sqrt


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _field(n):
    return tuple(
        _cos(3.0 * TAU * index / n)
        + 0.17 * _sin(7.0 * TAU * index / n)
        for index in range(n)
    )


def test_joukowski_endpoint_core_has_no_external_numerical_dependency() -> None:
    with open(joukowski_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_precise_qjet_fft_plan_round_trips_with_linear_storage() -> None:
    plan = QJetFFTPlan(64)
    values = tuple(
        complex(
            _cos(3.0 * TAU * index / 64),
            0.2 * _sin(5.0 * TAU * index / 64),
        )
        for index in range(64)
    )
    recovered = plan.ifft(plan.fft(values))
    assert _relative_error(values, recovered) < 3.0e-15
    assert plan.stored_twiddles == 63


def test_golden_map_uses_two_log_phi_and_has_the_exact_axes() -> None:
    mapping = JoukowskiMapQJet(1.0, GOLDEN_MU)
    assert abs(GOLDEN_MU - 2.0 * joukowski_module._log(PHI)) < 2.0e-15
    assert abs(mapping.radius - PHI * PHI) < 2.0e-15
    assert abs(mapping.axis_a - 3.0) < 2.0e-15
    assert abs(mapping.axis_b - _sqrt(5.0)) < 2.0e-15
    assert abs(mapping.eccentricity - 2.0 / 3.0) < 2.0e-15
    assert abs(mapping.modulation_ratio - PHI**-4) < 2.0e-15


def test_joukowski_chord_factorization_is_roundoff_accurate() -> None:
    qjet = golden_joukowski_ellipse_qjet(128)
    assert qjet.factorization_residual() < 3.0e-13


def test_static_modulated_cycle_matches_direct_pair_stream() -> None:
    for n in (16, 32, 64, 128):
        qjet = golden_joukowski_ellipse_qjet(n)
        values = _field(n)
        direct = qjet.direct_apply(values)
        static = qjet.apply(values)
        assert _relative_error(direct, static) < 8.0e-14
        assert qjet.constant_residual() == 0.0
        assert qjet.stats()["stored_dense_operator_matrix"] is False


def test_static_joukowski_operator_is_weighted_self_adjoint() -> None:
    qjet = golden_joukowski_ellipse_qjet(64)
    left = tuple(_cos(TAU * index / 64) for index in range(64))
    right = tuple(_sin(5.0 * TAU * index / 64) for index in range(64))
    q_left = qjet.apply(left)
    q_right = qjet.apply(right)
    lhs = qjet.weighted_inner(left, q_right)
    rhs = qjet.weighted_inner(q_left, right)
    assert abs(complex(lhs) - complex(rhs)) < 2.0e-12


def test_arclength_weighted_inverse_square_action_scales_as_inverse_length() -> None:
    unit = golden_joukowski_ellipse_qjet(64, scale=1.0)
    doubled = golden_joukowski_ellipse_qjet(64, scale=2.0)
    values = _field(64)
    unit_result = unit.apply(values)
    doubled_result = doubled.apply(values)
    assert _relative_error(
        unit_result,
        tuple(2.0 * complex(value) for value in doubled_result),
    ) < 2.0e-14


def test_golden_tail_is_static_and_cusp_needs_a_different_chart() -> None:
    golden = golden_joukowski_ellipse_qjet(64)
    assert golden.channel_radius == 19
    assert len(golden.channels) == 39
    assert golden.quotient_tail_bound() < 1.0e-16

    near_cusp = JoukowskiMapQJet(1.0, 0.005)
    try:
        StaticJoukowskiEllipseQJet(
            near_cusp,
            64,
            tolerance=2.0e-16,
            maximum_channel=64,
        )
    except ValueError as error:
        assert "Mellin endpoint chart" in str(error)
    else:
        raise AssertionError("near-cusp map should be routed to Mellin repayment")


def test_static_joukowski_annulus_matches_direct_without_radial_pairs() -> None:
    qjet = StaticJoukowskiAnnulusQJet(
        1.0,
        GOLDEN_MU,
        GOLDEN_MU + 0.4,
        4,
        16,
    )
    values = tuple(
        tuple(
            _cos(2.0 * TAU * column / qjet.n_theta)
            + 0.13 * _sin(TAU * row / qjet.n_scale)
            for column in range(qjet.n_theta)
        )
        for row in range(qjet.n_scale)
    )
    assert qjet.direct_relative_error(values) < 8.0e-14
    assert qjet.constant_residual() == 0.0
    stats = qjet.stats()
    assert stats["stored_dense_distance_matrix"] is False
    assert stats["stored_dense_operator_matrix"] is False
    assert stats["stored_base_symbol_entries"] < qjet.n_nodes * qjet.n_nodes


def test_joukowski_annulus_area_weighted_q2_is_scale_invariant() -> None:
    unit = StaticJoukowskiAnnulusQJet(
        1.0,
        GOLDEN_MU,
        GOLDEN_MU + 0.2,
        2,
        8,
    )
    doubled = StaticJoukowskiAnnulusQJet(
        2.0,
        GOLDEN_MU,
        GOLDEN_MU + 0.2,
        2,
        8,
    )
    values = tuple(
        tuple(
            _cos(TAU * column / unit.n_theta)
            for column in range(unit.n_theta)
        )
        for _row in range(unit.n_scale)
    )
    unit_result = tuple(value for row in unit.apply(values) for value in row)
    doubled_result = tuple(
        value for row in doubled.apply(values) for value in row
    )
    assert _relative_error(unit_result, doubled_result) < 2.0e-14


def test_static_joukowski_annulus_inverse_cube_includes_dtn_normalization() -> None:
    qjet = StaticJoukowskiAnnulusQJet(
        1.0,
        GOLDEN_MU,
        GOLDEN_MU + 0.3,
        3,
        16,
        kernel_power=3.0,
    )
    values = tuple(
        tuple(
            _cos(2.0 * TAU * column / qjet.n_theta)
            + 0.09 * _sin(TAU * row / qjet.n_scale)
            for column in range(qjet.n_theta)
        )
        for row in range(qjet.n_scale)
    )
    assert qjet.direct_relative_error(values) < 1.5e-13
    assert abs(qjet.normalization - 1.0 / TAU) < 2.0e-16
    assert qjet.stats()["kernel_power"] == 3.0
    assert qjet.constant_residual() == 0.0
    evaluation = qjet.evaluate(values)
    assert evaluation.ledger.status == "borrowed_repaid"
    assert evaluation.stats["stored_dense_operator_matrix"] is False


def test_branch_free_hurwitz_endpoint_evaluator_hits_exact_anchors() -> None:
    zeta_two, error_two = hurwitz_zeta_euler_maclaurin(2.0, 1.0)
    zeta_zero, error_zero = hurwitz_zeta_euler_maclaurin(0.0, 0.3)
    zeta_minus_one, error_minus_one = hurwitz_zeta_euler_maclaurin(-1.0, 0.3)
    assert abs(float(zeta_two) - PI * PI / 6.0) < 3.0e-15
    assert abs(float(zeta_zero) - 0.2) < 4.0e-15
    assert abs(float(zeta_minus_one) - 0.021666666666666667) < 2.0e-13
    assert error_two < 1.0e-22
    assert error_zero == 0.0
    assert error_minus_one == 0.0


def test_mellin_endpoint_repayment_is_static_and_has_the_expected_power() -> None:
    channel = MellinEndpointChannel(
        exponent=0.5,
        amplitude=2.0,
        phase=0.5,
        label="square-root cusp",
    )
    repayment = StaticMellinEndpointRepayment((channel,))
    coarse = repayment.evaluate(0.04)
    fine = repayment.evaluate(0.01)
    assert abs(complex(coarse["value"]) / complex(fine["value"]) - 2.0) < 2.0e-14
    assert fine["grid_refinement_iterations"] == 0
    assert fine["stored_dense_matrix"] is False
    assert fine["next_term_estimate"] < 1.0e-22
