import ast

import inverse_shape.scale_phase_convolution as convolution_module
from inverse_shape.scale_phase_convolution import (
    cone_convolution_qjet,
    cylinder_convolution_qjet,
    koenigs_tetration_cone_qjet,
    sphere_stereographic_convolution_qjet,
)


def _field(qjet):
    rows = []
    for scale in range(qjet.n_scale):
        row = []
        radius = qjet.radii[scale]
        z_value = qjet.z_values[scale]
        for phase in range(qjet.n_theta):
            theta = qjet.theta(phase, scale)
            x = radius * convolution_module._cos(theta)
            y = radius * convolution_module._sin(theta)
            row.append(x - 0.23 * y + 0.17 * z_value + 0.09j * x * z_value)
        rows.append(tuple(row))
    return tuple(rows)


def _max_abs(rows):
    return max(abs(complex(value)) for row in rows for value in row)


def _relative_difference(left, right):
    difference = max(
        abs(complex(a) - complex(b))
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )
    return difference / max(1.0, _max_abs(right))


def test_fast_scale_phase_kernel_has_no_external_numerical_import() -> None:
    with open(convolution_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.axisymmetric3d", "inverse_shape.quadrature"]


def test_exact_normal_form_fft_matches_physical_pair_stream() -> None:
    qjets = (
        cylinder_convolution_qjet(1.1, -1.0, 1.0, 4, 8),
        cone_convolution_qjet(0.6, -1.2, 0.7, 4, 8),
        sphere_stereographic_convolution_qjet(1.3, -2.0, 2.0, 4, 8),
        koenigs_tetration_cone_qjet(-0.35, 0.21, -1.0, 1.0, 1.0, 0.6, 4, 8),
    )
    for qjet in qjets:
        values = _field(qjet)
        assert _relative_difference(qjet.apply(values), qjet.direct_apply(values)) < 8.0e-14


def test_fast_normal_forms_annihilate_constants_and_are_weighted_self_adjoint() -> None:
    for qjet in (
        cylinder_convolution_qjet(0.8, -2.0, 1.0, 6, 16),
        cone_convolution_qjet(-0.4, -1.0, 1.1, 6, 16),
        sphere_stereographic_convolution_qjet(1.0, -2.5, 2.5, 6, 16),
    ):
        left = _field(qjet)
        right = tuple(
            tuple(complex(value).conjugate() + 0.2j for value in row)
            for row in reversed(left)
        )
        q_left = qjet.apply(left)
        q_right = qjet.apply(right)
        lhs = complex(qjet.weighted_inner(left, q_right))
        rhs = complex(qjet.weighted_inner(q_left, right))
        assert qjet.constant_residual() < 2.0e-13
        assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 5.0e-13


def test_repaid_fast_path_remains_self_adjoint_under_tetration_shear() -> None:
    qjet = koenigs_tetration_cone_qjet(
        -0.31,
        0.27,
        -1.0,
        1.0,
        1.1,
        0.52,
        6,
        16,
    )
    left = _field(qjet)
    right = tuple(
        tuple(complex(value).conjugate() - 0.13j for value in row)
        for row in reversed(left)
    )
    q_left = qjet.apply_repaid(left)
    q_right = qjet.apply_repaid(right)
    lhs = complex(qjet.weighted_inner(left, q_right))
    rhs = complex(qjet.weighted_inner(q_left, right))
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 2.0e-12
    constant = tuple(
        tuple(1.0 for _ in range(qjet.n_theta)) for _ in range(qjet.n_scale)
    )
    assert _max_abs(qjet.apply_repaid(constant)) < 3.0e-13


def test_generated_symbol_and_three_jets_have_linear_storage() -> None:
    for n_scale, n_theta in ((8, 16), (16, 32), (32, 64)):
        qjet = cone_convolution_qjet(0.5, -1.0, 1.0, n_scale, n_theta)
        stats = qjet.stats()
        assert stats["stored_dense_surface_matrix"] is False
        assert stats["stored_pair_kernel_table"] is False
        assert stats["generated_symbol_entries"] <= 4 * qjet.n_nodes
        assert stats["stored_three_jets"] == n_scale
        assert stats["stored_three_jet_scalars"] == 13 * n_scale
        assert stats["apply_complexity"] == "O(N log N)"
        assert stats["storage_complexity"] == "O(N)"


def test_koenigs_tetration_height_is_affine_in_scale_and_phase() -> None:
    alpha = -0.37
    beta = 0.23
    qjet = koenigs_tetration_cone_qjet(
        alpha,
        beta,
        -1.0,
        1.0,
        1.2,
        0.45,
        8,
        16,
    )
    for previous, following in zip(qjet.three_jets[:-1], qjet.three_jets[1:], strict=True):
        delta_height = following.coordinate - previous.coordinate
        log_ratio = convolution_module._log(following.radius / previous.radius)
        phase_delta = following.phase - previous.phase
        assert abs(log_ratio - alpha * delta_height) < 2.0e-14
        assert abs(phase_delta - beta * delta_height) < 2.0e-14
        assert following.radius_derivatives == (
            alpha * following.radius,
            alpha * alpha * following.radius,
            alpha**3 * following.radius,
        )


def test_fast_evaluation_emits_borrow_compute_repay_audit() -> None:
    qjet = sphere_stereographic_convolution_qjet(1.0, -2.0, 2.0, 8, 16)
    evaluation = qjet.evaluate(_field(qjet))
    assert evaluation.ledger.status == "borrowed_repaid"
    assert evaluation.stats["stored_dense_surface_matrix"] is False
    assert any("three-jet" in item for item in evaluation.ledger.borrowed)
    assert any("two-dimensional" in item for item in evaluation.ledger.computed)
