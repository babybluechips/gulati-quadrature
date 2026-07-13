import ast

import inverse_shape.axisymmetric3d as axisymmetric3d_module
from inverse_shape.axisymmetric3d import (
    build_axisymmetric_surface_qjet,
    cartesian_distance_squared,
    hyperbolic_scale_phase_distance_squared,
    scale_phase_distance_squared,
    spheroid_qjet,
    torus_qjet,
)


def _max_abs(rows):
    return max(abs(complex(value)) for row in rows for value in row)


def _max_difference(left, right):
    return max(
        abs(complex(a) - complex(b))
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def _mode_rayleigh(qjet, amplitudes, applied):
    numerator = 0.0 + 0.0j
    denominator = 0.0
    for index, (value, output) in enumerate(zip(amplitudes, applied, strict=True)):
        weight = qjet.node_area_weights[index]
        numerator += weight * complex(value).conjugate() * complex(output)
        denominator += weight * abs(complex(value)) ** 2
    return float((numerator / denominator).real)


def test_axisymmetric_kernel_has_no_external_numerical_dependency() -> None:
    with open(axisymmetric3d_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_scale_phase_cylinder_identity_matches_cartesian_coordinates() -> None:
    cases = (
        (1.0, -0.4, 0.1, 1.0, 0.9, 5.7),
        (0.13, 2.0, -1.2, 4.7, -3.0, 2.4),
        (18.0, 1.0e-4, 31.0, 0.025, -8.0, -17.0),
        (2.5, 4.0, 0.0, 2.5, 4.0, 0.125),
    )
    for case in cases:
        stable = scale_phase_distance_squared(*case)
        hyperbolic = hyperbolic_scale_phase_distance_squared(*case)
        cartesian = cartesian_distance_squared(*case)
        scale = max(1.0, abs(cartesian))
        assert abs(stable - cartesian) / scale < 2.0e-13
        assert abs(hyperbolic - cartesian) / scale < 2.0e-13


def test_streamed_angular_fft_matches_pair_stream_reference() -> None:
    qjet = spheroid_qjet(1.25, 0.8, n_meridian=4, n_theta=8)
    values = []
    for ring in range(qjet.n_rings):
        row = []
        for phase in range(qjet.n_theta):
            x, y, z = qjet.cartesian_point(ring, phase)
            row.append(x + 0.2 * y - 0.3 * z * z + 0.1j * x * z)
        values.append(tuple(row))
    values = tuple(values)

    modal = qjet.apply(values)
    direct = qjet.direct_apply(values)
    assert _max_difference(modal, direct) / max(_max_abs(direct), 1.0) < 3.0e-14
    assert qjet.stats()["stored_dense_surface_matrix"] is False
    assert qjet.stats()["stored_pair_kernel_table"] is False


def test_graph_diagonal_reproduces_constant_nullspace() -> None:
    for qjet in (
        spheroid_qjet(1.0, 1.7, 8, 16),
        torus_qjet(2.0, 0.45, 8, 16),
    ):
        assert qjet.constant_residual() < 2.0e-13


def test_weighted_surface_operator_is_self_adjoint() -> None:
    qjet = spheroid_qjet(1.4, 0.75, 6, 16)
    left = []
    right = []
    for ring in range(qjet.n_rings):
        left_row = []
        right_row = []
        for phase in range(qjet.n_theta):
            x, y, z = qjet.cartesian_point(ring, phase)
            left_row.append(x + 0.3j * y + 0.2 * z)
            right_row.append(x * z - 0.4 * y + 0.1j * (x - z))
        left.append(tuple(left_row))
        right.append(tuple(right_row))
    left = tuple(left)
    right = tuple(right)
    q_left = qjet.apply(left)
    q_right = qjet.apply(right)
    lhs = complex(qjet.weighted_inner(left, q_right))
    rhs = complex(qjet.weighted_inner(q_left, right))
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 2.0e-13


def test_full_surface_inverse_cube_converges_to_sphere_dtn_better_than_inverse_square() -> None:
    cube = spheroid_qjet(1.0, 1.0, 16, 32, kernel_power=3.0)
    square = spheroid_qjet(1.0, 1.0, 16, 32, kernel_power=2.0)
    amplitudes = tuple(0.5 * (3.0 * z * z - 1.0) for z in cube.z_values)
    cube_value = _mode_rayleigh(cube, amplitudes, cube.apply_azimuthal_mode(amplitudes, 0))
    square_value = _mode_rayleigh(
        square,
        amplitudes,
        square.apply_azimuthal_mode(amplitudes, 0),
    )
    assert abs(cube_value - 2.0) / 2.0 < 0.11
    assert abs(square_value - 2.0) / 2.0 > 0.20


def test_inverse_cube_has_exact_discrete_scaling_covariance() -> None:
    unit = spheroid_qjet(1.0, 1.0, 12, 32)
    large = spheroid_qjet(2.5, 2.5, 12, 32)
    unit_mode = tuple(z for z in unit.z_values)
    large_mode = tuple(z / 2.5 for z in large.z_values)
    unit_value = _mode_rayleigh(unit, unit_mode, unit.apply_azimuthal_mode(unit_mode, 0))
    large_value = _mode_rayleigh(
        large,
        large_mode,
        large.apply_azimuthal_mode(large_mode, 0),
    )
    assert abs(unit_value - 2.5 * large_value) < 2.0e-13


def test_azimuthal_reduction_recovers_inverse_square_meridional_order() -> None:
    separation = 0.025
    inverse_cube = build_axisymmetric_surface_qjet(
        (1.0, 1.0),
        (0.0, separation),
        (1.0, 1.0),
        2048,
        kernel_power=3.0,
    )
    inverse_square = build_axisymmetric_surface_qjet(
        (1.0, 1.0),
        (0.0, separation),
        (1.0, 1.0),
        2048,
        kernel_power=2.0,
    )
    cube_reduced = inverse_cube.reduced_meridional_kernel(0, 1)
    square_reduced = inverse_square.reduced_meridional_kernel(0, 1)

    assert abs(separation * separation * cube_reduced - 2.0) < 1.0e-3
    assert abs(separation * square_reduced - 3.141592653589793) < 3.0e-4


def test_phase_shift_equivariance_is_exact_to_fft_roundoff() -> None:
    qjet = spheroid_qjet(1.1, 0.9, 6, 16)
    values = []
    for ring in range(qjet.n_rings):
        row = []
        for phase in range(qjet.n_theta):
            x, y, z = qjet.cartesian_point(ring, phase)
            row.append(x + 0.4 * y * z)
        values.append(tuple(row))
    values = tuple(values)
    shift = 5
    shifted = tuple(
        tuple(row[(phase - shift) % qjet.n_theta] for phase in range(qjet.n_theta))
        for row in values
    )
    expected = tuple(
        tuple(row[(phase - shift) % qjet.n_theta] for phase in range(qjet.n_theta))
        for row in qjet.apply(values)
    )
    actual = qjet.apply(shifted)
    assert _max_difference(expected, actual) / max(1.0, _max_abs(expected)) < 2.0e-13
