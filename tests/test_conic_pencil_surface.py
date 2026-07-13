import ast

import inverse_shape.conic_pencil_surface as conic_module
from inverse_shape.conic_pencil_surface import (
    aircraft_conic_bundle_qjet,
    bent_conic_tube_qjet,
    straight_conic_tube_qjet,
    tapered_conic_tube_qjet,
    toroidal_conic_bundle_qjet,
    twisted_ellipse_tube_qjet,
)
from inverse_shape.quadrature import PI, TAU, _cos, _log, _sin


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _field(qjet):
    return tuple(
        _cos(3.0 * TAU * index / qjet.n_nodes)
        + 0.2 * _sin(7.0 * TAU * index / qjet.n_nodes)
        for index in range(qjet.n_nodes)
    )


def _direction(qjet):
    return tuple(
        tuple(
            0.025 * _sin(TAU * slice_index / qjet.n_slices + 0.2 * parameter)
            for parameter in range(qjet.parameter_count_per_slice)
        )
        for slice_index in range(qjet.n_slices)
    )


def test_conic_surface_kernel_has_no_external_numerical_dependency() -> None:
    with open(conic_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert sorted(imported) == [
        "inverse_shape.joukowski_endpoint",
        "inverse_shape.quadrature",
    ]


def test_generated_cylinder_area_and_linear_storage() -> None:
    qjet = straight_conic_tube_qjet(3.0, 1.0, 1.0, 12, 32)
    nodes = qjet.generate_nodes()
    assert abs(sum(nodes.weights) - 6.0 * PI) < 2.0e-12
    stats = qjet.stats()
    assert stats["stored_geometry_scalars"] == 36 * qjet.n_slices
    assert stats["generated_surface_entries_per_apply"] == qjet.n_nodes
    assert stats["stored_dense_distance_matrix"] is False
    assert stats["stored_dense_operator_matrix"] is False
    assert stats["stored_dense_reduced_hessian"] is False


def test_tree_matches_pair_stream_for_inverse_square_and_inverse_cube() -> None:
    shapes = (
        straight_conic_tube_qjet(3.0, 1.0, 0.7, 8, 16),
        tapered_conic_tube_qjet(3.0, 0.45, 1.0, 0.3, 0.7, 8, 16),
        bent_conic_tube_qjet(3.0, 1.2, 0.5, 0.35, 8, 16),
        twisted_ellipse_tube_qjet(3.0, 0.8, 0.35, 1.5, 8, 16),
        toroidal_conic_bundle_qjet(2.0, 0.4, 0.25, 8, 16),
        aircraft_conic_bundle_qjet(4.0, 8, 16),
    )
    for qjet in shapes:
        values = _field(qjet)
        for power in (2.0, 3.0):
            direct = qjet.apply(values, kernel_power=power, method="direct")
            tree = qjet.apply(
                values,
                kernel_power=power,
                method="tree",
                opening=0.30,
                leaf_size=8,
            )
            assert _relative_error(direct, tree) < 1.5e-3


def test_graph_nullspace_and_weighted_self_adjointness() -> None:
    qjet = bent_conic_tube_qjet(3.0, 1.1, 0.45, 0.3, 7, 14)
    constant = qjet.apply((1.0,) * qjet.n_nodes, method="tree")
    assert max(abs(complex(value)) for value in constant) < 1.0e-13

    nodes = qjet.generate_nodes()
    left = tuple(_cos(TAU * index / qjet.n_nodes) for index in range(qjet.n_nodes))
    right = tuple(_sin(3.0 * TAU * index / qjet.n_nodes) for index in range(qjet.n_nodes))
    q_left = qjet.apply(left, method="direct")
    q_right = qjet.apply(right, method="direct")
    lhs = sum(
        nodes.weights[index] * left[index] * complex(q_right[index]).real
        for index in range(qjet.n_nodes)
    )
    rhs = sum(
        nodes.weights[index] * complex(q_left[index]).real * right[index]
        for index in range(qjet.n_nodes)
    )
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 2.0e-13


def test_inverse_square_and_inverse_cube_have_distinct_scale_covariance() -> None:
    unit = straight_conic_tube_qjet(3.0, 1.0, 0.7, 7, 14)
    doubled = straight_conic_tube_qjet(6.0, 2.0, 1.4, 7, 14)
    values = _field(unit)
    unit_square = unit.apply(values, kernel_power=2.0, method="direct")
    doubled_square = doubled.apply(values, kernel_power=2.0, method="direct")
    unit_cube = unit.apply(values, kernel_power=3.0, method="direct")
    doubled_cube = doubled.apply(values, kernel_power=3.0, method="direct")
    assert _relative_error(unit_square, doubled_square) < 3.0e-14
    assert _relative_error(
        unit_cube,
        tuple(2.0 * complex(value) for value in doubled_cube),
    ) < 3.0e-14


def test_named_dtn_principal_path_includes_two_pi_normalization() -> None:
    qjet = straight_conic_tube_qjet(3.0, 0.8, 0.6, 7, 14)
    values = _field(qjet)
    raw = qjet.apply(values, kernel_power=3.0, method="direct")
    normalized = qjet.apply_dtn_principal(values, method="direct")
    assert _relative_error(
        normalized,
        tuple(complex(value) / TAU for value in raw),
    ) < 3.0e-14


def test_static_joukowski_slice_channel_matches_exact_surface_ring_pairs() -> None:
    qjet = twisted_ellipse_tube_qjet(3.0, 0.8, 0.35, 1.2, 5, 16)
    nodes = qjet.generate_nodes()
    values = _field(qjet)
    static = qjet.apply_same_slice_joukowski(values)
    direct = [0.0 for _ in range(qjet.n_nodes)]
    for slice_index in range(qjet.n_slices):
        start = slice_index * qjet.n_theta
        stop = start + qjet.n_theta
        for left in range(start, stop):
            for right in range(left + 1, stop):
                distance_squared = sum(
                    (
                        nodes.points[left][axis]
                        - nodes.points[right][axis]
                    )
                    ** 2
                    for axis in range(3)
                )
                kernel = 1.0 / distance_squared
                difference = values[left] - values[right]
                direct[left] += nodes.weights[right] * kernel * difference
                direct[right] -= nodes.weights[left] * kernel * difference
    assert _relative_error(direct, static) < 8.0e-14
    constant = qjet.apply_same_slice_joukowski((1.0,) * qjet.n_nodes)
    assert max(abs(complex(value)) for value in constant) == 0.0
    assert qjet.last_apply_stats["cross_slice_interactions_included"] is False
    assert qjet.last_apply_stats["stored_dense_matrix"] is False


def test_static_same_slice_cycle_fallback_matches_circular_rings() -> None:
    qjet = straight_conic_tube_qjet(3.0, 0.7, 0.7, 6, 16)
    nodes = qjet.generate_nodes()
    values = _field(qjet)
    static = qjet.apply_same_slice_joukowski(values)
    direct = [0.0 for _ in range(qjet.n_nodes)]
    for slice_index in range(qjet.n_slices):
        start = slice_index * qjet.n_theta
        stop = start + qjet.n_theta
        for left in range(start, stop):
            for right in range(left + 1, stop):
                distance_squared = sum(
                    (
                        nodes.points[left][axis]
                        - nodes.points[right][axis]
                    )
                    ** 2
                    for axis in range(3)
                )
                kernel = 1.0 / distance_squared
                difference = values[left] - values[right]
                direct[left] += nodes.weights[right] * kernel * difference
                direct[right] -= nodes.weights[left] * kernel * difference
    assert _relative_error(direct, static) < 8.0e-14


def test_static_same_slice_handles_transposed_ellipse_axes() -> None:
    qjet = straight_conic_tube_qjet(3.0, 0.35, 0.8, 5, 16)
    nodes = qjet.generate_nodes()
    values = _field(qjet)
    static = qjet.apply_same_slice_joukowski(values)
    direct = [0.0 for _ in range(qjet.n_nodes)]
    for slice_index in range(qjet.n_slices):
        start = slice_index * qjet.n_theta
        stop = start + qjet.n_theta
        for left in range(start, stop):
            for right in range(left + 1, stop):
                distance_squared = sum(
                    (
                        nodes.points[left][axis]
                        - nodes.points[right][axis]
                    )
                    ** 2
                    for axis in range(3)
                )
                kernel = 1.0 / distance_squared
                difference = values[left] - values[right]
                direct[left] += nodes.weights[right] * kernel * difference
                direct[right] -= nodes.weights[left] * kernel * difference
    assert _relative_error(direct, static) < 8.0e-14


def test_conic_pencil_log_determinant_curvature_identity() -> None:
    qjet = tapered_conic_tube_qjet(3.0, 0.5, 1.0, 0.3, 0.8, 6, 12)
    left = qjet.slice_jets[1]
    right = qjet.slice_jets[2]
    parameter = 0.37
    step = 1.0e-4
    center = conic_module.conic_pencil_certificate(left, right, parameter)
    before = conic_module.conic_pencil_certificate(left, right, parameter - step)
    after = conic_module.conic_pencil_certificate(left, right, parameter + step)
    numerical = -(
        _log(abs(after["determinant"]))
        - 2.0 * _log(abs(center["determinant"]))
        + _log(abs(before["determinant"]))
    ) / (step * step)
    assert abs(numerical - center["minus_second_log_det"]) < 2.0e-6
    assert center["degenerate"] is False


def test_shape_jvp_vjp_are_exact_adjoints_and_hessian_is_positive() -> None:
    qjet = bent_conic_tube_qjet(3.0, 1.2, 0.5, 0.35, 6, 12)
    nodes = qjet.generate_nodes()
    direction = _direction(qjet)
    displacement = qjet.jvp(direction, nodes)
    forces = tuple(
        (
            0.2 * _cos(TAU * index / qjet.n_nodes),
            0.1 * _sin(2.0 * TAU * index / qjet.n_nodes),
            0.15 * _cos(3.0 * TAU * index / qjet.n_nodes),
        )
        for index in range(qjet.n_nodes)
    )
    lhs = sum(
        nodes.weights[index]
        * sum(displacement[index][axis] * forces[index][axis] for axis in range(3))
        for index in range(qjet.n_nodes)
    )
    rhs = qjet.parameter_inner(direction, qjet.vjp(forces, nodes))
    assert abs(lhs - rhs) < 3.0e-15
    assert (
        qjet.shape_energy(
            direction,
            method="direct",
            ridge=1.0e-3,
            shell_smoothness=1.0e-2,
        )
        > 0.0
    )


def test_shape_jvp_matches_generated_surface_central_difference() -> None:
    qjet = twisted_ellipse_tube_qjet(3.0, 0.8, 0.35, 1.2, 6, 12)
    direction = _direction(qjet)
    exact = qjet.jvp(direction)
    step = 1.0e-6
    plus = qjet.deformed(direction, step=step).generate_nodes()
    minus = qjet.deformed(direction, step=-step).generate_nodes()
    numerical = tuple(
        tuple(
            (plus.points[index][axis] - minus.points[index][axis])
            / (2.0 * step)
            for axis in range(3)
        )
        for index in range(qjet.n_nodes)
    )
    error = sum(
        (exact[index][axis] - numerical[index][axis]) ** 2
        for index in range(qjet.n_nodes)
        for axis in range(3)
    ) ** 0.5
    scale = sum(
        exact[index][axis] ** 2
        for index in range(qjet.n_nodes)
        for axis in range(3)
    ) ** 0.5
    assert error / scale < 4.0e-9


def test_shape_load_solver_warps_the_conic_bundle_without_a_dense_hessian() -> None:
    qjet = bent_conic_tube_qjet(3.0, 1.0, 0.45, 0.32, 6, 12)
    load = tuple(
        tuple(
            (
                0.02 * _sin(PI * index / (qjet.n_slices - 1))
                if parameter == 0
                else -0.01 * _sin(PI * index / (qjet.n_slices - 1))
                if parameter == 4
                else 0.0
            )
            for parameter in range(qjet.parameter_count_per_slice)
        )
        for index in range(qjet.n_slices)
    )
    response, stats = qjet.solve_shape_load(
        load,
        method="direct",
        ridge=0.1,
        shell_smoothness=0.05,
        iterations=32,
    )
    deformed = qjet.deformed(response, step=0.25)
    assert stats["relative_residual"] < 1.0e-6
    assert stats["stored_dense_reduced_hessian"] is False
    assert max(abs(row[0]) for row in response) > 1.0e-2
    assert any(
        abs(new.center[0] - old.center[0]) > 1.0e-3
        for old, new in zip(qjet.slice_jets, deformed.slice_jets, strict=True)
    )
    assert any(
        abs(new.axes[1] - old.axes[1]) > 1.0e-4
        for old, new in zip(qjet.slice_jets, deformed.slice_jets, strict=True)
    )
