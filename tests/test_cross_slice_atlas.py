import ast

import inverse_shape.cross_slice_atlas as atlas_module
from inverse_shape.conic_pencil_surface import (
    aircraft_conic_bundle_qjet,
    bent_conic_tube_qjet,
    straight_conic_tube_qjet,
    tapered_conic_tube_qjet,
    toroidal_conic_bundle_qjet,
    twisted_ellipse_tube_qjet,
)
from inverse_shape.cross_slice_atlas import StaticCrossSliceAtlasQJet
from inverse_shape.quadrature import TAU, _cos, _sin


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _field(qjet):
    nodes = qjet.generate_nodes()
    return tuple(
        point[0]
        + 0.17 * point[1]
        - 0.09 * point[2] * point[2]
        + 0.04 * point[0] * point[2]
        for point in nodes.points
    )


def _shapes(n_slices=6, n_theta=16):
    return (
        straight_conic_tube_qjet(3.0, 0.7, 0.7, n_slices, n_theta),
        tapered_conic_tube_qjet(
            3.2,
            0.35,
            0.85,
            0.22,
            0.55,
            n_slices,
            n_theta,
        ),
        bent_conic_tube_qjet(
            3.0,
            1.25,
            0.48,
            0.31,
            n_slices,
            n_theta,
        ),
        twisted_ellipse_tube_qjet(
            3.2,
            0.78,
            0.30,
            1.6,
            n_slices,
            n_theta,
        ),
        toroidal_conic_bundle_qjet(
            2.0,
            0.40,
            0.23,
            n_slices,
            n_theta,
        ),
        aircraft_conic_bundle_qjet(4.0, n_slices, n_theta),
    )


def test_cross_atlas_has_no_external_numerical_dependency() -> None:
    with open(atlas_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert sorted(imported) == [
        "inverse_shape.conic_pencil_surface",
        "inverse_shape.quadrature",
    ]


def test_phase_difference_chart_is_exact_for_circular_cross_slices() -> None:
    qjet = straight_conic_tube_qjet(4.0, 0.6, 0.6, 7, 16)
    values = _field(qjet)
    atlas = StaticCrossSliceAtlasQJet(
        qjet,
        kernel_power=2.0,
        tolerance=1.0e-11,
        admissibility=0.2,
        local_slice_span=1,
    )
    assert all(not chart.direct for chart in atlas.phase_charts)
    assert all(len(chart.modes) == 1 for chart in atlas.phase_charts)
    direct = qjet.apply(values, kernel_power=2.0, method="direct")
    static = atlas.apply(values)
    assert _relative_error(direct, static) < 4.0e-14


def test_full_curved_atlas_matches_streamed_q2_and_q3_on_six_shapes() -> None:
    for qjet in _shapes():
        values = _field(qjet)
        for power in (2.0, 3.0):
            atlas = StaticCrossSliceAtlasQJet(
                qjet,
                kernel_power=power,
                tolerance=1.0e-10,
                admissibility=0.3,
                leaf_nodes=8,
                local_slice_span=1,
            )
            direct = qjet.apply(values, kernel_power=power, method="direct")
            static = atlas.apply(values)
            assert _relative_error(direct, static) < 8.0e-12
            assert atlas.constant_residual() < 2.0e-13
            stats = atlas.stats()
            assert stats["cross_pair_partition_residual"] == 0
            assert stats["temporary_pair_table_entries"] == 0
            assert stats["stored_dense_distance_matrix"] is False
            assert stats["stored_dense_operator_matrix"] is False


def test_larger_atlas_retains_low_rank_blocks_at_machine_scale() -> None:
    qjet = twisted_ellipse_tube_qjet(5.0, 0.5, 0.2, 1.8, 32, 16)
    values = tuple(
        _cos(3.0 * TAU * index / qjet.n_nodes)
        + 0.2 * _sin(7.0 * TAU * index / qjet.n_nodes)
        for index in range(qjet.n_nodes)
    )
    atlas = StaticCrossSliceAtlasQJet(
        qjet,
        kernel_power=2.0,
        tolerance=1.0e-10,
        admissibility=0.3,
        leaf_nodes=8,
        local_slice_span=1,
    )
    stats = atlas.stats()
    assert stats["low_rank_blocks"] > 0
    assert stats["stored_factor_entries"] < qjet.n_nodes * qjet.n_nodes
    assert atlas.direct_relative_error(values) < 2.0e-12


def test_atlas_is_weighted_self_adjoint_and_has_exact_constant_channel() -> None:
    qjet = bent_conic_tube_qjet(3.0, 1.3, 0.48, 0.30, 12, 16)
    atlas = StaticCrossSliceAtlasQJet(
        qjet,
        kernel_power=2.0,
        tolerance=1.0e-10,
        admissibility=0.3,
        leaf_nodes=8,
    )
    nodes = qjet.generate_nodes()
    left = tuple(
        _cos(TAU * index / qjet.n_nodes) for index in range(qjet.n_nodes)
    )
    right = tuple(
        _sin(3.0 * TAU * index / qjet.n_nodes)
        for index in range(qjet.n_nodes)
    )
    q_left = atlas.apply(left)
    q_right = atlas.apply(right)
    lhs = sum(
        nodes.weights[index]
        * left[index]
        * complex(q_right[index]).real
        for index in range(qjet.n_nodes)
    )
    rhs = sum(
        nodes.weights[index]
        * complex(q_left[index]).real
        * right[index]
        for index in range(qjet.n_nodes)
    )
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 8.0e-13
    assert atlas.constant_residual() == 0.0


def test_atlas_preserves_inverse_square_and_cube_scale_covariance() -> None:
    unit = straight_conic_tube_qjet(3.0, 0.7, 0.4, 7, 16)
    doubled = straight_conic_tube_qjet(6.0, 1.4, 0.8, 7, 16)
    values = _field(unit)
    for power, expected in ((2.0, 1.0), (3.0, 0.5)):
        unit_atlas = StaticCrossSliceAtlasQJet(
            unit,
            power,
            tolerance=1.0e-10,
            admissibility=0.2,
        )
        doubled_atlas = StaticCrossSliceAtlasQJet(
            doubled,
            power,
            tolerance=1.0e-10,
            admissibility=0.2,
        )
        reference = unit_atlas.apply(values)
        candidate = tuple(
            complex(value) / expected for value in doubled_atlas.apply(values)
        )
        assert _relative_error(reference, candidate) < 8.0e-13


def test_dtn_named_path_has_the_two_pi_normalization() -> None:
    qjet = aircraft_conic_bundle_qjet(4.0, 7, 16)
    values = _field(qjet)
    atlas = StaticCrossSliceAtlasQJet(
        qjet,
        kernel_power=3.0,
        tolerance=1.0e-10,
        admissibility=0.2,
    )
    raw = atlas.apply(values)
    normalized = atlas.apply_dtn_principal(values)
    assert _relative_error(
        normalized,
        tuple(complex(value) / TAU for value in raw),
    ) < 2.0e-15
