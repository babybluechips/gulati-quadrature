import ast

import inverse_shape.riesz_near_linear as riesz_module
from inverse_shape.quadrature import PI, TAU, _cos, _sin
from inverse_shape.riesz_near_linear import (
    NearLinearRieszQJet,
    ProductionRieszQJet,
)
from inverse_shape.testing.reference_pairwise import reference_weighted_distance_graph


def _sphere(latitude_count, longitude_count):
    points = []
    weights = []
    for latitude in range(latitude_count):
        theta = PI * (latitude + 0.5) / latitude_count
        sine = _sin(theta)
        cosine = _cos(theta)
        for longitude in range(longitude_count):
            phase = TAU * longitude / longitude_count
            points.append((sine * _cos(phase), sine * _sin(phase), cosine))
            weights.append(sine * PI / latitude_count * TAU / longitude_count)
    return tuple(points), tuple(weights)


def _field(points):
    return tuple(
        x + 0.2 * y - 0.1 * z * z + 0.03j * (y + z)
        for x, y, z in points
    )


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _double_sheet(side=8, gap=0.025):
    points = []
    weights = []
    for height in (-0.5 * gap, 0.5 * gap):
        for ix in range(side):
            x = -1.0 + 2.0 * (ix + 0.5) / side
            for iy in range(side):
                y = -1.0 + 2.0 * (iy + 0.5) / side
                points.append((x, y, height + 0.01 * x * y))
                weights.append(4.0 / (side * side))
    return tuple(points), tuple(weights)


def _logarithmic_cluster(count=128):
    points = []
    weights = []
    for index in range(count):
        scale = 2.0 ** (-(index % 48))
        branch = index // 48
        points.append(
            (
                scale,
                (branch + 1) * 1.0e-4 * scale,
                (index + 1) * 1.0e-7 * scale,
            )
        )
        weights.append(1.0 / count)
    return tuple(points), tuple(weights)


def _resolvable_multiscale_cluster(count=96):
    points = []
    weights = []
    for index in range(count):
        scale = 2.0 ** (-(index % 18))
        branch = index // 18
        points.append(
            (
                scale,
                (branch + 1) * 2.0e-3 * scale,
                (index + 1) * 2.0e-6 * scale,
            )
        )
        weights.append(1.0 / count)
    return tuple(points), tuple(weights)


def test_riesz_backend_has_no_external_numerical_dependency() -> None:
    with open(riesz_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert set(imported) == {
        "inverse_shape.arbitrary_surface",
        "inverse_shape.quadrature",
    }


def test_machine_precision_riesz_q2_uses_analytic_blocks() -> None:
    assert NearLinearRieszQJet is ProductionRieszQJet
    points, weights = _sphere(8, 12)
    values = _field(points)
    qjet = NearLinearRieszQJet(
        points,
        weights,
        kernel_power=2.0,
        tolerance=5.0e-13,
        maximum_order=16,
        leaf_size=4,
    )
    candidate = qjet.apply(values)
    reference = reference_weighted_distance_graph(points, weights, values, 2.0)
    assert _relative_error(reference, candidate) < 2.0e-13
    actual_inf = max(
        abs(complex(left) - complex(right))
        for left, right in zip(reference, candidate, strict=True)
    )
    assert actual_inf <= qjet.compression_inf_bound(values) + 2.0e-13
    stats = qjet.stats()
    assert stats["analytic_blocks"] > 0
    assert stats["analytic_pair_fraction"] > 0.2
    assert stats["hard_no_quadratic_contract"] is True
    assert stats["quadratic_fallback"] is False
    assert stats["method"] == "fixed_order_symmetric_gegenbauer_riesz_wspd"
    assert stats["storage_complexity"] == "O(N) for fixed order in 3D"
    assert not hasattr(qjet, "target_interactions")
    assert not hasattr(qjet, "target_direct_sources")


def test_riesz_q3_is_accurate_symmetric_and_constant_exact() -> None:
    points, weights = _sphere(8, 12)
    values = _field(points)
    qjet = NearLinearRieszQJet(
        points,
        weights,
        kernel_power=3.0,
        tolerance=2.0e-8,
        maximum_order=8,
        leaf_size=4,
    )
    candidate = qjet.apply(values)
    reference = reference_weighted_distance_graph(points, weights, values, 3.0)
    assert _relative_error(reference, candidate) < 2.0e-8
    assert qjet.constant_residual() == 0.0

    left = tuple(x - 0.1 * z for x, _y, z in points)
    right = tuple(y + 0.2 * z for _x, y, z in points)
    q_left = qjet.apply(left)
    q_right = qjet.apply(right)
    lhs = sum(
        weights[index] * left[index] * complex(q_right[index]).real
        for index in range(len(points))
    )
    rhs = sum(
        weights[index] * complex(q_left[index]).real * right[index]
        for index in range(len(points))
    )
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 3.0e-14


def test_near_field_and_analytic_work_scale_near_linearly() -> None:
    rows = []
    for latitudes, longitudes in ((6, 8), (8, 12), (12, 16), (16, 24)):
        points, weights = _sphere(latitudes, longitudes)
        qjet = NearLinearRieszQJet(
            points,
            weights,
            tolerance=1.0e-5,
            maximum_order=6,
            leaf_size=4,
        )
        stats = qjet.stats()
        rows.append(stats)
        assert stats["near_field_pairs"] <= stats["near_field_pair_budget"]
        assert stats["analytic_apply_units"] <= stats["analytic_apply_budget"]
        assert stats["pair_partition_residual"] == 0
    assert rows[-1]["analytic_pair_fraction"] > rows[0]["analytic_pair_fraction"]
    near_ratios = tuple(row["near_field_pairs"] / row["nodes"] for row in rows)
    assert max(near_ratios) < 90.0


def test_close_sheets_and_multiscale_cluster_never_trigger_pair_fallback() -> None:
    for points, weights in (_double_sheet(), _logarithmic_cluster()):
        qjet = NearLinearRieszQJet(
            points,
            weights,
            tolerance=2.0e-5,
            maximum_order=7,
            leaf_size=4,
        )
        values = _field(points)
        candidate = qjet.apply(values)
        assert all(
            value == value
            and abs(complex(value).real) != float("inf")
            and abs(complex(value).imag) != float("inf")
            for value in candidate
        )
        stats = qjet.stats()
        assert stats["quadratic_fallback"] is False
        assert stats["temporary_pair_table_entries"] == 0
        assert stats["stored_dense_operator_matrix"] is False


def test_centered_block_repayment_is_stable_across_five_decades() -> None:
    points, weights = _resolvable_multiscale_cluster()
    values = _field(points)
    for power in (2.0, 3.0):
        qjet = NearLinearRieszQJet(
            points,
            weights,
            kernel_power=power,
            tolerance=3.0e-13,
            maximum_order=16,
            leaf_size=4,
        )
        candidate = qjet.apply(values)
        reference = reference_weighted_distance_graph(
            points,
            weights,
            values,
            power,
        )
        assert _relative_error(reference, candidate) < 3.0e-12
        stats = qjet.stats()
        assert stats["quadratic_fallback"] is False
        assert stats["stored_dense_operator_matrix"] is False
