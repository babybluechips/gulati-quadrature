import ast

import inverse_shape.multivariate_resultant as resultant_module
from inverse_shape.multivariate_resultant import (
    MultivariateResultantPeeledJetQJet,
)
from inverse_shape.quadrature import TAU, _cos, _sin


def _geometry(count):
    points = tuple(
        (
            (1.0 + 0.1 * _cos(3.0 * TAU * index / count))
            * _cos(TAU * index / count),
            (1.0 + 0.1 * _cos(3.0 * TAU * index / count))
            * _sin(TAU * index / count),
            -0.7
            + 1.4 * (index + 0.5) / count
            + 0.07 * _sin(5.0 * TAU * index / count),
        )
        for index in range(count)
    )
    weights = tuple(0.8 + 0.05 * index for index in range(count))
    values = tuple(
        point[0] + 0.2 * point[1] - 0.1 * point[2] + 0.03j * index
        for index, point in enumerate(points)
    )
    return points, weights, values


def _direct_sums(points, residues):
    return tuple(
        sum(
            residues[right]
            / sum(
                (points[left][axis] - points[right][axis]) ** 2
                for axis in range(3)
            )
            for right in range(len(points))
            if right != left
        )
        for left in range(len(points))
    )


def _direct_q(points, weights, values):
    weight_sums = _direct_sums(points, weights)
    field_sums = _direct_sums(
        points,
        tuple(weights[index] * values[index] for index in range(len(points))),
    )
    return tuple(
        values[index] * weight_sums[index] - field_sums[index]
        for index in range(len(points))
    )


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def test_resultant_kernel_has_no_external_numerical_dependency() -> None:
    with open(resultant_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_sparse_multivariate_resultant_matches_direct_peeled_sums() -> None:
    points, weights, values = _geometry(6)
    weighted_field = tuple(
        weights[index] * values[index] for index in range(len(points))
    )
    qjet = MultivariateResultantPeeledJetQJet(
        points,
        weights,
        support_budget=20000,
        audit_mode="full",
        audit_tolerance=1.0e-12,
        fallback=False,
    )
    weight_sums, field_sums = qjet.generate_inverse_square_sums(
        (weights, weighted_field)
    )
    assert _relative_error(_direct_sums(points, weights), weight_sums) < 1.0e-12
    assert _relative_error(
        _direct_sums(points, weighted_field),
        field_sums,
    ) < 1.0e-12
    candidate = tuple(
        values[index] * complex(weight_sums[index]) - complex(field_sums[index])
        for index in range(len(points))
    )
    assert _relative_error(_direct_q(points, weights, values), candidate) < 1.0e-12
    stats = qjet.stats()
    assert stats["method"] == "sparse_multivariate_peeled_resultant"
    assert stats["resultant_degree"] == 12
    assert stats["stored_dense_matrix"] is False


def test_resultant_respects_rigid_motion_and_surface_scale_covariance() -> None:
    points, weights, values = _geometry(6)
    transformed = tuple(
        (
            2.0 * point[1] + 1.7,
            -2.0 * point[0] - 0.3,
            2.0 * point[2] + 0.8,
        )
        for point in points
    )
    transformed_weights = tuple(4.0 * value for value in weights)
    original = MultivariateResultantPeeledJetQJet(
        points,
        weights,
        audit_tolerance=1.0e-12,
        fallback=False,
    ).apply(values)
    candidate = MultivariateResultantPeeledJetQJet(
        transformed,
        transformed_weights,
        audit_tolerance=1.0e-12,
        fallback=False,
    ).apply(values)
    assert _relative_error(original, candidate) < 2.0e-12


def test_support_overflow_repays_by_exact_pair_stream() -> None:
    points, weights, values = _geometry(10)
    qjet = MultivariateResultantPeeledJetQJet(
        points,
        weights,
        support_budget=80,
        audit_mode="full",
        fallback=True,
    )
    candidate = qjet.apply(values)
    reference = _direct_q(points, weights, values)
    assert _relative_error(reference, candidate) < 2.0e-15
    stats = qjet.stats()
    assert stats["method"] == "exact_streamed_resultant_repayment"
    assert "support" in stats["fallback_reason"]
    assert stats["stored_dense_matrix"] is False


def test_ill_conditioned_resultant_fails_closed_after_full_audit() -> None:
    points, weights, values = _geometry(14)
    qjet = MultivariateResultantPeeledJetQJet(
        points,
        weights,
        support_budget=20000,
        audit_mode="full",
        audit_tolerance=1.0e-13,
        fallback=True,
    )
    candidate = qjet.apply(values)
    reference = _direct_q(points, weights, values)
    assert _relative_error(reference, candidate) < 2.0e-15
    stats = qjet.stats()
    assert stats["method"] == "exact_streamed_resultant_repayment"
    assert "audit" in stats["fallback_reason"]
