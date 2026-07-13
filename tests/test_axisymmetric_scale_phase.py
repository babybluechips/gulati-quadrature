import ast

import inverse_shape.axisymmetric_scale_phase as axisymmetric_module
import inverse_shape.quadrature as quadrature_module
from inverse_shape.axisymmetric_scale_phase import (
    AxisymmetricScalePhaseQJet,
    MeridianThreeJetSpline,
    StaticAxisymmetricModePlan,
    axisymmetric_qjet_from_three_jets,
)
from inverse_shape.testing.reference_pairwise import (
    reference_axisymmetric_mode,
    reference_axisymmetric_physical,
)


def _coordinates(count, start=-0.8, stop=0.8):
    return tuple(
        start + (stop - start) * index / (count - 1) for index in range(count)
    )


def _field(coordinates, n_theta):
    return tuple(
        tuple(
            0.3
            + 0.2 * coordinate
            + 0.17
            * quadrature_module._cos(
                quadrature_module.TAU * 3 * phase / n_theta
            )
            + 0.04j
            * (scale + 1)
            * quadrature_module._sin(
                quadrature_module.TAU * phase / n_theta
            )
            for phase in range(n_theta)
        )
        for scale, coordinate in enumerate(coordinates)
    )


def _relative_grid_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for left_row, right_row in zip(left, right, strict=True)
        for value, reference in zip(left_row, right_row, strict=True)
    )
    denominator = max(
        1.0,
        *(
            abs(complex(value))
            for reference_row in right
            for value in reference_row
        ),
    )
    return numerator / denominator


def _relative_vector_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for value, reference in zip(left, right, strict=True)
    )
    denominator = max(1.0, *(abs(complex(value)) for value in right))
    return numerator / denominator


def _weights(coordinates, meridian):
    step = (coordinates[-1] - coordinates[0]) / (len(coordinates) - 1)
    return tuple(
        meridian(value)[0] * step * (1.0 + 0.01 * index)
        for index, value in enumerate(coordinates)
    )


def test_axisymmetric_kernel_has_no_external_numerical_import() -> None:
    with open(axisymmetric_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == [
        "inverse_shape.quadrature",
        "inverse_shape.scale_phase_cauchy",
    ]


def test_closed_alias_repayment_matches_physical_pair_stream() -> None:
    coordinates = _coordinates(18)
    n_theta = 16
    meridians = (
        lambda value: (1.0, value),
        lambda value: (0.9 + 0.25 * value, value),
        lambda value: (
            quadrature_module._sqrt(1.4 * 1.4 - value * value),
            value,
        ),
        lambda value: (
            1.0 + 0.12 * quadrature_module._cos(4.0 * value),
            value + 0.09 * quadrature_module._sin(3.0 * value),
        ),
    )
    for meridian in meridians:
        qjet = AxisymmetricScalePhaseQJet(
            coordinates,
            meridian,
            n_theta,
            _weights(coordinates, meridian),
        )
        values = _field(coordinates, n_theta)
        error = _relative_grid_error(
            qjet.apply(values),
            reference_axisymmetric_physical(qjet, values),
        )
        assert error < 5.0e-14
        assert qjet.constant_residual() < 5.0e-13


def test_nested_meridian_modes_match_quadratic_mode_reference() -> None:
    coordinates = _coordinates(128, -1.0, 1.0)

    def meridian(value):
        return (
            1.1 + 0.16 * quadrature_module._cos(3.0 * value),
            value + 0.08 * quadrature_module._sin(5.0 * value),
        )

    plan = StaticAxisymmetricModePlan(coordinates, meridian, 16)
    values = tuple(
        quadrature_module._cos(0.2 * index)
        + 0.1j * quadrature_module._sin(0.13 * index)
        for index in range(len(coordinates))
    )
    plan_stats = plan.stats()
    assert plan_stats["compressed_pair_fraction"] > 0.7
    assert plan_stats["compiled_block_count"] <= plan_stats["static_block_budget"]
    assert plan_stats["compiled_exact_pairs"] <= plan_stats["exact_pair_budget"]
    assert plan_stats["compile_pair_visits"] <= plan_stats["compile_visit_budget"]
    assert plan_stats["quadratic_fallback"] is False
    assert not hasattr(plan, "direct_apply_mode")
    for mode in (0, 1, 4, 8):
        assert _relative_vector_error(
            plan.apply_mode(values, mode),
            reference_axisymmetric_mode(plan, values, mode),
        ) < 5.0e-13


def test_axisymmetric_storage_is_linear_and_never_global_dense() -> None:
    def meridian(value):
        return (1.0 + 0.1 * value, value)

    for count in (32, 64, 128):
        coordinates = _coordinates(count)
        qjet = AxisymmetricScalePhaseQJet(
            coordinates,
            meridian,
            16,
            _weights(coordinates, meridian),
        )
        stats = qjet.stats()
        assert stats["stored_dense_distance_matrix"] is False
        assert stats["stored_dense_operator_matrix"] is False
        assert stats["mode_plan"]["stored_dense_matrix"] is False
        assert stats["quadratic_fallback"] is False
        assert stats["mode_plan"]["quadratic_fallback"] is False
        assert not hasattr(qjet, "direct_apply")
        assert stats["storage_complexity"] == "O(p^2 N) with fixed p"


def test_axisymmetric_operator_is_weighted_self_adjoint() -> None:
    coordinates = _coordinates(32)

    def meridian(value):
        return (1.0 + 0.1 * value, value + 0.03 * value * value)

    qjet = AxisymmetricScalePhaseQJet(
        coordinates,
        meridian,
        16,
        _weights(coordinates, meridian),
    )
    left = _field(coordinates, 16)
    right = tuple(
        tuple(complex(value).conjugate() + 0.07j for value in row)
        for row in reversed(left)
    )
    lhs = complex(qjet.weighted_inner(left, qjet.apply(right)))
    rhs = complex(qjet.weighted_inner(qjet.apply(left), right))
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 2.0e-13


def test_meridian_three_jet_spline_reproduces_septic_geometry() -> None:
    coordinates = _coordinates(7, -1.0, 1.0)
    radius_coefficients = (1.2, 0.1, 0.04, -0.03, 0.02, 0.0, 0.01, -0.004)
    height_coefficients = (0.0, 1.0, -0.07, 0.03, 0.0, -0.01, 0.004, 0.002)

    def derivative(coefficients, coordinate, order):
        total = 0.0
        for degree in range(order, len(coefficients)):
            factor = 1.0
            for offset in range(order):
                factor *= degree - offset
            total += factor * coefficients[degree] * coordinate ** (degree - order)
        return total

    radius_jets = tuple(
        tuple(
            derivative(radius_coefficients, coordinate, order)
            for order in range(4)
        )
        for coordinate in coordinates
    )
    height_jets = tuple(
        tuple(
            derivative(height_coefficients, coordinate, order)
            for order in range(4)
        )
        for coordinate in coordinates
    )
    spline = MeridianThreeJetSpline(coordinates, radius_jets, height_jets)
    for index in range(41):
        coordinate = -1.0 + 2.0 * index / 40
        radius, height = spline(coordinate)
        assert abs(radius - derivative(radius_coefficients, coordinate, 0)) < 2.0e-13
        assert abs(height - derivative(height_coefficients, coordinate, 0)) < 2.0e-13
    qjet = axisymmetric_qjet_from_three_jets(
        coordinates,
        radius_jets,
        height_jets,
        8,
        tuple(0.1 + 0.01 * index for index in range(len(coordinates))),
    )
    source = _field(coordinates, 8)
    assert _relative_grid_error(
        qjet.apply(source),
        reference_axisymmetric_physical(qjet, source),
    ) < 5.0e-14
    assert qjet.stats()["meridian_geometry"]["source_jet_order"] == 3
