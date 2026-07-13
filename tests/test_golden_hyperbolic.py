import ast

import inverse_shape.golden_hyperbolic as golden_module
import inverse_shape.quadrature as quadrature_module
from inverse_shape.axisymmetric_scale_phase import (
    AxisymmetricScalePhaseQJet,
    MeridianThreeJetSpline,
)
from inverse_shape.golden_hyperbolic import (
    GOLDEN_ANALYTIC_RADIUS,
    GOLDEN_BERNSTEIN_RADIUS,
    GOLDEN_FOURIER_RATIO,
    GOLDEN_MU,
    GOLDEN_RADIAL_CONTRACTION,
    GOLDEN_TETRATION_MULTIPLIER,
    GoldenHyperbolicFrame,
    HolomorphicThreeJet,
    golden_hyperbolic_decay,
    golden_coordinate_geometric_factor,
    golden_reparameterize_meridian_jet,
    golden_tau_from_xi,
    golden_xi_from_tau,
    hyperbolic_cosh_golden,
    hyperbolic_cosh_upper,
    integer_trace_analytic_radius,
    integer_trace_bernstein_radius,
    integer_trace_half_length,
    meridian_upper_point,
    scale_phase_cosh_from_xi,
)
from inverse_shape.testing.reference_pairwise import (
    reference_axisymmetric_physical,
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


def test_golden_hyperbolic_core_has_only_foundational_imports() -> None:
    with open(golden_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == [
        "inverse_shape.joukowski_endpoint",
        "inverse_shape.quadrature",
    ]


def test_golden_constants_fix_trace_three_and_tetration_gauge() -> None:
    phi = golden_module.PHI
    assert abs(GOLDEN_MU - 2.0 * quadrature_module._log(phi)) < 2.0e-15
    assert abs(GOLDEN_RADIAL_CONTRACTION - phi**-2) < 2.0e-15
    assert abs(GOLDEN_FOURIER_RATIO - phi**-4) < 2.0e-15
    assert abs(
        GOLDEN_RADIAL_CONTRACTION
        - 4.0 * GOLDEN_TETRATION_MULTIPLIER**2
    ) < 2.0e-15
    trace = quadrature_module._exp(GOLDEN_MU) + quadrature_module._exp(
        -GOLDEN_MU
    )
    assert abs(trace - 3.0) < 3.0e-15
    assert abs(GOLDEN_ANALYTIC_RADIUS - quadrature_module._sqrt(5.0)) < 1.0e-15
    assert abs(GOLDEN_BERNSTEIN_RADIUS - phi**3) < 2.0e-15
    assert golden_coordinate_geometric_factor(24) < 9.0e-16


def test_trace_three_uniquely_maximizes_the_integer_trace_analytic_collar() -> None:
    radii = tuple(integer_trace_analytic_radius(trace) for trace in range(3, 20))
    bernstein = tuple(
        integer_trace_bernstein_radius(trace) for trace in range(3, 20)
    )
    assert all(left > right for left, right in zip(radii[:-1], radii[1:], strict=True))
    assert all(
        left > right
        for left, right in zip(bernstein[:-1], bernstein[1:], strict=True)
    )
    assert abs(integer_trace_half_length(3) - GOLDEN_MU) < 2.0e-15
    assert abs(radii[0] - GOLDEN_ANALYTIC_RADIUS) < 2.0e-15
    assert abs(bernstein[0] - GOLDEN_BERNSTEIN_RADIUS) < 2.0e-15


def test_golden_frame_maps_trace_three_endpoints_to_chebyshev_endpoints() -> None:
    frame = GoldenHyperbolicFrame(1j, 1j)
    upper = 1j * quadrature_module._exp(GOLDEN_MU)
    lower = 1j * quadrature_module._exp(-GOLDEN_MU)
    assert abs(frame.tau(upper) - 1.0) < 3.0e-15
    assert abs(frame.tau(lower) + 1.0) < 3.0e-15
    assert abs(frame.xi(upper) - GOLDEN_MU) < 3.0e-15
    assert abs(frame.xi(lower) + GOLDEN_MU) < 3.0e-15
    assert abs(hyperbolic_cosh_upper(1j, upper) - 1.5) < 3.0e-15
    assert abs(hyperbolic_cosh_golden(0.0, 1.0) - 1.5) < 3.0e-15
    assert abs(
        golden_hyperbolic_decay(0.0, 1.0)
        - GOLDEN_RADIAL_CONTRACTION
    ) < 3.0e-15


def test_golden_frame_is_biholomorphic_and_preserves_hyperbolic_distance() -> None:
    frame = GoldenHyperbolicFrame(1.7 + 0.8j, 0.31 + 0.93j)
    points = (-0.4 + 0.3j, 0.2 + 1.1j, 2.6 + 0.5j, 1.4 + 3.2j)
    certificate = frame.certificate()
    assert certificate["center_tau_residual"] < 2.0e-15
    assert certificate["center_xi_residual"] < 2.0e-15
    assert certificate["mapped_tangent_real_residual"] < 2.0e-15
    assert certificate["mapped_tangent_imaginary"] > 0.0
    for point in points:
        tau = frame.tau(point)
        xi = frame.xi(point)
        assert abs(tau) < GOLDEN_ANALYTIC_RADIUS
        assert abs(frame.inverse_tau(tau) - point) < 2.0e-14
        assert abs(golden_tau_from_xi(xi) - tau) < 2.0e-14
        assert abs(golden_xi_from_tau(tau) - xi) < 2.0e-14
    for first, second in zip(points[:-1], points[1:], strict=True):
        tau_first = frame.tau(first)
        tau_second = frame.tau(second)
        xi_first = frame.xi(first)
        xi_second = frame.xi(second)
        reference = hyperbolic_cosh_upper(first, second)
        assert abs(
            hyperbolic_cosh_golden(tau_first, tau_second) - reference
        ) / reference < 2.0e-14
        assert abs(
            scale_phase_cosh_from_xi(xi_first, xi_second) - reference
        ) / reference < 2.0e-14


def test_golden_frame_transports_arbitrary_holomorphic_three_jets_exactly() -> None:
    frame = GoldenHyperbolicFrame(0.7 + 1.3j, -0.4 + 0.9j)
    source = HolomorphicThreeJet(
        (
            -0.25 + 0.8j,
            0.7 - 0.2j,
            -0.3 + 0.11j,
            0.09 + 0.04j,
        )
    )
    transformed = frame.transform_jet(source)
    assert abs(transformed.value - frame.tau(source.value)) < 2.0e-15
    recovered = frame.inverse_transform_jet(transformed)
    for candidate, reference in zip(
        recovered.values,
        source.values,
        strict=True,
    ):
        assert abs(candidate - reference) / max(1.0, abs(reference)) < 3.0e-14
    assert transformed.stats()["stored_dense_matrix"] is False


def test_three_jets_close_under_golden_arclength_normalization() -> None:
    count = 13
    raw_coordinates = tuple(
        -GOLDEN_MU + 2.0 * GOLDEN_MU * index / (count - 1)
        for index in range(count)
    )
    normalized = []
    for coordinate in raw_coordinates:
        radius = quadrature_module._exp(coordinate)
        normalized.append(
            golden_reparameterize_meridian_jet(
                (radius, radius, radius, radius),
                (0.0, 0.0, 0.0, 0.0),
                coordinate,
            )
        )
    tau_values = tuple(value.coordinate for value in normalized)
    assert abs(tau_values[0] + 1.0) < 3.0e-15
    assert abs(tau_values[-1] - 1.0) < 3.0e-15
    spline = MeridianThreeJetSpline(
        tau_values,
        tuple(value.radius_jet for value in normalized),
        tuple(value.height_jet for value in normalized),
    )
    for index in range(32):
        tau = -1.0 + 2.0 * (index + 0.5) / 32.0
        expected_radius = (GOLDEN_ANALYTIC_RADIUS + tau) / (
            GOLDEN_ANALYTIC_RADIUS - tau
        )
        radius, height = spline(tau)
        assert abs(radius - expected_radius) < 3.0e-10
        assert abs(height) < 2.0e-14

    qjet = AxisymmetricScalePhaseQJet(
        tau_values,
        spline,
        8,
        tuple(0.1 + 0.01 * index for index in range(count)),
    )
    field = tuple(
        tuple(
            tau + 0.2 * quadrature_module._cos(
                quadrature_module.TAU * phase / 8
            )
            for phase in range(8)
        )
        for tau in tau_values
    )
    assert _relative_grid_error(
        qjet.apply(field),
        reference_axisymmetric_physical(qjet, field),
    ) < 8.0e-13


def test_meridian_upper_point_uses_height_plus_i_radius() -> None:
    assert meridian_upper_point(2.5, -0.75) == complex(-0.75, 2.5)
