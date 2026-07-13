import ast

import inverse_shape.pullback_generalization as pullback_module
from inverse_shape.axisymmetric3d import build_axisymmetric_surface_qjet, torus_qjet
from inverse_shape.pullback_generalization import (
    AxisymmetricMeridionalCosecantQJet,
    CosecantPullbackQJet,
    LagAveragedCirculantQJet,
    PeriodicCurveSamples,
    apply_physical_chord_qjet,
    equal_arclength_samples,
    streamed_pullback_diagnostics,
)
from inverse_shape.quadrature import TAU, _cos, _log, _sin


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _circle_samples(n, radius=1.0):
    return PeriodicCurveSamples(
        tuple(radius * complex(_cos(TAU * index / n), _sin(TAU * index / n)) for index in range(n)),
        TAU * radius,
    )


def _ellipse_samples(n):
    return equal_arclength_samples(
        lambda theta: complex(1.7 * _cos(theta), 0.65 * _sin(theta)),
        n,
        oversample_factor=64,
    )


def test_pullback_module_has_no_external_numerical_dependency() -> None:
    with open(pullback_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_both_circulant_paths_are_exact_on_a_circle() -> None:
    samples = _circle_samples(64, 1.35)
    values = tuple(
        _cos(3.0 * TAU * index / samples.n) + 0.2 * _sin(11.0 * TAU * index / samples.n)
        for index in range(samples.n)
    )
    direct = apply_physical_chord_qjet(samples, values)
    cosecant = CosecantPullbackQJet(samples)
    lagged = LagAveragedCirculantQJet(samples)

    assert _relative_error(direct, cosecant.apply(values)) < 2.0e-13
    assert _relative_error(direct, lagged.apply(values)) < 2.0e-13
    diagnostics = streamed_pullback_diagnostics(samples, cosecant, lagged)
    assert diagnostics["cosecant_far_relative_residual"] < 2.0e-13
    assert diagnostics["lag_average_far_relative_residual"] < 2.0e-13
    assert diagnostics["prime_form_far_relative_defect"] < 2.0e-12


def test_generic_ellipse_is_not_globally_circulant() -> None:
    samples = _ellipse_samples(96)
    values = tuple(
        _cos(5.0 * TAU * index / samples.n) + 0.15 * _sin(13.0 * TAU * index / samples.n)
        for index in range(samples.n)
    )
    direct = apply_physical_chord_qjet(samples, values)
    lagged = LagAveragedCirculantQJet(samples)
    diagnostics = streamed_pullback_diagnostics(
        samples,
        CosecantPullbackQJet(samples),
        lagged,
    )

    assert _relative_error(direct, lagged.apply(values)) > 1.0e-3
    assert diagnostics["lag_average_far_relative_residual"] > 0.05
    assert diagnostics["prime_form_far_relative_defect"] > 0.05


def test_sparse_cosecant_repayment_improves_high_mode_action() -> None:
    samples = _ellipse_samples(128)
    values = tuple(_cos(23.0 * TAU * index / samples.n) for index in range(samples.n))
    direct = apply_physical_chord_qjet(samples, values)
    cosecant = CosecantPullbackQJet(samples)
    raw_error = _relative_error(direct, cosecant.apply(values))
    repaid_error = _relative_error(direct, cosecant.apply_repaid(values, bandwidth=3))

    assert repaid_error < raw_error
    assert cosecant.stats(3)["stored_dense_matrix"] is False
    assert cosecant.stats(3)["sparse_repayment_edges"] == 3 * samples.n


def test_cosecant_subtraction_removes_the_diagonal_blowup() -> None:
    maxima = []
    for n in (64, 128, 256):
        samples = _ellipse_samples(n)
        cosecant = CosecantPullbackQJet(samples)
        maxima.append(
            max(
                abs(
                    1.0 / abs(samples.points[index] - samples.points[(index + 1) % n]) ** 2
                    - cosecant.weight(1)
                )
                for index in range(n)
            )
        )
    assert max(maxima) < 2.0
    assert maxima[-1] < 1.5 * maxima[0]


def test_axisymmetric_meridional_cosecant_is_matrix_free_and_repayment_helps() -> None:
    surface = torus_qjet(2.0, 0.45, n_meridian=48, n_theta=128)
    amplitudes = tuple(
        _cos(11.0 * TAU * (index + 0.5) / surface.n_rings) for index in range(surface.n_rings)
    )
    raw = AxisymmetricMeridionalCosecantQJet(surface, mode=0, bandwidth=0)
    repaid = AxisymmetricMeridionalCosecantQJet(surface, mode=0, bandwidth=2)

    assert repaid.relative_error(amplitudes) < raw.relative_error(amplitudes)
    assert repaid.stats()["stored_dense_matrix"] is False
    assert repaid.stats()["stored_sparse_edges"] == 2 * surface.n_rings
    constant = repaid.apply((1.0,) * surface.n_rings)
    assert max(abs(complex(value)) for value in constant) < 3.0e-13


def test_axisymmetric_cosecant_subtraction_leaves_predicted_log_channel() -> None:
    radius = 1.0
    ratios = []
    for separation in (0.1, 0.05, 0.025):
        surface = build_axisymmetric_surface_qjet(
            (radius, radius),
            (0.0, separation),
            (1.0, 1.0),
            8192,
        )
        reduced = surface.reduced_meridional_kernel(0, 1)
        ratio = (reduced - 2.0 / (separation * separation) + 3.0 / (8.0 * radius * radius)) / _log(
            8.0 * radius / separation
        )
        ratios.append(ratio)
    assert abs(ratios[-1] - 0.25) < 5.0e-5
    assert abs(ratios[-1] - 0.25) < abs(ratios[0] - 0.25)
