import inverse_shape.quadrature as quadrature_module
from inverse_shape.scale_phase_geometry import (
    CertifiedScalePhaseGeometryQJet,
    certify_scale_phase_chord,
)


def _embedded_annulus(rhos, n_theta):
    inverse_sqrt_two = 1.0 / quadrature_module._sqrt(2.0)
    inverse_sqrt_six = 1.0 / quadrature_module._sqrt(6.0)
    first = (inverse_sqrt_two, inverse_sqrt_two, 0.0)
    second = (-inverse_sqrt_six, inverse_sqrt_six, 2.0 * inverse_sqrt_six)
    center = (1.7, -0.4, 2.3)
    points = []
    for rho in rhos:
        radius = quadrature_module._exp(rho)
        for phase in range(n_theta):
            theta = quadrature_module.TAU * phase / n_theta
            cosine = quadrature_module._cos(theta)
            sine = quadrature_module._sin(theta)
            points.append(
                tuple(
                    center[axis]
                    + radius
                    * (cosine * first[axis] + sine * second[axis])
                    for axis in range(3)
                )
            )
    return tuple(points)


def _field(rhos, n_theta):
    return tuple(
        tuple(
            rho
            + 0.2 * quadrature_module._cos(
                quadrature_module.TAU * 3 * phase / n_theta
            )
            for phase in range(n_theta)
        )
        for rho in rhos
    )


def test_rigidly_embedded_scale_phase_chord_passes_full_audit() -> None:
    rhos = (-0.8, -0.3, 0.1, 0.55, 0.9)
    n_theta = 16
    points = _embedded_annulus(rhos, n_theta)
    certificate = certify_scale_phase_chord(
        points,
        rhos,
        n_theta,
        exhaustive=True,
    )
    assert certificate.accepted
    assert certificate.maximum_relative_residual < 8.0e-14
    assert certificate.audited_pairs == len(points) * (len(points) - 1) // 2


def test_non_scale_phase_bending_is_rejected() -> None:
    rhos = (-0.8, -0.3, 0.1, 0.55, 0.9)
    n_theta = 16
    points = list(_embedded_annulus(rhos, n_theta))
    for scale, rho in enumerate(rhos):
        displacement = 0.12 * rho * rho
        for phase in range(n_theta):
            index = scale * n_theta + phase
            point = points[index]
            points[index] = (point[0], point[1], point[2] + displacement)
    certificate = certify_scale_phase_chord(
        points,
        rhos,
        n_theta,
    )
    assert not certificate.accepted
    assert certificate.maximum_relative_residual > 1.0e-4


def test_certified_wrapper_applies_only_after_geometry_gate() -> None:
    rhos = (-0.7, -0.25, 0.2, 0.6)
    n_theta = 8
    points = _embedded_annulus(rhos, n_theta)
    weights = tuple(quadrature_module._exp(2.0 * rho) for rho in rhos)
    operator = CertifiedScalePhaseGeometryQJet(
        points,
        rhos,
        n_theta,
        weights,
        exhaustive_geometry_audit=True,
    )
    result = operator.apply(_field(rhos, n_theta))
    assert len(result) == len(rhos)
    assert operator.stats()["geometry_certificate"]["accepted"] is True
