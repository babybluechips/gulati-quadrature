"""Fail-closed geometric gate for the exact scale-phase chord normal form."""

from inverse_shape.quadrature import (
    TAU,
    _abs,
    _exp,
    _finite,
    _sin,
    _sqrt,
)
from inverse_shape.scale_phase_cauchy import ScalePhaseCauchyQJet


def _point3(value):
    point = tuple(float(component) for component in value)
    if len(point) != 3 or any(not _finite(component) for component in point):
        raise ValueError("each scale-phase point must have three finite entries")
    return point


def _distance_squared(left, right):
    return sum((left[axis] - right[axis]) ** 2 for axis in range(3))


def _normal_form_distance_squared(rho_i, theta_i, rho_j, theta_j):
    radius_i = _exp(rho_i)
    radius_j = _exp(rho_j)
    chord = 2.0 * _sin_squared_half(theta_i - theta_j)
    return (
        (radius_i - radius_j) ** 2
        + 2.0 * radius_i * radius_j * chord
    )


def _sin_squared_half(angle):
    sine = _sin(0.5 * angle)
    return sine * sine


class ScalePhaseChordCertificate:
    def __init__(
        self,
        maximum_relative_residual,
        relative_rms_residual,
        audited_pairs,
        worst_pair,
        tolerance,
        exhaustive,
    ):
        self.maximum_relative_residual = float(maximum_relative_residual)
        self.relative_rms_residual = float(relative_rms_residual)
        self.audited_pairs = int(audited_pairs)
        self.worst_pair = tuple(worst_pair)
        self.tolerance = float(tolerance)
        self.exhaustive = bool(exhaustive)
        self.accepted = self.maximum_relative_residual <= self.tolerance

    def as_dict(self):
        return {
            "accepted": self.accepted,
            "maximum_relative_residual": self.maximum_relative_residual,
            "relative_rms_residual": self.relative_rms_residual,
            "audited_pairs": self.audited_pairs,
            "worst_pair": self.worst_pair,
            "tolerance": self.tolerance,
            "exhaustive": self.exhaustive,
            "stored_pair_table": False,
            "stored_dense_matrix": False,
        }


def certify_scale_phase_chord(
    points,
    rhos,
    n_theta,
    tolerance=5.0e-13,
    exhaustive=False,
):
    """Audit the exact exponential chord identity without storing pair data."""

    point_values = tuple(_point3(value) for value in points)
    rho_values = tuple(float(value) for value in rhos)
    phase_count = int(n_theta)
    if len(rho_values) < 2:
        raise ValueError("at least two scale lines are required")
    if phase_count < 4:
        raise ValueError("n_theta must be at least four")
    expected = len(rho_values) * phase_count
    if len(point_values) != expected:
        raise ValueError("points must contain n_scale*n_theta entries")
    if any(not _finite(value) for value in rho_values):
        raise ValueError("rhos must be finite")
    tolerance_value = float(tolerance)
    if tolerance_value <= 0.0 or not _finite(tolerance_value):
        raise ValueError("tolerance must be positive and finite")

    maximum = 0.0
    squared_sum = 0.0
    audited = 0
    worst = (-1, -1)

    def audit(left, right):
        nonlocal maximum, squared_sum, audited, worst
        if left == right:
            return
        scale_i, phase_i = divmod(left, phase_count)
        scale_j, phase_j = divmod(right, phase_count)
        theta_i = TAU * phase_i / phase_count
        theta_j = TAU * phase_j / phase_count
        actual = _distance_squared(point_values[left], point_values[right])
        predicted = _normal_form_distance_squared(
            rho_values[scale_i],
            theta_i,
            rho_values[scale_j],
            theta_j,
        )
        residual = _abs(actual - predicted) / max(
            actual,
            predicted,
            1.0e-300,
        )
        squared_sum += residual * residual
        audited += 1
        if residual > maximum:
            maximum = residual
            worst = (left, right)

    if exhaustive:
        for left in range(expected):
            for right in range(left + 1, expected):
                audit(left, right)
    else:
        phase_offsets = tuple(
            sorted(
                {
                    1,
                    max(1, phase_count // 8),
                    max(1, phase_count // 4),
                    max(1, phase_count // 2),
                }
            )
        )
        for scale in range(len(rho_values)):
            start = scale * phase_count
            for phase in range(phase_count):
                for offset in phase_offsets:
                    audit(
                        start + phase,
                        start + (phase + offset) % phase_count,
                    )
        scale_offsets = []
        offset = 1
        while offset < len(rho_values):
            scale_offsets.append(offset)
            offset *= 2
        for left_scale in range(len(rho_values)):
            for offset in scale_offsets:
                right_scale = left_scale + offset
                if right_scale >= len(rho_values):
                    continue
                for phase in range(phase_count):
                    for phase_offset in (0, 1, phase_count // 4):
                        audit(
                            left_scale * phase_count + phase,
                            right_scale * phase_count
                            + (phase + phase_offset) % phase_count,
                        )
    rms = _sqrt(squared_sum / max(audited, 1))
    return ScalePhaseChordCertificate(
        maximum,
        rms,
        audited,
        worst,
        tolerance_value,
        exhaustive,
    )


class CertifiedScalePhaseGeometryQJet:
    """Scale-phase operator which refuses a geometrically invalid chart."""

    def __init__(
        self,
        points,
        rhos,
        n_theta,
        meridional_weights,
        geometry_tolerance=5.0e-13,
        exhaustive_geometry_audit=True,
        **qjet_options,
    ):
        self.certificate = certify_scale_phase_chord(
            points,
            rhos,
            n_theta,
            geometry_tolerance,
            exhaustive_geometry_audit,
        )
        if not self.certificate.accepted:
            residual = self.certificate.maximum_relative_residual
            raise ValueError(
                "physical chord metric is not the exact scale-phase normal "
                f"form (maximum relative residual {residual:.3e})"
            )
        self.qjet = ScalePhaseCauchyQJet(
            rhos,
            n_theta,
            meridional_weights,
            **qjet_options,
        )

    def apply(self, values):
        return self.qjet.apply(values)

    def evaluate(self, values):
        return self.qjet.evaluate(values)

    def stats(self):
        result = self.qjet.stats()
        result["geometry_certificate"] = self.certificate.as_dict()
        result["geometry_gate"] = "accepted_exact_scale_phase_chord"
        return result


__all__ = [
    "CertifiedScalePhaseGeometryQJet",
    "ScalePhaseChordCertificate",
    "certify_scale_phase_chord",
]
