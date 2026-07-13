"""Streamed pairwise references excluded from production QJet objects.

These routines intentionally cost quadratic time. They are small-grid
verification oracles only: production modules neither import nor call this
module.
"""

from inverse_shape.axisymmetric_scale_phase import _aliased_mode_coefficient
from inverse_shape.quadrature import (
    TAU,
    _abs,
    _clean_scalar,
    _fft_precise,
    _ifft_precise,
    _sin,
)
from inverse_shape.scale_phase_cauchy import _decay, _vector


def reference_axisymmetric_mode(plan, values, mode):
    row = _vector(values, plan.n, "values")
    output = []
    for left in range(plan.n):
        total = 0.0 + 0.0j
        compensation = 0.0 + 0.0j
        for right in range(plan.n):
            if left == right:
                continue
            contribution = (
                _aliased_mode_coefficient(
                    plan.samples[left],
                    plan.samples[right],
                    mode,
                    plan.n_theta,
                )
                * row[right]
            )
            corrected = contribution - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        output.append(_clean_scalar(total))
    return tuple(output)


def reference_axisymmetric_physical(qjet, values):
    rows = qjet._rows(values)
    output = [
        [0.0 + 0.0j for _ in range(qjet.n_theta)]
        for _ in range(qjet.n_scale)
    ]
    compensation = [
        [0.0 + 0.0j for _ in range(qjet.n_theta)]
        for _ in range(qjet.n_scale)
    ]

    def accumulate(scale, phase, contribution):
        corrected = contribution - compensation[scale][phase]
        updated = output[scale][phase] + corrected
        compensation[scale][phase] = (
            updated - output[scale][phase]
        ) - corrected
        output[scale][phase] = updated

    for scale_i in range(qjet.n_scale):
        radius_i, height_i = qjet.plan.samples[scale_i]
        for phase_i in range(qjet.n_theta):
            theta_i = TAU * phase_i / qjet.n_theta
            value_i = rows[scale_i][phase_i]
            for scale_j in range(scale_i, qjet.n_scale):
                radius_j, height_j = qjet.plan.samples[scale_j]
                phase_start = phase_i + 1 if scale_i == scale_j else 0
                for phase_j in range(phase_start, qjet.n_theta):
                    theta_j = TAU * phase_j / qjet.n_theta
                    sine = _sin(0.5 * (theta_i - theta_j))
                    distance_squared = (
                        (radius_i - radius_j) ** 2
                        + (height_i - height_j) ** 2
                        + 4.0 * radius_i * radius_j * sine * sine
                    )
                    difference = value_i - rows[scale_j][phase_j]
                    accumulate(
                        scale_i,
                        phase_i,
                        qjet.normalization
                        * qjet.meridional_weights[scale_j]
                        * qjet.theta_step
                        * difference
                        / distance_squared,
                    )
                    accumulate(
                        scale_j,
                        phase_j,
                        -qjet.normalization
                        * qjet.meridional_weights[scale_i]
                        * qjet.theta_step
                        * difference
                        / distance_squared,
                    )
    return tuple(
        tuple(_clean_scalar(value) for value in row) for row in output
    )


def reference_axisymmetric_spectral(qjet, values):
    rows = qjet._rows(values)
    transformed = tuple(tuple(_fft_precise(row)) for row in rows)
    output_modes = [
        [0.0 + 0.0j for _ in range(qjet.n_theta)]
        for _ in range(qjet.n_scale)
    ]
    cross = reference_axisymmetric_mode(
        qjet.plan,
        qjet.meridional_weights,
        0,
    )
    cross_row_sum = tuple(TAU * complex(value) for value in cross)
    for angular_index in range(qjet.n_theta):
        mode = min(angular_index, qjet.n_theta - angular_index)
        source = tuple(
            qjet.meridional_weights[scale]
            * transformed[scale][angular_index]
            for scale in range(qjet.n_scale)
        )
        potential = reference_axisymmetric_mode(qjet.plan, source, mode)
        cycle_eigenvalue = mode * (qjet.n_theta - mode) / 2.0
        for scale in range(qjet.n_scale):
            radius = qjet.plan.samples[scale][0]
            same_scale = (
                qjet.meridional_weights[scale]
                * qjet.theta_step
                * cycle_eigenvalue
                / (radius * radius)
            )
            output_modes[scale][angular_index] = qjet.normalization * (
                (cross_row_sum[scale] + same_scale)
                * transformed[scale][angular_index]
                - TAU * complex(potential[scale])
            )
    return tuple(
        tuple(_clean_scalar(value) for value in _ifft_precise(row))
        for row in output_modes
    )


def reference_scale_phase_mode(plan, values, rhos, mode):
    row = _vector(values, plan.n, "values")
    rho_values = tuple(float(value) for value in rhos)
    if len(rho_values) != plan.n:
        raise ValueError("rhos must contain one entry per Cauchy node")
    output = []
    mode_value = abs(int(mode))
    for left in range(plan.n):
        total = 0.0 + 0.0j
        for right in range(plan.n):
            if left == right:
                continue
            total += (
                _decay(
                    mode_value
                    * _abs(rho_values[left] - rho_values[right])
                )
                * row[right]
                / _abs(plan.nodes[left] - plan.nodes[right])
            )
        output.append(_clean_scalar(total))
    return tuple(output)


def reference_scale_phase_spectral(qjet, values):
    rows = qjet._rows(values)
    transformed = tuple(tuple(_fft_precise(row)) for row in rows)
    output_modes = [
        [0.0 + 0.0j for _ in range(qjet.n_theta)]
        for _ in range(qjet.n_scale)
    ]
    cross_row = reference_scale_phase_mode(
        qjet.plan,
        qjet.meridional_weights,
        qjet.rhos,
        0,
    )
    for angular_index in range(qjet.n_theta):
        mode = min(angular_index, qjet.n_theta - angular_index)
        source = tuple(
            qjet.meridional_weights[scale]
            * transformed[scale][angular_index]
            for scale in range(qjet.n_scale)
        )
        potential = reference_scale_phase_mode(
            qjet.plan,
            source,
            qjet.rhos,
            mode,
        )
        cycle_eigenvalue = mode * (qjet.n_theta - mode) / 2.0
        for scale in range(qjet.n_scale):
            same_scale = (
                qjet.meridional_weights[scale]
                * qjet.theta_step
                * cycle_eigenvalue
                / qjet.x_nodes[scale]
            )
            output_modes[scale][angular_index] = qjet.normalization * (
                (
                    TAU * complex(cross_row[scale])
                    + same_scale
                )
                * transformed[scale][angular_index]
                - TAU * complex(potential[scale])
            )
    return tuple(
        tuple(_clean_scalar(value) for value in _ifft_precise(row))
        for row in output_modes
    )


def reference_weighted_distance_graph(
    points,
    weights,
    values,
    kernel_power=2.0,
    normalization=1.0,
):
    """Stream an arbitrary weighted Euclidean graph for audit-sized grids."""

    point_rows = tuple(tuple(float(item) for item in point) for point in points)
    weight_rows = tuple(float(value) for value in weights)
    field = tuple(complex(value) for value in values)
    count = len(point_rows)
    if count < 2 or len(weight_rows) != count or len(field) != count:
        raise ValueError("points, weights, and values must have one common length")
    dimension = len(point_rows[0])
    if dimension < 1 or any(len(point) != dimension for point in point_rows):
        raise ValueError("reference points must have one common dimension")
    power = float(kernel_power)
    scale = float(normalization)
    if power <= 0.0:
        raise ValueError("kernel_power must be positive")
    output = [0.0 + 0.0j for _ in range(count)]
    compensation = [0.0 + 0.0j for _ in range(count)]

    def accumulate(index, contribution):
        corrected = contribution - compensation[index]
        updated = output[index] + corrected
        compensation[index] = (updated - output[index]) - corrected
        output[index] = updated

    for left in range(count):
        for right in range(left + 1, count):
            distance_squared = sum(
                (point_rows[left][axis] - point_rows[right][axis]) ** 2
                for axis in range(dimension)
            )
            if distance_squared <= 1.0e-28:
                raise ValueError("distinct reference points collide")
            kernel = scale * distance_squared ** (-0.5 * power)
            difference = field[left] - field[right]
            accumulate(
                left,
                weight_rows[right] * kernel * difference,
            )
            accumulate(
                right,
                -weight_rows[left] * kernel * difference,
            )
    return tuple(_clean_scalar(value) for value in output)


__all__ = [
    "reference_axisymmetric_mode",
    "reference_axisymmetric_physical",
    "reference_axisymmetric_spectral",
    "reference_scale_phase_mode",
    "reference_scale_phase_spectral",
    "reference_weighted_distance_graph",
]
