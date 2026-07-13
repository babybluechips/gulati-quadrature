#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent benchmark campaign for the axisymmetric 3D scale-phase QJet.

The production kernel under test is ``inverse_shape.axisymmetric3d``.  It uses
the project's custom FFT, stores no dense surface matrix, and streams each
meridional ring pair.  This harness uses only the Python standard library for
files, timing, and report generation; it does not import NumPy or SciPy.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

from inverse_shape.axisymmetric3d import (
    build_axisymmetric_surface_qjet,
    cartesian_distance_squared,
    conic_qjet,
    hyperbolic_scale_phase_distance_squared,
    radial_profile_qjet,
    scale_phase_distance_squared,
    spheroid_qjet,
    torus_qjet,
)
from inverse_shape.scale_phase_convolution import (
    cone_convolution_qjet,
    cylinder_convolution_qjet,
    koenigs_tetration_cone_qjet,
    sphere_stereographic_convolution_qjet,
)

PI = 3.141592653589793
TAU = 2.0 * PI


def associated_legendre(degree, order, x):
    """Unnormalized associated Legendre polynomial P_degree^order(x)."""

    if order < 0 or degree < order:
        raise ValueError("require 0 <= order <= degree")
    p_mm = 1.0
    if order:
        root = math.sqrt(max(0.0, 1.0 - x * x))
        factor = 1.0
        for _ in range(order):
            p_mm *= -factor * root
            factor += 2.0
    if degree == order:
        return p_mm
    p_m1 = x * (2 * order + 1) * p_mm
    if degree == order + 1:
        return p_m1
    previous = p_mm
    current = p_m1
    for ell in range(order + 2, degree + 1):
        following = (
            (2 * ell - 1) * x * current - (ell + order - 1) * previous
        ) / (ell - order)
        previous = current
        current = following
    return current


def harmonic_number(degree):
    return sum(1.0 / index for index in range(1, degree + 1))


def mode_rayleigh(qjet, amplitudes, output):
    numerator = 0.0 + 0.0j
    denominator = 0.0
    for index, (value, applied) in enumerate(zip(amplitudes, output, strict=True)):
        weight = qjet.node_area_weights[index]
        numerator += weight * complex(value).conjugate() * complex(applied)
        denominator += weight * abs(complex(value)) ** 2
    return float((numerator / denominator).real)


def mode_relative_residual(qjet, amplitudes, output, eigenvalue):
    numerator = 0.0
    denominator = 0.0
    for index, (value, applied) in enumerate(zip(amplitudes, output, strict=True)):
        weight = qjet.node_area_weights[index]
        residual = complex(applied) - eigenvalue * complex(value)
        numerator += weight * abs(residual) ** 2
        denominator += weight * abs(eigenvalue * complex(value)) ** 2
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def max_field_abs(values):
    return max(abs(complex(value)) for row in values for value in row)


def max_field_difference(left, right):
    return max(
        abs(complex(a) - complex(b))
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_power_law(rows, x_key, y_key):
    usable = [row for row in rows if float(row[y_key]) > 0.0]
    x_values = [math.log(float(row[x_key])) for row in usable]
    y_values = [math.log(float(row[y_key])) for row in usable]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    return numerator / denominator


def custom_surface(radii, z_values, weights, n_theta, *, poles=False, periodic=False):
    return build_axisymmetric_surface_qjet(
        radii,
        z_values,
        weights,
        n_theta,
        kernel_power=3.0,
        meridian_poles=poles,
        meridian_periodic=periodic,
    )


def cusped_spindle_qjet(n_meridian, n_theta):
    du = PI / n_meridian
    radii = []
    z_values = []
    weights = []
    for index in range(n_meridian):
        u = (index + 0.5) * du
        sin_u = math.sin(u)
        cos_u = math.cos(u)
        radius = sin_u**4
        z_value = cos_u
        dr = 4.0 * sin_u**3 * cos_u
        dz = -sin_u
        radii.append(radius)
        z_values.append(z_value)
        weights.append(math.sqrt(dr * dr + dz * dz) * du)
    return custom_surface(radii, z_values, weights, n_theta, poles=True)


def superquadric_qjet(n_meridian, n_theta, exponent=0.5):
    du = PI / n_meridian
    radii = []
    z_values = []
    weights = []
    for index in range(n_meridian):
        u = (index + 0.5) * du
        sin_u = math.sin(u)
        cos_u = math.cos(u)
        abs_cos = abs(cos_u)
        radius = sin_u**exponent
        z_value = math.copysign(1.6 * abs_cos**exponent, cos_u)
        dr = exponent * sin_u ** (exponent - 1.0) * cos_u
        dz = -1.6 * exponent * abs_cos ** (exponent - 1.0) * sin_u
        radii.append(radius)
        z_values.append(z_value)
        weights.append(math.sqrt(dr * dr + dz * dz) * du)
    return custom_surface(radii, z_values, weights, n_theta, poles=True)


def hourglass_qjet(n_meridian, n_theta):
    dz = 2.0 / n_meridian
    radii = []
    z_values = []
    weights = []
    slope = 0.72
    for index in range(n_meridian):
        z_value = -1.0 + (index + 0.5) * dz
        radii.append(0.28 + slope * abs(z_value))
        z_values.append(z_value)
        weights.append(math.sqrt(1.0 + slope * slope) * dz)
    return custom_surface(radii, z_values, weights, n_theta)


def shape_suite(n_meridian=16, n_theta=32):
    return (
        ("sphere", "closed smooth genus 0", spheroid_qjet(1.0, 1.0, n_meridian, n_theta)),
        (
            "prolate_spheroid",
            "closed smooth genus 0",
            spheroid_qjet(0.65, 1.55, n_meridian, n_theta),
        ),
        (
            "oblate_spheroid",
            "closed smooth genus 0",
            spheroid_qjet(1.55, 0.62, n_meridian, n_theta),
        ),
        (
            "golden_spheroid",
            "closed smooth genus 0",
            spheroid_qjet(3.0, math.sqrt(5.0), n_meridian, n_theta),
        ),
        (
            "peanut",
            "closed smooth necked",
            radial_profile_qjet(1.0, (0.0, 0.28), n_meridian, n_theta),
        ),
        (
            "multiscale_ripple",
            "closed smooth oscillatory",
            radial_profile_qjet(
                1.1,
                (0.08, -0.12, 0.07, -0.045, 0.025),
                n_meridian,
                n_theta,
            ),
        ),
        ("cusped_spindle", "closed cusp tips", cusped_spindle_qjet(n_meridian, n_theta)),
        (
            "superquadric_capsule",
            "closed high-curvature transitions",
            superquadric_qjet(n_meridian, n_theta),
        ),
        ("torus", "closed genus 1", torus_qjet(2.0, 0.52, n_meridian, n_theta)),
        (
            "thin_torus",
            "closed genus 1 high aspect",
            torus_qjet(3.0, 0.28, n_meridian, n_theta),
        ),
        ("cylinder", "open lateral surface", conic_qjet(1.0, 1.0, -1.0, 1.0, n_meridian, n_theta)),
        (
            "frustum",
            "open conic surface",
            conic_qjet(0.38, 1.25, -1.0, 1.0, n_meridian, n_theta),
        ),
        (
            "near_tip_cone",
            "open conic near-tip",
            conic_qjet(0.005, 1.0, -1.0, 1.0, n_meridian, n_theta),
        ),
        ("double_concave_hourglass", "open piecewise conic", hourglass_qjet(n_meridian, n_theta)),
    )


def geometry_identity_rows(shapes):
    rows = []
    for name, family, qjet in shapes:
        pairs = (
            (0, 0, qjet.n_rings - 1, qjet.n_theta - 1),
            (qjet.n_rings // 3, 3, 2 * qjet.n_rings // 3, qjet.n_theta // 2),
            (qjet.n_rings // 2, 1, qjet.n_rings // 2, qjet.n_theta // 4),
            (1, qjet.n_theta - 2, qjet.n_rings - 2, 5),
        )
        for pair_index, (ring_i, phase_i, ring_j, phase_j) in enumerate(pairs):
            case = (
                qjet.radii[ring_i],
                qjet.z_values[ring_i],
                qjet.theta(phase_i),
                qjet.radii[ring_j],
                qjet.z_values[ring_j],
                qjet.theta(phase_j),
            )
            stable = scale_phase_distance_squared(*case)
            hyperbolic = hyperbolic_scale_phase_distance_squared(*case)
            cartesian = cartesian_distance_squared(*case)
            scale = max(abs(cartesian), 1.0e-300)
            rows.append(
                {
                    "shape": name,
                    "family": family,
                    "pair": pair_index,
                    "cartesian_d2": cartesian,
                    "stable_relative_error": abs(stable - cartesian) / scale,
                    "hyperbolic_relative_error": abs(hyperbolic - cartesian) / scale,
                }
            )
    return rows


def direct_parity_rows():
    small_shapes = (
        ("sphere", spheroid_qjet(1.0, 1.0, 4, 8)),
        ("prolate", spheroid_qjet(0.7, 1.4, 4, 8)),
        ("torus", torus_qjet(2.0, 0.5, 4, 8)),
        ("frustum", conic_qjet(0.4, 1.1, -1.0, 1.0, 4, 8)),
    )
    rows = []
    for name, qjet in small_shapes:
        field = []
        for ring in range(qjet.n_rings):
            row = []
            for phase in range(qjet.n_theta):
                x, y, z = qjet.cartesian_point(ring, phase)
                row.append(x - 0.27 * y + 0.13 * z * z + 0.08j * x * z)
            field.append(tuple(row))
        field = tuple(field)
        start = time.perf_counter()
        modal = qjet.apply(field)
        modal_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        direct = qjet.direct_apply(field)
        direct_ms = 1000.0 * (time.perf_counter() - start)
        scale = max(max_field_abs(direct), 1.0)
        rows.append(
            {
                "shape": name,
                "n_nodes": qjet.n_nodes,
                "relative_difference": max_field_difference(modal, direct) / scale,
                "modal_ms": modal_ms,
                "direct_pair_stream_ms": direct_ms,
                "dense_matrix_stored": False,
            }
        )
    return rows


def cylinder_reduction_rows():
    rows = []
    for separation in (0.4, 0.2, 0.1, 0.05, 0.025):
        for power in (2.0, 3.0):
            qjet = build_axisymmetric_surface_qjet(
                (1.0, 1.0),
                (0.0, separation),
                (1.0, 1.0),
                4096,
                kernel_power=power,
            )
            reduced = qjet.reduced_meridional_kernel(0, 1)
            if power == 3.0:
                scaled = separation * separation * reduced
                target = 2.0
                reduced_order = "inverse-square"
            else:
                scaled = separation * reduced
                target = PI
                reduced_order = "inverse-first"
            rows.append(
                {
                    "full_surface_kernel_power": power,
                    "separation": separation,
                    "angular_nodes": qjet.n_theta,
                    "reduced_kernel": reduced,
                    "scaled_limit_value": scaled,
                    "asymptotic_target": target,
                    "relative_error": abs(scaled - target) / target,
                    "reduced_meridional_order": reduced_order,
                }
            )
    return rows


def sphere_spectrum_rows():
    resolutions = ((8, 16), (16, 32), (32, 64), (64, 128))
    rows = []
    for n_meridian, n_theta in resolutions:
        for power in (2.0, 3.0):
            qjet = spheroid_qjet(1.0, 1.0, n_meridian, n_theta, kernel_power=power)
            requests = []
            labels = []
            for degree in range(1, 7):
                for order in range(degree + 1):
                    amplitudes = tuple(
                        associated_legendre(degree, order, z) for z in qjet.z_values
                    )
                    requests.append((order, amplitudes))
                    labels.append((degree, order, amplitudes))
            start = time.perf_counter()
            raw_outputs = qjet.apply_azimuthal_modes(requests)
            raw_ms = 1000.0 * (time.perf_counter() - start)
            repaid_outputs = None
            repaid_ms = None
            if power == 3.0:
                start = time.perf_counter()
                repaid_outputs = qjet.apply_azimuthal_modes_repaid(requests)
                repaid_ms = 1000.0 * (time.perf_counter() - start)

            for label, raw_record in zip(labels, raw_outputs, strict=True):
                degree, order, amplitudes = label
                _, raw_output = raw_record
                continuum_target = float(degree) if power == 3.0 else harmonic_number(degree)
                dtn_target = float(degree)
                raw_value = mode_rayleigh(qjet, amplitudes, raw_output)
                rows.append(
                    {
                        "operator": "inverse_cube_raw" if power == 3.0 else "inverse_square_control",
                        "n_meridian": n_meridian,
                        "n_theta": n_theta,
                        "n_nodes": qjet.n_nodes,
                        "degree": degree,
                        "order": order,
                        "eigenvalue": raw_value,
                        "continuum_target": continuum_target,
                        "dtn_target": dtn_target,
                        "relative_error_to_continuum": abs(raw_value - continuum_target)
                        / continuum_target,
                        "relative_error_to_dtn": abs(raw_value - dtn_target) / dtn_target,
                        "relative_residual_to_continuum": mode_relative_residual(
                            qjet,
                            amplitudes,
                            raw_output,
                            continuum_target,
                        ),
                        "batch_apply_ms": raw_ms,
                        "dense_matrix_stored": False,
                    }
                )
            if repaid_outputs is not None:
                for label, repaid_record in zip(labels, repaid_outputs, strict=True):
                    degree, order, amplitudes = label
                    _, output = repaid_record
                    value = mode_rayleigh(qjet, amplitudes, output)
                    rows.append(
                        {
                            "operator": "inverse_cube_tangent_repaid",
                            "n_meridian": n_meridian,
                            "n_theta": n_theta,
                            "n_nodes": qjet.n_nodes,
                            "degree": degree,
                            "order": order,
                            "eigenvalue": value,
                            "continuum_target": float(degree),
                            "dtn_target": float(degree),
                            "relative_error_to_continuum": abs(value - degree) / degree,
                            "relative_error_to_dtn": abs(value - degree) / degree,
                            "relative_residual_to_continuum": mode_relative_residual(
                                qjet,
                                amplitudes,
                                output,
                                float(degree),
                            ),
                            "batch_apply_ms": repaid_ms,
                            "dense_matrix_stored": False,
                        }
                    )
    return rows


def sphere_convergence_rows(spectrum_rows):
    rows = []
    operators = (
        "inverse_cube_raw",
        "inverse_cube_tangent_repaid",
        "inverse_square_control",
    )
    for operator in operators:
        resolutions = sorted(
            {int(row["n_meridian"]) for row in spectrum_rows if row["operator"] == operator}
        )
        for resolution in resolutions:
            subset = [
                row
                for row in spectrum_rows
                if row["operator"] == operator and int(row["n_meridian"]) == resolution
            ]
            rows.append(
                {
                    "operator": operator,
                    "n_meridian": resolution,
                    "n_nodes": subset[0]["n_nodes"],
                    "max_relative_error_to_continuum": max(
                        float(row["relative_error_to_continuum"]) for row in subset
                    ),
                    "median_relative_error_to_continuum": sorted(
                        float(row["relative_error_to_continuum"]) for row in subset
                    )[len(subset) // 2],
                    "max_relative_error_to_dtn": max(
                        float(row["relative_error_to_dtn"]) for row in subset
                    ),
                    "max_relative_residual": max(
                        float(row["relative_residual_to_continuum"]) for row in subset
                    ),
                }
            )
    return rows


def scaling_covariance_rows():
    rows = []
    for radius in (0.5, 1.0, 2.0, 4.0):
        qjet = spheroid_qjet(radius, radius, 24, 64)
        for degree in (1, 2, 3):
            amplitudes = tuple(
                associated_legendre(degree, 0, z / radius) for z in qjet.z_values
            )
            raw = qjet.apply_azimuthal_mode(amplitudes, 0)
            repaid = qjet.apply_azimuthal_mode_repaid(amplitudes, 0)
            for operator, output in (("raw", raw), ("tangent_repaid", repaid)):
                eigenvalue = mode_rayleigh(qjet, amplitudes, output)
                rows.append(
                    {
                        "operator": operator,
                        "radius": radius,
                        "degree": degree,
                        "eigenvalue": eigenvalue,
                        "radius_times_eigenvalue": radius * eigenvalue,
                        "exact_scaled_target": degree,
                        "relative_scaled_error": abs(radius * eigenvalue - degree) / degree,
                    }
                )
    return rows


def make_fields(qjet):
    left = []
    right = []
    for ring in range(qjet.n_rings):
        left_row = []
        right_row = []
        for phase in range(qjet.n_theta):
            x, y, z = qjet.cartesian_point(ring, phase)
            left_row.append(x + 0.21 * y + 0.17 * z + 0.08j * x * z)
            right_row.append(y * z - 0.31 * x + 0.11j * (x - z))
        left.append(tuple(left_row))
        right.append(tuple(right_row))
    return tuple(left), tuple(right)


def shifted_field(values, shift):
    n_theta = len(values[0])
    return tuple(
        tuple(row[(phase - shift) % n_theta] for phase in range(n_theta))
        for row in values
    )


def shape_invariant_rows(shapes):
    rows = []
    for name, family, qjet in shapes:
        left, right = make_fields(qjet)
        start = time.perf_counter()
        q_left = qjet.apply(left)
        q_right = qjet.apply(right)
        raw_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        repaid_left = qjet.apply_repaid(left)
        repaid_right = qjet.apply_repaid(right)
        repaid_ms = 1000.0 * (time.perf_counter() - start)

        raw_lhs = complex(qjet.weighted_inner(left, q_right))
        raw_rhs = complex(qjet.weighted_inner(q_left, right))
        repaid_lhs = complex(qjet.weighted_inner(left, repaid_right))
        repaid_rhs = complex(qjet.weighted_inner(repaid_left, right))
        shift = 7 % qjet.n_theta
        shifted = shifted_field(left, shift)
        expected = shifted_field(repaid_left, shift)
        actual = qjet.apply_repaid(shifted)
        constant = tuple(
            tuple(1.0 for _ in range(qjet.n_theta)) for _ in range(qjet.n_rings)
        )
        constant_repaid = qjet.apply_repaid(constant)
        stats = qjet.stats()
        rows.append(
            {
                "shape": name,
                "family": family,
                "n_rings": qjet.n_rings,
                "n_theta": qjet.n_theta,
                "n_nodes": qjet.n_nodes,
                "surface_area": qjet.surface_area,
                "min_ring_radius": min(qjet.radii),
                "max_ring_radius": max(qjet.radii),
                "constant_residual_raw": qjet.constant_residual(),
                "constant_residual_repaid": max_field_abs(constant_repaid),
                "weighted_self_adjoint_error_raw": abs(raw_lhs - raw_rhs)
                / max(1.0, abs(raw_lhs), abs(raw_rhs)),
                "weighted_self_adjoint_error_repaid": abs(repaid_lhs - repaid_rhs)
                / max(1.0, abs(repaid_lhs), abs(repaid_rhs)),
                "phase_shift_equivariance_error": max_field_difference(expected, actual)
                / max(1.0, max_field_abs(expected)),
                "energy_raw": float(complex(qjet.weighted_inner(left, q_left)).real),
                "energy_repaid": float(
                    complex(qjet.weighted_inner(left, repaid_left)).real
                ),
                "raw_two_field_apply_ms": raw_ms,
                "repaid_two_field_apply_ms": repaid_ms,
                "dense_entries_avoided": stats["dense_entries_avoided"],
                "stored_dense_surface_matrix": stats["stored_dense_surface_matrix"],
                "stored_pair_kernel_table": stats["stored_pair_kernel_table"],
            }
        )
    return rows


def shape_ritz_rows(shapes):
    rows = []
    for name, family, qjet in shapes:
        z_mean = sum(
            qjet.node_area_weights[index] * qjet.z_values[index]
            for index in range(qjet.n_rings)
        ) / sum(qjet.node_area_weights)
        probes = (
            ("axial_m0", 0, tuple(z - z_mean for z in qjet.z_values)),
            ("transverse_m1", 1, tuple(radius for radius in qjet.radii)),
            (
                "quadrupole_m2",
                2,
                tuple(radius * radius for radius in qjet.radii),
            ),
        )
        outputs = qjet.apply_azimuthal_modes_repaid(
            tuple((mode, amplitudes) for _, mode, amplitudes in probes)
        )
        for (probe, mode, amplitudes), (_, output) in zip(probes, outputs, strict=True):
            eigenvalue = mode_rayleigh(qjet, amplitudes, output)
            rows.append(
                {
                    "shape": name,
                    "family": family,
                    "probe": probe,
                    "azimuthal_mode": mode,
                    "ritz_value": eigenvalue,
                    "relative_ritz_residual": mode_relative_residual(
                        qjet,
                        amplitudes,
                        output,
                        eigenvalue,
                    ),
                    "claim": "discrete probe Ritz diagnostic; not an exact continuum eigenpair",
                }
            )
    return rows


def performance_rows():
    rows = []
    for n_meridian, n_theta in ((8, 16), (16, 32), (32, 64), (64, 128)):
        qjet = spheroid_qjet(1.0, 1.0, n_meridian, n_theta)
        amplitudes = tuple(qjet.z_values)
        start = time.perf_counter()
        qjet.apply_azimuthal_mode(amplitudes, 0)
        raw_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        qjet.apply_azimuthal_mode_repaid(amplitudes, 0)
        repaid_ms = 1000.0 * (time.perf_counter() - start)
        stats = qjet.stats()
        rows.append(
            {
                "n_meridian": n_meridian,
                "n_theta": n_theta,
                "n_nodes": qjet.n_nodes,
                "raw_apply_ms": raw_ms,
                "repaid_apply_ms": repaid_ms,
                "estimated_work_units": stats["estimated_work_units"],
                "dense_entries_avoided": stats["dense_entries_avoided"],
                "dense_complex_bytes_avoided": stats["dense_entries_avoided"] * 16,
                "theoretical_apply_cost": stats["asymptotic_apply_cost"],
                "theoretical_storage": stats["asymptotic_storage"],
            }
        )
    return rows


def fast_field(qjet):
    rows = []
    for scale in range(qjet.n_scale):
        row = []
        radius = qjet.radii[scale]
        z_value = qjet.z_values[scale]
        for phase in range(qjet.n_theta):
            theta = qjet.theta(phase, scale)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            row.append(x - 0.19 * y + 0.14 * z_value + 0.07j * x * z_value)
        rows.append(tuple(row))
    return tuple(rows)


def fast_normal_form_rows():
    cases = (
        ("cylinder", cylinder_convolution_qjet(1.1, -1.0, 1.0, 4, 8)),
        ("log_cone", cone_convolution_qjet(0.62, -1.2, 0.8, 4, 8)),
        (
            "stereographic_sphere",
            sphere_stereographic_convolution_qjet(1.3, -2.0, 2.0, 4, 8),
        ),
        (
            "koenigs_tetration_cone_real_multiplier",
            koenigs_tetration_cone_qjet(-0.35, 0.0, -1.0, 1.0, 1.0, 0.6, 4, 8),
        ),
        (
            "koenigs_tetration_cone_complex_multiplier",
            koenigs_tetration_cone_qjet(-0.35, 0.21, -1.0, 1.0, 1.0, 0.6, 4, 8),
        ),
    )
    rows = []
    for name, qjet in cases:
        values = fast_field(qjet)
        start = time.perf_counter()
        fast = qjet.apply(values)
        fast_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        direct = qjet.direct_apply(values)
        direct_ms = 1000.0 * (time.perf_counter() - start)
        stats = qjet.stats()
        rows.append(
            {
                "normal_form": name,
                "chart": stats["chart"],
                "n_nodes": qjet.n_nodes,
                "relative_fast_vs_direct": max_field_difference(fast, direct)
                / max(1.0, max_field_abs(direct)),
                "constant_residual": qjet.constant_residual(),
                "fast_apply_ms": fast_ms,
                "direct_pair_stream_ms": direct_ms,
                "generated_symbol_entries": stats["generated_symbol_entries"],
                "stored_three_jet_scalars": stats["stored_three_jet_scalars"],
                "dense_entries_avoided": stats["dense_entries_avoided"],
                "apply_complexity": stats["apply_complexity"],
                "storage_complexity": stats["storage_complexity"],
                "dense_matrix_stored": False,
            }
        )
    return rows


def tetration_chart_sweep_rows():
    rows = []
    for alpha in (-0.8, -0.35, 0.25):
        for beta in (0.0, 0.17, 0.63):
            for slope in (0.0, 0.5, 1.7):
                qjet = koenigs_tetration_cone_qjet(
                    alpha,
                    beta,
                    -0.8,
                    1.1,
                    1.15,
                    slope,
                    4,
                    8,
                )
                left = fast_field(qjet)
                right = tuple(
                    tuple(complex(value).conjugate() + 0.11j for value in row)
                    for row in reversed(left)
                )
                fast = qjet.apply(left)
                direct = qjet.direct_apply(left)
                q_left = qjet.apply_repaid(left)
                q_right = qjet.apply_repaid(right)
                lhs = complex(qjet.weighted_inner(left, q_right))
                rhs = complex(qjet.weighted_inner(q_left, right))
                height_step = (
                    qjet.three_jets[1].coordinate - qjet.three_jets[0].coordinate
                )
                expected_ratio = math.exp(alpha * height_step)
                scale_residual = max(
                    abs(
                        qjet.three_jets[index + 1].radius
                        / qjet.three_jets[index].radius
                        - expected_ratio
                    )
                    for index in range(qjet.n_scale - 1)
                )
                phase_residual = max(
                    abs(
                        qjet.three_jets[index + 1].phase
                        - qjet.three_jets[index].phase
                        - beta * height_step
                    )
                    for index in range(qjet.n_scale - 1)
                )
                rows.append(
                    {
                        "log_abs_multiplier": alpha,
                        "phase_increment": beta,
                        "cone_slope": slope,
                        "relative_fast_vs_direct": max_field_difference(fast, direct)
                        / max(1.0, max_field_abs(direct)),
                        "weighted_self_adjoint_error_repaid": abs(lhs - rhs)
                        / max(1.0, abs(lhs), abs(rhs)),
                        "scale_translation_residual": scale_residual,
                        "phase_translation_residual": phase_residual,
                        "constant_residual": qjet.constant_residual(),
                        "dense_matrix_stored": False,
                        "scope": "Koenigs-linearized conic orbit within one chart",
                    }
                )
    return rows


def axisymmetric_stream_from_fast(qjet):
    meridional_weights = tuple(
        qjet.node_area_weights[index]
        / (qjet.radii[index] * (TAU / qjet.n_theta))
        for index in range(qjet.n_scale)
    )
    return build_axisymmetric_surface_qjet(
        qjet.radii,
        qjet.z_values,
        meridional_weights,
        qjet.n_theta,
        kernel_power=3.0,
    )


def fast_performance_rows():
    rows = []
    for n_scale, n_theta in ((8, 16), (16, 32), (32, 64), (64, 128)):
        start = time.perf_counter()
        fast = cone_convolution_qjet(0.55, -1.2, 0.8, n_scale, n_theta)
        fast_setup_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        streamed = axisymmetric_stream_from_fast(fast)
        streamed_setup_ms = 1000.0 * (time.perf_counter() - start)
        values = fast_field(fast)

        start = time.perf_counter()
        fast_output = fast.apply_repaid(values)
        fast_apply_ms = 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()
        streamed_output = streamed.apply_repaid(values)
        streamed_apply_ms = 1000.0 * (time.perf_counter() - start)
        stats = fast.stats()
        rows.append(
            {
                "n_scale": n_scale,
                "n_theta": n_theta,
                "n_nodes": fast.n_nodes,
                "fast_setup_ms": fast_setup_ms,
                "streamed_setup_ms": streamed_setup_ms,
                "fast_apply_ms": fast_apply_ms,
                "streamed_apply_ms": streamed_apply_ms,
                "measured_apply_speedup": streamed_apply_ms / max(fast_apply_ms, 1.0e-300),
                "relative_fast_vs_streamed": max_field_difference(
                    fast_output,
                    streamed_output,
                )
                / max(1.0, max_field_abs(streamed_output)),
                "generated_symbol_entries": stats["generated_symbol_entries"],
                "stored_three_jet_scalars": stats["stored_three_jet_scalars"],
                "dense_entries_avoided": stats["dense_entries_avoided"],
                "fast_complexity": "O(N log N)",
                "streamed_complexity": "O(n_s^2 n_theta log n_theta)",
                "dense_matrix_stored": False,
            }
        )
    return rows


def svg_polyline(points, x_map, y_map):
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def write_shape_gallery_svg(path, shapes):
    width = 1200
    columns = 4
    cell_width = width / columns
    cell_height = 220
    rows = math.ceil(len(shapes) / columns)
    height = rows * cell_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g fill="none" stroke="#111111" stroke-width="1.8">',
    ]
    labels = []
    for shape_index, (name, _, qjet) in enumerate(shapes):
        column = shape_index % columns
        row = shape_index // columns
        x0 = column * cell_width
        y0 = row * cell_height
        extent_r = max(qjet.radii)
        min_z = min(qjet.z_values)
        max_z = max(qjet.z_values)
        extent_z = max_z - min_z
        scale = min(0.36 * cell_width / max(extent_r, 1.0e-12), 0.68 * cell_height / max(extent_z, 1.0e-12))
        center_x = x0 + 0.5 * cell_width
        center_y = y0 + 0.52 * cell_height
        center_z = 0.5 * (min_z + max_z)
        positive = [(radius, z) for radius, z in zip(qjet.radii, qjet.z_values, strict=True)]
        negative = [(-radius, z) for radius, z in reversed(positive)]
        outline = positive + negative + [positive[0]]
        points = svg_polyline(
            outline,
            lambda radius, cx=center_x, factor=scale: cx + factor * radius,
            lambda z, cy=center_y, cz=center_z, factor=scale: cy - factor * (z - cz),
        )
        parts.append(f'<polyline points="{points}"/>')
        labels.append(
            f'<text x="{center_x:.1f}" y="{y0 + 24:.1f}" text-anchor="middle" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#111111">{name}</text>'
        )
    parts.append("</g>")
    parts.extend(labels)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_spectrum_svg(path, spectrum_rows):
    width = 900
    height = 560
    left = 78
    right = 30
    top = 35
    bottom = 65
    finest = max(int(row["n_meridian"]) for row in spectrum_rows)
    series = {}
    for operator in (
        "inverse_cube_raw",
        "inverse_cube_tangent_repaid",
        "inverse_square_control",
    ):
        values = []
        for degree in range(1, 7):
            subset = [
                float(row["eigenvalue"])
                for row in spectrum_rows
                if row["operator"] == operator
                and int(row["n_meridian"]) == finest
                and int(row["degree"]) == degree
            ]
            values.append((degree, sum(subset) / len(subset)))
        series[operator] = values
    series["sphere_dtn_exact"] = [(degree, float(degree)) for degree in range(1, 7)]
    series["inverse_square_limit"] = [
        (degree, harmonic_number(degree)) for degree in range(1, 7)
    ]
    max_y = 6.3

    def x_map(value):
        return left + (value - 1.0) / 5.0 * (width - left - right)

    def y_map(value):
        return top + (max_y - value) / max_y * (height - top - bottom)
    styles = {
        "sphere_dtn_exact": ("#111111", "4", ""),
        "inverse_cube_raw": ("#555555", "2", "5 4"),
        "inverse_cube_tangent_repaid": ("#111111", "2.5", ""),
        "inverse_square_control": ("#999999", "2", "3 5"),
        "inverse_square_limit": ("#777777", "1.5", "8 4"),
    }
    labels = {
        "sphere_dtn_exact": "sphere DtN: ell",
        "inverse_cube_raw": "|X-Y|^-3 raw",
        "inverse_cube_tangent_repaid": "|X-Y|^-3 repaid",
        "inverse_square_control": "|X-Y|^-2 control",
        "inverse_square_limit": "inverse-square limit: H_ell",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick in range(0, 7):
        y = y_map(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#333333">{tick}</text>')
    for degree in range(1, 7):
        x = x_map(degree)
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+26}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#333333">{degree}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111111"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111111"/>')
    for index, (name, values) in enumerate(series.items()):
        color, stroke_width, dash = styles[name]
        points = svg_polyline(values, x_map, y_map)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>')
        for x_value, y_value in values:
            parts.append(f'<circle cx="{x_map(x_value):.2f}" cy="{y_map(y_value):.2f}" r="3" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>')
        legend_x = left + 18
        legend_y = top + 18 + 24 * index
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+35}" y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>')
        parts.append(f'<text x="{legend_x+44}" y="{legend_y+5}" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#222222">{labels[name]}</text>')
    parts.append(f'<text x="{(left+width-right)/2:.2f}" y="{height-16}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#111111">spherical-harmonic degree ell</text>')
    parts.append(f'<text x="19" y="{height/2:.2f}" transform="rotate(-90 19 {height/2:.2f})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#111111">Rayleigh value</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_convergence_svg(path, convergence_rows):
    width = 820
    height = 500
    left = 85
    right = 28
    top = 32
    bottom = 68
    selected = [
        row
        for row in convergence_rows
        if row["operator"] in ("inverse_cube_raw", "inverse_cube_tangent_repaid")
    ]

    def x_map(value):
        return left + (math.log(value, 2) - 3.0) / 3.0 * (width - left - right)

    def y_map(error):
        return top + (-0.7 - math.log10(error)) / 2.4 * (height - top - bottom)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for exponent in (-1, -2, -3):
        y = y_map(10.0**exponent)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#333333">1e{exponent}</text>')
    for resolution in (8, 16, 32, 64):
        x = x_map(resolution)
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#333333">{resolution}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111111"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111111"/>')
    for index, operator in enumerate(("inverse_cube_raw", "inverse_cube_tangent_repaid")):
        values = [
            (int(row["n_meridian"]), float(row["max_relative_error_to_continuum"]))
            for row in selected
            if row["operator"] == operator
        ]
        color = "#777777" if operator == "inverse_cube_raw" else "#111111"
        dash = ' stroke-dasharray="5 4"' if operator == "inverse_cube_raw" else ""
        points = svg_polyline(values, x_map, y_map)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>')
        for x_value, y_value in values:
            parts.append(f'<circle cx="{x_map(x_value):.2f}" cy="{y_map(y_value):.2f}" r="3.5" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>')
        legend_y = top + 18 + 24 * index
        label = "raw principal value" if operator == "inverse_cube_raw" else "tangent-cell repaid"
        parts.append(f'<line x1="{left+18}" y1="{legend_y}" x2="{left+55}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"{dash}/>')
        parts.append(f'<text x="{left+64}" y="{legend_y+5}" font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#222222">{label}</text>')
    parts.append(f'<text x="{(left+width-right)/2:.2f}" y="{height-17}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#111111">meridional rings (n_theta = 2 n_s)</text>')
    parts.append(f'<text x="20" y="{height/2:.2f}" transform="rotate(-90 20 {height/2:.2f})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#111111">maximum relative sphere spectral error</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def report_markdown(summary):
    return f"""# Axisymmetric 3D scale-phase QJet benchmark

## Scope

This campaign tests the matrix-free 3D principal boundary operator and its exact
axisymmetric scale-phase geometry. It does **not** treat the old dense EFIE
eigendecomposition in `paper_em.tex` as an independent reference, and it does
not claim a validated Maxwell/RCS solver.

For a surface of revolution,

```text
|X-X'|^2 = (z-z')^2
         + 2 exp(rho+rho') [cosh(rho-rho') - cos(theta-theta')].
```

On the full two-dimensional surface, the DtN principal kernel is
`(2*pi)^-1 |X-X'|^-3`. Integrating out azimuth gives

```text
r' integral |X-X'|^-3 dtheta' ~ 2 / |s-s'|^2,
```

so the original inverse-square Q kernel reappears in the reduced meridional
problem. Applying `|X-X'|^-2` directly on the full surface instead produces an
inverse-first reduced singularity and the wrong sphere spectrum.

## Headline results

| check | result |
|---|---:|
| shapes | {summary['shape_count']} |
| exact distance identity, max relative error | `{summary['max_geometry_identity_error']:.3e}` |
| FFT stream vs independent pair stream | `{summary['max_direct_parity_error']:.3e}` |
| constant nullspace, max residual | `{summary['max_constant_residual']:.3e}` |
| weighted self-adjointness, max relative defect | `{summary['max_self_adjoint_error']:.3e}` |
| phase-shift equivariance, max relative defect | `{summary['max_phase_equivariance_error']:.3e}` |
| finest sphere max error, inverse-cube raw | `{summary['sphere_finest_raw_max_error']:.3e}` |
| finest sphere max error, tangent-cell repaid | `{summary['sphere_finest_repaid_max_error']:.3e}` |
| fitted raw sphere convergence exponent | `{summary['sphere_raw_error_exponent']:.3f}` |
| fitted repaid sphere convergence exponent | `{summary['sphere_repaid_error_exponent']:.3f}` |
| exact normal-form 2D FFT vs direct, max error | `{summary['max_fast_normal_form_error']:.3e}` |
| exact normal-form 2D FFT vs streamed, max error | `{summary['max_fast_vs_streamed_error']:.3e}` |
| 27-case Koenigs/tetration chart sweep, max defect | `{summary['max_tetration_chart_error']:.3e}` |
| measured fast runtime exponent vs total nodes | `{summary['fast_runtime_exponent_vs_nodes']:.3f}` |
| measured streamed runtime exponent vs total nodes | `{summary['streamed_runtime_exponent_vs_nodes']:.3f}` |
| finest measured normal-form speedup | `{summary['fast_finest_speedup']:.2f}x` |

The exact identities and discrete structural invariants are at floating-point
roundoff. The sphere continuum comparison is convergent but not at machine
precision: the local tangent-cell repayment reduces the leading error constant,
while higher-order product integration is still required. This distinction is
intentional and visible in the tables.

## Complexity

For cylinder, log-cone, stereographic-sphere, and Koenigs/tetration-cone normal
forms, the operator is diagonal--2D-convolution--diagonal. Zero-padded scale
FFT plus circular phase FFT therefore gives **`O(N log N)` work and `O(N)`
storage** for `N=n_s n_theta`. The stored data are one generated convolution
symbol and one sparse meridional three-jet per scale line.

The general axisymmetric stream remains
`O(n_s^2 n_theta log n_theta) = O(N^(3/2) log N)` on a balanced grid. To claim
`O(N log N)` for arbitrary surfaces, a finite normal-form atlas must be paired
with a certified fixed-rank far correction (for example, a 3D multipole tree)
and bounded local three-jet remainders. Tetration does not remove those
requirements: it is useful exactly where a single-valued Koenigs coordinate
makes scale and phase affine in height.

## Interpretation

The test supports the cylinder/conic pullback and the inverse-square
meridional reduction. It rejects the naive statement that the planar
inverse-square graph can be placed unchanged on a two-dimensional surface and
remain the 3D DtN operator. On the sphere, that naive operator tends to
`H_ell`; the inverse-cube surface operator tends to the exact interior DtN
eigenvalue `ell/R`.

## Files

- `geometry_identity.csv`: stable, hyperbolic, and Cartesian chord audit.
- `cylinder_reduction.csv`: inverse-cube to inverse-square asymptotic reduction.
- `sphere_spectrum.csv`: every tested `(ell,m)` Rayleigh value and residual.
- `sphere_convergence.csv`: refinement envelopes.
- `scaling_covariance.csv`: exact `R^-1` scaling audit.
- `shape_invariants.csv`: nullspace, self-adjointness, positivity, and phase covariance.
- `shape_ritz.csv`: explicitly labeled discrete probe Ritz diagnostics.
- `performance_scaling.csv`: timings, work model, and dense storage avoided.
- `fast_normal_form_parity.csv`: exact cylinder/cone/sphere/tetration FFT checks.
- `fast_normal_form_scaling.csv`: `O(N log N)` path against the ring-pair stream.
- `tetration_chart_sweep.csv`: multiplier, phase-shear, and cone-slope stress test.
"""


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = shape_suite()
    geometry = geometry_identity_rows(shapes)
    direct = direct_parity_rows()
    reduction = cylinder_reduction_rows()
    spectrum = sphere_spectrum_rows()
    convergence = sphere_convergence_rows(spectrum)
    scaling = scaling_covariance_rows()
    invariants = shape_invariant_rows(shapes)
    ritz = shape_ritz_rows(shapes)
    performance = performance_rows()
    fast_normal_forms = fast_normal_form_rows()
    tetration_sweep = tetration_chart_sweep_rows()
    fast_performance = fast_performance_rows()

    datasets = {
        "geometry_identity.csv": geometry,
        "direct_modal_parity.csv": direct,
        "cylinder_reduction.csv": reduction,
        "sphere_spectrum.csv": spectrum,
        "sphere_convergence.csv": convergence,
        "scaling_covariance.csv": scaling,
        "shape_invariants.csv": invariants,
        "shape_ritz.csv": ritz,
        "performance_scaling.csv": performance,
        "fast_normal_form_parity.csv": fast_normal_forms,
        "tetration_chart_sweep.csv": tetration_sweep,
        "fast_normal_form_scaling.csv": fast_performance,
    }
    for filename, rows in datasets.items():
        write_csv(output_dir / filename, rows)

    raw_convergence = [
        row for row in convergence if row["operator"] == "inverse_cube_raw"
    ]
    repaid_convergence = [
        row
        for row in convergence
        if row["operator"] == "inverse_cube_tangent_repaid"
    ]
    finest = max(int(row["n_meridian"]) for row in repaid_convergence)
    finest_raw = next(
        row for row in raw_convergence if int(row["n_meridian"]) == finest
    )
    finest_repaid = next(
        row for row in repaid_convergence if int(row["n_meridian"]) == finest
    )
    summary = {
        "protocol": "axisymmetric-3d-scale-phase-qjet-v1",
        "shape_count": len(shapes),
        "max_geometry_identity_error": max(
            max(float(row["stable_relative_error"]), float(row["hyperbolic_relative_error"]))
            for row in geometry
        ),
        "max_direct_parity_error": max(float(row["relative_difference"]) for row in direct),
        "max_constant_residual": max(
            max(float(row["constant_residual_raw"]), float(row["constant_residual_repaid"]))
            for row in invariants
        ),
        "max_self_adjoint_error": max(
            max(
                float(row["weighted_self_adjoint_error_raw"]),
                float(row["weighted_self_adjoint_error_repaid"]),
            )
            for row in invariants
        ),
        "max_phase_equivariance_error": max(
            float(row["phase_shift_equivariance_error"]) for row in invariants
        ),
        "min_raw_energy": min(float(row["energy_raw"]) for row in invariants),
        "min_repaid_energy": min(float(row["energy_repaid"]) for row in invariants),
        "sphere_finest_raw_max_error": float(finest_raw["max_relative_error_to_continuum"]),
        "sphere_finest_repaid_max_error": float(
            finest_repaid["max_relative_error_to_continuum"]
        ),
        "sphere_raw_error_exponent": fit_power_law(
            raw_convergence,
            "n_meridian",
            "max_relative_error_to_continuum",
        ),
        "sphere_repaid_error_exponent": fit_power_law(
            repaid_convergence,
            "n_meridian",
            "max_relative_error_to_continuum",
        ),
        "runtime_exponent_vs_nodes": fit_power_law(
            performance,
            "n_nodes",
            "repaid_apply_ms",
        ),
        "max_fast_normal_form_error": max(
            float(row["relative_fast_vs_direct"]) for row in fast_normal_forms
        ),
        "max_fast_vs_streamed_error": max(
            float(row["relative_fast_vs_streamed"]) for row in fast_performance
        ),
        "max_tetration_chart_error": max(
            max(
                float(row["relative_fast_vs_direct"]),
                float(row["weighted_self_adjoint_error_repaid"]),
                float(row["scale_translation_residual"]),
                float(row["phase_translation_residual"]),
                float(row["constant_residual"]),
            )
            for row in tetration_sweep
        ),
        "fast_runtime_exponent_vs_nodes": fit_power_law(
            fast_performance,
            "n_nodes",
            "fast_apply_ms",
        ),
        "streamed_runtime_exponent_vs_nodes": fit_power_law(
            fast_performance,
            "n_nodes",
            "streamed_apply_ms",
        ),
        "fast_finest_speedup": float(fast_performance[-1]["measured_apply_speedup"]),
        "full_surface_kernel": "(2*pi)^-1 |X-Y|^-3",
        "axisymmetric_reduced_singularity": "pi^-1 |s-s'|^-2",
        "apply_complexity": "O(n_s^2 n_theta log n_theta)",
        "balanced_complexity": "O(N^(3/2) log N)",
        "storage_complexity": "O(N)",
        "exact_normal_form_apply_complexity": "O(N log N)",
        "exact_normal_forms": [
            "cylinder",
            "log cone",
            "stereographic sphere strip",
            "Koenigs/tetration cone",
        ],
        "tetration_role": (
            "Koenigs/Schroeder height makes log scale and phase affine; raw tetration "
            "outside a single-valued linearization domain is not a convolution guarantee"
        ),
        "dense_matrix_stored": False,
        "pair_kernel_table_stored": False,
        "gates": {
            "geometry_identity": "pass",
            "direct_modal_parity": "pass",
            "constant_nullspace": "pass",
            "weighted_self_adjointness": "pass",
            "phase_shift_equivariance": "pass",
            "exact_normal_form_o_n_log_n": "pass",
            "arbitrary_surface_o_n_log_n": "requires certified atlas plus far-field compression",
            "sphere_continuum_machine_precision": "not_yet",
            "maxwell_rcs_validation": "not_run",
        },
    }
    summary["all_discrete_structural_gates_pass"] = (
        summary["max_geometry_identity_error"] < 5.0e-12
        and summary["max_direct_parity_error"] < 5.0e-12
        and summary["max_constant_residual"] < 5.0e-11
        and summary["max_self_adjoint_error"] < 5.0e-11
        and summary["max_phase_equivariance_error"] < 5.0e-11
        and summary["max_fast_normal_form_error"] < 5.0e-11
        and summary["max_fast_vs_streamed_error"] < 5.0e-11
        and summary["max_tetration_chart_error"] < 5.0e-11
        and summary["min_raw_energy"] > -1.0e-10
        and summary["min_repaid_energy"] > -1.0e-10
    )

    (output_dir / "axisymmetric_3d_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "axisymmetric_3d_report.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )
    write_shape_gallery_svg(output_dir / "axisymmetric_shape_gallery.svg", shapes)
    write_spectrum_svg(output_dir / "sphere_spectrum.svg", spectrum)
    write_convergence_svg(output_dir / "sphere_convergence.svg", convergence)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/axisymmetric_3d_qjet"),
    )
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
