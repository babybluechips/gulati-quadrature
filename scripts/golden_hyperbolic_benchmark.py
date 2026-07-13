#!/usr/bin/env python3
"""Benchmark the trace-three golden hyperbolic normalization."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.axisymmetric_scale_phase import (  # noqa: E402
    AxisymmetricScalePhaseQJet,
)
from inverse_shape.golden_hyperbolic import (  # noqa: E402
    GOLDEN_ANALYTIC_RADIUS,
    GOLDEN_BERNSTEIN_RADIUS,
    GOLDEN_MU,
    GOLDEN_RADIAL_CONTRACTION,
    GoldenHyperbolicFrame,
    golden_arclength_coordinate,
    golden_coordinate_geometric_factor,
    golden_xi_from_tau,
    hyperbolic_cosh_golden,
    hyperbolic_cosh_upper,
    integer_trace_analytic_radius,
    integer_trace_bernstein_radius,
    integer_trace_half_length,
    meridian_upper_point,
    scale_phase_cosh_from_xi,
)
from inverse_shape.golden_hyperbolic_atlas import (  # noqa: E402
    GoldenHyperbolicJetAtlas,
)
from inverse_shape.quadrature import PI, TAU, _cos, _exp, _sin  # noqa: E402
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_axisymmetric_physical,
    reference_axisymmetric_spectral,
)


OUT = ROOT / "outputs" / "golden_hyperbolic"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_grid_error(left, right):
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


def trace_rows():
    rows = []
    for trace in range(3, 13):
        bernstein = integer_trace_bernstein_radius(trace)
        order = 0
        while bernstein ** (-order) > 2.0e-16:
            order += 1
        rows.append(
            {
                "integer_trace": trace,
                "half_length": integer_trace_half_length(trace),
                "analytic_disk_radius": integer_trace_analytic_radius(trace),
                "bernstein_radius": bernstein,
                "order_for_2e-16_geometric_factor": order,
            }
        )
    return rows


def frame_cases():
    return (
        (
            "cylinder",
            lambda value: (1.0, value),
            lambda _value: (0.0, 1.0),
            0.0,
            (-0.9, 0.9),
        ),
        (
            "cone",
            lambda value: (0.9 + 0.25 * value, value),
            lambda _value: (0.25, 1.0),
            0.0,
            (-0.8, 0.8),
        ),
        (
            "sphere_cap",
            lambda value: ((1.45 * 1.45 - value * value) ** 0.5, value),
            lambda value: (
                -value / (1.45 * 1.45 - value * value) ** 0.5,
                1.0,
            ),
            0.0,
            (-0.8, 0.8),
        ),
        (
            "corrugated",
            lambda value: (
                1.0 + 0.15 * _cos(5.0 * value),
                value + 0.08 * _sin(4.0 * value),
            ),
            lambda value: (
                -0.75 * _sin(5.0 * value),
                1.0 + 0.32 * _cos(4.0 * value),
            ),
            0.0,
            (-0.75, 0.75),
        ),
        (
            "double_neck",
            lambda value: (
                0.85 - 0.27 * _cos(2.0 * PI * value / 1.8),
                value,
            ),
            lambda value: (
                0.27 * (2.0 * PI / 1.8) * _sin(2.0 * PI * value / 1.8),
                1.0,
            ),
            0.0,
            (-0.7, 0.7),
        ),
        (
            "cusp_right_chart",
            lambda value: (0.35 + 0.55 * value**0.75, value),
            lambda value: (0.55 * 0.75 * value**-0.25, 1.0),
            0.45,
            (0.08, 0.85),
        ),
        (
            "airfoil_body",
            lambda value: (
                0.18 + 0.9 * max(1.0 - value * value, 0.0) ** 0.5,
                value + 0.06 * (1.0 - value * value),
            ),
            lambda value: (
                -0.9 * value / max(1.0 - value * value, 1.0e-30) ** 0.5,
                1.0 - 0.12 * value,
            ),
            0.0,
            (-0.8, 0.8),
        ),
    )


def frame_rows():
    rows = []
    for name, meridian, derivative, center, limits in frame_cases():
        radius, height = meridian(center)
        radius_derivative, height_derivative = derivative(center)
        frame = GoldenHyperbolicFrame.from_meridian(
            radius,
            height,
            radius_derivative,
            height_derivative,
        )
        points = tuple(
            meridian_upper_point(*meridian(limits[0] + (limits[1] - limits[0]) * index / 32))
            for index in range(33)
        )
        inverse_error = 0.0
        maximum_disk_fraction = 0.0
        distance_error = 0.0
        strip_error = 0.0
        for point in points:
            tau = frame.tau(point)
            inverse_error = max(inverse_error, abs(frame.inverse_tau(tau) - point))
            maximum_disk_fraction = max(
                maximum_disk_fraction,
                abs(tau) / GOLDEN_ANALYTIC_RADIUS,
            )
        for first, second in zip(points[:-1], points[1:], strict=True):
            reference = hyperbolic_cosh_upper(first, second)
            tau_first = frame.tau(first)
            tau_second = frame.tau(second)
            xi_first = frame.xi(first)
            xi_second = frame.xi(second)
            distance_error = max(
                distance_error,
                abs(
                    hyperbolic_cosh_golden(tau_first, tau_second)
                    - reference
                )
                / reference,
            )
            strip_error = max(
                strip_error,
                abs(
                    scale_phase_cosh_from_xi(xi_first, xi_second)
                    - reference
                )
                / reference,
            )
        rows.append(
            {
                "shape": name,
                "inverse_error": inverse_error,
                "distance_relative_error": distance_error,
                "strip_relative_error": strip_error,
                "maximum_disk_fraction": maximum_disk_fraction,
                "center_residual": frame.certificate()["center_tau_residual"],
            }
        )
    return rows


def polynomial_jets(coordinates):
    radius = tuple(
        (
            1.2 + 0.1 * value * value,
            0.2 * value,
            0.2,
            0.0,
        )
        for value in coordinates
    )
    height = tuple(
        (
            value + 0.05 * value**3,
            1.0 + 0.15 * value * value,
            0.3 * value,
            0.3,
        )
        for value in coordinates
    )
    return radius, height


def corrugated_jets(coordinates):
    radius = tuple(
        (
            1.0 + 0.12 * _cos(3.0 * value),
            -0.36 * _sin(3.0 * value),
            -1.08 * _cos(3.0 * value),
            3.24 * _sin(3.0 * value),
        )
        for value in coordinates
    )
    height = tuple(
        (
            value + 0.06 * _sin(4.0 * value),
            1.0 + 0.24 * _cos(4.0 * value),
            -0.96 * _sin(4.0 * value),
            -3.84 * _cos(4.0 * value),
        )
        for value in coordinates
    )
    return radius, height


def atlas_rows():
    cases = []
    geodesic_coordinates = tuple(
        -2.0 * GOLDEN_MU + 4.0 * GOLDEN_MU * index / 16
        for index in range(17)
    )
    cases.append(
        (
            "long_geodesic",
            geodesic_coordinates,
            tuple(
                (
                    _exp(value),
                    _exp(value),
                    _exp(value),
                    _exp(value),
                )
                for value in geodesic_coordinates
            ),
            ((0.0, 0.0, 0.0, 0.0),) * len(geodesic_coordinates),
        )
    )
    polynomial_coordinates = tuple(-0.8 + 1.6 * index / 16 for index in range(17))
    polynomial_radius, polynomial_height = polynomial_jets(polynomial_coordinates)
    cases.append(
        (
            "polynomial_meridian",
            polynomial_coordinates,
            polynomial_radius,
            polynomial_height,
        )
    )
    corrugated_coordinates = tuple(-1.0 + 2.0 * index / 24 for index in range(25))
    corrugated_radius, corrugated_height = corrugated_jets(corrugated_coordinates)
    cases.append(
        (
            "corrugated_meridian",
            corrugated_coordinates,
            corrugated_radius,
            corrugated_height,
        )
    )
    rows = []
    for name, coordinates, radius_jets, height_jets in cases:
        atlas = GoldenHyperbolicJetAtlas(coordinates, radius_jets, height_jets)
        maximum_qjet_error = 0.0
        maximum_block_budget_fraction = 0.0
        maximum_exact_pair_budget_fraction = 0.0
        maximum_visit_budget_fraction = 0.0
        for patch in atlas.patches:
            qjet = patch.qjet(16, 8)
            field = tuple(
                tuple(
                    coordinate + 0.2 * _cos(TAU * 3 * phase / 8)
                    for phase in range(8)
                )
                for coordinate in qjet.coordinates
            )
            maximum_qjet_error = max(
                maximum_qjet_error,
                relative_grid_error(
                    qjet.apply(field),
                    reference_axisymmetric_physical(qjet, field),
                ),
            )
            plan_stats = qjet.plan.stats()
            maximum_block_budget_fraction = max(
                maximum_block_budget_fraction,
                plan_stats["compiled_block_count"]
                / plan_stats["static_block_budget"],
            )
            maximum_exact_pair_budget_fraction = max(
                maximum_exact_pair_budget_fraction,
                plan_stats["compiled_exact_pairs"]
                / plan_stats["exact_pair_budget"],
            )
            maximum_visit_budget_fraction = max(
                maximum_visit_budget_fraction,
                plan_stats["compile_pair_visits"]
                / plan_stats["compile_visit_budget"],
            )
        stats = atlas.stats()
        rows.append(
            {
                "shape": name,
                "source_nodes": stats["source_nodes"],
                "patch_count": stats["patch_count"],
                "total_hyperbolic_length": stats["total_hyperbolic_length"],
                "maximum_patch_span": stats["maximum_patch_span"],
                "maximum_qjet_relative_error": maximum_qjet_error,
                "maximum_block_budget_fraction": (
                    maximum_block_budget_fraction
                ),
                "maximum_exact_pair_budget_fraction": (
                    maximum_exact_pair_budget_fraction
                ),
                "maximum_visit_budget_fraction": (
                    maximum_visit_budget_fraction
                ),
                "quadratic_fallback": False,
                "stored_dense_matrix": False,
            }
        )
    return rows


def _chebyshev_interpolation_error(order):
    nodes = []
    weights = []
    values = []
    for index in range(order):
        angle = PI * (index + 0.5) / order
        node = _cos(angle)
        nodes.append(node)
        weights.append((-1.0 if index % 2 else 1.0) * _sin(angle))
        values.append(
            (GOLDEN_ANALYTIC_RADIUS + node)
            / (GOLDEN_ANALYTIC_RADIUS - node)
        )
    maximum = 0.0
    for index in range(2001):
        target = -1.0 + 2.0 * index / 2000
        numerators = tuple(
            weight / (target - node)
            for node, weight in zip(nodes, weights, strict=True)
        )
        denominator = sum(numerators)
        approximate = sum(
            numerator * value
            for numerator, value in zip(numerators, values, strict=True)
        ) / denominator
        exact = (
            GOLDEN_ANALYTIC_RADIUS + target
        ) / (GOLDEN_ANALYTIC_RADIUS - target)
        maximum = max(maximum, abs(approximate - exact) / exact)
    return maximum


def interpolation_rows():
    return tuple(
        {
            "order": order,
            "measured_relative_error": _chebyshev_interpolation_error(order),
            "golden_geometric_factor": golden_coordinate_geometric_factor(order),
        }
        for order in (8, 12, 16, 20, 24, 28, 32)
    )


def parameterization_rows():
    rows = []
    count = 128
    n_theta = 8
    raw_coordinates = tuple(-1.0 + 2.0 * index / (count - 1) for index in range(count))
    for distortion in (0.3, 0.03, 0.003):
        def signed_distance(value, distortion=distortion):
            return GOLDEN_MU * (
                distortion * value + (1.0 - distortion) * value**3
            )

        raw_distances = tuple(signed_distance(value) for value in raw_coordinates)
        golden_coordinates = tuple(
            golden_arclength_coordinate(value) for value in raw_distances
        )
        uniform_arclength = tuple(
            -GOLDEN_MU + 2.0 * GOLDEN_MU * index / (count - 1)
            for index in range(count)
        )
        uniform_golden = tuple(-1.0 + 2.0 * index / (count - 1) for index in range(count))

        def raw_meridian(value, signed_distance=signed_distance):
            return _exp(signed_distance(value)), 0.0

        def arclength_meridian(value):
            return _exp(value), 0.0

        def golden_meridian(value):
            return (
                (GOLDEN_ANALYTIC_RADIUS + value)
                / (GOLDEN_ANALYTIC_RADIUS - value),
                0.0,
            )

        configurations = (
            ("raw_uniform_parameter", raw_coordinates, raw_meridian, raw_distances),
            ("same_nodes_arclength", raw_distances, arclength_meridian, raw_distances),
            ("same_nodes_golden", golden_coordinates, golden_meridian, raw_distances),
            ("uniform_arclength", uniform_arclength, arclength_meridian, uniform_arclength),
            (
                "uniform_golden",
                uniform_golden,
                golden_meridian,
                tuple(golden_xi_from_tau(value).real for value in uniform_golden),
            ),
        )
        for name, coordinates, meridian, physical_distances in configurations:
            compile_start = time.perf_counter()
            qjet = AxisymmetricScalePhaseQJet(
                coordinates,
                meridian,
                n_theta,
                (0.01,) * count,
            )
            compile_ms = 1000.0 * (time.perf_counter() - compile_start)
            field = tuple(
                tuple(
                    physical_distances[index]
                    + 0.2 * _cos(TAU * 3 * phase / n_theta)
                    for phase in range(n_theta)
                )
                for index in range(count)
            )
            start = time.perf_counter()
            candidate = qjet.apply(field)
            apply_ms = 1000.0 * (time.perf_counter() - start)
            reference = reference_axisymmetric_spectral(qjet, field)
            spacings = tuple(
                physical_distances[index + 1] - physical_distances[index]
                for index in range(count - 1)
            )
            stats = qjet.plan.stats()
            rows.append(
                {
                    "distortion": distortion,
                    "parameterization": name,
                    "spacing_ratio": max(spacings) / min(spacings),
                    "compile_ms": compile_ms,
                    "apply_ms": apply_ms,
                    "relative_error": relative_grid_error(candidate, reference),
                    "compressed_pair_fraction": stats[
                        "compressed_pair_fraction"
                    ],
                    "stored_dense_matrix": False,
                }
            )
    return rows


def write_report(summary):
    lines = [
        "# Golden hyperbolic normalization benchmark",
        "",
        "The golden coordinate is the scaled Cayley map",
        "",
        "```text",
        "tau = sqrt(5) (g(w)-i)/(g(w)+i).",
        "```",
        "",
        "It maps the whole hyperbolic upper half-plane to `|tau| < sqrt(5)` "
        "and sends signed distance `+/-2 log(phi)` to `+/-1`.",
        "",
        "## Integer-trace optimality",
        "",
        "| trace | half length | disk radius | Bernstein radius | order at 2e-16 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["trace_rows"]:
        lines.append(
            f"| {row['integer_trace']} | {row['half_length']:.6f} | "
            f"{row['analytic_disk_radius']:.6f} | "
            f"{row['bernstein_radius']:.6f} | "
            f"{row['order_for_2e-16_geometric_factor']} |"
        )
    lines.extend(
        [
            "",
            "Trace three uniquely maximizes both radii. Its Bernstein factor "
            "is `sqrt(5)+2=phi^3`; order 24 has geometric factor "
            f"`{golden_coordinate_geometric_factor(24):.3e}`.",
            "",
            "## Frame invariance",
            "",
            "| shape | inverse error | distance error | strip error | disk fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["frame_rows"]:
        lines.append(
            f"| {row['shape']} | {row['inverse_error']:.3e} | "
            f"{row['distance_relative_error']:.3e} | "
            f"{row['strip_relative_error']:.3e} | "
            f"{row['maximum_disk_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Three-jet atlas",
            "",
            "| shape | source nodes | patches | total length | max patch | Q error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["atlas_rows"]:
        lines.append(
            f"| {row['shape']} | {row['source_nodes']} | "
            f"{row['patch_count']} | {row['total_hyperbolic_length']:.6f} | "
            f"{row['maximum_patch_span']:.6f} | "
            f"{row['maximum_qjet_relative_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Uniform hyperbolic sampling removes parameter-induced node "
            "clustering. Golden compactification keeps that property while "
            "fixing a canonical algebraic interval and analytic collar. A "
            "coordinate change on the same badly clustered nodes does not "
            "repair the sampling; the atlas must generate the nodes from the "
            "normalized jets.",
            "",
            "The tetration input is only the fixed-point multiplier "
            "`1/(2 phi)`. It fixes `phi^-2 = 4 mu_T^2`; the geometric map "
            "itself is the branch-free Cayley/PSL(2,R) normalization, not raw "
            "complex tetration.",
            "",
            "## Production execution contract",
            "",
            "Production objects expose only fixed-rank QJet applies. Static "
            "compilation is capped at `64N` pair visits, `16N` block "
            "records, and `64N` exact local pairs per mode. A cap violation "
            "raises; there is no quadratic fallback. Streamed pairwise "
            "oracles live only in `inverse_shape.testing`.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "golden_trace_three_hyperbolic_cayley_atlas",
        "golden_mu": GOLDEN_MU,
        "golden_radial_contraction": GOLDEN_RADIAL_CONTRACTION,
        "analytic_disk_radius": GOLDEN_ANALYTIC_RADIUS,
        "bernstein_radius": GOLDEN_BERNSTEIN_RADIUS,
        "trace_rows": trace_rows(),
        "frame_rows": frame_rows(),
        "atlas_rows": atlas_rows(),
        "interpolation_rows": interpolation_rows(),
        "parameterization_rows": parameterization_rows(),
        "quadratic_fallback": False,
        "reference_oracles_in_production_objects": False,
        "stored_dense_matrix": False,
    }
    write_csv(OUT / "integer_trace_optimality.csv", summary["trace_rows"])
    write_csv(OUT / "frame_invariance.csv", summary["frame_rows"])
    write_csv(OUT / "atlas.csv", summary["atlas_rows"])
    write_csv(OUT / "interpolation.csv", summary["interpolation_rows"])
    write_csv(OUT / "parameterization.csv", summary["parameterization_rows"])
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
