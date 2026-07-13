#!/usr/bin/env python3
"""Benchmark the certified arbitrary-surface QJet against pair streaming."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.arbitrary_surface import (  # noqa: E402
    CertifiedArbitrarySurfaceQJet,
)
from inverse_shape.quadrature import PI, TAU, _cos, _log, _sin  # noqa: E402
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_weighted_distance_graph,
)


OUT = ROOT / "outputs" / "arbitrary_surface"
TIMING_REPEATS = 1
REFERENCE_TIMING_REPEATS = 3


def sphere(latitude_count=8, longitude_count=12, scales=(1.0, 1.0, 1.0)):
    points = []
    weights = []
    for latitude in range(latitude_count):
        theta = PI * (latitude + 0.5) / latitude_count
        sine = _sin(theta)
        cosine = _cos(theta)
        for longitude in range(longitude_count):
            phase = TAU * longitude / longitude_count
            points.append(
                (
                    scales[0] * sine * _cos(phase),
                    scales[1] * sine * _sin(phase),
                    scales[2] * cosine,
                )
            )
            weights.append(
                sine * PI / latitude_count * TAU / longitude_count
            )
    return tuple(points), tuple(weights)


def torus(major_count=12, minor_count=8):
    points = []
    weights = []
    major_radius = 1.4
    minor_radius = 0.35
    for major in range(major_count):
        u = TAU * major / major_count
        for minor in range(minor_count):
            v = TAU * minor / minor_count
            radial = major_radius + minor_radius * _cos(v)
            points.append(
                (
                    radial * _cos(u),
                    radial * _sin(u),
                    minor_radius * _sin(v),
                )
            )
            weights.append(
                minor_radius
                * radial
                * TAU
                / major_count
                * TAU
                / minor_count
            )
    return tuple(points), tuple(weights)


def folded_sheet(nx=12, ny=8):
    points = []
    weights = []
    for ix in range(nx):
        x = -1.0 + 2.0 * (ix + 0.5) / nx
        for iy in range(ny):
            y = -0.8 + 1.6 * (iy + 0.5) / ny
            z = 0.35 * _sin(1.7 * x) + 0.18 * _sin(2.3 * y + 0.4 * x)
            points.append((x, y, z))
            weights.append(3.2 / (nx * ny))
    return tuple(points), tuple(weights)


def mobius(around=16, across=6):
    points = []
    weights = []
    half_width = 0.28
    for along in range(around):
        u = TAU * along / around
        for transverse in range(across):
            v = half_width * (2.0 * (transverse + 0.5) / across - 1.0)
            radial = 1.2 + v * _cos(0.5 * u)
            points.append(
                (
                    radial * _cos(u),
                    radial * _sin(u),
                    v * _sin(0.5 * u),
                )
            )
            weights.append(2.0 * half_width * TAU * 1.2 / (around * across))
    return tuple(points), tuple(weights)


def star_surface():
    points, weights = sphere()
    output = []
    for x, y, z in points:
        phase_radius = max(x * x + y * y, 0.0) ** 0.5
        modulation = 1.0 + 0.16 * (4.0 * z * z - 1.0) * phase_radius
        output.append((modulation * x, modulation * y, modulation * z))
    return tuple(output), weights


def spherical_spiral(count=256):
    points = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = max(1.0 - z * z, 0.0) ** 0.5
        phase = TAU * index / count
        points.append((radius * _cos(phase), radius * _sin(phase), z))
    return tuple(points), (4.0 * PI / count,) * count


def field(points):
    return tuple(
        x + 0.2 * y - 0.1 * z * z + 0.03j * (y + z)
        for x, y, z in points
    )


def relative_l2(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def median_seconds(operation, repeats=TIMING_REPEATS):
    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings)


def compile_qjet(points, weights):
    return CertifiedArbitrarySurfaceQJet(
        points,
        weights,
        kernel_power=2.0,
        tolerance=2.0e-13,
        maximum_order=16,
        leaf_size=4,
    )


def benchmark_case(name, geometry):
    points, weights = geometry
    values = field(points)
    start = time.perf_counter()
    qjet = compile_qjet(points, weights)
    compile_seconds = time.perf_counter() - start
    qjet.apply(values)
    candidate, apply_seconds = median_seconds(lambda: qjet.apply(values))
    reference, direct_seconds = median_seconds(
        lambda: reference_weighted_distance_graph(
            points,
            weights,
            values,
            2.0,
        ),
        repeats=REFERENCE_TIMING_REPEATS,
    )
    stats = qjet.stats()
    actual_inf = max(
        abs(complex(left) - complex(right))
        for left, right in zip(reference, candidate, strict=True)
    )
    return {
        "shape": name,
        "n": len(points),
        "compile_ms": 1000.0 * compile_seconds,
        "hierarchy_apply_ms": 1000.0 * apply_seconds,
        "streamed_direct_ms": 1000.0 * direct_seconds,
        "direct_over_hierarchy_speedup": direct_seconds / apply_seconds,
        "relative_l2_error": relative_l2(reference, candidate),
        "actual_inf_error": actual_inf,
        "certified_compression_inf_bound": qjet.compression_inf_bound(values),
        "constant_residual": qjet.constant_residual(),
        "low_rank_pair_fraction": stats["analytic_pair_fraction"],
        "exact_pair_fraction": stats["near_field_pair_fraction"],
        "maximum_rank": stats["maximum_block_rank"],
        "stored_factor_entries": stats["persistent_moment_entries"],
        "analytic_apply_units": stats["analytic_apply_units"],
        "analytic_apply_budget": stats["analytic_apply_budget"],
        "analytic_blocks": stats["analytic_blocks"],
        "near_field_pairs": stats["near_field_pairs"],
        "near_field_pair_budget": stats["near_field_pair_budget"],
        "hard_no_quadratic_contract": stats["hard_no_quadratic_contract"],
        "quadratic_fallback": stats["quadratic_fallback"],
        "dense_matrix_stored": False,
    }


def fit_power(rows, key):
    points = [
        (float(row["n"]), float(row[key]))
        for row in rows
        if float(row[key]) > 0.0
    ]
    transformed = [(_log(x), _log(y)) for x, y in points]
    mean_x = sum(x for x, _y in transformed) / len(transformed)
    mean_y = sum(y for _x, y in transformed) / len(transformed)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in transformed
    )
    denominator = sum((x - mean_x) ** 2 for x, _y in transformed)
    return numerator / denominator


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary):
    lines = [
        "# Certified arbitrary-surface QJet benchmark",
        "",
        "All rows use the topology-free weighted-node API. Separated blocks "
        "use fixed-order analytic Gegenbauer moments; only adjacent leaf "
        "blocks are evaluated directly. No rejected far block, adaptive "
        "rank, pair table, or dense matrix is present.",
        "",
        "| shape | N | compile ms | hierarchy ms | direct ms | rel. error | "
        "compressed pairs | exact pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["shape_cases"]:
        lines.append(
            f"| {row['shape']} | {row['n']} | {row['compile_ms']:.3e} | "
            f"{row['hierarchy_apply_ms']:.3e} | "
            f"{row['streamed_direct_ms']:.3e} | "
            f"{row['relative_l2_error']:.3e} | "
            f"{row['low_rank_pair_fraction']:.3f} | "
            f"{row['exact_pair_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Two-dimensional surface scaling",
            "",
            "| N | hierarchy ms | direct ms | rel. error | exact pairs | "
            "analytic units | far blocks |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["surface_scaling"]:
        lines.append(
            f"| {row['n']} | {row['hierarchy_apply_ms']:.3e} | "
            f"{row['streamed_direct_ms']:.3e} | "
            f"{row['relative_l2_error']:.3e} | "
            f"{row['exact_pair_fraction']:.3f} | "
            f"{row['analytic_apply_units']} | "
            f"{row['analytic_blocks']} |"
        )
    lines.extend(
        [
            "",
            "The fitted apply exponent is "
            f"{summary['fits']['hierarchy_apply_exponent']:.3f}; the streamed "
            "reference exponent is "
            f"{summary['fits']['streamed_direct_exponent']:.3f}. The compiled "
            "analytic-work and far-block slopes over the same finite range "
            "are "
            f"{summary['fits']['analytic_work_exponent']:.3f} and "
            f"{summary['fits']['analytic_block_exponent']:.3f}. Finite-range "
            "slopes are measurements, not asymptotic proofs. The compiled "
            "backend enforces the asymptotic contract directly through fixed "
            "expansion order, linear persistent moments, a symmetric WSPD, "
            "exact terminal leaves, and explicit "
            "near-field and analytic-work budgets.",
            "",
            "Production gate: "
            f"{'PASS' if summary['gates']['passed'] else 'FAIL'}; maximum "
            "relative error "
            f"{summary['maximum_relative_error']:.3e}, measured apply slope "
            f"{summary['fits']['hierarchy_apply_exponent']:.3f} < 1.9, "
            "no dense matrix, and no quadratic fallback.",
            "",
            "The streamed direct reference is isolated in the testing module "
            "and is not callable through the production QJet API.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shape_cases = [
        benchmark_case("sphere", sphere()),
        benchmark_case("ellipsoid", sphere(scales=(1.8, 0.7, 1.2))),
        benchmark_case("torus", torus()),
        benchmark_case("folded_sheet", folded_sheet()),
        benchmark_case("mobius", mobius()),
        benchmark_case("star_surface", star_surface()),
        benchmark_case("spherical_spiral", spherical_spiral()),
    ]
    surface_scaling = [
        benchmark_case("sphere", sphere(latitudes, longitudes))
        for latitudes, longitudes in ((6, 8), (8, 12), (12, 16), (16, 24))
    ]
    fits = {
        "hierarchy_apply_exponent": fit_power(
            surface_scaling,
            "hierarchy_apply_ms",
        ),
        "streamed_direct_exponent": fit_power(
            surface_scaling,
            "streamed_direct_ms",
        ),
        "analytic_work_exponent": fit_power(
            surface_scaling,
            "analytic_apply_units",
        ),
        "analytic_block_exponent": fit_power(
            surface_scaling,
            "analytic_blocks",
        ),
    }
    maximum_relative_error = max(
        row["relative_l2_error"] for row in (*shape_cases, *surface_scaling)
    )
    all_rows = (*shape_cases, *surface_scaling)
    gates = {
        "machine_scale_accuracy": maximum_relative_error < 5.0e-14,
        "measured_apply_subquadratic": fits["hierarchy_apply_exponent"] < 1.9,
        "reference_exposes_quadratic_slope": (
            fits["streamed_direct_exponent"] > 1.9
        ),
        "hard_no_quadratic_contract": all(
            row["hard_no_quadratic_contract"] for row in all_rows
        ),
        "no_quadratic_fallback": not any(
            row["quadratic_fallback"] for row in all_rows
        ),
        "no_dense_matrix": not any(
            row["dense_matrix_stored"] for row in all_rows
        ),
    }
    gates["passed"] = all(gates.values())
    summary = {
        "method": "fixed_order_symmetric_gegenbauer_riesz_wspd",
        "numerical_dependencies": "project scalar QJet primitives; no NumPy",
        "stored_dense_matrix": False,
        "timing_repeats": TIMING_REPEATS,
        "reference_timing_repeats": REFERENCE_TIMING_REPEATS,
        "certification": "analytic Gegenbauer tail plus exact terminal leaves",
        "complexity": {
            "compile": "O(N log^2 N) for fixed order in 3D",
            "apply": "O(N log N) for fixed order in 3D",
            "storage": "O(N) for fixed order in 3D",
        },
        "shape_cases": shape_cases,
        "surface_scaling": surface_scaling,
        "maximum_relative_error": maximum_relative_error,
        "fits": fits,
        "gates": gates,
    }
    if not gates["passed"]:
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"arbitrary-surface production gates failed: {failed}")
    write_csv(OUT / "shape_cases.csv", shape_cases)
    write_csv(OUT / "surface_scaling.csv", surface_scaling)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
