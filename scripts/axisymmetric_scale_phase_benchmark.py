#!/usr/bin/env python3
"""Benchmark exact-alias nested Q on general surfaces of revolution."""

from __future__ import annotations

import csv
import json
import statistics
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
from inverse_shape.quadrature import PI, TAU, _cos, _log, _sin, _sqrt  # noqa: E402
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_axisymmetric_physical,
)


OUT = ROOT / "outputs" / "axisymmetric_scale_phase"
N_THETA = 8
SCALE_SIZES = (16, 32, 64, 128, 256, 512)


def coordinates(count, start=-0.9, stop=0.9):
    return tuple(
        start + (stop - start) * index / (count - 1) for index in range(count)
    )


def field(values, n_theta):
    return tuple(
        tuple(
            0.4
            + 0.2 * coordinate
            + 0.13 * _cos(TAU * 3 * phase / n_theta)
            + 0.03j * (index + 1) * _sin(TAU * phase / n_theta)
            for phase in range(n_theta)
        )
        for index, coordinate in enumerate(values)
    )


def weights(values, meridian):
    step = (values[-1] - values[0]) / (len(values) - 1)
    return tuple(
        meridian(value)[0] * step * (1.0 + 0.01 * index)
        for index, value in enumerate(values)
    )


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


def median_seconds(operation, repeats=3):
    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings)


def fit_power(rows, size_key, time_key):
    points = [
        (_log(float(row[size_key])), _log(float(row[time_key])))
        for row in rows
        if row[time_key] is not None and row[time_key] > 0.0
    ]
    mean_x = sum(x for x, _y in points) / len(points)
    mean_y = sum(y for _x, y in points) / len(points)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    )
    denominator = sum((x - mean_x) ** 2 for x, _y in points)
    return numerator / denominator


def meridian_cases():
    return (
        ("cylinder", lambda value: (1.0, value)),
        ("cone", lambda value: (0.9 + 0.28 * value, value)),
        (
            "sphere_cap",
            lambda value: (_sqrt(1.45 * 1.45 - value * value), value),
        ),
        (
            "corrugated",
            lambda value: (
                1.0 + 0.15 * _cos(5.0 * value),
                value + 0.08 * _sin(4.0 * value),
            ),
        ),
        (
            "double_neck",
            lambda value: (0.85 - 0.27 * _cos(2.0 * PI * value / 1.8), value),
        ),
        (
            "cusp_meridian",
            lambda value: (0.35 + 0.55 * abs(value) ** 0.75, value),
        ),
        (
            "airfoil_body",
            lambda value: (
                0.18 + 0.9 * _sqrt(max(1.0 - value * value, 0.0)),
                value + 0.06 * (1.0 - value * value),
            ),
        ),
    )


def accuracy_campaign():
    rows = []
    values = coordinates(19, -0.85, 0.85)
    n_theta = 16
    source = field(values, n_theta)
    for name, meridian in meridian_cases():
        compile_start = time.perf_counter()
        qjet = AxisymmetricScalePhaseQJet(
            values,
            meridian,
            n_theta,
            weights(values, meridian),
        )
        compile_ms = 1000.0 * (time.perf_counter() - compile_start)
        candidate, fast_seconds = median_seconds(
            lambda qjet=qjet, source=source: qjet.apply(source)
        )
        reference, direct_seconds = median_seconds(
            lambda qjet=qjet, source=source: (
                reference_axisymmetric_physical(qjet, source)
            ),
        )
        stats = qjet.stats()
        rows.append(
            {
                "shape": name,
                "n_nodes": qjet.n_nodes,
                "compile_ms": compile_ms,
                "nested_ms": 1000.0 * fast_seconds,
                "direct_ms": 1000.0 * direct_seconds,
                "relative_error": relative_grid_error(candidate, reference),
                "constant_residual": qjet.constant_residual(),
                "compressed_pair_fraction": stats["mode_plan"][
                    "compressed_pair_fraction"
                ],
                "stored_dense_matrix": False,
            }
        )
    return rows


def scaling_campaign():
    rows = []

    def meridian(value):
        return (
            1.0 + 0.14 * _cos(4.0 * value),
            value + 0.07 * _sin(3.0 * value),
        )

    for count in SCALE_SIZES:
        values = coordinates(count)
        compile_start = time.perf_counter()
        qjet = AxisymmetricScalePhaseQJet(
            values,
            meridian,
            N_THETA,
            weights(values, meridian),
        )
        compile_ms = 1000.0 * (time.perf_counter() - compile_start)
        source = field(values, N_THETA)
        candidate, nested_seconds = median_seconds(
            lambda qjet=qjet, source=source: qjet.apply(source)
        )
        if count <= 128:
            reference, direct_seconds = median_seconds(
                lambda qjet=qjet, source=source: (
                    reference_axisymmetric_physical(qjet, source)
                ),
            )
            direct_ms = 1000.0 * direct_seconds
            error = relative_grid_error(candidate, reference)
        else:
            direct_ms = None
            error = None
        stats = qjet.stats()["mode_plan"]
        rows.append(
            {
                "n_scale": count,
                "n_theta": N_THETA,
                "n_nodes": qjet.n_nodes,
                "compile_ms": compile_ms,
                "nested_ms": 1000.0 * nested_seconds,
                "direct_ms": direct_ms,
                "relative_error": error,
                "constant_residual": qjet.constant_residual(),
                "compressed_pair_fraction": stats["compressed_pair_fraction"],
                "stored_mode_tile_entries": stats[
                    "stored_mode_tile_entries"
                ],
                "stored_dense_matrix": False,
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary):
    lines = [
        "# Axisymmetric scale-phase benchmark",
        "",
        "Every cross-ring mode uses the hyperbolic meridian distance. The "
        "closed alias quotient includes every finite-angle Fourier rung, so "
        "the independent reference is the physical all-pairs graph.",
        "",
        "| shape | nodes | nested ms | direct ms | relative error | constant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["accuracy_rows"]:
        lines.append(
            f"| {row['shape']} | {row['n_nodes']} | "
            f"{row['nested_ms']:.3e} | {row['direct_ms']:.3e} | "
            f"{row['relative_error']:.3e} | "
            f"{row['constant_residual']:.3e} |"
        )
    lines.extend(
        [
            "",
            "| nodes | nested ms | direct ms | relative error | compressed pairs |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["scaling_rows"]:
        direct = (
            f"{row['direct_ms']:.3e}"
            if row["direct_ms"] is not None
            else "not run"
        )
        error = (
            f"{row['relative_error']:.3e}"
            if row["relative_error"] is not None
            else "not run"
        )
        lines.append(
            f"| {row['n_nodes']} | {row['nested_ms']:.3e} | {direct} | "
            f"{error} | {row['compressed_pair_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The nested apply exponent is "
            f"{summary['fits']['nested_time_exponent']:.3f}; its tail fit is "
            f"{summary['fits']['nested_tail_exponent']:.3f}. The physical "
            "pair stream fits exponent "
            f"{summary['fits']['direct_time_exponent']:.3f}. All retained "
            "mode tiles have fixed size and total linear storage in the "
            "surface-node count.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy_rows = accuracy_campaign()
    scaling_rows = scaling_campaign()
    direct_rows = [row for row in scaling_rows if row["direct_ms"] is not None]
    summary = {
        "method": "axisymmetric_hyperbolic_exact_alias_qjet",
        "accuracy_rows": accuracy_rows,
        "scaling_rows": scaling_rows,
        "fits": {
            "nested_time_exponent": fit_power(
                scaling_rows,
                "n_nodes",
                "nested_ms",
            ),
            "nested_tail_exponent": fit_power(
                scaling_rows[-4:],
                "n_nodes",
                "nested_ms",
            ),
            "direct_time_exponent": fit_power(
                direct_rows,
                "n_nodes",
                "direct_ms",
            ),
        },
        "stored_dense_matrix": False,
        "stored_pair_table": False,
        "apply_complexity": "O(p^2 N + N log n_theta), fixed p",
        "storage_complexity": "O(p^2 N), fixed p",
    }
    write_csv(OUT / "accuracy.csv", accuracy_rows)
    write_csv(OUT / "scaling.csv", scaling_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
