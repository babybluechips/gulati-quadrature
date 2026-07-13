#!/usr/bin/env python3
# ruff: noqa: E501
"""Benchmark the static Joukowski endpoint compiler without dense matrices."""

import csv
import json
from pathlib import Path
from time import perf_counter

from inverse_shape.conic_pencil_surface import twisted_ellipse_tube_qjet
from inverse_shape.joukowski_endpoint import (
    GOLDEN_MU,
    JoukowskiMapQJet,
    MellinEndpointChannel,
    StaticJoukowskiAnnulusQJet,
    StaticJoukowskiEllipseQJet,
    StaticMellinEndpointRepayment,
    golden_joukowski_ellipse_qjet,
)
from inverse_shape.quadrature import TAU, _abs, _cos, _log, _sin, _sqrt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "joukowski_static_endpoint"


def relative_error(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def fitted_exponent(rows, time_key):
    filtered = [row for row in rows if row[time_key] > 0.0]
    x_values = [_log(float(row["nodes"])) for row in filtered]
    y_values = [_log(float(row[time_key])) for row in filtered]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    return numerator / denominator


def ellipse_field(n):
    return tuple(
        _cos(3.0 * TAU * index / n)
        + 0.17 * _sin(7.0 * TAU * index / n)
        + 0.03 * _cos(13.0 * TAU * index / n)
        for index in range(n)
    )


def benchmark_golden_scaling():
    rows = []
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        start = perf_counter()
        qjet = golden_joukowski_ellipse_qjet(n)
        compile_ms = 1000.0 * (perf_counter() - start)
        values = ellipse_field(n)
        start = perf_counter()
        static = qjet.apply(values)
        apply_ms = 1000.0 * (perf_counter() - start)
        direct_ms = 0.0
        error = 0.0
        cartesian_reference_error = 0.0
        if n <= 1024:
            start = perf_counter()
            direct = qjet.direct_apply_factored(values)
            direct_ms = 1000.0 * (perf_counter() - start)
            error = relative_error(direct, static)
            cartesian = qjet.direct_apply(values)
            cartesian_reference_error = relative_error(direct, cartesian)
        rows.append(
            {
                "nodes": n,
                "compile_ms": compile_ms,
                "static_apply_ms": apply_ms,
                "direct_ms": direct_ms,
                "static_direct_relative_error": error,
                "cartesian_reference_relative_error": cartesian_reference_error,
                "channel_radius": qjet.channel_radius,
                "fft_channels": len(qjet.channels),
                "tail_bound": qjet.quotient_tail_bound(),
                "constant_residual": qjet.constant_residual(),
                "dense_entries_avoided": n * n,
                "stored_dense_matrix": False,
            }
        )
    return rows


def benchmark_ellipse_family():
    rows = []
    for name, mu in (
        ("nearly_circular", 2.0),
        ("golden", GOLDEN_MU),
        ("moderately_slender", 0.55),
        ("near_cusp_but_regular", 0.30),
    ):
        mapping = JoukowskiMapQJet(1.0, mu)
        start = perf_counter()
        qjet = StaticJoukowskiEllipseQJet(mapping, 128)
        compile_ms = 1000.0 * (perf_counter() - start)
        values = ellipse_field(128)
        start = perf_counter()
        static = qjet.apply(values)
        static_ms = 1000.0 * (perf_counter() - start)
        start = perf_counter()
        direct = qjet.direct_apply_factored(values)
        direct_ms = 1000.0 * (perf_counter() - start)
        rows.append(
            {
                "shape": name,
                "mu": mu,
                "axis_a": mapping.axis_a,
                "axis_b": mapping.axis_b,
                "eccentricity": mapping.eccentricity,
                "quotient_ratio": mapping.modulation_ratio,
                "channel_radius": qjet.channel_radius,
                "fft_channels": len(qjet.channels),
                "tail_bound": qjet.quotient_tail_bound(),
                "compile_ms": compile_ms,
                "static_apply_ms": static_ms,
                "direct_ms": direct_ms,
                "relative_error": relative_error(direct, static),
                "cartesian_reference_relative_error": relative_error(
                    direct,
                    qjet.direct_apply(values),
                ),
                "stored_dense_matrix": False,
            }
        )
    return rows


def annulus_field(qjet):
    return tuple(
        tuple(
            _cos(2.0 * TAU * column / qjet.n_theta)
            + 0.13 * _sin(TAU * row / qjet.n_scale)
            for column in range(qjet.n_theta)
        )
        for row in range(qjet.n_scale)
    )


def benchmark_annulus():
    rows = []
    for kernel_power in (2.0, 3.0):
        for n_scale, n_theta in ((2, 8), (4, 16), (8, 32)):
            start = perf_counter()
            qjet = StaticJoukowskiAnnulusQJet(
                1.0,
                GOLDEN_MU,
                GOLDEN_MU + 0.4,
                n_scale,
                n_theta,
                kernel_power=kernel_power,
            )
            compile_ms = 1000.0 * (perf_counter() - start)
            values = annulus_field(qjet)
            start = perf_counter()
            static = qjet.apply(values)
            static_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            direct = qjet.direct_apply(values)
            direct_ms = 1000.0 * (perf_counter() - start)
            rows.append(
                {
                    "kernel_power": kernel_power,
                    "normalization": qjet.normalization,
                    "n_scale": n_scale,
                    "n_theta": n_theta,
                    "nodes": qjet.n_nodes,
                    "compile_ms": compile_ms,
                    "static_apply_ms": static_ms,
                    "direct_ms": direct_ms,
                    "relative_error": relative_error(
                        tuple(value for row in direct for value in row),
                        tuple(value for row in static for value in row),
                    ),
                    "channel_radius": qjet.channel_radius,
                    "modulation_channels": len(qjet.channels),
                    "tail_bound": qjet.quotient_tail_bound(),
                    "constant_residual": qjet.constant_residual(),
                    "dense_entries_avoided": qjet.n_nodes * qjet.n_nodes,
                    "stored_dense_matrix": False,
                }
            )
    return rows


def benchmark_conic_integration():
    qjet = twisted_ellipse_tube_qjet(3.0, 0.8, 0.35, 1.2, 8, 32)
    nodes = qjet.generate_nodes()
    values = tuple(
        point[0] + 0.2 * point[1] - 0.1 * point[2] * point[2]
        for point in nodes.points
    )
    start = perf_counter()
    static = qjet.apply_same_slice_joukowski(values)
    static_ms = 1000.0 * (perf_counter() - start)
    start = perf_counter()
    direct = [0.0 for _ in range(qjet.n_nodes)]
    pair_count = 0
    for slice_index in range(qjet.n_slices):
        first = slice_index * qjet.n_theta
        stop = first + qjet.n_theta
        for left in range(first, stop):
            for right in range(left + 1, stop):
                distance_squared = sum(
                    (nodes.points[left][axis] - nodes.points[right][axis]) ** 2
                    for axis in range(3)
                )
                kernel = 1.0 / distance_squared
                difference = values[left] - values[right]
                direct[left] += nodes.weights[right] * kernel * difference
                direct[right] -= nodes.weights[left] * kernel * difference
                pair_count += 1
    direct_ms = 1000.0 * (perf_counter() - start)
    return {
        "nodes": qjet.n_nodes,
        "slices": qjet.n_slices,
        "n_theta": qjet.n_theta,
        "static_apply_ms": static_ms,
        "direct_same_slice_ms": direct_ms,
        "direct_pairs_replaced": pair_count,
        "relative_error": relative_error(direct, static),
        "cycle_fft_channels": qjet.last_apply_stats["cycle_fft_channels"],
        "maximum_tail_bound": qjet.last_apply_stats["maximum_joukowski_tail_bound"],
        "cross_slice_interactions_included": False,
        "stored_dense_matrix": False,
    }


def benchmark_endpoint_repayment():
    channels = (
        MellinEndpointChannel(0.5, 2.0, 0.5, "square-root cusp"),
        MellinEndpointChannel(2.0 / 3.0, -0.3, 0.25, "reentrant edge"),
        MellinEndpointChannel(1.25, 0.08, 0.75, "higher corner mode"),
    )
    repayment = StaticMellinEndpointRepayment(channels)
    return tuple(
        {
            "step": step,
            **repayment.evaluate(step),
        }
        for step in (0.08, 0.04, 0.02, 0.01)
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _log_point(value, minimum, maximum, low, high):
    coordinate = (_log(value) - _log(minimum)) / (_log(maximum) - _log(minimum))
    return low + coordinate * (high - low)


def write_scaling_svg(path, rows):
    timed = [row for row in rows if row["direct_ms"] > 0.0]
    min_n = min(row["nodes"] for row in rows)
    max_n = max(row["nodes"] for row in rows)
    times = [row["static_apply_ms"] for row in rows]
    times.extend(row["direct_ms"] for row in timed)
    min_time = min(times)
    max_time = max(times)

    def point(row, key):
        return (
            _log_point(row["nodes"], min_n, max_n, 90.0, 930.0),
            430.0 - _log_point(row[key], min_time, max_time, 0.0, 330.0),
        )

    static_points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (point(row, "static_apply_ms") for row in rows)
    )
    direct_points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (point(row, "direct_ms") for row in timed)
    )
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="520" viewBox="0 0 1020 520">',
        '<rect width="1020" height="520" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111;letter-spacing:0}.title{font-size:20px;font-weight:600}.label{font-size:14px}</style>',
        '<text x="510" y="34" text-anchor="middle" class="title">Golden Joukowski inverse-square action: measured scaling</text>',
        '<line x1="90" y1="430" x2="930" y2="430" stroke="#111"/>',
        '<line x1="90" y1="100" x2="90" y2="430" stroke="#111"/>',
        f'<polyline points="{static_points}" fill="none" stroke="#111" stroke-width="2"/>',
        f'<polyline points="{direct_points}" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="7 5"/>',
        '<text x="750" y="78" class="label">black: static FFT endpoint compiler</text>',
        '<text x="750" y="98" class="label" fill="#666">grey: direct pair stream</text>',
        '<text x="510" y="486" text-anchor="middle" class="label">nodes (log scale)</text>',
        '<text x="25" y="270" text-anchor="middle" class="label" transform="rotate(-90 25 270)">apply time (log scale)</text>',
        '</svg>',
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def compact(value):
    return f"{float(value):.3e}"


def write_report(path, summary, scaling, family, annulus, conic):
    lines = [
        "# Static Joukowski endpoint benchmark",
        "",
        "The inverse-square and normalized inverse-cube physical chords are compiled from the exact Joukowski factorization. No target/source pair table is retained. The singular difference coordinate is handled by the foundational QJet FFT; the smooth sum-coordinate quotient is a finite geometric channel list whose tail is checked before application.",
        "",
        "## Headline",
        "",
        f"- golden ellipse maximum static/direct error: `{compact(summary['maximum_golden_error'])}`",
        f"- maximum Cartesian-reference cancellation diagnostic: `{compact(summary['maximum_cartesian_reference_error'])}`",
        f"- exterior annulus Q2 maximum static/direct error: `{compact(summary['maximum_annulus_q2_error'])}`",
        f"- normalized exterior annulus Q3 maximum static/direct error: `{compact(summary['maximum_annulus_q3_error'])}`",
        f"- conic-surface same-slice error: `{compact(conic['relative_error'])}`",
        f"- fitted static apply exponent: `{summary['static_exponent']:.3f}`",
        f"- fitted direct exponent: `{summary['direct_exponent']:.3f}`",
        f"- golden modulation channels: `{summary['golden_channels']}`",
        f"- golden quotient tail bound: `{compact(summary['golden_tail_bound'])}`",
        "- dense matrices stored: `no`",
        "",
        "## Golden scaling",
        "",
        "| nodes | compile ms | static apply ms | direct ms | stable error | Cartesian reference loss | dense entries avoided |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scaling:
        lines.append(
            f"| {row['nodes']} | {row['compile_ms']:.3f} | {row['static_apply_ms']:.3f} | {row['direct_ms']:.3f} | `{compact(row['static_direct_relative_error'])}` | `{compact(row['cartesian_reference_relative_error'])}` | {row['dense_entries_avoided']} |"
        )
    lines.extend(
        [
            "",
            "## Eccentricity audit",
            "",
            "| chart | mu | eccentricity | channels | tail | error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in family:
        lines.append(
            f"| {row['shape']} | {row['mu']:.5f} | {row['eccentricity']:.5f} | {row['fft_channels']} | `{compact(row['tail_bound'])}` | `{compact(row['relative_error'])}` |"
        )
    lines.extend(
        [
            "",
            "## Two-dimensional exterior chart",
            "",
            "| power | grid | nodes | static ms | direct ms | error | channels |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in annulus:
        lines.append(
            f"| {row['kernel_power']:.0f} | {row['n_scale']}x{row['n_theta']} | {row['nodes']} | {row['static_apply_ms']:.3f} | {row['direct_ms']:.3f} | `{compact(row['relative_error'])}` | {row['modulation_channels']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This implementation removes the `1e-3` Barnes-Hut error on the compiled Joukowski channels. The 3D conic integration currently replaces same-slice singular interactions only. A complete arbitrary curved-surface operator still needs statically compiled cross-slice chart residuals, tangent-cell repayment, and the lower-order DtN geometry channel. Near `mu=0`, the geometric series is rejected and the map must hand off to the Mellin cusp channels rather than increasing the Fourier rank without bound.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaling = benchmark_golden_scaling()
    family = benchmark_ellipse_family()
    annulus = benchmark_annulus()
    conic = benchmark_conic_integration()
    endpoint = benchmark_endpoint_repayment()
    direct_rows = [row for row in scaling if row["direct_ms"] > 0.0]
    summary = {
        "maximum_golden_error": max(row["static_direct_relative_error"] for row in direct_rows),
        "maximum_cartesian_reference_error": max(row["cartesian_reference_relative_error"] for row in direct_rows),
        "maximum_annulus_q2_error": max(
            row["relative_error"] for row in annulus if row["kernel_power"] == 2.0
        ),
        "maximum_annulus_q3_error": max(
            row["relative_error"] for row in annulus if row["kernel_power"] == 3.0
        ),
        "static_exponent": fitted_exponent(scaling, "static_apply_ms"),
        "direct_exponent": fitted_exponent(direct_rows, "direct_ms"),
        "golden_channels": scaling[0]["fft_channels"],
        "golden_tail_bound": scaling[0]["tail_bound"],
        "maximum_constant_residual": max(row["constant_residual"] for row in scaling),
        "conic_integration": conic,
        "stored_dense_matrix": False,
        "complexity": "O(L N log N + L^2 N) compiled charts; O(N) storage at fixed tolerance",
    }
    write_csv(OUT / "golden_scaling.csv", scaling)
    write_csv(OUT / "ellipse_family.csv", family)
    write_csv(OUT / "annulus_scaling.csv", annulus)
    write_csv(OUT / "conic_same_slice.csv", [conic])
    write_csv(
        OUT / "mellin_endpoint.csv",
        [
            {
                "step": row["step"],
                "value": row["value"],
                "next_term_estimate": row["next_term_estimate"],
                "channel_count": row["channel_count"],
                "grid_refinement_iterations": row["grid_refinement_iterations"],
                "stored_dense_matrix": row["stored_dense_matrix"],
            }
            for row in endpoint
        ],
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        OUT / "report.md",
        summary,
        scaling,
        family,
        annulus,
        conic,
    )
    write_scaling_svg(OUT / "scaling.svg", scaling)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
