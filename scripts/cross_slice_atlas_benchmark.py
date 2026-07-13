#!/usr/bin/env python3
# ruff: noqa: E501
"""Benchmark the static curved cross-slice atlas against streamed Q2/Q3."""

import csv
import json
from pathlib import Path
from time import perf_counter

from inverse_shape.conic_pencil_surface import (
    aircraft_conic_bundle_qjet,
    bent_conic_tube_qjet,
    straight_conic_tube_qjet,
    tapered_conic_tube_qjet,
    toroidal_conic_bundle_qjet,
    twisted_ellipse_tube_qjet,
)
from inverse_shape.cross_slice_atlas import StaticCrossSliceAtlasQJet
from inverse_shape.quadrature import TAU, _abs, _cos, _log, _sin, _sqrt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cross_slice_atlas"


def relative_error(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def field(qjet):
    nodes = qjet.generate_nodes()
    return tuple(
        point[0]
        + 0.17 * point[1]
        - 0.09 * point[2] * point[2]
        + 0.04 * point[0] * point[2]
        + 0.03 * _sin(5.0 * TAU * index / qjet.n_nodes)
        for index, point in enumerate(nodes.points)
    )


def shape_factory(name, n_slices, n_theta):
    if name == "circular_cylinder":
        return straight_conic_tube_qjet(3.0, 0.7, 0.7, n_slices, n_theta)
    if name == "elliptic_taper":
        return tapered_conic_tube_qjet(
            3.2, 0.35, 0.85, 0.22, 0.55, n_slices, n_theta
        )
    if name == "bent_tube":
        return bent_conic_tube_qjet(
            3.0, 1.25, 0.48, 0.31, n_slices, n_theta
        )
    if name == "twisted_ellipse":
        return twisted_ellipse_tube_qjet(
            3.2, 0.78, 0.30, 1.6, n_slices, n_theta
        )
    if name == "toroidal_bundle":
        return toroidal_conic_bundle_qjet(
            2.0, 0.40, 0.23, n_slices, n_theta
        )
    if name == "aircraft_body":
        return aircraft_conic_bundle_qjet(4.0, n_slices, n_theta)
    raise ValueError(f"unknown shape: {name}")


SHAPES = (
    "circular_cylinder",
    "elliptic_taper",
    "bent_tube",
    "twisted_ellipse",
    "toroidal_bundle",
    "aircraft_body",
)


def timed(function, repeats=1):
    best = float("inf")
    value = None
    for _ in range(repeats):
        start = perf_counter()
        candidate = function()
        elapsed = 1000.0 * (perf_counter() - start)
        if elapsed < best:
            best = elapsed
            value = candidate
    return value, best


def shape_suite():
    rows = []
    for shape in SHAPES:
        qjet = shape_factory(shape, 12, 16)
        values = field(qjet)
        for power in (2.0, 3.0):
            start = perf_counter()
            atlas = StaticCrossSliceAtlasQJet(
                qjet,
                kernel_power=power,
                tolerance=1.0e-10,
                admissibility=0.3,
                leaf_nodes=8,
                local_slice_span=1,
            )
            compile_ms = 1000.0 * (perf_counter() - start)
            first, first_apply_ms = timed(lambda: atlas.apply(values))
            static, warm_apply_ms = timed(lambda: atlas.apply(values), repeats=3)
            direct, direct_ms = timed(
                lambda: qjet.apply(values, kernel_power=power, method="direct"),
                repeats=2,
            )
            tree, tree_ms = timed(
                lambda: qjet.apply(
                    values,
                    kernel_power=power,
                    method="tree",
                    opening=0.30,
                    leaf_size=8,
                )
            )
            stats = atlas.stats()
            rows.append(
                {
                    "shape": shape,
                    "kernel_power": power,
                    "nodes": qjet.n_nodes,
                    "compile_ms": compile_ms,
                    "first_apply_ms": first_apply_ms,
                    "warm_apply_ms": warm_apply_ms,
                    "direct_ms": direct_ms,
                    "tree_ms": tree_ms,
                    "atlas_relative_error": relative_error(direct, static),
                    "first_warm_residual": relative_error(first, static),
                    "tree_relative_error": relative_error(direct, tree),
                    "constant_residual": atlas.constant_residual(),
                    "phase_fft_charts": stats["phase_fft_charts"],
                    "phase_direct_charts": stats["phase_direct_charts"],
                    "phase_channels": stats["phase_channels"],
                    "low_rank_blocks": stats["low_rank_blocks"],
                    "total_rank": stats["total_rank"],
                    "maximum_rank": stats["maximum_rank"],
                    "exact_cross_pair_fraction": stats[
                        "exact_cross_pair_fraction"
                    ],
                    "low_rank_pair_fraction": stats["low_rank_pair_fraction"],
                    "phase_chart_pair_fraction": stats[
                        "phase_chart_pair_fraction"
                    ],
                    "compile_kernel_samples": stats["compile_kernel_samples"],
                    "stored_factor_entries": stats["stored_factor_entries"],
                    "stored_phase_channel_entries": stats[
                        "stored_phase_channel_entries"
                    ],
                    "cross_pair_partition_residual": stats[
                        "cross_pair_partition_residual"
                    ],
                    "stored_dense_matrix": False,
                }
            )
    return rows


def scaling_suite():
    rows = []
    for n_slices in (8, 16, 32, 64):
        qjet = twisted_ellipse_tube_qjet(
            5.0, 0.5, 0.2, 1.8, n_slices, 16
        )
        values = tuple(
            _cos(3.0 * TAU * index / qjet.n_nodes)
            + 0.2 * _sin(7.0 * TAU * index / qjet.n_nodes)
            for index in range(qjet.n_nodes)
        )
        start = perf_counter()
        atlas = StaticCrossSliceAtlasQJet(
            qjet,
            kernel_power=2.0,
            tolerance=1.0e-10,
            admissibility=0.3,
            leaf_nodes=8,
            local_slice_span=1,
        )
        compile_ms = 1000.0 * (perf_counter() - start)
        atlas.apply(values)
        static, warm_ms = timed(lambda: atlas.apply(values), repeats=3)
        direct, direct_ms = timed(
            lambda: qjet.apply(values, kernel_power=2.0, method="direct")
        )
        stats = atlas.stats()
        rows.append(
            {
                "n_slices": n_slices,
                "n_theta": qjet.n_theta,
                "nodes": qjet.n_nodes,
                "compile_ms": compile_ms,
                "warm_apply_ms": warm_ms,
                "direct_ms": direct_ms,
                "relative_error": relative_error(direct, static),
                "low_rank_blocks": stats["low_rank_blocks"],
                "total_rank": stats["total_rank"],
                "low_rank_pair_fraction": stats["low_rank_pair_fraction"],
                "exact_cross_pair_fraction": stats[
                    "exact_cross_pair_fraction"
                ],
                "compile_kernel_samples": stats["compile_kernel_samples"],
                "stored_factor_entries": stats["stored_factor_entries"],
                "dense_entries_avoided": qjet.n_nodes * qjet.n_nodes,
                "stored_dense_matrix": False,
            }
        )
    return rows


def fitted_exponent(rows, key):
    x_values = [_log(float(row["nodes"])) for row in rows]
    y_values = [_log(max(float(row[key]), 1.0e-12)) for row in rows]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    return numerator / denominator


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path, scaling):
    width = 920
    height = 440
    left = 78
    right = 28
    top = 36
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_logs = [_log(row["nodes"]) for row in scaling]
    values = [
        row[key]
        for row in scaling
        for key in ("warm_apply_ms", "direct_ms")
    ]
    y_logs = [_log(max(value, 1.0e-9)) for value in values]
    x_min, x_max = min(x_logs), max(x_logs)
    y_min, y_max = min(y_logs), max(y_logs)

    def x_position(node_count):
        return left + plot_width * (_log(node_count) - x_min) / (x_max - x_min)

    def y_position(value):
        return top + plot_height * (y_max - _log(max(value, 1.0e-9))) / (y_max - y_min)

    atlas_points = " ".join(
        f"{x_position(row['nodes']):.2f},{y_position(row['warm_apply_ms']):.2f}"
        for row in scaling
    )
    direct_points = " ".join(
        f"{x_position(row['nodes']):.2f},{y_position(row['direct_ms']):.2f}"
        for row in scaling
    )
    labels = []
    for row in scaling:
        x = x_position(row["nodes"])
        labels.append(
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="middle">{row["nodes"]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font:14px Helvetica,Arial,sans-serif;fill:#111}} .axis{{stroke:#111;stroke-width:1}} .grid{{stroke:#d4d4d4;stroke-width:1}} </style>
<text x="{left}" y="22" font-weight="bold">Curved cross-slice atlas: warm apply scaling</text>
<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>
<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>
<polyline points="{direct_points}" fill="none" stroke="#999" stroke-width="2"/>
<polyline points="{atlas_points}" fill="none" stroke="#111" stroke-width="2.5"/>
{''.join(labels)}
<text x="{width/2:.1f}" y="{height-7}" text-anchor="middle">surface nodes N</text>
<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle">milliseconds (log scale)</text>
<line x1="{width-245}" y1="28" x2="{width-215}" y2="28" stroke="#111" stroke-width="2.5"/><text x="{width-205}" y="33">static atlas</text>
<line x1="{width-115}" y1="28" x2="{width-85}" y2="28" stroke="#999" stroke-width="2"/><text x="{width-75}" y="33">direct</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_report(path, shape_rows, scaling, summary):
    lines = [
        "# Static curved cross-slice atlas benchmark",
        "",
        "The local singular channel is the exact complete-slice Joukowski/cycle operator. Adjacent cross slices use phase-difference/modulation FFT charts. Separated product patches use symmetric adaptive-cross factors, and terminal failures are streamed exactly. No distance or operator matrix is stored.",
        "",
        "## Accuracy against the independent streamed reference",
        "",
        "| shape | p | nodes | atlas rel. error | old quadrupole rel. error | warm ms | direct ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shape_rows:
        lines.append(
            f"| {row['shape']} | {row['kernel_power']:.0f} | {row['nodes']} | {row['atlas_relative_error']:.3e} | {row['tree_relative_error']:.3e} | {row['warm_apply_ms']:.3f} | {row['direct_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scaling and storage",
            "",
            "| nodes | compile ms | warm ms | direct ms | rel. error | low-rank pair fraction | exact residual pair fraction | factor entries | N^2 entries avoided |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scaling:
        lines.append(
            f"| {row['nodes']} | {row['compile_ms']:.3f} | {row['warm_apply_ms']:.3f} | {row['direct_ms']:.3f} | {row['relative_error']:.3e} | {row['low_rank_pair_fraction']:.3f} | {row['exact_cross_pair_fraction']:.3f} | {row['stored_factor_entries']} | {row['dense_entries_avoided']} |"
        )
    lines.extend(
        [
            "",
            "## Cost statement",
            "",
            "For retained blocks the work is `O(sum_b r_b(m_b+n_b))`; local phase charts cost `O(L_local N log n_theta)` and exact terminal repayment costs `O(P_near)`. Thus a bounded-rank, bounded-neighbor atlas is `O(r N log N + L_local N log n_theta + N n_theta)` with `O(N + sum_b r_b(m_b+n_b))` storage. Arbitrary folded geometry can force rank growth or many exact terminal pairs, so this implementation does not claim an unconditional subquadratic worst-case bound.",
            "",
            f"Fitted warm-apply exponent on this four-size run: `{summary['atlas_apply_exponent']:.3f}`. Streamed direct exponent: `{summary['direct_apply_exponent']:.3f}`.",
            "",
            "The ACA residual is sampled, not a continuum theorem. The reported errors use the independent streamed all-pairs action, while the exact pair-partition checksum verifies that every cross-slice pair is represented once.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shapes = shape_suite()
    scaling = scaling_suite()
    summary = {
        "shape_count": len(SHAPES),
        "kernel_powers": [2, 3],
        "maximum_atlas_relative_error": max(
            row["atlas_relative_error"] for row in shapes
        ),
        "maximum_tree_relative_error": max(
            row["tree_relative_error"] for row in shapes
        ),
        "maximum_constant_residual": max(
            row["constant_residual"] for row in shapes
        ),
        "maximum_scaling_relative_error": max(
            row["relative_error"] for row in scaling
        ),
        "atlas_apply_exponent": fitted_exponent(scaling, "warm_apply_ms"),
        "direct_apply_exponent": fitted_exponent(scaling, "direct_ms"),
        "largest_low_rank_pair_fraction": max(
            row["low_rank_pair_fraction"] for row in scaling
        ),
        "stored_dense_matrix": False,
        "numerical_substrate": "project QJet FFT and scalar kernels; no NumPy",
        "complexity": "O(r N log N + L_local N log n_theta + P_near + N n_theta) under bounded atlas ranks/neighbors; geometry-dependent worst case",
    }
    write_csv(OUT / "shape_suite.csv", shapes)
    write_csv(OUT / "scaling.csv", scaling)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(OUT / "scaling.svg", scaling)
    write_report(OUT / "report.md", shapes, scaling, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
