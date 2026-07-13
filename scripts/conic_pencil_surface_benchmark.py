#!/usr/bin/env python3
# ruff: noqa: E501
"""Benchmark the conic-pencil surface QJet without dense matrices."""

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
from inverse_shape.quadrature import PI, TAU, _abs, _log, _sin, _sqrt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conic_pencil_surface_qjet"


def relative_error(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def fitted_exponent(rows, x_key, y_key):
    filtered = [row for row in rows if row[y_key] > 0.0]
    x_values = [_log(float(row[x_key])) for row in filtered]
    y_values = [_log(float(row[y_key])) for row in filtered]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    return numerator / denominator


def shape_factory(name, n_slices, n_theta):
    if name == "circular_cylinder":
        return straight_conic_tube_qjet(3.0, 0.8, 0.8, n_slices, n_theta)
    if name == "elliptic_taper":
        return tapered_conic_tube_qjet(
            3.2,
            0.35,
            0.95,
            0.22,
            0.62,
            n_slices,
            n_theta,
        )
    if name == "bent_tube":
        return bent_conic_tube_qjet(
            3.0,
            1.35,
            0.48,
            0.32,
            n_slices,
            n_theta,
        )
    if name == "twisted_ellipse":
        return twisted_ellipse_tube_qjet(
            3.2,
            0.82,
            0.31,
            1.8,
            n_slices,
            n_theta,
        )
    if name == "toroidal_bundle":
        return toroidal_conic_bundle_qjet(
            2.0,
            0.42,
            0.24,
            n_slices,
            n_theta,
        )
    if name == "aircraft_body":
        return aircraft_conic_bundle_qjet(4.2, n_slices, n_theta)
    raise ValueError(f"unknown shape: {name}")


SHAPES = (
    "circular_cylinder",
    "elliptic_taper",
    "bent_tube",
    "twisted_ellipse",
    "toroidal_bundle",
    "aircraft_body",
)


def physical_field(qjet):
    nodes = qjet.generate_nodes()
    return tuple(
        point[0] + 0.2 * point[1] - 0.15 * point[2] * point[2] + 0.05 * point[0] * point[2]
        for point in nodes.points
    )


def benchmark_shapes():
    rows = []
    certificate_rows = []
    for shape in SHAPES:
        for n_slices, n_theta in ((8, 16), (16, 24), (32, 32)):
            qjet = shape_factory(shape, n_slices, n_theta)
            nodes = qjet.generate_nodes()
            values = physical_field(qjet)
            certificates = qjet.pencil_certificates()
            for edge, certificate in enumerate(certificates):
                certificate_rows.append(
                    {
                        "shape": shape,
                        "n_slices": n_slices,
                        "edge": edge,
                        "determinant": certificate["determinant"],
                        "minus_second_log_det": certificate["minus_second_log_det"],
                        "degenerate": certificate["degenerate"],
                    }
                )

            results = {}
            for kernel_power, label in ((2.0, "inverse_square"), (3.0, "inverse_cube")):
                start = perf_counter()
                if kernel_power == 2.0:
                    direct = qjet.apply_inverse_square_metric(values, method="direct")
                else:
                    direct = qjet.apply_dtn_principal(values, method="direct")
                direct_ms = 1000.0 * (perf_counter() - start)
                start = perf_counter()
                if kernel_power == 2.0:
                    tree = qjet.apply_inverse_square_metric(
                        values,
                        method="tree",
                        opening=0.30,
                        leaf_size=8,
                    )
                else:
                    tree = qjet.apply_dtn_principal(
                        values,
                        method="tree",
                        opening=0.30,
                        leaf_size=8,
                    )
                tree_ms = 1000.0 * (perf_counter() - start)
                results[label] = {
                    "relative_error": relative_error(direct, tree),
                    "direct_ms": direct_ms,
                    "tree_ms": tree_ms,
                    "speedup": direct_ms / max(tree_ms, 1.0e-12),
                    "tree_direct_pairs": qjet.last_apply_stats["direct_pairs"],
                    "tree_accepted_blocks": qjet.last_apply_stats["accepted_blocks"],
                }
            constant = qjet.apply(
                (1.0,) * qjet.n_nodes,
                kernel_power=2.0,
                method="tree",
                opening=0.30,
                leaf_size=8,
            )
            stats = qjet.stats()
            rows.append(
                {
                    "shape": shape,
                    "n_slices": n_slices,
                    "n_theta": n_theta,
                    "nodes": qjet.n_nodes,
                    "surface_area": sum(nodes.weights),
                    "inverse_square_relative_error": results["inverse_square"]["relative_error"],
                    "inverse_cube_relative_error": results["inverse_cube"]["relative_error"],
                    "inverse_square_direct_ms": results["inverse_square"]["direct_ms"],
                    "inverse_square_tree_ms": results["inverse_square"]["tree_ms"],
                    "inverse_square_speedup": results["inverse_square"]["speedup"],
                    "inverse_cube_direct_ms": results["inverse_cube"]["direct_ms"],
                    "inverse_cube_tree_ms": results["inverse_cube"]["tree_ms"],
                    "inverse_cube_speedup": results["inverse_cube"]["speedup"],
                    "tree_direct_pairs": results["inverse_square"]["tree_direct_pairs"],
                    "tree_accepted_blocks": results["inverse_square"]["tree_accepted_blocks"],
                    "constant_residual": max(_abs(complex(value)) for value in constant),
                    "stored_geometry_scalars": stats["stored_geometry_scalars"],
                    "generated_entries": stats["generated_surface_entries_per_apply"],
                    "dense_entries_avoided": qjet.n_nodes * qjet.n_nodes,
                    "stored_dense_matrix": False,
                }
            )
    return rows, certificate_rows


def benchmark_scaling():
    rows = []
    for n_slices in (8, 12, 16, 24, 32, 48, 64, 96):
        qjet = aircraft_conic_bundle_qjet(4.2, n_slices, 24)
        values = physical_field(qjet)
        start = perf_counter()
        tree = qjet.apply(
            values,
            kernel_power=2.0,
            method="tree",
            opening=0.30,
            leaf_size=8,
        )
        tree_ms = 1000.0 * (perf_counter() - start)
        direct_ms = 0.0
        direct_error = 0.0
        if qjet.n_nodes <= 768:
            start = perf_counter()
            direct = qjet.apply(
                values,
                kernel_power=2.0,
                method="direct",
            )
            direct_ms = 1000.0 * (perf_counter() - start)
            direct_error = relative_error(direct, tree)
        rows.append(
            {
                "n_slices": n_slices,
                "n_theta": qjet.n_theta,
                "nodes": qjet.n_nodes,
                "tree_ms": tree_ms,
                "direct_ms": direct_ms,
                "tree_relative_error": direct_error,
                "stored_dense_matrix": False,
            }
        )
    return rows


def benchmark_warp_response():
    qjet = bent_conic_tube_qjet(3.0, 1.2, 0.48, 0.32, 8, 16)
    load = tuple(
        tuple(
            (
                0.025 * _sin(PI * index / (qjet.n_slices - 1))
                if parameter == 0
                else -0.012 * _sin(PI * index / (qjet.n_slices - 1))
                if parameter == 4
                else 0.006 * _sin(TAU * index / (qjet.n_slices - 1))
                if parameter == 7
                else 0.0
            )
            for parameter in range(qjet.parameter_count_per_slice)
        )
        for index in range(qjet.n_slices)
    )
    start = perf_counter()
    response, solver = qjet.solve_shape_load(
        load,
        method="direct",
        ridge=0.12,
        shell_smoothness=0.06,
        iterations=40,
        tolerance=1.0e-9,
    )
    solve_ms = 1000.0 * (perf_counter() - start)
    deformed = qjet.deformed(response, step=0.3)
    initial_nodes = qjet.generate_nodes()
    final_nodes = deformed.generate_nodes()
    max_node_motion = max(
        _sqrt(
            sum(
                (final_nodes.points[index][axis] - initial_nodes.points[index][axis]) ** 2
                for axis in range(3)
            )
        )
        for index in range(qjet.n_nodes)
    )
    return (
        qjet,
        deformed,
        response,
        {
            "nodes": qjet.n_nodes,
            "parameters": qjet.parameter_count,
            "solve_ms": solve_ms,
            "iterations": solver["iterations"],
            "relative_residual": solver["relative_residual"],
            "max_center_response": max(abs(row[0]) for row in response),
            "max_log_axis_response": max(abs(row[4]) for row in response),
            "max_rotation_response": max(abs(row[7]) for row in response),
            "max_node_motion_at_step_0_3": max_node_motion,
            "initial_area": sum(initial_nodes.weights),
            "deformed_area": sum(final_nodes.weights),
            "stored_dense_reduced_hessian": False,
        },
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project(point, scale, offset_x, offset_y):
    horizontal = point[0] + 0.35 * point[1]
    vertical = point[2] + 0.16 * point[1]
    return offset_x + scale * horizontal, offset_y - scale * vertical


def svg_polyline(points, stroke, width, opacity=1.0):
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{data}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" opacity="{opacity}"/>'
    )


def surface_lines(qjet, scale, offset_x, offset_y, stroke, width, opacity):
    nodes = qjet.generate_nodes()
    lines = []
    for slice_index in range(qjet.n_slices):
        ring = [
            project(
                nodes.points[slice_index * qjet.n_theta + phase],
                scale,
                offset_x,
                offset_y,
            )
            for phase in range(qjet.n_theta)
        ]
        ring.append(ring[0])
        lines.append(svg_polyline(ring, stroke, width, opacity))
    phase_stride = max(1, qjet.n_theta // 8)
    for phase in range(0, qjet.n_theta, phase_stride):
        path = [
            project(
                nodes.points[slice_index * qjet.n_theta + phase],
                scale,
                offset_x,
                offset_y,
            )
            for slice_index in range(qjet.n_slices)
        ]
        if qjet.periodic:
            path.append(path[0])
        lines.append(svg_polyline(path, stroke, width, opacity))
    return lines


def write_warp_svg(path, initial, deformed):
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520">',
        '<rect width="1200" height="520" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#111;letter-spacing:0} .small{font-size:14px} .title{font-size:20px;font-weight:600}</style>",
        '<text x="40" y="40" class="title">Conic-pencil surface response under the inverse-square reduced Hessian</text>',
        '<text x="260" y="82" class="small" text-anchor="middle">initial conic bundle</text>',
        '<text x="900" y="82" class="small" text-anchor="middle">J* Q J response: bend + pinch + twist</text>',
    ]
    lines.extend(surface_lines(initial, 105.0, 260.0, 300.0, "#9a9a9a", 1.1, 0.9))
    lines.extend(surface_lines(deformed, 105.0, 900.0, 300.0, "#111111", 1.25, 1.0))
    lines.extend(
        [
            '<line x1="520" y1="285" x2="675" y2="285" stroke="#111" stroke-width="1.4"/>',
            '<path d="M675 285 l-12 -6 l0 12 z" fill="#111"/>',
            '<text x="597" y="268" class="small" text-anchor="middle">matrix-free solve</text>',
            '<text x="600" y="480" class="small" text-anchor="middle">stored state: conic value/3-jets and SU(2) rotors; no distance matrix or reduced Hessian</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact(value):
    return f"{float(value):.3e}"


def summarize(rows, scaling_rows, certificate_rows, warp):
    finest = [row for row in rows if row["n_slices"] == 32]
    direct_scaling = [row for row in scaling_rows if row["direct_ms"] > 0.0]
    return {
        "shape_count": len(SHAPES),
        "quadrature_cases": len(rows),
        "max_inverse_square_tree_error": max(row["inverse_square_relative_error"] for row in rows),
        "max_inverse_cube_tree_error": max(row["inverse_cube_relative_error"] for row in rows),
        "max_constant_residual": max(row["constant_residual"] for row in rows),
        "finest_inverse_square_speedup_min": min(row["inverse_square_speedup"] for row in finest),
        "finest_inverse_cube_speedup_min": min(row["inverse_cube_speedup"] for row in finest),
        "fitted_tree_runtime_exponent": fitted_exponent(
            scaling_rows,
            "nodes",
            "tree_ms",
        ),
        "fitted_direct_runtime_exponent": fitted_exponent(
            direct_scaling,
            "nodes",
            "direct_ms",
        ),
        "minimum_pencil_determinant": min(abs(row["determinant"]) for row in certificate_rows),
        "maximum_pencil_curvature": max(
            abs(row["minus_second_log_det"]) for row in certificate_rows
        ),
        "warp_response": warp,
        "stored_dense_matrix": False,
        "continuum_tangent_cell_repayment": False,
        "global_dtn_geometry_correction": False,
        "representation": "36 conic value/three-jet scalars per slice plus field vectors",
        "complexity_scope": (
            "expected O(N log N) at fixed accuracy and bounded geometry; "
            "near-contact can degrade toward O(N^2)"
        ),
    }


def write_report(path, summary, rows, warp):
    finest = [row for row in rows if row["n_slices"] == 32]
    lines = [
        "# Conic-pencil surface QJet benchmark",
        "",
        "## Construction",
        "",
        "The retained geometry is a bundle of moving conics",
        "",
        "```text",
        "X(u,theta) = c(u) + a(u) cos(theta) e1(u) + b(u) sin(theta) e2(u).",
        "```",
        "",
        "Each slice stores value/three-jets of the center, SU(2) frame rotor, and two log axes: 36 scalars. Surface nodes and area weights are generated only during an apply; meridional tangents are differentiated directly from those jets. No dense distance matrix, surface operator, or reduced shape Hessian is retained.",
        "",
        "The inverse-square operator supplies the shape metric. Shape parameters are lowered by",
        "",
        "```text",
        "delta p -> J delta p -> Q_inverse_square(J delta p) -> J* Q J delta p.",
        "```",
        "",
        "The normalized `(2*pi)^-1 |X-Y|^-3` action is reported separately as the discretized off-diagonal three-dimensional DtN principal channel.",
        "",
        "## Headline checks",
        "",
        f"- shapes: `{summary['shape_count']}`; refinement cases: `{summary['quadrature_cases']}`",
        f"- maximum inverse-square tree/direct error: `{compact(summary['max_inverse_square_tree_error'])}`",
        f"- maximum inverse-cube tree/direct error: `{compact(summary['max_inverse_cube_tree_error'])}`",
        f"- maximum constant residual: `{compact(summary['max_constant_residual'])}`",
        f"- fitted tree runtime exponent: `{summary['fitted_tree_runtime_exponent']:.3f}`",
        f"- fitted streamed-direct runtime exponent: `{summary['fitted_direct_runtime_exponent']:.3f}`",
        f"- minimum conic-pencil determinant magnitude: `{compact(summary['minimum_pencil_determinant'])}`",
        "- dense matrices stored: `no`",
        "",
        "## Finest tested grids",
        "",
        "| shape | nodes | Q2 rel. err. | Q3/(2pi) rel. err. | Q2 speedup | Q3 speedup | dense entries avoided |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in finest:
        lines.append(
            "| {shape} | {nodes} | `{q2}` | `{q3}` | `{s2:.2f}x` | `{s3:.2f}x` | `{dense}` |".format(
                shape=row["shape"],
                nodes=row["nodes"],
                q2=compact(row["inverse_square_relative_error"]),
                q3=compact(row["inverse_cube_relative_error"]),
                s2=row["inverse_square_speedup"],
                s3=row["inverse_cube_speedup"],
                dense=row["dense_entries_avoided"],
            )
        )
    lines.extend(
        [
            "",
            "## Shape response",
            "",
            f"A bend/pinch/twist load on `{warp['nodes']}` generated surface nodes and `{warp['parameters']}` conic parameters converged in `{warp['iterations']}` matrix-free CG iterations.",
            f"The relative residual was `{compact(warp['relative_residual'])}` and maximum generated node motion at step 0.3 was `{compact(warp['max_node_motion_at_step_0_3'])}`.",
            "",
            "The deformation is not a free unconstrained repulsion. It is the response of the conic parameter bundle under the positive reduced metric `J*QJ`; the ridge stabilizes its translation nullspace and shell smoothness controls slice-to-slice oscillation.",
            "",
            "## What the three source papers contribute",
            "",
            "- The cone paper supplies shell ordering and the warning that nearest-shell block tridiagonality belongs to a local stencil; its direct Schur sweep is not a fast all-pairs quadrature.",
            "- The discriminant paper supplies `-d_lambda^2 log det(A0+lambda A1)`, an O(1) inverse-square certificate for each 3x3 conic pencil. It detects chart degeneration but is not the surface distance matrix.",
            "- The SU(2) paper supplies frame transport and exact group convolution on genuine subgroup orbits. Here rotors transport local frames; generic bent conic bundles are not falsely declared group convolutions.",
            "",
            "## Complexity and limitation",
            "",
            "The quadrupole tree uses exact leaf interactions and generated far moments. At fixed accuracy and bounded reach it targets `O(N log N)` work and uses `O(N)` storage. The measured exponent in this Python campaign is reported above and is not relabeled as `O(N log N)`. Close sheets, cusps, or collapsing conic pencils grow the near list and can approach the streamed `O(N^2)` reference cost.",
            "",
            "The Q3 comparison audits only tree compression against the independent direct discretization. It is not a continuum DtN accuracy result: tangent-cell singular repayment and the lower-order geometry operator have not yet been added to this 3D path. Open tube examples are surface patches; the toroidal bundle is the closed-surface case.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, certificate_rows = benchmark_shapes()
    scaling_rows = benchmark_scaling()
    initial, deformed, response, warp = benchmark_warp_response()
    summary = summarize(rows, scaling_rows, certificate_rows, warp)
    write_csv(OUT / "surface_quadrature.csv", rows)
    write_csv(OUT / "pencil_certificates.csv", certificate_rows)
    write_csv(OUT / "tree_scaling.csv", scaling_rows)
    write_csv(OUT / "warp_response.csv", [warp])
    (OUT / "conic_pencil_surface_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(OUT / "conic_pencil_surface_report.md", summary, rows, warp)
    write_warp_svg(OUT / "conic_pencil_warp_response.svg", initial, deformed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
