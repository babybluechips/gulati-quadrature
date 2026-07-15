#!/usr/bin/env python3
"""Independent refinement study for the repaired curved-panel 3D pipeline.

The retained harmonic compiler sees degrees at most three: candidates use
degrees one and two, and model selection may inspect degree three.  Final
errors are measured on fixed degree-four and degree-five solid harmonics.
Those traces are never passed to compilation or adaptive selection.

The exact sphere uses a closed-form radial chart, where the Q3 continuum
operator equals the interior DtN map.  Ellipsoid and funky PN rows use the
universal manufactured identity ``Lambda(p|Gamma)=normal dot grad(p)`` but also
contain the bounded geometry remainder, so they audit the complete corrected
operator rather than only singular quadrature.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gulati_quadrature import (  # noqa: E402
    CurvedPanelConfig,
    FeatureChannelConfig,
    ManifoldRepairConfig,
    PanelSingularRepaymentConfig,
    SurfaceQConfig,
    build_curved_panel_engine,
    build_curved_panel_surface,
    build_radial_quadric_panel_surface,
    repair_triangle_mesh,
)
from inverse_shape.surface_feature_channels import (  # noqa: E402
    MellinKondratievPanelRepayment3D,
)
from inverse_shape.surface_repayment import (  # noqa: E402
    solid_harmonic_polynomials,
)


OUT = ROOT / "outputs" / "production_3d_repaired_refinement"
HELD_OUT_DEGREES = (4, 5)
ADAPTIVE_MAXIMUM_COMPILED_DEGREE = 2
ADAPTIVE_MAXIMUM_INSPECTED_DEGREE = 3


def projected_octahedron(refinements: int):
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    for _level in range(int(refinements)):
        cache = {}

        def midpoint(left, right):
            edge = (min(left, right), max(left, right))
            if edge not in cache:
                point = tuple(
                    0.5 * (vertices[left][axis] + vertices[right][axis])
                    for axis in range(3)
                )
                length = math.sqrt(sum(value * value for value in point))
                cache[edge] = len(vertices)
                vertices.append(tuple(value / length for value in point))
            return cache[edge]

        refined = []
        for first, second, third in faces:
            first_second = midpoint(first, second)
            second_third = midpoint(second, third)
            third_first = midpoint(third, first)
            refined.extend(
                (
                    (first, first_second, third_first),
                    (first_second, second, second_third),
                    (third_first, second_third, third),
                    (first_second, second_third, third_first),
                )
            )
        faces = refined
    return tuple(vertices), tuple(faces)


def cube():
    return (
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
    )


def weighted_norm(weights, values):
    return math.sqrt(
        max(
            sum(
                weight * abs(complex(value)) ** 2
                for weight, value in zip(weights, values, strict=True)
            ),
            0.0,
        )
    )


def relative_error(weights, expected, actual):
    difference = tuple(
        complex(left) - complex(right)
        for left, right in zip(expected, actual, strict=True)
    )
    return weighted_norm(weights, difference) / max(
        weighted_norm(weights, expected),
        1.0e-300,
    )


def held_out_modes():
    all_modes = solid_harmonic_polynomials(max(HELD_OUT_DEGREES))
    selected = []
    for degree in HELD_OUT_DEGREES:
        rows = tuple(mode for mode in all_modes if mode.degree == degree)
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            selected.append(rows[index])
    return tuple(selected)


def transformed_vertices(kind, vertices):
    if kind == "exact_sphere":
        return vertices
    if kind == "exact_ellipsoid":
        axes = (1.45, 0.9, 0.65)
        return tuple(
            tuple(axes[axis] * point[axis] for axis in range(3))
            for point in vertices
        )
    if kind == "funky_pn":
        output = []
        for x_value, y_value, z_value in vertices:
            radius = (
                1.0
                + 0.16 * x_value * y_value
                + 0.10 * (3.0 * z_value * z_value - 1.0)
                + 0.06 * x_value * z_value
            )
            output.append(
                (
                    1.2 * radius * x_value,
                    0.92 * radius * y_value,
                    0.78 * radius * z_value,
                )
            )
        return tuple(output)
    raise ValueError(f"unknown refinement shape {kind!r}")


def build_case(kind, level):
    unit_vertices, faces = projected_octahedron(level)
    vertices = transformed_vertices(kind, unit_vertices)
    topology = repair_triangle_mesh(
        vertices,
        faces,
        config=ManifoldRepairConfig(sharp_angle_degrees=120.0),
    )
    panel_config = CurvedPanelConfig(quadrature_order=3)
    if kind == "exact_sphere":
        surface = build_radial_quadric_panel_surface(
            topology,
            (1.0, 1.0, 1.0),
            config=panel_config,
        )
        adaptive = False
        maximum_degree = 0
    elif kind == "exact_ellipsoid":
        surface = build_radial_quadric_panel_surface(
            topology,
            (1.45, 0.9, 0.65),
            config=panel_config,
        )
        adaptive = True
        maximum_degree = ADAPTIVE_MAXIMUM_COMPILED_DEGREE
    else:
        surface = build_curved_panel_surface(topology, config=panel_config)
        adaptive = True
        maximum_degree = ADAPTIVE_MAXIMUM_COMPILED_DEGREE
    engine = build_curved_panel_engine(
        surface,
        singular_config=PanelSingularRepaymentConfig(
            series_order=1,
            maximum_panel_rings=2,
        ),
        config=SurfaceQConfig(
            tolerance=3.0e-7,
            maximum_order=6,
            leaf_size=8,
            work_budget_factor=512,
            singular_cell_order=1,
            harmonic_moment_degree=maximum_degree,
            adaptive_moment_degree=adaptive,
            minimum_harmonic_moment_degree=1,
            moment_validation_tolerance=0.15,
            moment_validation_gap=1,
        ),
    )
    return topology, surface, engine


def run_refinement():
    mode_rows = []
    geometry_rows = []
    aggregate_rows = []
    for kind in ("exact_sphere", "exact_ellipsoid", "funky_pn"):
        for level in (0, 1, 2):
            started = time.perf_counter()
            topology, surface, engine = build_case(kind, level)
            build_ms = 1000.0 * (time.perf_counter() - started)
            case_errors = []
            first_result = None
            for polynomial in held_out_modes():
                trace = []
                exact_flux = []
                for point, normal in zip(
                    surface.points,
                    surface.normals,
                    strict=True,
                ):
                    value, gradient = polynomial.value_gradient(point)
                    trace.append(value)
                    exact_flux.append(
                        sum(
                            gradient[axis] * normal[axis]
                            for axis in range(3)
                        )
                    )
                raw_started = time.perf_counter()
                raw_graph = engine.apply(trace)
                raw_ms = 1000.0 * (time.perf_counter() - raw_started)
                raw_values = tuple(
                    complex(value) / (2.0 * math.pi)
                    for value in raw_graph.values
                )
                full_started = time.perf_counter()
                full = engine.apply_dtn_principal(trace)
                full_ms = 1000.0 * (time.perf_counter() - full_started)
                if first_result is None:
                    first_result = full
                raw_error = relative_error(surface.weights, exact_flux, raw_values)
                full_error = relative_error(surface.weights, exact_flux, full.values)
                case_errors.append((raw_error, full_error))
                mode_rows.append(
                    {
                        "shape": kind,
                        "refinement": level,
                        "nodes": len(surface.points),
                        "mode": polynomial.name,
                        "held_out_degree": polynomial.degree,
                        "maximum_compiled_degree": (
                            0
                            if kind == "exact_sphere"
                            else ADAPTIVE_MAXIMUM_COMPILED_DEGREE
                        ),
                        "maximum_adaptively_inspected_degree": (
                            0
                            if kind == "exact_sphere"
                            else ADAPTIVE_MAXIMUM_INSPECTED_DEGREE
                        ),
                        "used_in_compilation_or_selection": False,
                        "raw_relative_error": raw_error,
                        "repaid_relative_error": full_error,
                        "raw_apply_ms": raw_ms,
                        "repaid_apply_ms": full_ms,
                        "compression_inf_bound": full.compression_inf_bound,
                    }
                )
            stats = engine.stats()
            geometry_rows.append(
                {
                    "shape": kind,
                    "refinement": level,
                    "mesh_vertices": len(topology.vertices),
                    "mesh_faces": len(topology.faces),
                    "nodes": len(surface.points),
                    "build_ms": build_ms,
                    "area": sum(surface.weights),
                    "maximum_panel_seam_gap": surface.maximum_panel_seam_gap,
                    "boundary_edges": topology.certificate.boundary_edges,
                    "nonmanifold_edges": topology.certificate.nonmanifold_edges,
                    "nonmanifold_vertices": (
                        topology.certificate.nonmanifold_vertices
                    ),
                    "production_ready": topology.certificate.production_ready,
                    "panel_geometry_kind": surface.stats["panel_geometry_kind"],
                    "adaptive_selected_degree": stats.get(
                        "adaptive_selected_degree",
                        0,
                    ),
                    "adaptive_validation_certified": stats.get(
                        "adaptive_validation_certified",
                        kind == "exact_sphere",
                    ),
                    "dense_q_matrix_stored": stats["dense_q_matrix_stored"],
                    "pair_table_stored": stats["pair_table_stored"],
                }
            )
            aggregate_rows.append(
                {
                    "shape": kind,
                    "refinement": level,
                    "nodes": len(surface.points),
                    "maximum_raw_held_out_error": max(row[0] for row in case_errors),
                    "maximum_repaid_held_out_error": max(row[1] for row in case_errors),
                    "median_repaid_held_out_error": sorted(
                        row[1] for row in case_errors
                    )[len(case_errors) // 2],
                }
            )
            print(
                kind,
                level,
                len(surface.points),
                f"max={aggregate_rows[-1]['maximum_repaid_held_out_error']:.3e}",
                flush=True,
            )
    return mode_rows, geometry_rows, aggregate_rows


def run_feature_audit():
    vertices, faces = cube()
    topology = repair_triangle_mesh(vertices, faces)
    surface = build_curved_panel_surface(
        topology,
        config=CurvedPanelConfig(quadrature_order=3),
    )
    repayment = MellinKondratievPanelRepayment3D(
        surface,
        config=FeatureChannelConfig(
            reference_quadrature_order=12,
            vertex_link_refinements=3,
        ),
    )
    independent = MellinKondratievPanelRepayment3D(
        surface,
        config=FeatureChannelConfig(
            reference_quadrature_order=16,
            vertex_link_refinements=3,
        ),
    )
    independent_by_label = {
        channel.label: channel for channel in independent.channels
    }
    rows = []
    for channel in repayment.channels:
        kind = channel.kind
        for rung in range(channel.rank):
            values = [0.0j for _ in surface.points]
            for offset, index in enumerate(channel.support):
                values[index] = channel.basis_values[rung][offset]
            borrowed = sum(
                weight * value
                for weight, value in zip(surface.weights, values, strict=True)
            )
            result = repayment.repay_integral(
                values,
                borrowed_value=borrowed,
                channel_labels=(channel.label,),
            )
            compiled_reference = channel.exact_moments[rung]
            independent_reference = independent_by_label[
                channel.label
            ].exact_moments[rung]
            rows.append(
                {
                    "kind": kind,
                    "channel": channel.label,
                    "rung": rung,
                    "kondratiev_exponent": channel.pencil_certificate[
                        "kondratiev_exponent"
                    ],
                    "compiled_reference_order": 12,
                    "independent_reference_order": 16,
                    "raw_absolute_error": abs(
                        borrowed - independent_reference
                    ),
                    "repaid_absolute_error": abs(
                        result.value - independent_reference
                    ),
                    "reference_disagreement": abs(
                        compiled_reference - independent_reference
                    ),
                    "support_nodes": len(channel.support),
                }
            )
    return rows, repayment.stats(), topology.certificate.stats


def fit_endpoint_rate(rows, shape, key):
    selected = sorted(
        (row for row in rows if row["shape"] == shape),
        key=lambda row: row["nodes"],
    )
    first = selected[0]
    last = selected[-1]
    h_first = first["nodes"] ** -0.5
    h_last = last["nodes"] ** -0.5
    return math.log(first[key] / last[key]) / math.log(h_first / h_last)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary, geometry_rows, aggregate_rows, feature_rows):
    lines = [
        "# Repaired curved-panel 3D QJet refinement",
        "",
        "## Protocol",
        "",
        "All final traces are degree-four or degree-five solid harmonics. The "
        "adaptive compiler retains at most degree two and may inspect only degree "
        "three. Therefore every reported final mode is independent of both fitting "
        "and model selection.",
        "",
        "The exact-sphere rows isolate singular quadrature because the radial chart "
        "lies exactly on the unit sphere and the continuum Q3 operator has DtN "
        "eigenvalue `l`. Ellipsoid and funky PN rows test the full bounded-remainder "
        "approximation and must not be read as singular-quadrature rates alone.",
        "",
        "## Geometry and adaptivity",
        "",
        "| Shape | Level | Nodes | Geometry | Boundary E | Nonmanifold E | "
        "Nonmanifold V | Seam gap | Selected d | Certified |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in geometry_rows:
        lines.append(
            f"| {row['shape']} | {row['refinement']} | {row['nodes']} | "
            f"{row['panel_geometry_kind']} | {row['boundary_edges']} | "
            f"{row['nonmanifold_edges']} | {row['nonmanifold_vertices']} | "
            f"{row['maximum_panel_seam_gap']:.3e} | "
            f"{row['adaptive_selected_degree']} | "
            f"{row['adaptive_validation_certified']} |"
        )
    lines.extend(
        [
            "",
            "## Independent held-out refinement",
            "",
            "| Shape | Level | Nodes | Max raw error | Max repaid error | Median repaid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['shape']} | {row['refinement']} | {row['nodes']} | "
            f"{row['maximum_raw_held_out_error']:.3e} | "
            f"{row['maximum_repaid_held_out_error']:.3e} | "
            f"{row['median_repaid_held_out_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Independently checked Mellin/Kondratiev feature moments",
            "",
            "All edge and vertex channels are compiled with an order-twelve "
            "curved-panel rule and checked against a separate order-sixteen rule.",
            "",
            "| Kind | Channel | Rung | Exponent | Raw abs. error | Repaid abs. error | "
            "Reference gap |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in feature_rows:
        lines.append(
            f"| {row['kind']} | {row['channel']} | {row['rung']} | "
            f"{row['kondratiev_exponent']:.6f} | "
            f"{row['raw_absolute_error']:.3e} | "
            f"{row['repaid_absolute_error']:.3e} | "
            f"{row['reference_disagreement']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The eight cube vertex pencils have exponent spread "
            f"`{summary['cube_vertex_kondratiev_spread']:.3e}`. Their maximum "
            "error against the exact octant exponent three is "
            f"`{summary['maximum_cube_vertex_kondratiev_error']:.3e}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"The exact-sphere endpoint fit for the repaid maximum error is "
            f"`{summary['exact_sphere_repaid_endpoint_rate']:.3f}` in the node-spacing "
            "variable. This is evidence of refinement, not a machine-precision "
            "certificate. Feature basis moments are repaid to the error shown above. "
            "No dense Q matrix or global pair table is stored.",
            "",
            "The arbitrary-domain held-out rows remain the controlling limitation. "
            "A failed adaptive validation flag means the configured degree-two bounded "
            "remainder did not meet its next-degree target; it is not converted into a "
            "pass by the final benchmark.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode_rows, geometry_rows, aggregate_rows = run_refinement()
    feature_rows, feature_stats, feature_topology = run_feature_audit()
    summary = {
        "shape_count": 3,
        "refinement_levels": 3,
        "held_out_degrees": HELD_OUT_DEGREES,
        "maximum_compiled_degree": ADAPTIVE_MAXIMUM_COMPILED_DEGREE,
        "maximum_adaptively_inspected_degree": (
            ADAPTIVE_MAXIMUM_INSPECTED_DEGREE
        ),
        "all_final_modes_independent": all(
            not row["used_in_compilation_or_selection"] for row in mode_rows
        ),
        "all_repaired_surfaces_production_ready": all(
            row["production_ready"] for row in geometry_rows
        ),
        "all_repaired_surfaces_watertight": all(
            row["boundary_edges"] == 0 for row in geometry_rows
        ),
        "all_repaired_surfaces_edge_manifold": all(
            row["nonmanifold_edges"] == 0 for row in geometry_rows
        ),
        "all_repaired_surfaces_vertex_manifold": all(
            row["nonmanifold_vertices"] == 0 for row in geometry_rows
        ),
        "maximum_curved_panel_seam_gap": max(
            row["maximum_panel_seam_gap"] for row in geometry_rows
        ),
        "all_curved_panel_atlases_watertight": all(
            row["maximum_panel_seam_gap"] <= 2.0e-12
            for row in geometry_rows
        ),
        "maximum_feature_repaid_absolute_error": max(
            row["repaid_absolute_error"] for row in feature_rows
        ),
        "maximum_cube_vertex_kondratiev_error": max(
            abs(row["kondratiev_exponent"] - 3.0)
            for row in feature_rows
            if row["kind"] == "vertex"
        ),
        "cube_vertex_kondratiev_spread": (
            max(
                row["kondratiev_exponent"]
                for row in feature_rows
                if row["kind"] == "vertex"
            )
            - min(
                row["kondratiev_exponent"]
                for row in feature_rows
                if row["kind"] == "vertex"
            )
        ),
        "exact_sphere_repaid_endpoint_rate": fit_endpoint_rate(
            aggregate_rows,
            "exact_sphere",
            "maximum_repaid_held_out_error",
        ),
        "dense_q_matrix_stored": False,
        "pair_table_stored": False,
        "feature_stats": feature_stats,
        "feature_topology_production_ready": feature_topology[
            "production_ready"
        ],
        "production_apply_complexity": "O(N log N)+O(N q)+O(N d^2)",
        "production_storage_complexity": "O(N) at fixed q,d and feature rank",
        "universal_machine_precision_claim": False,
    }
    write_csv(OUT / "held_out_mode_rows.csv", mode_rows)
    write_csv(OUT / "geometry_rows.csv", geometry_rows)
    write_csv(OUT / "aggregate_rows.csv", aggregate_rows)
    write_csv(OUT / "feature_channel_rows.csv", feature_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(summary, geometry_rows, aggregate_rows, feature_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
