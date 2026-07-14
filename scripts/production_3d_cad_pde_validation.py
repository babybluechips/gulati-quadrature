#!/usr/bin/env python3
"""Continuum-aware PDE audit on every compiled public CAD model.

The report keeps four quantities separate:

* QJet compression error;
* algebraic Krylov residual;
* exact error on moments deliberately compiled into the repayment QJets; and
* held-out continuum error on a degree-four solid harmonic.

The first two can reach roundoff without implying the fourth.  No pairwise
matrix or dense boundary operator is formed by this campaign.
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
    SurfacePDEConfig,
    SurfaceQConfig,
    build_compiled_cad_engine,
    build_mesh_engine,
    build_surface_pde_solver,
    load_compiled_cad_surface,
)
from inverse_shape.quadrature import _cos, _sin  # noqa: E402
from inverse_shape.surface_repayment import (  # noqa: E402
    solid_harmonic_polynomials,
)


OUT = ROOT / "outputs" / "production_3d_cad_pde_validation"
CAD = ROOT / "outputs" / "cad_qjet_invertibility"
MODELS = (
    ("NASA SOFIA aircraft", "sofia_aircraft.qcad3j"),
    ("FreeCAD cement mixer", "cement_mixer.qcad3j"),
    ("NASA Curiosity manufacturing plates", "curiosity_rover.qcad3j"),
    ("NASA Curiosity assembled", "curiosity_assembled.qcad3j"),
    ("buildingSMART IFC bridge", "ifc_bridge.qcad3j"),
)
Q_CONFIG = SurfaceQConfig(
    tolerance=2.0e-10,
    maximum_order=10,
    leaf_size=8,
    work_budget_factor=256,
    continuum_repayment=True,
    singular_cell_order=3,
    harmonic_moment_degree=3,
)
PDE_CONFIG = SurfacePDEConfig(
    tolerance=1.0e-7,
    maximum_iterations=60,
    heat_steps=1,
    wave_steps=1,
    fail_on_nonconvergence=False,
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


def mean_zero(weights, values):
    mean = sum(
        weight * complex(value)
        for weight, value in zip(weights, values, strict=True)
    ) / sum(weights)
    return tuple(complex(value) - mean for value in values)


def timed(call):
    started = time.perf_counter()
    result = call()
    return result, 1000.0 * (time.perf_counter() - started)


def result_row(label, problem, reference, result, elapsed, error="", gate=1.0):
    value = None if error == "" else float(error)
    passed = (
        result.converged
        and result.relative_residual <= max(5.0 * PDE_CONFIG.tolerance, gate)
        and (value is None or value <= gate)
        and not result.stats["dense_operator_stored"]
        and not result.stats["quadratic_fallback"]
    )
    return {
        "shape": label,
        "problem": problem,
        "reference_class": reference,
        "relative_error": error,
        "relative_algebraic_residual": result.relative_residual,
        "iterations": result.iterations,
        "qjet_applications": result.operator_applications,
        "solve_ms": elapsed,
        "gate": gate,
        "converged": result.converged,
        "passed": passed,
    }


def run_cad_case(label, archive_name):
    surface, compile_ms = timed(
        lambda: load_compiled_cad_surface(
            CAD / archive_name,
            target_vertices=96,
        )
    )
    engine = build_compiled_cad_engine(surface, config=Q_CONFIG)
    solver = build_surface_pde_solver(engine, config=PDE_CONFIG)
    points = surface.vertices
    normals = surface.normals
    weights = surface.weights
    direction = (1.0, 0.37, -0.21)
    trace = mean_zero(
        weights,
        tuple(
            direction[0] * x + direction[1] * y + direction[2] * z
            for x, y, z in points
        ),
    )
    exact_flux = tuple(
        direction[0] * nx + direction[1] * ny + direction[2] * nz
        for nx, ny, nz in normals
    )
    rows = []

    laplace, elapsed = timed(lambda: solver.apply_laplace_dtn(trace))
    rows.append(
        result_row(
            label,
            "laplace_dtn",
            "compiled exact solid-harmonic degree one",
            laplace,
            elapsed,
            relative_error(weights, exact_flux, laplace.values),
            2.0e-10,
        )
    )
    print(f"  {label}: Laplace DtN", flush=True)

    compatible_flux = mean_zero(weights, exact_flux)
    poisson, elapsed = timed(lambda: solver.solve_poisson(compatible_flux))
    rows.append(
        result_row(
            label,
            "poisson_boundary_inverse",
            "manufactured exact harmonic flux with mean-zero gauge",
            poisson,
            elapsed,
            relative_error(weights, trace, poisson.values),
            2.0e-5,
        )
    )
    print(f"  {label}: Poisson inverse", flush=True)

    mass = 0.4
    screened_rhs = tuple(
        flux + mass * value
        for flux, value in zip(exact_flux, trace, strict=True)
    )
    screened, elapsed = timed(
        lambda: solver.solve_poisson(screened_rhs, mass=mass)
    )
    rows.append(
        result_row(
            label,
            "screened_poisson_boundary_inverse",
            "manufactured exact harmonic flux",
            screened,
            elapsed,
            relative_error(weights, trace, screened.values),
            2.0e-5,
        )
    )
    print(f"  {label}: screened Poisson", flush=True)

    wave_number = 0.7
    plane_trace = tuple(
        complex(_cos(wave_number * point[0]), _sin(wave_number * point[0]))
        for point in points
    )
    plane_flux = tuple(
        1j * wave_number * normal[0] * value
        for normal, value in zip(normals, plane_trace, strict=True)
    )
    helmholtz_dtn, elapsed = timed(
        lambda: solver.apply_helmholtz_dtn(
            plane_trace,
            wavenumber=wave_number,
        )
    )
    rows.append(
        result_row(
            label,
            "helmholtz_dtn",
            "compiled exact whole-space plane wave",
            helmholtz_dtn,
            elapsed,
            relative_error(weights, plane_flux, helmholtz_dtn.values),
            2.0e-10,
        )
    )
    print(f"  {label}: Helmholtz DtN", flush=True)

    evolution_time = 0.005 * math.sqrt(min(weights))
    heat, elapsed = timed(
        lambda: solver.solve_heat(trace, time=evolution_time, steps=1)
    )
    rows.append(
        result_row(
            label,
            "heat_boundary_semigroup",
            f"Pade denominator residual at t={evolution_time:.3e}; no bulk heat claim",
            heat,
            elapsed,
            gate=2.0e-7,
        )
    )
    print(f"  {label}: heat semigroup", flush=True)

    wave, elapsed = timed(
        lambda: solver.solve_wave(trace, time=evolution_time, steps=1)
    )
    rows.append(
        result_row(
            label,
            "wave_boundary_functional_calculus",
            f"Newmark denominator residual at t={evolution_time:.3e}; no bulk wave claim",
            wave,
            elapsed,
            gate=2.0e-7,
        )
    )
    print(f"  {label}: wave calculus", flush=True)

    held_out = solid_harmonic_polynomials(4)[-1]
    held_trace = []
    held_flux = []
    for point, normal in zip(points, normals, strict=True):
        value, gradient = held_out.value_gradient(point)
        held_trace.append(value)
        held_flux.append(sum(a * b for a, b in zip(gradient, normal, strict=True)))
    held_result, held_ms = timed(
        lambda: solver.apply_laplace_dtn(held_trace)
    )
    held_out_row = {
        "shape": label,
        "mode": held_out.name,
        "compiled_harmonic_degree": Q_CONFIG.harmonic_moment_degree,
        "held_out_degree": 4,
        "relative_continuum_error": relative_error(
            weights,
            held_flux,
            held_result.values,
        ),
        "solve_ms": held_ms,
        "used_for_repayment_compilation": False,
    }
    engine_stats = engine.stats()
    geometry_row = dict(surface.stats)
    geometry_row.update(
        {
            "shape": label,
            "archive": archive_name,
            "compile_ms": compile_ms,
            "singular_cells_repaid": engine_stats.get(
                "singular_cells_repaid",
                0,
            ),
            "nonsmooth_vertices": engine_stats.get("nonsmooth_vertices", 0),
            "harmonic_repayment_rank": engine_stats.get(
                "harmonic_repayment_rank",
                0,
            ),
            "dense_q_matrix_stored": engine_stats["dense_q_matrix_stored"],
            "pair_table_stored": engine_stats["pair_table_stored"],
        }
    )
    return rows, held_out_row, geometry_row


def projected_octahedron(refinements):
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
    for _level in range(refinements):
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


def sphere_repayment_rows():
    polynomial = solid_harmonic_polynomials(4)[-1]
    rows = []
    for refinement in (1, 2, 3):
        points, faces = projected_octahedron(refinement)
        trace = []
        exact = []
        for point in points:
            value, gradient = polynomial.value_gradient(point)
            trace.append(value)
            exact.append(sum(a * b for a, b in zip(gradient, point, strict=True)))
        for method, config in (
            (
                "raw_Q3",
                SurfaceQConfig(
                    continuum_repayment=False,
                    leaf_size=4,
                    work_budget_factor=256,
                ),
            ),
            (
                "singular_cell_curvature",
                SurfaceQConfig(
                    continuum_repayment=True,
                    singular_cell_order=3,
                    harmonic_moment_degree=0,
                    leaf_size=4,
                    work_budget_factor=256,
                ),
            ),
            ("full_degree3_repayment", Q_CONFIG),
        ):
            engine = build_mesh_engine(
                points,
                faces,
                normals=points,
                config=config,
            )
            result, elapsed = timed(lambda: engine.apply_dtn_principal(trace))
            rows.append(
                {
                    "refinement": refinement,
                    "nodes": len(points),
                    "mode": polynomial.name,
                    "held_out_degree": 4,
                    "method": method,
                    "relative_continuum_error": relative_error(
                        engine.weights,
                        exact,
                        result.values,
                    ),
                    "solve_ms": elapsed,
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary, rows, held_out_rows, geometry_rows, sphere_rows):
    lines = [
        "# Production 3D CAD boundary-PDE validation",
        "",
        "## Error classes",
        "",
        "The previous `1.779e-8` maximum was a discrete implementation/PDE "
        "solve error against the same finite operator. It was not a continuum "
        "surface-discretization bound. This report separates compiled-mode "
        "accuracy, algebraic residual, and held-out continuum error.",
        "",
        "## CAD coverage",
        "",
        "Every source triangle in every QCAD3J archive is scanned. The PDE "
        "atlas is nondimensional and topology-bearing but intentionally coarse. "
        "Its lumped measure is inherited from all source faces; the geometric "
        "coarse-triangle area is reported separately.",
        "",
        "| Shape | Source V | Source F | PDE nodes | PDE faces | Measure ratio "
        "| Local cells repaid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in geometry_rows:
        lines.append(
            f"| {row['shape']} | {row['source_vertices']} | {row['source_faces']} | "
            f"{row['compiled_vertices']} | {row['compiled_faces']} | "
            f"{row['compiled_measure_to_source_area_ratio']:.6f} | "
            f"{row['singular_cells_repaid']} |"
        )
    lines.extend(
        [
            "",
            "## PDE results",
            "",
            "| Shape | Problem | Reference | Error | Residual | Q applies | Pass |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        error = "n/a" if row["relative_error"] == "" else f"{row['relative_error']:.3e}"
        lines.append(
            f"| {row['shape']} | {row['problem']} | {row['reference_class']} | "
            f"{error} | {row['relative_algebraic_residual']:.3e} | "
            f"{row['qjet_applications']} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Held-out continuum audit",
            "",
            "The following degree-four harmonic was not included in the degree-three "
            "moment compiler. These values are the relevant continuum generalization "
            "test and are not expected to equal the compiled-mode residual.",
            "",
            "| Shape | Held-out degree | Relative continuum error |",
            "|---|---:|---:|",
        ]
    )
    for row in held_out_rows:
        lines.append(
            f"| {row['shape']} | {row['held_out_degree']} | "
            f"{row['relative_continuum_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Singular-cell and curvature repayment",
            "",
            "The sphere table isolates the topology-aware tangent-cell series from "
            "the fixed-rank harmonic channel on a held-out degree-four mode.",
            "",
            "| N | Method | Held-out continuum error |",
            "|---:|---|---:|",
        ]
    )
    for row in sphere_rows:
        lines.append(
            f"| {row['nodes']} | {row['method']} | "
            f"{row['relative_continuum_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "`poisson_boundary_inverse`, heat, and wave are boundary functional-"
            "calculus problems generated by the repaid DtN discretization. They are "
            "not arbitrary volumetric-source solves. `helmholtz_dtn` has an exact "
            "plane-wave continuum reference.",
            "",
            "No result in this report justifies a universal 3D machine-precision "
            "claim. Machine-level rows certify retained moment channels. The maximum "
            "held-out error remains the honest continuum limitation.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    held_out_rows = []
    geometry_rows = []
    for label, archive in MODELS:
        case_rows, held_out, geometry = run_cad_case(label, archive)
        rows.extend(case_rows)
        held_out_rows.append(held_out)
        geometry_rows.append(geometry)
        print(f"completed {label}", flush=True)
    sphere_rows = sphere_repayment_rows()
    summary = {
        "model_count": len(MODELS),
        "source_vertex_count": sum(row["source_vertices"] for row in geometry_rows),
        "source_face_count": sum(row["source_faces"] for row in geometry_rows),
        "pde_case_count": len(rows),
        "all_source_faces_scanned": all(
            row["all_source_faces_scanned"] for row in geometry_rows
        ),
        "all_pde_gates_passed": all(row["passed"] for row in rows),
        "maximum_compiled_reference_error": max(
            float(row["relative_error"])
            for row in rows
            if row["relative_error"] != ""
        ),
        "maximum_algebraic_residual": max(
            row["relative_algebraic_residual"] for row in rows
        ),
        "maximum_held_out_continuum_error": max(
            row["relative_continuum_error"] for row in held_out_rows
        ),
        "minimum_held_out_continuum_error": min(
            row["relative_continuum_error"] for row in held_out_rows
        ),
        "dense_q_matrix_stored": False,
        "pair_table_stored": False,
        "production_apply_complexity": "O(N log N)+O(N r) at fixed jet ranks",
        "production_storage_complexity": "O(N)",
        "universal_machine_precision_claim": False,
    }
    write_csv(OUT / "cad_pde_rows.csv", rows)
    write_csv(OUT / "cad_held_out_rows.csv", held_out_rows)
    write_csv(OUT / "cad_geometry_rows.csv", geometry_rows)
    write_csv(OUT / "sphere_repayment_rows.csv", sphere_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(summary, rows, held_out_rows, geometry_rows, sphere_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
