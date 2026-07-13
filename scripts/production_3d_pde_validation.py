#!/usr/bin/env python3
"""Independent validation of the production 3D boundary PDE calculus.

The campaign keeps two error classes separate:

1. discrete implementation error, measured against an independently streamed
   pairwise ``Q_3/(2*pi)`` operator on audit-sized geometries; and
2. continuum discretization error, measured against exact spherical-harmonic
   solutions on the unit sphere.

The pairwise reference is verification-only and is never imported by the
production package.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for location in (SRC, SCRIPTS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from gulati_quadrature import (  # noqa: E402
    SurfacePDEConfig,
    SurfaceQConfig,
    build_surface_engine,
    build_surface_pde_solver,
)
from inverse_shape.quadrature import BorrowComputeRepayLedger  # noqa: E402
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_weighted_distance_graph,
)
from production_3d_qjet_extended_validation import (  # noqa: E402
    car_assembly,
    faceted_octahedron,
    folded_sheet,
    sphere,
    torus,
)


OUT = ROOT / "outputs" / "production_3d_pde_validation"
NORMALIZATION = 1.0 / (2.0 * math.pi)
Q_CONFIG = SurfaceQConfig(
    kernel_power=3.0,
    tolerance=3.0e-13,
    maximum_order=16,
    leaf_size=4,
    work_budget_factor=192,
)
PDE_CONFIG = SurfacePDEConfig(
    tolerance=2.0e-9,
    maximum_iterations=180,
    heat_steps=1,
    wave_steps=2,
    fail_on_nonconvergence=False,
)
DISCRETE_GATES = {
    "laplace_dtn": 2.0e-10,
    "poisson": 2.0e-7,
    "screened_poisson": 2.0e-8,
    "helmholtz": 2.0e-6,
    "heat": 2.0e-7,
    "wave": 2.0e-7,
}


@dataclass(frozen=True)
class _ReferenceConfig:
    kernel_power: float = 3.0


@dataclass(frozen=True)
class _ReferenceEvaluation:
    values: tuple[complex, ...]
    compression_inf_bound: float = 0.0
    ledger: BorrowComputeRepayLedger | None = None


class StreamedReferenceSurfaceEngine:
    """Quadratic-time, matrix-free audit oracle excluded from production."""

    def __init__(self, points, weights):
        self.points = tuple(tuple(float(value) for value in point) for point in points)
        self.weights = tuple(float(value) for value in weights)
        self.n = len(self.points)
        self.config = _ReferenceConfig()

    def apply_dtn_principal(self, values):
        output = reference_weighted_distance_graph(
            self.points,
            self.weights,
            values,
            kernel_power=3.0,
            normalization=NORMALIZATION,
        )
        return _ReferenceEvaluation(tuple(complex(value) for value in output))


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


def relative_l2(weights, reference, candidate):
    difference = tuple(
        complex(left) - complex(right)
        for left, right in zip(reference, candidate, strict=True)
    )
    return weighted_norm(weights, difference) / max(
        weighted_norm(weights, reference),
        1.0e-300,
    )


def project_mean_zero(weights, values):
    mean = sum(
        weight * value for weight, value in zip(weights, values, strict=True)
    ) / sum(weights)
    return tuple(value - mean for value in values)


def smooth_field(points):
    return tuple(
        x + 0.17 * y - 0.11 * z + 0.06 * x * y + 0.03j * (y - z)
        for x, y, z in points
    )


def direct_apply(engine, values):
    return engine.apply_dtn_principal(values).values


def timed(call):
    started = time.perf_counter()
    result = call()
    return result, 1000.0 * (time.perf_counter() - started)


def result_row(shape, nodes, problem, reference_kind, expected, result, weights, elapsed_ms):
    error = relative_l2(weights, expected, result.values)
    gate = DISCRETE_GATES[problem]
    return {
        "shape": shape,
        "nodes": nodes,
        "problem": problem,
        "reference": reference_kind,
        "relative_solution_error": error,
        "relative_equation_residual": result.relative_residual,
        "compression_inf_bound": result.compression_inf_bound,
        "iterations": result.iterations,
        "qjet_applications": result.operator_applications,
        "solve_ms": elapsed_ms,
        "error_gate": gate,
        "converged": result.converged,
        "dense_operator_stored": result.stats["dense_operator_stored"],
        "quadratic_fallback": result.stats["quadratic_fallback"],
        "passed": (
            result.converged
            and error <= gate
            and not result.stats["dense_operator_stored"]
            and not result.stats["quadratic_fallback"]
        ),
    }


def validate_discrete_shape(name, geometry):
    points, weights = geometry
    production = build_surface_engine(points, weights, config=Q_CONFIG)
    reference = StreamedReferenceSurfaceEngine(points, weights)
    solver = build_surface_pde_solver(production, config=PDE_CONFIG)
    reference_solver = build_surface_pde_solver(reference, config=PDE_CONFIG)
    mean_zero = project_mean_zero(weights, smooth_field(points))
    shifted = tuple(value + 0.37 for value in mean_zero)
    rows = []

    exact_first = direct_apply(reference, mean_zero)
    result, elapsed = timed(lambda: solver.apply_laplace_dtn(mean_zero))
    rows.append(
        result_row(
            name,
            len(points),
            "laplace_dtn",
            "independent streamed pairwise Q3",
            exact_first,
            result,
            weights,
            elapsed,
        )
    )

    result, elapsed = timed(lambda: solver.solve_poisson(exact_first))
    rows.append(
        result_row(
            name,
            len(points),
            "poisson",
            "manufactured with independent streamed pairwise Q3",
            mean_zero,
            result,
            weights,
            elapsed,
        )
    )

    mass = 0.45
    exact_shifted = direct_apply(reference, shifted)
    screened_rhs = tuple(
        value + mass * target
        for value, target in zip(exact_shifted, shifted, strict=True)
    )
    result, elapsed = timed(
        lambda: solver.solve_poisson(screened_rhs, mass=mass)
    )
    rows.append(
        result_row(
            name,
            len(points),
            "screened_poisson",
            "manufactured with independent streamed pairwise Q3",
            shifted,
            result,
            weights,
            elapsed,
        )
    )

    wave_number = 0.65
    damping = 0.8
    exact_second = direct_apply(reference, exact_first)
    helmholtz_rhs = tuple(
        value - wave_number * wave_number * target + 1j * damping * target
        for value, target in zip(exact_second, mean_zero, strict=True)
    )
    result, elapsed = timed(
        lambda: solver.solve_helmholtz(
            helmholtz_rhs,
            wavenumber=wave_number,
            damping=damping,
        )
    )
    rows.append(
        result_row(
            name,
            len(points),
            "helmholtz",
            "manufactured with independent streamed pairwise Q3",
            mean_zero,
            result,
            weights,
            elapsed,
        )
    )

    reference_heat = reference_solver.solve_heat(mean_zero, time=0.06, steps=1)
    result, elapsed = timed(lambda: solver.solve_heat(mean_zero, time=0.06, steps=1))
    rows.append(
        result_row(
            name,
            len(points),
            "heat",
            "same Pade integrator over independent streamed pairwise Q3",
            reference_heat.values,
            result,
            weights,
            elapsed,
        )
    )

    reference_wave = reference_solver.solve_wave(mean_zero, time=0.08, steps=2)
    result, elapsed = timed(lambda: solver.solve_wave(mean_zero, time=0.08, steps=2))
    rows.append(
        result_row(
            name,
            len(points),
            "wave",
            "same Newmark integrator over independent streamed pairwise Q3",
            reference_wave.values,
            result,
            weights,
            elapsed,
        )
    )
    return rows


def sphere_mode(points, degree):
    if degree == 1:
        return tuple(z for _x, _y, z in points)
    if degree == 2:
        return tuple(0.5 * (3.0 * z * z - 1.0) for _x, _y, z in points)
    raise ValueError("sphere audit implements degrees one and two")


def analytic_row(nodes, degree, problem, expected, result, weights, elapsed_ms):
    return {
        "nodes": nodes,
        "degree": degree,
        "exact_dtn_eigenvalue": float(degree),
        "problem": problem,
        "reference": "exact unit-sphere spherical harmonic",
        "relative_continuum_error": relative_l2(weights, expected, result.values),
        "relative_equation_residual": result.relative_residual,
        "iterations": result.iterations,
        "qjet_applications": result.operator_applications,
        "solve_ms": elapsed_ms,
        "converged": result.converged,
    }


def validate_sphere_continuum():
    rows = []
    for meridians, phases in ((4, 8), (6, 12), (8, 16), (12, 24)):
        points, weights = sphere(meridians, phases)
        engine = build_surface_engine(points, weights, config=Q_CONFIG)
        solver = build_surface_pde_solver(engine, config=PDE_CONFIG)
        for degree in (1, 2):
            mode = sphere_mode(points, degree)
            expected = tuple(degree * value for value in mode)
            result, elapsed = timed(lambda: solver.apply_laplace_dtn(mode))
            rows.append(
                analytic_row(
                    len(points),
                    degree,
                    "laplace_dtn",
                    expected,
                    result,
                    weights,
                    elapsed,
                )
            )

    points, weights = sphere(6, 12)
    engine = build_surface_engine(points, weights, config=Q_CONFIG)
    solver = build_surface_pde_solver(engine, config=PDE_CONFIG)
    degree = 1
    eigenvalue = 1.0
    mode = sphere_mode(points, degree)

    result, elapsed = timed(lambda: solver.solve_poisson(mode))
    rows.append(
        analytic_row(
            len(points),
            degree,
            "poisson",
            mode,
            result,
            weights,
            elapsed,
        )
    )

    mass = 0.45
    screened_rhs = tuple((eigenvalue + mass) * value for value in mode)
    result, elapsed = timed(lambda: solver.solve_poisson(screened_rhs, mass=mass))
    rows.append(
        analytic_row(
            len(points),
            degree,
            "screened_poisson",
            mode,
            result,
            weights,
            elapsed,
        )
    )

    wave_number = 0.65
    damping = 0.8
    helmholtz_symbol = eigenvalue * eigenvalue - wave_number * wave_number + 1j * damping
    helmholtz_rhs = tuple(helmholtz_symbol * value for value in mode)
    result, elapsed = timed(
        lambda: solver.solve_helmholtz(
            helmholtz_rhs,
            wavenumber=wave_number,
            damping=damping,
        )
    )
    rows.append(
        analytic_row(
            len(points),
            degree,
            "helmholtz",
            mode,
            result,
            weights,
            elapsed,
        )
    )

    heat_time = 0.06
    result, elapsed = timed(lambda: solver.solve_heat(mode, time=heat_time, steps=1))
    heat_expected = tuple(math.exp(-heat_time * eigenvalue) * value for value in mode)
    rows.append(
        analytic_row(
            len(points),
            degree,
            "heat",
            heat_expected,
            result,
            weights,
            elapsed,
        )
    )

    wave_time = 0.08
    result, elapsed = timed(lambda: solver.solve_wave(mode, time=wave_time, steps=2))
    wave_expected = tuple(math.cos(wave_time * eigenvalue) * value for value in mode)
    rows.append(
        analytic_row(
            len(points),
            degree,
            "wave",
            wave_expected,
            result,
            weights,
            elapsed,
        )
    )
    return rows


def observed_orders(rows, degree):
    selected = [
        row
        for row in rows
        if row["problem"] == "laplace_dtn" and row["degree"] == degree
    ]
    selected.sort(key=lambda row: row["nodes"])
    output = []
    for left, right in zip(selected, selected[1:], strict=False):
        mesh_ratio = math.sqrt(right["nodes"] / left["nodes"])
        order = math.log(
            left["relative_continuum_error"] / right["relative_continuum_error"]
        ) / math.log(mesh_ratio)
        output.append(order)
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary, discrete_rows, analytic_rows):
    lines = [
        "# Production 3D boundary PDE validation",
        "",
        "## Scope",
        "",
        "The tested generator is `A = Q_3/(2*pi)`. The named heat, Poisson, "
        "Helmholtz, and wave problems are boundary equations generated by `A`; "
        "they are not arbitrary volumetric-source solves.",
        "",
        "Two errors are reported separately. Discrete implementation error uses "
        "an independently streamed pairwise operator. Continuum error uses the "
        "exact unit-sphere identity `Lambda Y_lm = l Y_lm`.",
        "",
        "## Discrete audit",
        "",
        f"- Shapes: {summary['discrete_shape_count']}",
        f"- PDE cases: {summary['discrete_case_count']}",
        f"- Maximum relative solution error: `{summary['maximum_discrete_error']:.6e}`",
        f"- All configured gates passed: `{summary['all_discrete_gates_passed']}`",
        f"- Dense operator stored: `{summary['dense_operator_stored']}`",
        f"- Quadratic fallback: `{summary['quadratic_fallback']}`",
        "",
        "| Shape | PDE | N | Relative error | Residual | Q applies |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in discrete_rows:
        lines.append(
            f"| {row['shape']} | {row['problem']} | {row['nodes']} | "
            f"{row['relative_solution_error']:.3e} | "
            f"{row['relative_equation_residual']:.3e} | "
            f"{row['qjet_applications']} |"
        )
    lines.extend(
        [
            "",
            "## Continuum sphere audit",
            "",
            f"- Best degree-one Laplace DtN error: "
            f"`{summary['best_sphere_degree_one_error']:.6e}`",
            f"- Best degree-two Laplace DtN error: "
            f"`{summary['best_sphere_degree_two_error']:.6e}`",
            f"- Median observed degree-one refinement order: "
            f"`{summary['median_degree_one_order']:.4f}`",
            "",
            "| PDE | l | N | Continuum error | Algebraic residual |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in analytic_rows:
        lines.append(
            f"| {row['problem']} | {row['degree']} | {row['nodes']} | "
            f"{row['relative_continuum_error']:.3e} | "
            f"{row['relative_equation_residual']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            "The QJet compression and the matrix-free PDE algebra agree with the "
            "independent discrete operator at the configured tolerances. The "
            "continuum sphere error is substantially larger and converges only "
            "algebraically for the present lumped-node singular quadrature. "
            "Therefore this campaign does **not** establish universal machine-precision "
            "3D PDE accuracy. A high-order local tangent-cell/curvature repayment, "
            "plus frequency-dependent lower-order operators for true Helmholtz and "
            "volume-source channels for bulk Poisson/heat, remains necessary.",
            "",
            "## Complexity",
            "",
            "For `k` QJet applications, each solve uses `O(k N log N)` time and "
            "`O(N)` auxiliary storage at fixed QJet order. The pairwise oracle is "
            "`O(N^2)` and appears only in this audit script.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometries = (
        ("sphere", sphere(5, 8)),
        ("torus", torus(8, 5)),
        ("open folded sheet", folded_sheet(8, 5)),
        ("faceted octahedron", faceted_octahedron(1)),
        ("car assembly", car_assembly(4, 6, 2, 2)),
    )
    discrete_rows = []
    for name, geometry in geometries:
        discrete_rows.extend(validate_discrete_shape(name, geometry))
    analytic_rows = validate_sphere_continuum()
    first_orders = observed_orders(analytic_rows, 1)
    sorted_orders = sorted(first_orders)
    median_order = sorted_orders[len(sorted_orders) // 2]
    summary = {
        "operator": "A = Q_3 / (2*pi)",
        "pde_scope": "boundary functional calculus",
        "problems": [
            "laplace_dtn",
            "poisson",
            "screened_poisson",
            "helmholtz",
            "heat",
            "wave",
        ],
        "discrete_shape_count": len(geometries),
        "discrete_case_count": len(discrete_rows),
        "maximum_discrete_error": max(
            row["relative_solution_error"] for row in discrete_rows
        ),
        "maximum_discrete_residual": max(
            row["relative_equation_residual"] for row in discrete_rows
        ),
        "all_discrete_gates_passed": all(row["passed"] for row in discrete_rows),
        "dense_operator_stored": any(
            row["dense_operator_stored"] for row in discrete_rows
        ),
        "quadratic_fallback": any(row["quadratic_fallback"] for row in discrete_rows),
        "best_sphere_degree_one_error": min(
            row["relative_continuum_error"]
            for row in analytic_rows
            if row["problem"] == "laplace_dtn" and row["degree"] == 1
        ),
        "best_sphere_degree_two_error": min(
            row["relative_continuum_error"]
            for row in analytic_rows
            if row["problem"] == "laplace_dtn" and row["degree"] == 2
        ),
        "degree_one_observed_orders": first_orders,
        "median_degree_one_order": median_order,
        "continuum_machine_precision_claim": False,
        "volume_source_solver_claim": False,
        "production_operator_complexity": "O(N log N) per apply, O(N) storage",
        "iterative_solve_complexity": "O(k N log N), O(N) auxiliary storage",
    }
    write_csv(OUT / "discrete_pde_rows.csv", discrete_rows)
    write_csv(OUT / "sphere_continuum_rows.csv", analytic_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(summary, discrete_rows, analytic_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_discrete_gates_passed"]:
        raise SystemExit("one or more discrete PDE validation gates failed")


if __name__ == "__main__":
    main()
