#!/usr/bin/env python3
"""Final production Q pipeline and FEM on funky planar boundaries.

This benchmark intentionally uses the final matrix-free Q path from
``final_q_machine_precision_pipeline.py`` for Q evaluations. FEM is a
NumPy/SciPy radial-fan Schur-complement baseline, not ground truth. The
reported Q/FEM norms are pairwise disagreement diagnostics unless an analytic or
overresolved reference is supplied by another suite.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

import final_q_machine_precision_pipeline as q

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.fem import (  # noqa: E402
    best_weighted_scalar,
    build_fem_boundary_dtn,
    relative_inf,
    relative_weighted_l2,
)


OUT = ROOT / "outputs" / "final_q_vs_fem_funky"
N_BOUNDARY = 256
FEM_RADIAL_LEVELS = 12
APPLY_REPEATS = 7


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    start = perf_counter()
    value = fn()
    return value, 1000.0 * (perf_counter() - start)


def timed_median(fn: Callable[[], object], repeats: int = APPLY_REPEATS) -> tuple[object, float]:
    timings: list[float] = []
    value: object | None = None
    for _ in range(repeats):
        value, elapsed_ms = timed(fn)
        timings.append(elapsed_ms)
    return value, median(timings)


def polygon_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return 0.5 * total


def sample_shape(shape: q.LaurentShape, n: int) -> tuple[list[tuple[float, float]], list[float], list[complex]]:
    points: list[tuple[float, float]] = []
    speeds: list[float] = []
    values: list[complex] = []
    for index in range(n):
        theta = q.TAU * index / n
        w = q.unit(theta)
        z = shape.psi(w)
        points.append((z.real, z.imag))
        speeds.append(max(abs(shape.dpsi(w)), 1.0e-14))
        values.append(
            complex(
                math.cos(4.0 * theta + 0.17)
                + 0.35 * math.sin(7.0 * theta - 0.31)
                - 0.18 * math.cos(11.0 * theta + 0.43),
                0.0,
            )
        )
    if polygon_area(points) < 0.0:
        points.reverse()
        speeds.reverse()
        values.reverse()
    return points, speeds, values


def metric_repay(output: list[complex], speeds: list[float]) -> list[complex]:
    return [output[index] / speeds[index] for index in range(len(output))]


def solve_final_q(problem: str, values: list[complex], params: dict[str, float], speeds: list[float]) -> list[complex]:
    circle_output = q.solve_cycle_problem(problem, values, params)
    return metric_repay(circle_output, speeds)


def real_imag(value: complex) -> tuple[float, float]:
    z_value = complex(value)
    return float(z_value.real), float(z_value.imag)


def run_shape(shape: q.LaurentShape) -> list[dict[str, object]]:
    points, speeds, values = sample_shape(shape, N_BOUNDARY)
    (_, q_build_ms) = timed(lambda: (points, speeds, values))
    fem, fem_build_ms = timed(lambda: build_fem_boundary_dtn(points, radial_levels=FEM_RADIAL_LEVELS))
    rows: list[dict[str, object]] = []
    for problem, params in q.pde_problem_suite().items():
        q_values, q_apply_ms = timed_median(lambda problem=problem, params=params: solve_final_q(problem, values, params, speeds))
        fem_values, fem_apply_ms = timed_median(
            lambda problem=problem, params=params: fem.solve_boundary_problem(problem, values, **params)
        )
        q_array = np.asarray(tuple(q_values), dtype=np.complex128)
        fem_array = np.asarray(tuple(fem_values), dtype=np.complex128)
        alpha = best_weighted_scalar(q_array, fem_array, fem.boundary_mass)
        scaled_q = alpha * q_array
        alpha_real, alpha_imag = real_imag(alpha)
        rows.append(
            {
                "shape": shape.name,
                "family": shape.family,
                "problem": problem,
                "n_boundary": N_BOUNDARY,
                "q_method": "final_production_generated_q_custom_fft_metric_repay",
                "fem_method": "volumetric_p1_radial_fan_schur_dtn",
                "q_time_big_o": "O(n log n)",
                "q_storage_big_o": "O(n)",
                "fem_build_big_o": "sparse factorization plus dense boundary eigensolve; roughly O(N_mesh^alpha + n^3)",
                "fem_apply_big_o": "O(n^2)",
                "fem_storage_big_o": "O(N_mesh + n^2)",
                "q_build_ms": q_build_ms,
                "q_apply_ms": q_apply_ms,
                "q_total_cold_ms": q_build_ms + q_apply_ms,
                "fem_build_ms": fem_build_ms,
                "fem_apply_ms": fem_apply_ms,
                "fem_total_cold_ms": fem_build_ms + fem_apply_ms,
                "fem_nodes": fem.mesh.node_count,
                "fem_triangles": fem.mesh.triangle_count,
                "fem_radial_levels": fem.mesh.radial_levels,
                "raw_relative_l2_vs_fem": relative_weighted_l2(q_array, fem_array, fem.boundary_mass),
                "raw_pairwise_l2_disagreement": relative_weighted_l2(q_array, fem_array, fem.boundary_mass),
                "raw_relative_inf_vs_fem": relative_inf(q_array, fem_array),
                "raw_pairwise_inf_disagreement": relative_inf(q_array, fem_array),
                "best_scalar_real": alpha_real,
                "best_scalar_imag": alpha_imag,
                "best_scaled_relative_l2_vs_fem": relative_weighted_l2(scaled_q, fem_array, fem.boundary_mass),
                "best_scaled_pairwise_l2_disagreement": relative_weighted_l2(scaled_q, fem_array, fem.boundary_mass),
                "best_scaled_relative_inf_vs_fem": relative_inf(scaled_q, fem_array),
                "best_scaled_pairwise_inf_disagreement": relative_inf(scaled_q, fem_array),
                "q_output_inf_norm": float(np.max(np.abs(q_array))),
                "fem_output_inf_norm": float(np.max(np.abs(fem_array))),
                "reference_status": "pairwise_q_fem_disagreement_only",
                "fem_is_ground_truth": False,
                "dense_matrix_stored_by_q": False,
                "dense_matrix_stored_by_fem_baseline": True,
            }
        )
    return rows


def safe_run_shape(shape: q.LaurentShape) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    try:
        return run_shape(shape), None
    except Exception as exc:  # noqa: BLE001 - benchmark records geometry/FEM failures.
        return [], {
            "shape": shape.name,
            "family": shape.family,
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(rows: list[dict[str, object]], failures: list[dict[str, object]]) -> dict[str, object]:
    def med(key: str, selected: list[dict[str, object]]) -> float | None:
        vals = [float(row[key]) for row in selected if math.isfinite(float(row[key]))]
        return None if not vals else float(median(vals))

    by_problem: dict[str, dict[str, object]] = {}
    for problem in sorted({str(row["problem"]) for row in rows}):
        selected = [row for row in rows if row["problem"] == problem]
        by_problem[problem] = {
            "rows": len(selected),
            "median_q_apply_ms": med("q_apply_ms", selected),
            "median_fem_apply_ms": med("fem_apply_ms", selected),
            "median_q_cold_ms": med("q_total_cold_ms", selected),
            "median_fem_cold_ms": med("fem_total_cold_ms", selected),
            "median_best_scaled_relative_l2_vs_fem": med("best_scaled_relative_l2_vs_fem", selected),
            "median_best_scaled_pairwise_l2_disagreement": med("best_scaled_pairwise_l2_disagreement", selected),
            "max_best_scaled_relative_l2_vs_fem": max(float(row["best_scaled_relative_l2_vs_fem"]) for row in selected),
            "max_best_scaled_pairwise_l2_disagreement": max(
                float(row["best_scaled_pairwise_l2_disagreement"]) for row in selected
            ),
            "q_apply_faster_count": sum(1 for row in selected if float(row["q_apply_ms"]) < float(row["fem_apply_ms"])),
            "fem_apply_faster_count": sum(1 for row in selected if float(row["fem_apply_ms"]) < float(row["q_apply_ms"])),
            "q_cold_faster_count": sum(1 for row in selected if float(row["q_total_cold_ms"]) < float(row["fem_total_cold_ms"])),
            "fem_cold_faster_count": sum(1 for row in selected if float(row["fem_total_cold_ms"]) < float(row["q_total_cold_ms"])),
        }
        by_problem[problem]["q_apply_win_count"] = by_problem[problem]["q_apply_faster_count"]
        by_problem[problem]["fem_apply_win_count"] = by_problem[problem]["fem_apply_faster_count"]
        by_problem[problem]["q_cold_win_count"] = by_problem[problem]["q_cold_faster_count"]
        by_problem[problem]["fem_cold_win_count"] = by_problem[problem]["fem_cold_faster_count"]
    by_shape: dict[str, dict[str, object]] = {}
    for shape in sorted({str(row["shape"]) for row in rows}):
        selected = [row for row in rows if row["shape"] == shape]
        by_shape[shape] = {
            "rows": len(selected),
            "family": selected[0]["family"],
            "median_q_apply_ms": med("q_apply_ms", selected),
            "median_fem_apply_ms": med("fem_apply_ms", selected),
            "median_best_scaled_relative_l2_vs_fem": med("best_scaled_relative_l2_vs_fem", selected),
            "median_best_scaled_pairwise_l2_disagreement": med("best_scaled_pairwise_l2_disagreement", selected),
            "max_best_scaled_relative_l2_vs_fem": max(float(row["best_scaled_relative_l2_vs_fem"]) for row in selected),
            "max_best_scaled_pairwise_l2_disagreement": max(
                float(row["best_scaled_pairwise_l2_disagreement"]) for row in selected
            ),
        }
    return {
        "case_count": len(rows),
        "shape_count": len(by_shape),
        "problem_count": len(by_problem),
        "failure_count": len(failures),
        "truth_status": "FEM is a competitor baseline, not a reference solution",
        "reference_status": "pairwise Q/FEM disagreement only; analytic-reference suites must be used for error claims",
        "median_q_apply_ms": med("q_apply_ms", rows),
        "median_fem_apply_ms": med("fem_apply_ms", rows),
        "median_q_cold_ms": med("q_total_cold_ms", rows),
        "median_fem_cold_ms": med("fem_total_cold_ms", rows),
        "median_best_scaled_relative_l2_vs_fem": med("best_scaled_relative_l2_vs_fem", rows),
        "median_best_scaled_pairwise_l2_disagreement": med("best_scaled_pairwise_l2_disagreement", rows),
        "max_best_scaled_relative_l2_vs_fem": max(float(row["best_scaled_relative_l2_vs_fem"]) for row in rows),
        "max_best_scaled_pairwise_l2_disagreement": max(
            float(row["best_scaled_pairwise_l2_disagreement"]) for row in rows
        ),
        "q_apply_faster_count": sum(1 for row in rows if float(row["q_apply_ms"]) < float(row["fem_apply_ms"])),
        "fem_apply_faster_count": sum(1 for row in rows if float(row["fem_apply_ms"]) < float(row["q_apply_ms"])),
        "q_cold_faster_count": sum(1 for row in rows if float(row["q_total_cold_ms"]) < float(row["fem_total_cold_ms"])),
        "fem_cold_faster_count": sum(1 for row in rows if float(row["fem_total_cold_ms"]) < float(row["q_total_cold_ms"])),
        "q_apply_win_count": sum(1 for row in rows if float(row["q_apply_ms"]) < float(row["fem_apply_ms"])),
        "fem_apply_win_count": sum(1 for row in rows if float(row["fem_apply_ms"]) < float(row["q_apply_ms"])),
        "q_cold_win_count": sum(1 for row in rows if float(row["q_total_cold_ms"]) < float(row["fem_total_cold_ms"])),
        "fem_cold_win_count": sum(1 for row in rows if float(row["fem_total_cold_ms"]) < float(row["q_total_cold_ms"])),
        "by_problem": by_problem,
        "by_shape": by_shape,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for shape in q.shapes():
        shape_rows, failure = safe_run_shape(shape)
        rows.extend(shape_rows)
        if failure is not None:
            failures.append(failure)
    if not rows:
        raise SystemExit("no successful FEM comparison rows")
    payload = {
        "parameters": {
            "boundary_samples": N_BOUNDARY,
            "fem_radial_levels": FEM_RADIAL_LEVELS,
            "apply_repeats": APPLY_REPEATS,
            "q_baseline": "final production generated-Q custom FFT with metric repayment",
            "fem_baseline": "volumetric P1 radial-fan FEM; boundary Schur-complement DtN; generalized Steklov eigensystem",
            "q_time_big_o": "O(n log n) per boundary PDE apply",
            "q_storage_big_o": "O(n), no dense Q matrix",
            "fem_apply_big_o": "O(n^2) after build in this implementation",
            "fem_storage_big_o": "O(N_mesh + n^2)",
            "comparison_norm": "mass-weighted pairwise Q/FEM boundary disagreement; also best scalar rescale",
            "fem_is_ground_truth": False,
            "truth_policy": (
                "This harness does not declare a winner by accuracy. Use analytic manufactured "
                "or independently overresolved suites for error claims."
            ),
        },
        "summary": summarize(rows, failures),
        "rows": rows,
        "failures": failures,
    }
    write_csv(OUT / "final_q_vs_fem_funky_rows.csv", rows)
    if failures:
        write_csv(OUT / "final_q_vs_fem_funky_failures.csv", failures)
    (OUT / "final_q_vs_fem_funky.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT / "final_q_vs_fem_funky.json"),
                "rows": len(rows),
                "failures": len(failures),
                "shape_count": payload["summary"]["shape_count"],
                "median_q_apply_ms": payload["summary"]["median_q_apply_ms"],
                "median_fem_apply_ms": payload["summary"]["median_fem_apply_ms"],
                "q_apply_faster_count": payload["summary"]["q_apply_faster_count"],
                "fem_apply_faster_count": payload["summary"]["fem_apply_faster_count"],
                "median_best_scaled_pairwise_l2_disagreement": payload["summary"][
                    "median_best_scaled_pairwise_l2_disagreement"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
