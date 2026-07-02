#!/usr/bin/env python3
"""Head-to-head Q/DtN versus volumetric FEM on funky planar domains."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from inverse_shape.fem import (
    best_weighted_scalar,
    build_fem_boundary_dtn,
    relative_inf,
    relative_weighted_l2,
)
from inverse_shape.q_dtn import (
    build_boundary_pullback_qjet,
    build_harmonic_moment_corrected_planar_qjet,
    build_planar_domain_qjet,
    ellipse_qjet_map,
    radial_fourier_qjet_map,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "q_dtn_funky_fem_head_to_head.json"


def timed(fn):
    start = perf_counter()
    value = fn()
    return value, 1000.0 * (perf_counter() - start)


def radial_points(n, base, cos_coefficients=(), sin_coefficients=()):
    return build_boundary_pullback_qjet(
        n,
        radial_fourier_qjet_map(
            base,
            cos_coefficients=cos_coefficients,
            sin_coefficients=sin_coefficients,
        ),
    ).points


def ellipse_points(n, a, b):
    return build_boundary_pullback_qjet(n, ellipse_qjet_map(a, b)).points


def square_points(n):
    out = []
    per_side = n // 4
    for side in range(4):
        for k in range(per_side):
            t = k / per_side
            if side == 0:
                out.append((1.0, -1.0 + 2.0 * t))
            elif side == 1:
                out.append((1.0 - 2.0 * t, 1.0))
            elif side == 2:
                out.append((-1.0, 1.0 - 2.0 * t))
            else:
                out.append((-1.0 + 2.0 * t, -1.0))
    return tuple(out[:n])


def star_polygon_points(n, spikes=5):
    out = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        radius = 1.0 + 0.32 * (1.0 if (index * spikes // n) % 2 == 0 else -1.0)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return tuple(out)


def cardioid_points(n):
    out = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        radius = 1.0 - 0.82 * math.cos(theta)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return tuple(out)


def rounded_square_points(n, exponent=4.0):
    out = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        c = math.cos(theta)
        s = math.sin(theta)
        denom = (abs(c) ** exponent + abs(s) ** exponent) ** (1.0 / exponent)
        out.append((c / denom, s / denom))
    return tuple(out)


def boundary_signal(n):
    out = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        out.append(
            math.cos(4.0 * theta + 0.17)
            + 0.35 * math.sin(7.0 * theta - 0.31)
            - 0.18 * math.cos(11.0 * theta + 0.43)
        )
    return out


def shape_suite():
    return [
        ("ellipse_3_to_1", "smooth", lambda n: ellipse_points(n, 3.0, 1.0)),
        (
            "funky_flower_curve",
            "smooth_nonconvex",
            lambda n: radial_points(
                n,
                1.0,
                cos_coefficients=(0.0, 0.28, 0.0, -0.09),
                sin_coefficients=(0.0, 0.0, 0.06),
            ),
        ),
        ("rounded_square_superellipse", "smooth_high_curvature", rounded_square_points),
        ("square_polygon", "corner", square_points),
        ("star_polygon", "corner_vertex_scattering", star_polygon_points),
        ("cardioid_cusp", "cusp_endpoint", cardioid_points),
    ]


def problem_suite():
    return [
        ("laplace_dtn", {}),
        ("heat", {"time": 0.02, "max_steps": 48}),
        ("poisson", {"mass": 0.4, "iterations": 96, "tolerance": 1.0e-10}),
        ("helmholtz", {"wavenumber": 1.0, "damping": 0.25, "iterations": 24}),
        ("wave", {"time": 0.05, "max_steps": 48}),
    ]


def real_imag(value):
    z_value = complex(value)
    return float(z_value.real), float(z_value.imag)


def run_shape(shape, family, make_points, *, n_boundary, fem_radial_levels):
    points = make_points(n_boundary)
    values = boundary_signal(n_boundary)
    raw_qjet, raw_q_build_ms = timed(lambda: build_planar_domain_qjet(points))
    qjet, q_build_ms = timed(lambda: build_harmonic_moment_corrected_planar_qjet(points, moment_degree=2))
    fem, fem_build_ms = timed(lambda: build_fem_boundary_dtn(points, radial_levels=fem_radial_levels))
    rows = []
    for problem, params in problem_suite():
        raw_q_result, raw_q_ms = timed(
            lambda problem=problem, params=params: raw_qjet.solve_boundary_problem(problem, values, **params)
        )
        q_result, q_ms = timed(
            lambda problem=problem, params=params: qjet.solve_boundary_problem(problem, values, **params)
        )
        fem_result, fem_apply_ms = timed(
            lambda problem=problem, params=params: fem.solve_boundary_problem(problem, values, **params)
        )
        q_values = np.asarray(tuple(q_result.values), dtype=np.complex128)
        fem_values = np.asarray(fem_result, dtype=np.complex128)
        alpha = best_weighted_scalar(q_values, fem_values, fem.boundary_mass)
        scaled_q_values = alpha * q_values
        alpha_real, alpha_imag = real_imag(alpha)
        rows.append(
            {
                "shape": shape,
                "family": family,
                "problem": problem,
                "n_boundary": n_boundary,
                "q_method": "matrix_free_harmonic_moment_zeta_repaid_planar_qjet",
                "fem_method": "volumetric_p1_radial_fan_schur_dtn",
                "q_status": q_result.ledger.status,
                "q_error_type": q_result.stats.get("q_error_type"),
                "q_recommended_q": q_result.stats.get("recommended_q"),
                "q_operator_bound": q_result.stats.get("operator_bound"),
                "correction_rank": q_result.stats.get("correction_rank"),
                "harmonic_moment_rank": q_result.stats.get("harmonic_moment_rank"),
                "zeta_tail_rank": q_result.stats.get("zeta_tail_rank"),
                "zeta_tail_degree": q_result.stats.get("zeta_tail_degree"),
                "raw_q_build_ms": raw_q_build_ms,
                "raw_q_apply_ms": raw_q_ms,
                "q_build_ms": q_build_ms,
                "q_apply_ms": q_ms,
                "q_total_cold_ms": q_build_ms + q_ms,
                "q_work_units": q_result.work_units,
                "fem_build_ms": fem_build_ms,
                "fem_apply_ms": fem_apply_ms,
                "fem_total_cold_ms": fem_build_ms + fem_apply_ms,
                "fem_nodes": fem.mesh.node_count,
                "fem_triangles": fem.mesh.triangle_count,
                "fem_radial_levels": fem.mesh.radial_levels,
                "fem_min_steklov": float(np.min(fem.eigenvalues)),
                "fem_max_steklov": float(np.max(fem.eigenvalues)),
                "raw_relative_l2_vs_fem": relative_weighted_l2(q_values, fem_values, fem.boundary_mass),
                "raw_chord_q_relative_l2_vs_fem": relative_weighted_l2(
                    raw_q_result.values,
                    fem_values,
                    fem.boundary_mass,
                ),
                "raw_relative_inf_vs_fem": relative_inf(q_values, fem_values),
                "best_scalar_real": alpha_real,
                "best_scalar_imag": alpha_imag,
                "best_scaled_relative_l2_vs_fem": relative_weighted_l2(
                    scaled_q_values,
                    fem_values,
                    fem.boundary_mass,
                ),
                "best_scaled_relative_inf_vs_fem": relative_inf(scaled_q_values, fem_values),
                "q_output_inf_norm": float(np.max(np.abs(q_values))),
                "fem_output_inf_norm": float(np.max(np.abs(fem_values))),
                "dense_matrix_stored_by_q": False,
                "dense_matrix_stored_by_fem_baseline": True,
            }
        )
    return rows


def median(values):
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def summarize(rows):
    by_problem = {}
    for problem in sorted({row["problem"] for row in rows}):
        selected = [row for row in rows if row["problem"] == problem]
        by_problem[problem] = {
            "rows": len(selected),
            "median_raw_relative_l2_vs_fem": median(row["raw_relative_l2_vs_fem"] for row in selected),
            "max_raw_relative_l2_vs_fem": max(row["raw_relative_l2_vs_fem"] for row in selected),
            "median_best_scaled_relative_l2_vs_fem": median(
                row["best_scaled_relative_l2_vs_fem"] for row in selected
            ),
            "max_best_scaled_relative_l2_vs_fem": max(
                row["best_scaled_relative_l2_vs_fem"] for row in selected
            ),
            "median_q_apply_ms": median(row["q_apply_ms"] for row in selected),
            "median_fem_apply_ms": median(row["fem_apply_ms"] for row in selected),
            "median_fem_cold_ms": median(row["fem_total_cold_ms"] for row in selected),
        }
    by_shape = {}
    for shape in sorted({row["shape"] for row in rows}):
        selected = [row for row in rows if row["shape"] == shape]
        by_shape[shape] = {
            "rows": len(selected),
            "family": selected[0]["family"],
            "median_raw_relative_l2_vs_fem": median(row["raw_relative_l2_vs_fem"] for row in selected),
            "max_raw_relative_l2_vs_fem": max(row["raw_relative_l2_vs_fem"] for row in selected),
            "median_best_scaled_relative_l2_vs_fem": median(
                row["best_scaled_relative_l2_vs_fem"] for row in selected
            ),
            "max_best_scaled_relative_l2_vs_fem": max(
                row["best_scaled_relative_l2_vs_fem"] for row in selected
            ),
            "median_q_apply_ms": median(row["q_apply_ms"] for row in selected),
            "median_fem_apply_ms": median(row["fem_apply_ms"] for row in selected),
            "fem_nodes": selected[0]["fem_nodes"],
            "fem_triangles": selected[0]["fem_triangles"],
        }
    return {
        "case_count": len(rows),
        "shape_count": len(by_shape),
        "problem_count": len(by_problem),
        "q_failure_count": sum(1 for row in rows if row["q_status"] != "borrowed_repaid"),
        "median_raw_relative_l2_vs_fem": median(row["raw_relative_l2_vs_fem"] for row in rows),
        "max_raw_relative_l2_vs_fem": max(row["raw_relative_l2_vs_fem"] for row in rows),
        "median_best_scaled_relative_l2_vs_fem": median(
            row["best_scaled_relative_l2_vs_fem"] for row in rows
        ),
        "max_best_scaled_relative_l2_vs_fem": max(row["best_scaled_relative_l2_vs_fem"] for row in rows),
        "median_q_apply_ms": median(row["q_apply_ms"] for row in rows),
        "median_fem_apply_ms": median(row["fem_apply_ms"] for row in rows),
        "median_fem_cold_ms": median(row["fem_total_cold_ms"] for row in rows),
        "by_problem": by_problem,
        "by_shape": by_shape,
    }


def main():
    n_boundary = 64
    fem_radial_levels = 32
    rows = []
    for shape, family, make_points in shape_suite():
        rows.extend(
            run_shape(
                shape,
                family,
                make_points,
                n_boundary=n_boundary,
                fem_radial_levels=fem_radial_levels,
            )
        )
    payload = {
        "parameters": {
            "boundary_samples": n_boundary,
            "fem_radial_levels": fem_radial_levels,
            "boundary_signal": "cos(4 theta + 0.17) + 0.35 sin(7 theta - 0.31) - 0.18 cos(11 theta + 0.43)",
            "fem_baseline": "volumetric P1 radial-fan FEM; boundary Schur-complement DtN; generalized Steklov eigensystem for operator functions",
            "q_baseline": "matrix-free planar chord QJet with borrow-compute-repay ledger",
            "comparison_norm": "mass-weighted boundary L2 against FEM boundary point values",
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "rows": len(rows),
                "median_raw_relative_l2_vs_fem": payload["summary"]["median_raw_relative_l2_vs_fem"],
                "max_raw_relative_l2_vs_fem": payload["summary"]["max_raw_relative_l2_vs_fem"],
                "median_best_scaled_relative_l2_vs_fem": payload["summary"][
                    "median_best_scaled_relative_l2_vs_fem"
                ],
                "q_failure_count": payload["summary"]["q_failure_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
