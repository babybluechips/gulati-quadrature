"""Benchmark arbitrary-planar-domain Q/DtN with error-corrected quadrature."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

from inverse_shape.q_dtn import (
    build_boundary_pullback_qjet,
    build_planar_domain_qjet,
    ellipse_qjet_map,
    radial_fourier_qjet_map,
)
from inverse_shape.quadrature import (
    log_layer_multipole_zeta_q_borrow_compute_repay,
    log_layer_trapezoid,
    outward_unit_normals,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "q_dtn_arbitrary_planar_benchmark.json"


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


def cosine_mode(n, mode):
    return [math.cos(2.0 * math.pi * mode * index / n) for index in range(n)]


def density(n, mode=3):
    return [1.0 + 0.25 * math.cos(2.0 * math.pi * mode * index / n + 0.2) for index in range(n)]


def near_target(points, sample_index=0, delta_over_h=0.08):
    normals = outward_unit_normals(points)
    n = len(points)
    length = 0.0
    for index in range(n):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % n]
        length += math.hypot(x1 - x0, y1 - y0)
    h = length / n
    base = points[sample_index]
    normal = normals.row_tuple(sample_index)
    return (
        base[0] + delta_over_h * h * normal[0],
        base[1] + delta_over_h * h * normal[1],
    )


def rel_error(value, reference):
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def shape_suite(n):
    return [
        ("ellipse_3_to_1", "smooth", lambda m: ellipse_points(m, 3.0, 1.0)),
        (
            "funky_flower_curve",
            "smooth_nonconvex",
            lambda m: radial_points(m, 1.0, cos_coefficients=(0.0, 0.28, 0.0, -0.09), sin_coefficients=(0.0, 0.0, 0.06)),
        ),
        ("square_polygon", "corner", square_points),
        ("star_polygon", "corner_vertex_scattering", star_polygon_points),
        ("cardioid_cusp", "cusp_endpoint", cardioid_points),
    ]


def pde_rows():
    rows = []
    n = 64
    for shape, family, make_points in shape_suite(n):
        points = make_points(n)
        qjet = build_planar_domain_qjet(points)
        values = cosine_mode(n, 4)
        for problem, params in (
            ("laplace_dtn", {}),
            ("heat", {"time": 0.02, "max_steps": 24}),
            ("poisson", {"mass": 0.4, "iterations": 36}),
            ("helmholtz", {"wavenumber": 1.0, "damping": 0.25, "iterations": 10}),
            ("wave", {"time": 0.05, "max_steps": 24}),
        ):
            result, ms = timed(lambda problem=problem, params=params: qjet.solve_boundary_problem(problem, values, **params))
            stats = result.stats
            rows.append(
                {
                    "shape": shape,
                    "family": family,
                    "problem": problem,
                    "n": n,
                    "status": result.ledger.status,
                    "method": "planar_chord_qjet_error_corrected",
                    "q_error_type": stats.get("q_error_type"),
                    "recommended_q": stats.get("recommended_q"),
                    "operator_bound": stats.get("operator_bound"),
                    "work_units": result.work_units,
                    "ms": ms,
                    "output_inf_norm": max(abs(complex(value)) for value in result.values),
                    "dense_matrix_stored": False,
                }
            )
    return rows


def quadrature_rows():
    rows = []
    n0 = 64
    for shape, family, make_points in shape_suite(n0):
        levels = [n0, 2 * n0, 4 * n0]
        point_sets = [make_points(n) for n in levels]
        density_sets = [density(n) for n in levels]
        target = near_target(point_sets[0], sample_index=max(1, n0 // 9), delta_over_h=0.06)
        sample_indices = [max(1, n // 9) for n in levels]
        reference_points = make_points(2048)
        reference_density = density(2048)
        reference, reference_ms = timed(lambda: log_layer_trapezoid(reference_points, reference_density, target))
        result, ms = timed(
            lambda: log_layer_multipole_zeta_q_borrow_compute_repay(
                point_sets,
                density_sets,
                target,
                sample_indices=sample_indices,
                order=14,
                leaf_size=16,
                theta=0.45,
            )
        )
        rows.append(
            {
                "shape": shape,
                "family": family,
                "n_levels": levels,
                "status": result.ledger.status,
                "method": "multipole_zeta_q_error_correction",
                "relative_error_vs_ref": rel_error(result.value, reference),
                "estimated_zeta_exponent": result.estimated_zeta_exponent,
                "moment_build_units": result.moment_build_units,
                "cached_target_work_units": result.cached_target_work_units,
                "single_target_work_units": result.single_target_work_units,
                "ms": ms,
                "reference_ms": reference_ms,
                "dense_matrix_stored": False,
            }
        )
    return rows


def main():
    data = {
        "parameters": {
            "pde_n": 64,
            "quadrature_levels": [64, 128, 256],
            "reference_n": 2048,
            "protocol": "borrow_compute_repay: planar chord QJet plus multipole/zeta error correction",
            "pdf_method": "Q = pi*Lambda_Gamma + K0; circle principal calculus plus bounded/corner correction channels",
        },
        "pde_rows": pde_rows(),
        "quadrature_rows": quadrature_rows(),
    }
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "pde_rows": len(data["pde_rows"]), "quadrature_rows": len(data["quadrature_rows"])}, indent=2))


if __name__ == "__main__":
    main()
