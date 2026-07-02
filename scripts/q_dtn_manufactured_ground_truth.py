#!/usr/bin/env python3
"""Manufactured harmonic ground-truth tests for funky-domain DtN maps."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from inverse_shape.fem import build_fem_boundary_dtn, boundary_lumped_mass, relative_weighted_l2
from inverse_shape.geometry import as_points, polygon_area
from inverse_shape.q_dtn import (
    build_boundary_pullback_qjet,
    build_harmonic_moment_corrected_planar_qjet,
    build_planar_domain_qjet,
    ellipse_weighted_dtn,
    ellipse_qjet_map,
    radial_fourier_qjet_map,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "q_dtn_manufactured_ground_truth.json"


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


def field_linear_x(x, y):
    return x, (1.0, 0.0)


def field_quadratic_saddle(x, y):
    return x * x - y * y, (2.0 * x, -2.0 * y)


def field_external_log_combo(x, y):
    sources = ((4.25, 2.8, 1.0), (-3.7, 2.6, -0.35), (3.6, -3.1, 0.22))
    value = 0.0
    gx = 0.0
    gy = 0.0
    for sx, sy, weight in sources:
        dx = x - sx
        dy = y - sy
        radius2 = dx * dx + dy * dy
        value += 0.5 * weight * math.log(radius2)
        gx += weight * dx / radius2
        gy += weight * dy / radius2
    return value, (gx, gy)


def field_suite():
    return [
        ("linear_x", field_linear_x),
        ("quadratic_saddle", field_quadratic_saddle),
        ("external_log_combo", field_external_log_combo),
    ]


def edge_outward_normal(left, right, orientation):
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("duplicate adjacent boundary points")
    if orientation >= 0.0:
        return (dy / length, -dx / length), length
    return (-dy / length, dx / length), length


def manufactured_boundary_data(points, field):
    pts = as_points(points)
    if polygon_area(pts) == 0.0:
        raise ValueError("zero-area polygonal boundary")
    values = []
    for x, y in pts:
        value, _ = field(float(x), float(y))
        values.append(float(value))
    return values


def exact_weak_flux(points, field):
    pts = as_points(points)
    orientation = 1.0 if polygon_area(pts) >= 0.0 else -1.0
    out = np.zeros(len(pts), dtype=np.float64)
    # Four-point Gauss-Legendre is ample for these analytic manufactured fields
    # and avoids assigning a single point normal at a corner.
    gauss_x = (
        0.06943184420297371,
        0.33000947820757187,
        0.6699905217924281,
        0.9305681557970262,
    )
    gauss_w = (
        0.17392742256872692,
        0.3260725774312731,
        0.3260725774312731,
        0.17392742256872692,
    )
    for index in range(len(pts)):
        left = pts[index]
        right = pts[(index + 1) % len(pts)]
        normal, length = edge_outward_normal(left, right, orientation)
        for s, weight in zip(gauss_x, gauss_w, strict=True):
            point = (1.0 - s) * left + s * right
            _, gradient = field(float(point[0]), float(point[1]))
            normal_flux = gradient[0] * normal[0] + gradient[1] * normal[1]
            contribution = weight * length * normal_flux
            out[index] += contribution * (1.0 - s)
            out[(index + 1) % len(pts)] += contribution * s
    return out


def exact_smooth_ellipse_flux(n, a, b, field):
    out = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        x = a * math.cos(theta)
        y = b * math.sin(theta)
        _, gradient = field(x, y)
        raw_normal = (math.cos(theta) / a, math.sin(theta) / b)
        normal_scale = math.hypot(raw_normal[0], raw_normal[1])
        out.append((gradient[0] * raw_normal[0] + gradient[1] * raw_normal[1]) / normal_scale)
    return out


def fit_power(ns, errors):
    clean = [(float(n), float(error)) for n, error in zip(ns, errors, strict=True) if error > 0.0 and math.isfinite(error)]
    if len(clean) < 2:
        return None
    xs = [math.log(n) for n, _ in clean]
    ys = [math.log(error) for _, error in clean]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom <= 0.0:
        return None
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = ybar - slope * xbar
    fitted = [intercept + slope * x for x in xs]
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - yhat) ** 2 for y, yhat in zip(ys, fitted, strict=True))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return {"power": slope, "r2": r2, "fit_count": len(clean)}


def run_case(shape, family, make_points, n, field_name, field):
    points = make_points(n)
    values = manufactured_boundary_data(points, field)
    mass = boundary_lumped_mass(points)
    exact_weak = exact_weak_flux(points, field)
    exact_nodal_flux = exact_weak / mass
    qjet, q_build_ms = timed(lambda: build_planar_domain_qjet(points))
    corrected_qjet, corrected_q_build_ms = timed(
        lambda: build_harmonic_moment_corrected_planar_qjet(
            points,
            moment_degree=2,
            zeta_tail_degree=None,
        )
    )
    fem, fem_build_ms = timed(lambda: build_fem_boundary_dtn(points, radial_levels=max(12, n // 2)))
    q_result, q_ms = timed(lambda: qjet.apply_dtn(values))
    corrected_q_result, corrected_q_ms = timed(lambda: corrected_qjet.apply_dtn(values))
    fem_flux, fem_ms = timed(lambda: fem.apply_dtn(values))
    raw_q_flux = np.asarray(tuple(q_result.values), dtype=np.complex128)
    corrected_q_flux = np.asarray(tuple(corrected_q_result.values), dtype=np.complex128)
    fem_flux = np.asarray(tuple(fem_flux), dtype=np.complex128)
    raw_q_error = relative_weighted_l2(raw_q_flux, exact_nodal_flux, mass)
    corrected_q_error = relative_weighted_l2(corrected_q_flux, exact_nodal_flux, mass)
    fem_error = relative_weighted_l2(fem_flux, exact_nodal_flux, mass)
    corrected_q_vs_fem = relative_weighted_l2(corrected_q_flux, fem_flux, mass)
    ellipse_weighted_error = None
    ellipse_weighted_smooth_error = None
    if shape == "ellipse_3_to_1":
        ellipse_flux = ellipse_weighted_dtn(values, 3.0, 1.0)
        ellipse_weighted_error = relative_weighted_l2(ellipse_flux, exact_nodal_flux, mass)
        ellipse_weighted_smooth_error = relative_weighted_l2(
            ellipse_flux,
            exact_smooth_ellipse_flux(n, 3.0, 1.0, field),
            mass,
        )
    exact_norm = math.sqrt(float(np.sum(mass * np.abs(exact_nodal_flux) ** 2)))
    winner = "corrected_q" if corrected_q_error < fem_error else "fem"
    return {
        "shape": shape,
        "family": family,
        "n": n,
        "field": field_name,
        "q_status": corrected_q_result.ledger.status,
        "q_error_type": corrected_q_result.stats.get("q_error_type"),
        "q_recommended_q": corrected_q_result.stats.get("recommended_q"),
        "raw_q_relative_l2_to_exact": raw_q_error,
        "q_relative_l2_to_exact": corrected_q_error,
        "corrected_q_relative_l2_to_exact": corrected_q_error,
        "ellipse_weighted_q_relative_l2_to_polygon_exact": ellipse_weighted_error,
        "ellipse_weighted_q_relative_l2_to_smooth_exact": ellipse_weighted_smooth_error,
        "fem_relative_l2_to_exact": fem_error,
        "q_vs_fem_relative_l2": corrected_q_vs_fem,
        "winner": winner,
        "q_error_over_fem_error": corrected_q_error / max(fem_error, 1.0e-14),
        "raw_q_build_ms": q_build_ms,
        "raw_q_apply_ms": q_ms,
        "q_build_ms": corrected_q_build_ms,
        "q_apply_ms": corrected_q_ms,
        "q_work_units": corrected_q_result.work_units,
        "correction_rank": corrected_q_result.stats.get("correction_rank"),
        "harmonic_moment_rank": corrected_q_result.stats.get("harmonic_moment_rank"),
        "zeta_tail_rank": corrected_q_result.stats.get("zeta_tail_rank"),
        "zeta_tail_degree": corrected_q_result.stats.get("zeta_tail_degree"),
        "fem_build_ms": fem_build_ms,
        "fem_apply_ms": fem_ms,
        "fem_nodes": fem.mesh.node_count,
        "fem_triangles": fem.mesh.triangle_count,
        "fem_radial_levels": fem.mesh.radial_levels,
        "exact_flux_weighted_l2_norm": exact_norm,
        "dense_matrix_stored_by_q": False,
        "dense_matrix_stored_by_fem_baseline": True,
        "truth": "manufactured_harmonic_exact_weak_boundary_flux",
    }


def median(values):
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def summarize(rows):
    by_shape = {}
    for shape in sorted({row["shape"] for row in rows}):
        selected = [row for row in rows if row["shape"] == shape]
        by_shape[shape] = {
            "family": selected[0]["family"],
            "rows": len(selected),
            "raw_q_median_relative_l2_to_exact": median(row["raw_q_relative_l2_to_exact"] for row in selected),
            "q_wins": sum(1 for row in selected if row["winner"] == "corrected_q"),
            "fem_wins": sum(1 for row in selected if row["winner"] == "fem"),
            "median_q_relative_l2_to_exact": median(row["q_relative_l2_to_exact"] for row in selected),
            "median_fem_relative_l2_to_exact": median(row["fem_relative_l2_to_exact"] for row in selected),
            "max_q_relative_l2_to_exact": max(row["q_relative_l2_to_exact"] for row in selected),
            "max_fem_relative_l2_to_exact": max(row["fem_relative_l2_to_exact"] for row in selected),
        }
    by_field = {}
    for field in sorted({row["field"] for row in rows}):
        selected = [row for row in rows if row["field"] == field]
        by_field[field] = {
            "rows": len(selected),
            "raw_q_median_relative_l2_to_exact": median(row["raw_q_relative_l2_to_exact"] for row in selected),
            "q_wins": sum(1 for row in selected if row["winner"] == "corrected_q"),
            "fem_wins": sum(1 for row in selected if row["winner"] == "fem"),
            "median_q_relative_l2_to_exact": median(row["q_relative_l2_to_exact"] for row in selected),
            "median_fem_relative_l2_to_exact": median(row["fem_relative_l2_to_exact"] for row in selected),
        }
    fits = []
    for shape in sorted({row["shape"] for row in rows}):
        for field in sorted({row["field"] for row in rows}):
            selected = sorted(
                [row for row in rows if row["shape"] == shape and row["field"] == field],
                key=lambda row: row["n"],
            )
            if not selected:
                continue
            ns = [row["n"] for row in selected]
            q_fit = fit_power(ns, [row["q_relative_l2_to_exact"] for row in selected])
            fem_fit = fit_power(ns, [row["fem_relative_l2_to_exact"] for row in selected])
            fits.append(
                {
                    "shape": shape,
                    "field": field,
                    "q_error_power": None if q_fit is None else q_fit["power"],
                    "q_error_r2": None if q_fit is None else q_fit["r2"],
                    "fem_error_power": None if fem_fit is None else fem_fit["power"],
                    "fem_error_r2": None if fem_fit is None else fem_fit["r2"],
                    "fit_count": len(selected),
                }
            )
    ellipse_weighted_polygon = [
        row["ellipse_weighted_q_relative_l2_to_polygon_exact"]
        for row in rows
        if row["ellipse_weighted_q_relative_l2_to_polygon_exact"] is not None
    ]
    ellipse_weighted_smooth = [
        row["ellipse_weighted_q_relative_l2_to_smooth_exact"]
        for row in rows
        if row["ellipse_weighted_q_relative_l2_to_smooth_exact"] is not None
    ]
    return {
        "case_count": len(rows),
        "shape_count": len(by_shape),
        "field_count": len(by_field),
        "q_failure_count": sum(1 for row in rows if row["q_status"] != "borrowed_repaid"),
        "q_wins": sum(1 for row in rows if row["winner"] == "corrected_q"),
        "fem_wins": sum(1 for row in rows if row["winner"] == "fem"),
        "median_raw_q_relative_l2_to_exact": median(row["raw_q_relative_l2_to_exact"] for row in rows),
        "max_raw_q_relative_l2_to_exact": max(row["raw_q_relative_l2_to_exact"] for row in rows),
        "median_q_relative_l2_to_exact": median(row["q_relative_l2_to_exact"] for row in rows),
        "median_fem_relative_l2_to_exact": median(row["fem_relative_l2_to_exact"] for row in rows),
        "max_q_relative_l2_to_exact": max(row["q_relative_l2_to_exact"] for row in rows),
        "max_fem_relative_l2_to_exact": max(row["fem_relative_l2_to_exact"] for row in rows),
        "ellipse_weighted_q_median_relative_l2_to_polygon_exact": median(ellipse_weighted_polygon),
        "ellipse_weighted_q_max_relative_l2_to_polygon_exact": max(ellipse_weighted_polygon, default=None),
        "ellipse_weighted_q_median_relative_l2_to_smooth_exact": median(ellipse_weighted_smooth),
        "ellipse_weighted_q_max_relative_l2_to_smooth_exact": max(ellipse_weighted_smooth, default=None),
        "by_shape": by_shape,
        "by_field": by_field,
        "convergence_fits": fits,
    }


def main():
    ns = [32, 64, 128]
    rows = []
    for shape, family, make_points in shape_suite():
        for n in ns:
            for field_name, field in field_suite():
                rows.append(run_case(shape, family, make_points, n, field_name, field))
    payload = {
        "parameters": {
            "boundary_samples": ns,
            "fem_radial_levels": "max(12, n/2)",
            "truth": "analytic manufactured harmonic fields; exact weak boundary flux integrated on polygon edges",
            "comparison_norm": "mass-lumped boundary L2 on nodal flux exact_weak_flux / lumped_mass",
            "fields": [name for name, _ in field_suite()],
            "q_policy": "matrix-free planar chord QJet; no dense Q matrix stored",
            "q_correction": "harmonic moment reproduction through degree 2 plus projected zeta-tail harmonic multipoles",
            "fem_policy": "volumetric P1 radial-fan FEM Schur complement baseline",
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
                "q_wins": payload["summary"]["q_wins"],
                "fem_wins": payload["summary"]["fem_wins"],
                "median_q_relative_l2_to_exact": payload["summary"]["median_q_relative_l2_to_exact"],
                "median_fem_relative_l2_to_exact": payload["summary"]["median_fem_relative_l2_to_exact"],
                "q_failure_count": payload["summary"]["q_failure_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
