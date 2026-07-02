#!/usr/bin/env python3
"""Fit empirical scaling exponents for point-QBX.

Fits two useful finite-range laws:

* order sweep:      error ~= C exp(-alpha * order)
* sample sweep:     error ~= C qbx_n^-p

The fits are empirical over the selected near-boundary benchmark regime; they
are meant to identify active scaling behavior and floors, not prove asymptotic
rates.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.quadrature import (  # noqa: E402
    log_layer_multipole_zeta_q_borrow_compute_repay,
    log_layer_qbx_auto,
    log_layer_trapezoid,
    outward_unit_normals,
)

TAU = 2.0 * math.pi


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def perimeter(points):
    return sum(dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def resample_closed_curve(points, n):
    lengths = [dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    total = sum(lengths)
    if total <= 0.0:
        raise ValueError("degenerate curve")
    out = []
    edge = 0
    edge_start = 0.0
    for k in range(n):
        target = total * k / n
        while edge + 1 < len(points) and edge_start + lengths[edge] < target:
            edge_start += lengths[edge]
            edge += 1
        span = lengths[edge]
        alpha = 0.0 if span <= 0.0 else (target - edge_start) / span
        a = points[edge]
        b = points[(edge + 1) % len(points)]
        out.append((a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1])))
    return out


def radial_dense(samples, radius_fn):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        radius = radius_fn(theta)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def star_terms_dense(samples, terms):
    def radius(theta):
        value = 1.0
        for mode, cos_coeff, sin_coeff in terms:
            value += cos_coeff * math.cos(mode * theta) + sin_coeff * math.sin(mode * theta)
        return value

    return radial_dense(samples, radius)


def ellipse_dense(samples):
    return [(1.8 * math.cos(TAU * i / samples), 0.55 * math.sin(TAU * i / samples)) for i in range(samples)]


def superellipse_dense(samples):
    exponent = 0.5
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        c = math.cos(theta)
        s = math.sin(theta)
        out.append(
            (
                1.35 * (1.0 if c >= 0.0 else -1.0) * abs(c) ** exponent,
                0.82 * (1.0 if s >= 0.0 else -1.0) * abs(s) ** exponent,
            )
        )
    return out


def square_polygon():
    return [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def star_polygon():
    out = []
    for i in range(10):
        radius = 1.22 if i % 2 == 0 else 0.58
        out.append((radius * math.cos(TAU * i / 10), radius * math.sin(TAU * i / 10)))
    return out


def naca0012_dense(samples):
    half = max(32, samples // 2)
    xs = [0.5 * (1.0 - math.cos(math.pi * i / (half - 1))) for i in range(half)]

    def thickness(x):
        t = 0.12
        return 5.0 * t * (
            0.2969 * math.sqrt(max(x, 0.0))
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x * x * x
            - 0.1015 * x * x * x * x
        )

    upper = [(2.0 * (x - 0.5), 2.0 * thickness(x)) for x in reversed(xs)]
    lower = [(2.0 * (x - 0.5), -2.0 * thickness(x)) for x in xs[1:-1]]
    return upper + lower


def density_values(n):
    return [
        1.0
        + 0.35 * math.cos(3.0 * TAU * i / n + 0.2)
        - 0.20 * math.sin(5.0 * TAU * i / n - 0.1)
        for i in range(n)
    ]


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, 1000.0 * (time.perf_counter() - start)


def linear_fit(xs, ys):
    if len(xs) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "count": len(xs)}
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    sxx = sum((x - xbar) * (x - xbar) for x in xs)
    if sxx <= 0.0:
        return {"slope": 0.0, "intercept": ybar, "r2": 0.0, "count": len(xs)}
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    ss_tot = sum((y - ybar) * (y - ybar) for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "count": len(xs)}


def active_rows(rows, error_key):
    positive = [row for row in rows if row[error_key] > 0.0 and math.isfinite(row[error_key])]
    if len(positive) <= 3:
        return positive
    floor = min(row[error_key] for row in positive)
    threshold = max(2.0 * floor, 1.0e-13)
    active = [row for row in positive if row[error_key] >= threshold]
    return active if len(active) >= 2 else positive


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def target_for_shape(coarse, delta_over_h):
    index = len(coarse) // 7
    h = perimeter(coarse) / len(coarse)
    normal = tuple(outward_unit_normals(coarse)[index])
    return (coarse[index][0] + delta_over_h * h * normal[0], coarse[index][1] + delta_over_h * h * normal[1])


def nearest_index(points, target):
    return min(range(len(points)), key=lambda index: dist(points[index], target))


def evaluate_qbx(points, density, target, reference, order, radius_factor):
    value, elapsed_ms = timed(lambda: log_layer_qbx_auto(points, density, target, order=order, radius_factor=radius_factor))
    return abs(value - reference) / max(abs(reference), 1.0e-14), elapsed_ms


def evaluate_multipole_zeta(dense_points, target, reference, base_n, order, leaf_size):
    levels = (base_n, 2 * base_n, 4 * base_n)
    point_sets = [resample_closed_curve(dense_points, level_n) for level_n in levels]
    density_sets = [density_values(level_n) for level_n in levels]
    sample_indices = [nearest_index(points, target) for points in point_sets]
    evaluation, elapsed_ms = timed(
        lambda: log_layer_multipole_zeta_q_borrow_compute_repay(
            point_sets,
            density_sets,
            target,
            sample_indices=sample_indices,
            order=order,
            leaf_size=leaf_size,
        )
    )
    error = abs(evaluation.value - reference) / max(abs(reference), 1.0e-14)
    return error, elapsed_ms, evaluation


def run_shape(name, dense_points, parameters):
    coarse = resample_closed_curve(dense_points, parameters["coarse_n"])
    target = target_for_shape(coarse, parameters["delta_over_h"])
    reference_points = resample_closed_curve(dense_points, parameters["reference_n"])
    reference_density = density_values(parameters["reference_n"])
    reference, reference_ms = timed(lambda: log_layer_trapezoid(reference_points, reference_density, target))

    order_points = resample_closed_curve(dense_points, parameters["order_sweep_qbx_n"])
    order_density = density_values(parameters["order_sweep_qbx_n"])
    order_rows = []
    for order in parameters["orders"]:
        error, elapsed_ms = evaluate_qbx(
            order_points,
            order_density,
            target,
            reference,
            order,
            parameters["order_sweep_radius_factor"],
        )
        order_rows.append({"shape": name, "order": order, "relative_error": error, "ms": elapsed_ms})

    sample_rows = []
    for qbx_n in parameters["sample_counts"]:
        sample_points = resample_closed_curve(dense_points, qbx_n)
        sample_density = density_values(qbx_n)
        error, elapsed_ms = evaluate_qbx(
            sample_points,
            sample_density,
            target,
            reference,
            parameters["sample_sweep_order"],
            parameters["sample_sweep_radius_factor"],
        )
        sample_rows.append({"shape": name, "qbx_n": qbx_n, "relative_error": error, "ms": elapsed_ms})

    zeta_rows = []
    for base_n in parameters["zeta_base_counts"]:
        error, elapsed_ms, evaluation = evaluate_multipole_zeta(
            dense_points,
            target,
            reference,
            base_n,
            parameters["zeta_order"],
            parameters["zeta_leaf_size"],
        )
        zeta_rows.append(
            {
                "shape": name,
                "base_n": base_n,
                "levels": [base_n, 2 * base_n, 4 * base_n],
                "relative_error": error,
                "ms": elapsed_ms,
                "estimated_zeta_exponent": evaluation.estimated_zeta_exponent,
                "cached_target_work_units": evaluation.cached_target_work_units,
                "moment_build_units": evaluation.moment_build_units,
                "single_target_work_units": evaluation.single_target_work_units,
            }
        )

    order_active = active_rows(order_rows, "relative_error")
    order_fit = linear_fit(
        [row["order"] for row in order_active],
        [math.log(row["relative_error"]) for row in order_active],
    )
    sample_active = active_rows(sample_rows, "relative_error")
    sample_fit = linear_fit(
        [math.log(row["qbx_n"]) for row in sample_active],
        [math.log(row["relative_error"]) for row in sample_active],
    )
    order_time_fit = linear_fit(
        [math.log(row["order"]) for row in order_rows],
        [math.log(max(row["ms"], 1.0e-9)) for row in order_rows],
    )
    sample_time_fit = linear_fit(
        [math.log(row["qbx_n"]) for row in sample_rows],
        [math.log(max(row["ms"], 1.0e-9)) for row in sample_rows],
    )
    qbx_work_fit = linear_fit(
        [math.log(row["qbx_n"]) for row in sample_rows],
        [math.log(row["qbx_n"] * parameters["sample_sweep_order"]) for row in sample_rows],
    )
    zeta_active = active_rows(zeta_rows, "relative_error")
    zeta_fit = linear_fit(
        [math.log(row["base_n"]) for row in zeta_active],
        [math.log(row["relative_error"]) for row in zeta_active],
    )
    zeta_time_fit = linear_fit(
        [math.log(row["base_n"]) for row in zeta_rows],
        [math.log(max(row["ms"], 1.0e-9)) for row in zeta_rows],
    )
    zeta_cached_work_fit = linear_fit(
        [math.log(row["base_n"]) for row in zeta_rows],
        [math.log(max(row["cached_target_work_units"], 1.0e-9)) for row in zeta_rows],
    )
    zeta_single_work_fit = linear_fit(
        [math.log(row["base_n"]) for row in zeta_rows],
        [math.log(max(row["single_target_work_units"], 1.0e-9)) for row in zeta_rows],
    )

    return {
        "shape": name,
        "reference_ms": reference_ms,
        "order_rows": order_rows,
        "sample_rows": sample_rows,
        "zeta_rows": zeta_rows,
        "fits": {
            "order_error_alpha": -order_fit["slope"],
            "order_error_effective_ratio": math.exp(order_fit["slope"]),
            "order_error_r2": order_fit["r2"],
            "order_error_fit_count": order_fit["count"],
            "sample_error_power": -sample_fit["slope"],
            "sample_error_r2": sample_fit["r2"],
            "sample_error_fit_count": sample_fit["count"],
            "order_time_power": order_time_fit["slope"],
            "sample_time_power": sample_time_fit["slope"],
            "sample_work_power": qbx_work_fit["slope"],
            "zeta_error_power": -zeta_fit["slope"],
            "zeta_error_r2": zeta_fit["r2"],
            "zeta_error_fit_count": zeta_fit["count"],
            "zeta_time_power": zeta_time_fit["slope"],
            "zeta_cached_work_power": zeta_cached_work_fit["slope"],
            "zeta_single_work_power": zeta_single_work_fit["slope"],
        },
    }


def main():
    parameters = {
        "coarse_n": 512,
        "reference_n": 65536,
        "delta_over_h": 0.1,
        "order_sweep_radius_factor": 128.0,
        "sample_sweep_radius_factor": 4.0,
        "orders": [8, 12, 16, 24, 32, 48, 64, 80],
        "order_sweep_qbx_n": 8192,
        "sample_counts": [512, 1024, 2048, 4096, 8192, 16384],
        "sample_sweep_order": 64,
        "zeta_base_counts": [128, 256, 512, 1024, 2048],
        "zeta_order": 18,
        "zeta_leaf_size": 32,
    }
    dense_samples = 8192
    shapes = [
        ("ellipse", ellipse_dense(dense_samples)),
        ("conformal_hard_like", star_terms_dense(dense_samples, ((2, 0.32, 0.0), (5, -0.10, 0.0)))),
        ("pinched_star", star_terms_dense(dense_samples, ((2, 0.22, 0.0), (5, 0.08, -0.06), (7, -0.04, 0.03)))),
        ("skew_propeller", star_terms_dense(dense_samples, ((3, 0.0, 0.18), (4, -0.12, 0.0), (8, 0.03, -0.05)))),
        ("rounded_square", superellipse_dense(dense_samples)),
        ("square_polygon", square_polygon()),
        ("star_polygon", star_polygon()),
        ("naca0012_airfoil", naca0012_dense(dense_samples)),
    ]
    shape_results = [run_shape(name, points, parameters) for name, points in shapes]
    summary = {
        "shape_count": len(shape_results),
        "median_order_error_alpha": median([row["fits"]["order_error_alpha"] for row in shape_results]),
        "median_order_effective_ratio": median([row["fits"]["order_error_effective_ratio"] for row in shape_results]),
        "median_sample_error_power": median([row["fits"]["sample_error_power"] for row in shape_results]),
        "median_order_time_power": median([row["fits"]["order_time_power"] for row in shape_results]),
        "median_sample_time_power": median([row["fits"]["sample_time_power"] for row in shape_results]),
        "median_sample_work_power": median([row["fits"]["sample_work_power"] for row in shape_results]),
        "median_zeta_error_power": median([row["fits"]["zeta_error_power"] for row in shape_results]),
        "median_zeta_time_power": median([row["fits"]["zeta_time_power"] for row in shape_results]),
        "median_zeta_cached_work_power": median([row["fits"]["zeta_cached_work_power"] for row in shape_results]),
        "median_zeta_single_work_power": median([row["fits"]["zeta_single_work_power"] for row in shape_results]),
    }
    payload = {"parameters": parameters, "summary": summary, "shape_results": shape_results}
    output = ROOT / "docs" / "assets" / "qbx_scaling_fit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "shape,alpha_order,effective_ratio,qbx_sample_power,zeta_sample_power,"
        "qbx_time_power,zeta_time_power,qbx_work_power,zeta_cached_work_power,zeta_single_work_power,"
        "order_r2,qbx_sample_r2,zeta_sample_r2"
    )
    for result in shape_results:
        fit = result["fits"]
        print(
            f"{result['shape']},{fit['order_error_alpha']:.4f},"
            f"{fit['order_error_effective_ratio']:.4f},{fit['sample_error_power']:.3f},"
            f"{fit['zeta_error_power']:.3f},{fit['sample_time_power']:.3f},"
            f"{fit['zeta_time_power']:.3f},{fit['sample_work_power']:.3f},"
            f"{fit['zeta_cached_work_power']:.3f},{fit['zeta_single_work_power']:.3f},"
            f"{fit['order_error_r2']:.3f},{fit['sample_error_r2']:.3f},{fit['zeta_error_r2']:.3f}"
        )
    print("summary=" + json.dumps(summary, sort_keys=True))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
