#!/usr/bin/env python3
"""Head-to-head QBX benchmark against trapezoid and QJet local bridge."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.quadrature import (  # noqa: E402
    circle_log_layer_spectral,
    circle_log_layer_trapezoid,
    log_layer_local_bridge,
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


def square_polygon():
    return [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def star_polygon():
    return [
        ((1.22 if i % 2 == 0 else 0.58) * math.cos(TAU * i / 10), (1.22 if i % 2 == 0 else 0.58) * math.sin(TAU * i / 10))
        for i in range(10)
    ]


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


def evaluate_shape(name, dense_points, *, n, qbx_n, ref_n, ratios, qbx_order, radius_factor):
    coarse = resample_closed_curve(dense_points, n)
    qbx_points = resample_closed_curve(dense_points, qbx_n)
    reference_points = resample_closed_curve(dense_points, ref_n)
    coarse_density = density_values(n)
    qbx_density = density_values(qbx_n)
    reference_density = density_values(ref_n)
    sample_index = n // 7
    h = perimeter(coarse) / n
    normal = tuple(outward_unit_normals(coarse)[sample_index])
    rows = []

    qbx_sample_index = min(
        range(qbx_n),
        key=lambda index: dist(qbx_points[index], coarse[sample_index]),
    )

    for ratio in ratios:
        target = (
            coarse[sample_index][0] + ratio * h * normal[0],
            coarse[sample_index][1] + ratio * h * normal[1],
        )
        reference, reference_ms = timed(lambda: log_layer_trapezoid(reference_points, reference_density, target))
        trapezoid, trapezoid_ms = timed(lambda: log_layer_trapezoid(coarse, coarse_density, target))
        bridge, bridge_ms = timed(lambda: log_layer_local_bridge(coarse, coarse_density, target, sample_index=sample_index))
        qbx_same, qbx_same_ms = timed(
            lambda: log_layer_qbx_auto(
                coarse,
                coarse_density,
                target,
                sample_index=sample_index,
                order=qbx_order,
                radius_factor=radius_factor,
            )
        )
        qbx_refined, qbx_refined_ms = timed(
            lambda: log_layer_qbx_auto(
                qbx_points,
                qbx_density,
                target,
                sample_index=qbx_sample_index,
                order=qbx_order,
                radius_factor=radius_factor,
            )
        )
        scale = max(abs(reference), 1.0e-14)
        trap_error = abs(trapezoid - reference) / scale
        bridge_error = abs(bridge - reference) / scale
        qbx_same_error = abs(qbx_same - reference) / scale
        qbx_refined_error = abs(qbx_refined - reference) / scale
        rows.append(
            {
                "shape": name,
                "n": n,
                "qbx_n": qbx_n,
                "reference_n": ref_n,
                "delta_over_h": ratio,
                "trapezoid_relative_error": trap_error,
                "bridge_relative_error": bridge_error,
                "qbx_same_n_relative_error": qbx_same_error,
                "qbx_refined_relative_error": qbx_refined_error,
                "bridge_vs_trapezoid": trap_error / bridge_error if bridge_error > 0.0 else float("inf"),
                "qbx_same_vs_trapezoid": trap_error / qbx_same_error if qbx_same_error > 0.0 else float("inf"),
                "qbx_refined_vs_trapezoid": trap_error / qbx_refined_error if qbx_refined_error > 0.0 else float("inf"),
                "reference_ms": reference_ms,
                "trapezoid_ms": trapezoid_ms,
                "bridge_ms": bridge_ms,
                "qbx_same_n_ms": qbx_same_ms,
                "qbx_refined_ms": qbx_refined_ms,
            }
        )
    return rows


def circle_rows():
    n = 4096
    phase = 0.7
    density = [math.cos(TAU * i / n) for i in range(n)]
    rows = []
    for delta in (1.0e-4, 1.0e-6, 1.0e-8):
        point = (1.0 + delta) * complex(math.cos(phase), math.sin(phase))
        exact = -math.pi * math.cos(phase) / abs(point)
        trapezoid, trapezoid_ms = timed(lambda: circle_log_layer_trapezoid(density, point))
        spectral, spectral_ms = timed(lambda: circle_log_layer_spectral(density, point))
        scale = max(abs(exact), 1.0e-14)
        rows.append(
            {
                "shape": "circle_exact_spectral",
                "n": n,
                "delta": delta,
                "trapezoid_relative_error": abs(trapezoid - exact) / scale,
                "spectral_qjet_relative_error": abs(spectral - exact) / scale,
                "trapezoid_ms": trapezoid_ms,
                "spectral_qjet_ms": spectral_ms,
            }
        )
    return rows


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main():
    n = 512
    qbx_n = 4096
    ref_n = 65536
    qbx_order = 50
    radius_factor = 4.0
    ratios = (0.5, 0.2, 0.1, 0.05)
    dense_samples = 8192
    shape_defs = [
        ("ellipse", ellipse_dense(dense_samples)),
        ("conformal_hard_like", star_terms_dense(dense_samples, ((2, 0.32, 0.0), (5, -0.10, 0.0)))),
        ("pinched_star", star_terms_dense(dense_samples, ((2, 0.22, 0.0), (5, 0.08, -0.06), (7, -0.04, 0.03)))),
        ("skew_propeller", star_terms_dense(dense_samples, ((3, 0.0, 0.18), (4, -0.12, 0.0), (8, 0.03, -0.05)))),
        ("rounded_square", superellipse_dense(dense_samples)),
        ("square_polygon", square_polygon()),
        ("star_polygon", star_polygon()),
        ("naca0012_airfoil", naca0012_dense(dense_samples)),
    ]
    rows = []
    for name, points in shape_defs:
        rows.extend(
            evaluate_shape(
                name,
                points,
                n=n,
                qbx_n=qbx_n,
                ref_n=ref_n,
                ratios=ratios,
                qbx_order=qbx_order,
                radius_factor=radius_factor,
            )
        )
    payload = {
        "parameters": {
            "n": n,
            "qbx_n": qbx_n,
            "reference_n": ref_n,
            "qbx_order": qbx_order,
            "qbx_radius_factor": radius_factor,
            "delta_over_h": list(ratios),
        },
        "summary": {
            "case_count": len(rows),
            "median_bridge_vs_trapezoid": median([row["bridge_vs_trapezoid"] for row in rows]),
            "median_qbx_same_vs_trapezoid": median([row["qbx_same_vs_trapezoid"] for row in rows]),
            "median_qbx_refined_vs_trapezoid": median([row["qbx_refined_vs_trapezoid"] for row in rows]),
            "max_bridge_relative_error": max(row["bridge_relative_error"] for row in rows),
            "max_qbx_same_n_relative_error": max(row["qbx_same_n_relative_error"] for row in rows),
            "max_qbx_refined_relative_error": max(row["qbx_refined_relative_error"] for row in rows),
        },
        "rows": rows,
        "circle_exact_rows": circle_rows(),
    }
    output = ROOT / "docs" / "assets" / "qbx_head_to_head_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("shape,delta_over_h,trap_rel,bridge_rel,qbx_same_rel,qbx_refined_rel,bridge_x,qbx_same_x,qbx_refined_x")
    for row in rows:
        print(
            f"{row['shape']},{row['delta_over_h']},"
            f"{row['trapezoid_relative_error']:.6e},{row['bridge_relative_error']:.6e},"
            f"{row['qbx_same_n_relative_error']:.6e},{row['qbx_refined_relative_error']:.6e},"
            f"{row['bridge_vs_trapezoid']:.2f},{row['qbx_same_vs_trapezoid']:.2f},"
            f"{row['qbx_refined_vs_trapezoid']:.2f}"
        )
    print("\nsummary=" + json.dumps(payload["summary"], sort_keys=True))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
