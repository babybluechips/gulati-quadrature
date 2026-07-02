#!/usr/bin/env python3
"""Hard-case benchmark harness for the planar QJet quadrature engine.

The engine under test is the 2D logarithmic layer-potential quadrature engine
in ``inverse_shape.quadrature``.  This script intentionally treats cylinders as
circle cross-sections and reports true 3D surface families as unsupported,
rather than silently projecting them into a different mathematical problem.
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
    build_boundary_qjet,
    circle_log_layer_spectral,
    circle_log_layer_trapezoid,
    log_layer_local_bridge,
    log_layer_trapezoid,
    near_singular_circle_table,
    outward_unit_normals,
)


TAU = 2.0 * math.pi


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def perimeter(points: list[tuple[float, float]]) -> float:
    return sum(dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def resample_closed_curve(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    if n < 3:
        raise ValueError("n must be at least 3")
    lengths = [dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    total = sum(lengths)
    if total <= 0.0:
        raise ValueError("degenerate curve")
    out: list[tuple[float, float]] = []
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


def radial_dense(samples: int, radius_fn) -> list[tuple[float, float]]:
    points = []
    for i in range(samples):
        theta = TAU * i / samples
        r = radius_fn(theta)
        points.append((r * math.cos(theta), r * math.sin(theta)))
    return points


def ellipse_dense(samples: int) -> list[tuple[float, float]]:
    return [(1.8 * math.cos(TAU * i / samples), 0.55 * math.sin(TAU * i / samples)) for i in range(samples)]


def flower_dense(samples: int) -> list[tuple[float, float]]:
    return radial_dense(
        samples,
        lambda t: 1.0 + 0.32 * math.cos(3.0 * t) - 0.18 * math.sin(5.0 * t) + 0.08 * math.cos(8.0 * t),
    )


def kidney_dense(samples: int) -> list[tuple[float, float]]:
    return radial_dense(
        samples,
        lambda t: 1.0 + 0.38 * math.cos(t) - 0.24 * math.sin(2.0 * t) + 0.08 * math.cos(4.0 * t),
    )


def superellipse_dense(samples: int) -> list[tuple[float, float]]:
    exponent = 0.5
    points = []
    for i in range(samples):
        theta = TAU * i / samples
        c = math.cos(theta)
        s = math.sin(theta)
        x = 1.35 * (1.0 if c >= 0.0 else -1.0) * abs(c) ** exponent
        y = 0.82 * (1.0 if s >= 0.0 else -1.0) * abs(s) ** exponent
        points.append((x, y))
    return points


def square_polygon() -> list[tuple[float, float]]:
    return [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def star_polygon() -> list[tuple[float, float]]:
    points = []
    for i in range(10):
        theta = TAU * i / 10
        radius = 1.22 if i % 2 == 0 else 0.58
        points.append((radius * math.cos(theta), radius * math.sin(theta)))
    return points


def naca0012_dense(samples: int) -> list[tuple[float, float]]:
    half = max(32, samples // 2)
    xs = [0.5 * (1.0 - math.cos(math.pi * i / (half - 1))) for i in range(half)]

    def thickness(x: float) -> float:
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


def airplane_silhouette_dense(samples: int) -> list[tuple[float, float]]:
    return radial_dense(
        samples,
        lambda t: 1.0
        + 0.26 * math.cos(t)
        - 0.18 * math.cos(2.0 * t)
        + 0.20 * math.sin(3.0 * t)
        - 0.12 * math.cos(5.0 * t),
    )


def density_values(n: int) -> list[float]:
    return [
        1.0
        + 0.35 * math.cos(3.0 * TAU * i / n + 0.2)
        - 0.20 * math.sin(5.0 * TAU * i / n - 0.1)
        for i in range(n)
    ]


def add_scaled(point, vector, scale):
    return (point[0] + scale * vector[0], point[1] + scale * vector[1])


def timed(callable_):
    start = time.perf_counter()
    value = callable_()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return value, elapsed_ms


def planar_case(name: str, dense_points: list[tuple[float, float]], *, n: int, ref_n: int, delta_over_h: float):
    coarse = resample_closed_curve(dense_points, n)
    reference = resample_closed_curve(dense_points, ref_n)
    density = density_values(n)
    ref_density = density_values(ref_n)
    sample_index = n // 7
    h = perimeter(coarse) / n
    normal = tuple(outward_unit_normals(coarse)[sample_index])
    target = add_scaled(coarse[sample_index], normal, delta_over_h * h)

    ref_value, ref_ms = timed(lambda: log_layer_trapezoid(reference, ref_density, target))
    trap_value, trap_ms = timed(lambda: log_layer_trapezoid(coarse, density, target))
    bridge_value, bridge_ms = timed(
        lambda: log_layer_local_bridge(coarse, density, target, sample_index=sample_index)
    )
    qjet = build_boundary_qjet(coarse)
    q_values = [math.cos(2.0 * TAU * i / n) - 0.25 * math.sin(7.0 * TAU * i / n) for i in range(n)]
    _, q_apply_ms = timed(lambda: qjet.apply(q_values))

    trap_error = abs(trap_value - ref_value)
    bridge_error = abs(bridge_value - ref_value)
    improvement = trap_error / bridge_error if bridge_error > 0.0 else float("inf")
    return {
        "case": name,
        "status": "ok",
        "dimension": "2D planar boundary",
        "n": n,
        "ref_n": ref_n,
        "delta_over_h": delta_over_h,
        "reference_ms": ref_ms,
        "trapezoid_ms": trap_ms,
        "bridge_ms": bridge_ms,
        "q_apply_ms": q_apply_ms,
        "trapezoid_abs_error": float(trap_error),
        "bridge_abs_error": float(bridge_error),
        "bridge_improvement": float(improvement),
    }


def circle_spectral_rows():
    rows = []
    n = 4096
    phase = 0.7
    density = [math.cos(TAU * i / n) for i in range(n)]
    for delta in (1e-2, 1e-4, 1e-6, 1e-8):
        point = (1.0 + delta) * complex(math.cos(phase), math.sin(phase))
        exact = -math.pi * math.cos(phase) / abs(point)
        trap, trap_ms = timed(lambda: circle_log_layer_trapezoid(density, point))
        spectral, spectral_ms = timed(lambda: circle_log_layer_spectral(density, point))
        scale = max(abs(exact), 2.2250738585072014e-308)
        rows.append(
            {
                "case": "circle_or_cylinder_cross_section",
                "status": "ok",
                "dimension": "2D planar boundary",
                "n": n,
                "delta": delta,
                "trapezoid_ms": trap_ms,
                "spectral_ms": spectral_ms,
                "trapezoid_relative_error": abs(trap - exact) / scale,
                "spectral_relative_error": abs(spectral - exact) / scale,
            }
        )
    return rows


def unsupported_surface_rows():
    reason = (
        "unsupported by the current planar logarithmic layer engine: requires a "
        "3D surface kernel, surface QJets, and surface quadrature reference"
    )
    return [
        {"case": "finite_cylinder_surface", "status": "unsupported", "dimension": "3D surface", "reason": reason},
        {"case": "cube_polyhedron", "status": "unsupported", "dimension": "3D surface", "reason": reason},
        {"case": "tetrahedron_polyhedron", "status": "unsupported", "dimension": "3D surface", "reason": reason},
        {"case": "airplane_3d_mesh", "status": "unsupported", "dimension": "3D surface", "reason": reason},
        {"case": "torus_higher_genus_surface", "status": "unsupported", "dimension": "3D surface", "reason": reason},
    ]


def main() -> int:
    n = 512
    ref_n = 32768
    delta_over_h = 0.05
    dense_samples = 8192
    cases = [
        ("ellipse_3p3_to_1", ellipse_dense(dense_samples)),
        ("funky_flower_curve", flower_dense(dense_samples)),
        ("kidney_nonconvex_curve", kidney_dense(dense_samples)),
        ("rounded_square_superellipse", superellipse_dense(dense_samples)),
        ("square_polygon", square_polygon()),
        ("star_polygon", star_polygon()),
        ("naca0012_airfoil", naca0012_dense(dense_samples)),
        ("airplane_planar_silhouette", airplane_silhouette_dense(dense_samples)),
    ]

    planar_rows = [planar_case(name, points, n=n, ref_n=ref_n, delta_over_h=delta_over_h) for name, points in cases]
    circle_rows = circle_spectral_rows()
    unsupported_rows = unsupported_surface_rows()
    payload = {
        "engine": "self-contained planar QJet quadrature",
        "planar_hard_cases": planar_rows,
        "circle_spectral_cases": circle_rows,
        "surface_support_audit": unsupported_rows,
        "existing_pressure_table": near_singular_circle_table(n=1024, deltas=(1e-2, 1e-4, 1e-6)),
    }

    output = ROOT / "docs" / "assets" / "qjet_quadrature_hard_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("case,status,n,ref_n,delta_over_h,trap_err,bridge_err,improvement,trap_ms,bridge_ms,q_apply_ms")
    for row in planar_rows:
        print(
            f"{row['case']},{row['status']},{row['n']},{row['ref_n']},{row['delta_over_h']},"
            f"{row['trapezoid_abs_error']:.6e},{row['bridge_abs_error']:.6e},"
            f"{row['bridge_improvement']:.3f},{row['trapezoid_ms']:.3f},"
            f"{row['bridge_ms']:.3f},{row['q_apply_ms']:.3f}"
        )
    print("\ncase,status,n,delta,trap_rel_err,spectral_rel_err,trap_ms,spectral_ms")
    for row in circle_rows:
        print(
            f"{row['case']},{row['status']},{row['n']},{row['delta']},"
            f"{row['trapezoid_relative_error']:.6e},{row['spectral_relative_error']:.6e},"
            f"{row['trapezoid_ms']:.3f},{row['spectral_ms']:.3f}"
        )
    print("\nunsupported_surface_case,status,reason")
    for row in unsupported_rows:
        print(f"{row['case']},{row['status']},{row['reason']}")
    print(f"\njson={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
