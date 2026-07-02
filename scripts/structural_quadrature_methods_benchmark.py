#!/usr/bin/env python3
"""Compare structurally different near-singular log-layer quadratures."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.quadrature import (  # noqa: E402
    log_layer_adaptive_panel_borrow_compute_repay,
    log_layer_local_bridge,
    log_layer_multipole_zeta_q_borrow_compute_repay,
    log_layer_qbx_auto,
    log_layer_singularity_subtraction_borrow_compute_repay,
    log_layer_trapezoid,
    outward_unit_normals,
    q_spectral_error_signature,
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


def cardioid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        radius = 1.0 - math.cos(theta)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def astroid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        c = math.cos(theta)
        s = math.sin(theta)
        out.append((c * c * c, s * s * s))
    return out


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


def safe_eval(fn):
    try:
        value, elapsed_ms = timed(fn)
        stats = getattr(value, "stats", {})
        if hasattr(value, "value"):
            value = value.value
        return {"ok": True, "value": value, "ms": elapsed_ms, "error": "", "stats": stats}
    except Exception as exc:  # noqa: BLE001 - benchmark records numerical failure modes.
        return {"ok": False, "value": None, "ms": 0.0, "error": f"{type(exc).__name__}: {exc}", "stats": {}}


def nearest_index(points, target):
    return min(range(len(points)), key=lambda index: dist(points[index], target))


def target_from_sample(points, sample_index, ratio):
    h = perimeter(points) / len(points)
    normal = tuple(outward_unit_normals(points)[sample_index])
    base = points[sample_index]
    return (base[0] + ratio * h * normal[0], base[1] + ratio * h * normal[1])


def sample_index_for_mode(n, mode):
    if mode == "tip":
        return 0
    if mode == "near_tip":
        return max(2, n // 128)
    if mode == "smooth":
        return n // 7
    raise ValueError("unknown target mode")


def relative_error(result, reference):
    if not result["ok"]:
        return None
    return abs(result["value"] - reference) / max(abs(reference), 1.0e-14)


def method_payload(name, result, reference, trap_error):
    rel = relative_error(result, reference)
    payload = {
        f"{name}_ok": result["ok"],
        f"{name}_ms": result["ms"],
        f"{name}_relative_error": rel,
        f"{name}_failure": result["error"],
        f"{name}_stats": result["stats"],
        f"{name}_work_units": result["stats"].get("work_units"),
    }
    if rel is not None and rel > 0.0 and trap_error is not None:
        payload[f"{name}_improvement_vs_trap"] = trap_error / rel
    else:
        payload[f"{name}_improvement_vs_trap"] = None
    return payload


def multipole_zeta_eval(dense_points, target, *, levels, order, leaf_size):
    point_sets = []
    density_sets = []
    sample_indices = []
    for level_n in levels:
        points = resample_closed_curve(dense_points, level_n)
        point_sets.append(points)
        density_sets.append(density_values(level_n))
        sample_indices.append(nearest_index(points, target))
    return log_layer_multipole_zeta_q_borrow_compute_repay(
        point_sets,
        density_sets,
        target,
        sample_indices=sample_indices,
        order=order,
        leaf_size=leaf_size,
    )


def run_case(shape_name, family, dense_points, target_mode, ratio, parameters, q_signature):
    n = parameters["n"]
    qbx_n = parameters["qbx_n"]
    reference_n = parameters["reference_n"]
    coarse = resample_closed_curve(dense_points, n)
    qbx_points = resample_closed_curve(dense_points, qbx_n)
    reference_points = resample_closed_curve(dense_points, reference_n)
    coarse_density = density_values(n)
    qbx_density = density_values(qbx_n)
    reference_density = density_values(reference_n)
    sample_index = sample_index_for_mode(n, target_mode)
    target = target_from_sample(coarse, sample_index, ratio)
    qbx_index = nearest_index(qbx_points, target)

    reference, reference_ms = timed(lambda: log_layer_trapezoid(reference_points, reference_density, target))
    trapezoid = safe_eval(lambda: log_layer_trapezoid(coarse, coarse_density, target))
    trap_error = relative_error(trapezoid, reference)
    singularity = safe_eval(
        lambda: log_layer_singularity_subtraction_borrow_compute_repay(
            coarse,
            coarse_density,
            target,
            sample_index=sample_index,
            window=parameters["singularity_window"],
        )
    )
    adaptive_panel = safe_eval(
        lambda: log_layer_adaptive_panel_borrow_compute_repay(
            coarse,
            coarse_density,
            target,
            sample_index=sample_index,
            panel_radius=parameters["panel_radius"],
            subdivisions=parameters["panel_subdivisions"],
        )
    )
    gulati_bridge = safe_eval(
        lambda: log_layer_local_bridge(coarse, coarse_density, target, sample_index=sample_index)
    )
    multipole_zeta = safe_eval(
        lambda: multipole_zeta_eval(
            dense_points,
            target,
            levels=parameters["multipole_zeta_levels"],
            order=parameters["multipole_zeta_order"],
            leaf_size=parameters["multipole_zeta_leaf_size"],
        )
    )
    qbx_refined = safe_eval(
        lambda: log_layer_qbx_auto(
            qbx_points,
            qbx_density,
            target,
            sample_index=qbx_index,
            order=parameters["qbx_order"],
            radius_factor=parameters["qbx_radius_factor"],
        )
    )

    row = {
        "shape": shape_name,
        "family": family,
        "target_mode": target_mode,
        "delta_over_h": ratio,
        "q_spectral_error_type": q_signature.error_type,
        "q_spectral_symbol_power": q_signature.symbol_power,
        "q_spectral_median_pair_split": q_signature.median_pair_split,
        "q_spectral_max_pair_split": q_signature.max_pair_split,
        "q_spectral_symbol_variation": q_signature.normalized_symbol_variation,
        "q_spectral_recommended_q": q_signature.recommended_q,
        "n": n,
        "qbx_n": qbx_n,
        "reference_n": reference_n,
        "reference_ms": reference_ms,
        "reference": float(reference.real) if isinstance(reference, complex) else float(reference),
    }
    for name, result in (
        ("trapezoid", trapezoid),
        ("singularity_subtraction", singularity),
        ("adaptive_panel", adaptive_panel),
        ("gulati_q_bridge", gulati_bridge),
        ("multipole_zeta_q", multipole_zeta),
        ("qbx_refined", qbx_refined),
    ):
        row.update(method_payload(name, result, reference, trap_error))
    row["trapezoid_work_units"] = n
    row["gulati_q_bridge_work_units"] = n + 1
    row["qbx_refined_work_units"] = qbx_n * parameters["qbx_order"]
    return row


def median(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fmt(value):
    if value is None:
        return "failed"
    return f"{value:.6e}" if isinstance(value, float) else str(value)


def fmt_ratio(value):
    if value is None:
        return "failed"
    return f"{value:.2f}"


def method_summary(rows, method):
    errors = [row[f"{method}_relative_error"] for row in rows]
    return {
        "median_relative_error": median(errors),
        "max_relative_error": max(value for value in errors if value is not None),
        "median_improvement_vs_trap": median([row[f"{method}_improvement_vs_trap"] for row in rows]),
        "median_ms": median([row[f"{method}_ms"] for row in rows]),
        "median_work_units": median([row[f"{method}_work_units"] for row in rows]),
        "failure_count": sum(not row[f"{method}_ok"] for row in rows),
    }


def main():
    parameters = {
        "n": 512,
        "qbx_n": 4096,
        "reference_n": 65536,
        "singularity_window": 10,
        "panel_radius": 8,
        "panel_subdivisions": 24,
        "multipole_zeta_levels": [512, 1024, 2048],
        "multipole_zeta_order": 18,
        "multipole_zeta_leaf_size": 32,
        "qbx_order": 60,
        "qbx_radius_factor": 4.0,
        "delta_over_h": [0.2, 0.1, 0.05],
    }
    dense_samples = 8192
    shape_defs = [
        ("ellipse", "smooth", ellipse_dense(dense_samples), ("smooth",)),
        (
            "conformal_hard_like",
            "smooth",
            star_terms_dense(dense_samples, ((2, 0.32, 0.0), (5, -0.10, 0.0))),
            ("smooth",),
        ),
        ("rounded_square", "cornered_smooth", superellipse_dense(dense_samples), ("smooth",)),
        ("square_polygon", "polygon", square_polygon(), ("smooth",)),
        ("star_polygon", "polygon", star_polygon(), ("smooth",)),
        ("naca0012_airfoil", "airfoil", naca0012_dense(dense_samples), ("smooth",)),
        ("cardioid_single_cusp", "cusp", cardioid_dense(dense_samples), ("tip", "near_tip")),
        ("astroid_four_cusps", "cusp", astroid_dense(dense_samples), ("tip", "near_tip")),
    ]

    rows = []
    signatures = {}
    for shape_name, family, points, target_modes in shape_defs:
        signature = q_spectral_error_signature(
            resample_closed_curve(points, parameters["n"]),
            mode_start=4,
            mode_stop=32,
        )
        signatures[shape_name] = signature.stats
        for target_mode in target_modes:
            for ratio in parameters["delta_over_h"]:
                rows.append(run_case(shape_name, family, points, target_mode, ratio, parameters, signature))

    methods = (
        "trapezoid",
        "singularity_subtraction",
        "adaptive_panel",
        "gulati_q_bridge",
        "multipole_zeta_q",
        "qbx_refined",
    )
    summary = {
        "case_count": len(rows),
        "methods": {method: method_summary(rows, method) for method in methods},
        "families": {
            family: {method: method_summary([row for row in rows if row["family"] == family], method) for method in methods}
            for family in sorted({row["family"] for row in rows})
        },
    }
    payload = {
        "parameters": parameters,
        "q_spectral_signatures": signatures,
        "summary": summary,
        "rows": rows,
    }
    output = ROOT / "docs" / "assets" / "structural_quadrature_methods_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "shape,family,q_error_type,target,delta_h,trap_rel,sub_rel,panel_rel,gulati_rel,zeta_rel,qbx_rel,"
        "sub_x,panel_x,gulati_x,zeta_x,qbx_x"
    )
    for row in rows:
        print(
            f"{row['shape']},{row['family']},{row['q_spectral_error_type']},"
            f"{row['target_mode']},{row['delta_over_h']},"
            f"{fmt(row['trapezoid_relative_error'])},"
            f"{fmt(row['singularity_subtraction_relative_error'])},"
            f"{fmt(row['adaptive_panel_relative_error'])},"
            f"{fmt(row['gulati_q_bridge_relative_error'])},"
            f"{fmt(row['multipole_zeta_q_relative_error'])},"
            f"{fmt(row['qbx_refined_relative_error'])},"
            f"{fmt_ratio(row['singularity_subtraction_improvement_vs_trap'])},"
            f"{fmt_ratio(row['adaptive_panel_improvement_vs_trap'])},"
            f"{fmt_ratio(row['gulati_q_bridge_improvement_vs_trap'])},"
            f"{fmt_ratio(row['multipole_zeta_q_improvement_vs_trap'])},"
            f"{fmt_ratio(row['qbx_refined_improvement_vs_trap'])}"
        )
    compact = {
        method: {
            "median_x": summary["methods"][method]["median_improvement_vs_trap"],
            "median_rel": summary["methods"][method]["median_relative_error"],
            "median_work": summary["methods"][method]["median_work_units"],
            "failures": summary["methods"][method]["failure_count"],
        }
        for method in methods
    }
    print("summary=" + json.dumps(compact, sort_keys=True))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
