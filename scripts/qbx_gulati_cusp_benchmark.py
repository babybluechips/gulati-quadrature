#!/usr/bin/env python3
"""Benchmark QBX and Gulati-Q local bridge on cusped planar curves."""

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
    log_layer_local_bridge,
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


def cardioid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        r = 1.0 - math.cos(theta)
        out.append((r * math.cos(theta), r * math.sin(theta)))
    return out


def astroid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        c = math.cos(theta)
        s = math.sin(theta)
        out.append((c * c * c, s * s * s))
    return out


def deltoid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        out.append(
            (
                2.0 * math.cos(theta) + math.cos(2.0 * theta),
                2.0 * math.sin(theta) - math.sin(2.0 * theta),
            )
        )
    return out


def nephroid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        out.append(
            (
                3.0 * math.cos(theta) - math.cos(3.0 * theta),
                3.0 * math.sin(theta) - math.sin(3.0 * theta),
            )
        )
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
        return {"ok": True, "value": value, "ms": elapsed_ms, "error": ""}
    except Exception as exc:  # noqa: BLE001 - benchmark records numerical failure modes.
        return {"ok": False, "value": None, "ms": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def target_from_sample(points, sample_index, ratio):
    h = perimeter(points) / len(points)
    normal = tuple(outward_unit_normals(points)[sample_index])
    base = points[sample_index]
    return (base[0] + ratio * h * normal[0], base[1] + ratio * h * normal[1])


def nearest_index(points, target):
    return min(range(len(points)), key=lambda index: dist(points[index], target))


def relative_error(result, reference):
    if not result["ok"]:
        return None
    return abs(result["value"] - reference) / max(abs(reference), 1.0e-14)


def multipole_zeta_refined_q(dense_points, target, *, base_n, levels, order=16, leaf_size=32):
    level_points = []
    level_density = []
    sample_indices = []
    for level_n in levels:
        points = resample_closed_curve(dense_points, level_n)
        level_points.append(points)
        level_density.append(density_values(level_n))
        sample_indices.append(nearest_index(points, target))
    evaluation = log_layer_multipole_zeta_q_borrow_compute_repay(
        level_points,
        level_density,
        target,
        sample_indices=sample_indices,
        order=order,
        leaf_size=leaf_size,
    )
    stats = evaluation.stats
    stats["base_n"] = base_n
    return evaluation.value, stats


def method_payload(name, result, reference, trap_error):
    rel = relative_error(result, reference)
    payload = {
        f"{name}_ok": result["ok"],
        f"{name}_ms": result["ms"],
        f"{name}_relative_error": rel,
        f"{name}_failure": result["error"],
    }
    if rel is not None and rel > 0.0 and trap_error is not None:
        payload[f"{name}_improvement_vs_trap"] = trap_error / rel
    else:
        payload[f"{name}_improvement_vs_trap"] = None
    return payload


def run_case(shape_name, dense_points, target_mode, ratio, *, n, qbx_n, reference_n, qbx_order, radius_factor):
    coarse = resample_closed_curve(dense_points, n)
    qbx_points = resample_closed_curve(dense_points, qbx_n)
    reference_points = resample_closed_curve(dense_points, reference_n)
    coarse_density = density_values(n)
    qbx_density = density_values(qbx_n)
    reference_density = density_values(reference_n)

    if target_mode == "tip":
        sample_index = 0
    elif target_mode == "near_tip":
        sample_index = max(2, n // 128)
    elif target_mode == "smooth_control":
        sample_index = n // 7
    else:
        raise ValueError("unknown target mode")

    target = target_from_sample(coarse, sample_index, ratio)
    qbx_index = nearest_index(qbx_points, coarse[sample_index])

    reference, reference_ms = timed(lambda: log_layer_trapezoid(reference_points, reference_density, target))
    trapezoid = safe_eval(lambda: log_layer_trapezoid(coarse, coarse_density, target))
    gulati = safe_eval(
        lambda: log_layer_local_bridge(coarse, coarse_density, target, sample_index=sample_index)
    )
    qbx_same = safe_eval(
        lambda: log_layer_qbx_auto(
            coarse,
            coarse_density,
            target,
            sample_index=sample_index,
            order=qbx_order,
            radius_factor=radius_factor,
        )
    )
    qbx_refined = safe_eval(
        lambda: log_layer_qbx_auto(
            qbx_points,
            qbx_density,
            target,
            sample_index=qbx_index,
            order=qbx_order,
            radius_factor=radius_factor,
        )
    )
    zeta_refined = safe_eval(
        lambda: multipole_zeta_refined_q(
            dense_points,
            target,
            base_n=n,
            levels=(n, 2 * n, 4 * n),
            order=18,
            leaf_size=32,
        )
    )
    if zeta_refined["ok"]:
        zeta_value, zeta_stats = zeta_refined["value"]
        zeta_refined["value"] = zeta_value
    else:
        zeta_stats = {}
    qjet = build_boundary_qjet(coarse)
    probe = [math.cos(2.0 * TAU * i / n) - 0.25 * math.sin(7.0 * TAU * i / n) for i in range(n)]
    _, q_apply_ms = timed(lambda: qjet.apply(probe))

    trap_error = relative_error(trapezoid, reference)
    row = {
        "shape": shape_name,
        "target_mode": target_mode,
        "delta_over_h": ratio,
        "n": n,
        "qbx_n": qbx_n,
        "reference_n": reference_n,
        "qbx_order": qbx_order,
        "qbx_radius_factor": radius_factor,
        "reference_ms": reference_ms,
        "q_apply_ms": q_apply_ms,
        "reference": float(reference.real) if isinstance(reference, complex) else float(reference),
    }
    row.update(method_payload("trapezoid", trapezoid, reference, trap_error))
    row.update(method_payload("gulati_q_bridge", gulati, reference, trap_error))
    row.update(method_payload("multipole_zeta_q", zeta_refined, reference, trap_error))
    row["multipole_zeta_q_stats"] = zeta_stats
    row["multipole_zeta_q_cached_target_work_units"] = zeta_stats.get("cached_target_work_units")
    row["multipole_zeta_q_moment_build_units"] = zeta_stats.get("moment_build_units")
    row["multipole_zeta_q_single_target_work_units"] = zeta_stats.get("single_target_work_units")
    row.update(method_payload("qbx_same_n", qbx_same, reference, trap_error))
    row.update(method_payload("qbx_refined", qbx_refined, reference, trap_error))
    row["qbx_refined_work_units"] = qbx_n * qbx_order
    row["qbx_same_n_work_units"] = n * qbx_order
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
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def fmt_ratio(value):
    if value is None:
        return "failed"
    return f"{value:.2f}"


def fmt_units(value):
    if value is None:
        return "failed"
    return str(int(value))


def main():
    n = 512
    qbx_n = 4096
    reference_n = 131072
    qbx_order = 60
    radius_factor = 4.0
    dense_samples = 16384
    shapes = [
        ("cardioid_single_cusp", cardioid_dense(dense_samples)),
        ("astroid_four_cusps", astroid_dense(dense_samples)),
        ("deltoid_three_cusps", deltoid_dense(dense_samples)),
        ("nephroid_two_cusps", nephroid_dense(dense_samples)),
    ]
    target_modes = ("tip", "near_tip", "smooth_control")
    ratios = (0.2, 0.1, 0.05)

    rows = []
    for shape_name, dense_points in shapes:
        for target_mode in target_modes:
            for ratio in ratios:
                rows.append(
                    run_case(
                        shape_name,
                        dense_points,
                        target_mode,
                        ratio,
                        n=n,
                        qbx_n=qbx_n,
                        reference_n=reference_n,
                        qbx_order=qbx_order,
                        radius_factor=radius_factor,
                    )
                )

    tip_rows = [row for row in rows if row["target_mode"] == "tip"]
    near_rows = [row for row in rows if row["target_mode"] == "near_tip"]
    control_rows = [row for row in rows if row["target_mode"] == "smooth_control"]
    summary = {
        "case_count": len(rows),
        "tip_median_gulati_q_bridge_improvement": median(
            [row["gulati_q_bridge_improvement_vs_trap"] for row in tip_rows]
        ),
        "tip_median_multipole_zeta_q_improvement": median(
            [row["multipole_zeta_q_improvement_vs_trap"] for row in tip_rows]
        ),
        "tip_median_qbx_refined_improvement": median(
            [row["qbx_refined_improvement_vs_trap"] for row in tip_rows]
        ),
        "near_tip_median_gulati_q_bridge_improvement": median(
            [row["gulati_q_bridge_improvement_vs_trap"] for row in near_rows]
        ),
        "near_tip_median_multipole_zeta_q_improvement": median(
            [row["multipole_zeta_q_improvement_vs_trap"] for row in near_rows]
        ),
        "near_tip_median_qbx_refined_improvement": median(
            [row["qbx_refined_improvement_vs_trap"] for row in near_rows]
        ),
        "control_median_gulati_q_bridge_improvement": median(
            [row["gulati_q_bridge_improvement_vs_trap"] for row in control_rows]
        ),
        "control_median_multipole_zeta_q_improvement": median(
            [row["multipole_zeta_q_improvement_vs_trap"] for row in control_rows]
        ),
        "control_median_qbx_refined_improvement": median(
            [row["qbx_refined_improvement_vs_trap"] for row in control_rows]
        ),
        "max_gulati_q_bridge_relative_error": max(
            row["gulati_q_bridge_relative_error"] for row in rows if row["gulati_q_bridge_relative_error"] is not None
        ),
        "max_multipole_zeta_q_relative_error": max(
            row["multipole_zeta_q_relative_error"] for row in rows if row["multipole_zeta_q_relative_error"] is not None
        ),
        "max_qbx_refined_relative_error": max(
            row["qbx_refined_relative_error"] for row in rows if row["qbx_refined_relative_error"] is not None
        ),
        "median_multipole_zeta_q_ms": median([row["multipole_zeta_q_ms"] for row in rows]),
        "median_qbx_refined_ms": median([row["qbx_refined_ms"] for row in rows]),
        "median_multipole_zeta_q_cached_target_work_units": median(
            [row["multipole_zeta_q_cached_target_work_units"] for row in rows]
        ),
        "median_multipole_zeta_q_moment_build_units": median(
            [row["multipole_zeta_q_moment_build_units"] for row in rows]
        ),
        "median_multipole_zeta_q_single_target_work_units": median(
            [row["multipole_zeta_q_single_target_work_units"] for row in rows]
        ),
        "median_qbx_refined_work_units": median([row["qbx_refined_work_units"] for row in rows]),
        "failure_count": sum(
            not row[f"{method}_ok"]
            for row in rows
            for method in ("trapezoid", "gulati_q_bridge", "multipole_zeta_q", "qbx_same_n", "qbx_refined")
        ),
    }
    payload = {
        "parameters": {
            "n": n,
            "qbx_n": qbx_n,
            "reference_n": reference_n,
            "qbx_order": qbx_order,
            "qbx_radius_factor": radius_factor,
            "multipole_zeta_q_order": 18,
            "multipole_zeta_q_leaf_size": 32,
            "multipole_zeta_q_levels": [n, 2 * n, 4 * n],
            "delta_over_h": list(ratios),
            "target_modes": list(target_modes),
        },
        "summary": summary,
        "rows": rows,
    }
    output = ROOT / "docs" / "assets" / "qbx_gulati_cusp_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "shape,target,delta_h,trap_rel,gulati_rel,zeta_q_rel,qbx_same_rel,qbx_ref_rel,"
        "gulati_x,zeta_q_x,qbx_ref_x,zeta_cached_units,zeta_single_units,qbx_units"
    )
    for row in rows:
        print(
            f"{row['shape']},{row['target_mode']},{row['delta_over_h']},"
            f"{fmt(row['trapezoid_relative_error'])},"
            f"{fmt(row['gulati_q_bridge_relative_error'])},"
            f"{fmt(row['multipole_zeta_q_relative_error'])},"
            f"{fmt(row['qbx_same_n_relative_error'])},"
            f"{fmt(row['qbx_refined_relative_error'])},"
            f"{fmt_ratio(row['gulati_q_bridge_improvement_vs_trap'])},"
            f"{fmt_ratio(row['multipole_zeta_q_improvement_vs_trap'])},"
            f"{fmt_ratio(row['qbx_refined_improvement_vs_trap'])},"
            f"{fmt_units(row['multipole_zeta_q_cached_target_work_units'])},"
            f"{fmt_units(row['multipole_zeta_q_single_target_work_units'])},"
            f"{fmt_units(row['qbx_refined_work_units'])}"
        )
    print("summary=" + json.dumps(summary, sort_keys=True))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
