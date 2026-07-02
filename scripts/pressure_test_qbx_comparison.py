"""Compare local Gulati bridge correction with point-QBX on hard shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from inverse_shape.geometry import BoundaryCurve
from inverse_shape.quadrature import (
    log_layer_local_bridge,
    log_layer_qbx,
    log_layer_trapezoid,
    outward_unit_normals,
)

HardShapeTerms = tuple[tuple[int, float, float], ...]

HARD_SHAPES: dict[str, HardShapeTerms] = {
    "peanut": ((2, 0.32, 0.0), (5, -0.10, 0.0)),
    "teardrop": ((1, 0.35, 0.0), (3, 0.0, 0.12), (4, -0.06, 0.0)),
    "wavy_star": ((5, 0.18, 0.0), (7, 0.0, 0.09), (11, 0.06, 0.0)),
    "asymmetric_gear": (
        (3, 0.24, 0.0),
        (4, 0.0, -0.18),
        (8, 0.10, 0.0),
        (9, 0.0, -0.06),
    ),
    "three_lobed": ((2, 0.0, -0.22), (3, 0.31, 0.0), (6, 0.0, 0.08)),
}


def hard_shape_points(terms: HardShapeTerms, n: int) -> np.ndarray:
    """Return approximately arclength-spaced samples of a hard star shape."""

    dense_n = max(8 * n, 8192)
    theta = np.linspace(0.0, 2.0 * np.pi, dense_n, endpoint=False)
    radius = np.ones_like(theta)
    for mode, cos_coeff, sin_coeff in terms:
        radius += cos_coeff * np.cos(mode * theta) + sin_coeff * np.sin(mode * theta)
    if np.any(radius <= 0.0):
        raise ValueError("radial function became non-positive")
    dense = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return BoundaryCurve(dense).resample(n).points


def density_samples(n: int) -> np.ndarray:
    """Smooth positive arclength-periodic density used in all cases."""

    tau = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    return 1.0 + 0.35 * np.cos(3.0 * tau + 0.2) - 0.20 * np.sin(5.0 * tau - 0.1)


def target_and_center(
    points: np.ndarray,
    sample_index: int,
    delta: float,
    *,
    qbx_radius_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    normal = outward_unit_normals(points)[sample_index]
    target = points[sample_index] + delta * normal
    center = points[sample_index] + qbx_radius_factor * delta * normal
    return target, center


def run_comparison(
    *,
    coarse_n: int,
    qbx_n: int,
    reference_n: int,
    qbx_order: int,
    qbx_radius_factor: float,
    delta_over_h: tuple[float, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for shape_name, terms in HARD_SHAPES.items():
        coarse_points = hard_shape_points(terms, coarse_n)
        qbx_points = hard_shape_points(terms, qbx_n)
        reference_points = hard_shape_points(terms, reference_n)
        coarse_density = density_samples(coarse_n)
        qbx_density = density_samples(qbx_n)
        reference_density = density_samples(reference_n)
        sample_index = coarse_n // 7
        h = float(np.linalg.norm(np.roll(coarse_points, -1, axis=0) - coarse_points, axis=1).mean())

        for ratio in delta_over_h:
            delta = ratio * h
            target, center = target_and_center(
                coarse_points,
                sample_index,
                delta,
                qbx_radius_factor=qbx_radius_factor,
            )
            reference = log_layer_trapezoid(reference_points, reference_density, target)
            trapezoid = log_layer_trapezoid(coarse_points, coarse_density, target)
            bridge = log_layer_local_bridge(
                coarse_points,
                coarse_density,
                target,
                sample_index=sample_index,
            )
            qbx = log_layer_qbx(
                qbx_points,
                qbx_density,
                target,
                center,
                order=qbx_order,
            )
            scale = max(abs(reference), 1e-14)
            trap_error = abs(trapezoid - reference)
            bridge_error = abs(bridge - reference)
            qbx_error = abs(qbx - reference)
            rows.append(
                {
                    "shape": shape_name,
                    "coarse_n": coarse_n,
                    "qbx_n": qbx_n,
                    "reference_n": reference_n,
                    "delta_over_h": ratio,
                    "delta": delta,
                    "reference": float(np.real(reference)),
                    "trapezoid_relative_error": float(trap_error / scale),
                    "bridge_relative_error": float(bridge_error / scale),
                    "qbx_relative_error": float(qbx_error / scale),
                    "bridge_improvement": float(trap_error / max(bridge_error, 1e-300)),
                    "qbx_improvement": float(trap_error / max(qbx_error, 1e-300)),
                }
            )

    qbx_max = max(row["qbx_relative_error"] for row in rows)
    bridge_better = sum(
        row["bridge_relative_error"] < row["trapezoid_relative_error"] for row in rows
    )
    if qbx_max > 1e-6:
        failures.append(f"QBX baseline exceeded 1e-6 relative error: {qbx_max:.3e}")
    if bridge_better == 0:
        failures.append("bridge correction did not improve any hard case")

    return {
        "passed": not failures,
        "failures": failures,
        "parameters": {
            "coarse_n": coarse_n,
            "qbx_n": qbx_n,
            "reference_n": reference_n,
            "qbx_order": qbx_order,
            "qbx_radius_factor": qbx_radius_factor,
            "delta_over_h": list(delta_over_h),
        },
        "summary": {
            "case_count": len(rows),
            "bridge_better_count": bridge_better,
            "max_trapezoid_relative_error": max(row["trapezoid_relative_error"] for row in rows),
            "max_bridge_relative_error": max(row["bridge_relative_error"] for row in rows),
            "max_qbx_relative_error": qbx_max,
            "median_bridge_improvement": float(np.median([row["bridge_improvement"] for row in rows])),
            "median_qbx_improvement": float(np.median([row["qbx_improvement"] for row in rows])),
        },
        "rows": rows,
    }


def _parse_delta_over_h(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-n", type=int, default=1024)
    parser.add_argument("--qbx-n", type=int, default=32768)
    parser.add_argument("--reference-n", type=int, default=131072)
    parser.add_argument("--qbx-order", type=int, default=50)
    parser.add_argument("--qbx-radius-factor", type=float, default=4.0)
    parser.add_argument(
        "--delta-over-h",
        type=_parse_delta_over_h,
        default=_parse_delta_over_h("0.5,0.2,0.1,0.05"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.coarse_n < 64:
        raise SystemExit("--coarse-n must be at least 64")
    if args.qbx_n < args.coarse_n:
        raise SystemExit("--qbx-n must be at least --coarse-n")
    if args.reference_n < args.qbx_n:
        raise SystemExit("--reference-n must be at least --qbx-n")
    if args.qbx_radius_factor <= 1.0:
        raise SystemExit("--qbx-radius-factor must exceed 1")

    payload = run_comparison(
        coarse_n=args.coarse_n,
        qbx_n=args.qbx_n,
        reference_n=args.reference_n,
        qbx_order=args.qbx_order,
        qbx_radius_factor=args.qbx_radius_factor,
        delta_over_h=args.delta_over_h,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
