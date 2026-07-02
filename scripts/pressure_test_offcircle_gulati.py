"""Pressure-test Gulati diagnostics on non-circular closed planar curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from inverse_shape.datasets import piecewise_curved_boundary
from inverse_shape.geometry import BoundaryCurve, StarShapeModel, perimeter, resample_closed_curve
from inverse_shape.operators import gulati_laplacian
from inverse_shape.quadrature import (
    arclength_scaled_gulati_eigenvalues,
    gulati_incidence_factor,
    gulati_weyl_pair_ratios,
    near_boundary_gulati_coercivity_table,
)


def ellipse_points(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 8 * n, endpoint=False)
    dense = np.column_stack([1.5 * np.cos(theta), 0.7 * np.sin(theta)])
    return resample_closed_curve(dense, n)


def smooth_star_points(n: int) -> np.ndarray:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.18, -0.05, 0.03]),
        sin=np.array([0.04, 0.06, -0.02]),
    )
    return BoundaryCurve(model.boundary_points(8 * n)).resample(n).points


def piecewise_points(n: int) -> np.ndarray:
    return BoundaryCurve(piecewise_curved_boundary(240)).resample(n).normalized().points


def shape_suite(n: int) -> dict[str, np.ndarray]:
    return {
        "ellipse": ellipse_points(n),
        "smooth_star": smooth_star_points(n),
        "piecewise_curved": piecewise_points(n),
    }


def conservation_psd_rows(n: int) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, points in shape_suite(n).items():
        gu = gulati_laplacian(points)
        eigenvalues = np.linalg.eigvalsh((gu + gu.T) / 2.0)
        factor = gulati_incidence_factor(points)
        rows[name] = {
            "n": float(n),
            "perimeter": perimeter(points),
            "constant_residual_inf": float(np.linalg.norm(gu @ np.ones(n), ord=np.inf)),
            "min_eigenvalue": float(eigenvalues[0]),
            "first_positive_eigenvalue": float(eigenvalues[1]),
            "cauchy_gram_error_inf": float(np.linalg.norm(factor.T @ factor - gu, ord=np.inf)),
        }
    return rows


def coercivity_rows(n: int) -> dict[str, list[dict[str, float]]]:
    return {
        name: near_boundary_gulati_coercivity_table(points, sample_index=0)
        for name, points in shape_suite(n).items()
    }


def weyl_rows(n: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, points in shape_suite(n).items():
        ratios = gulati_weyl_pair_ratios(points, mode_start=8, mode_stop=16)
        ratio_values = [row["ratio"] for row in ratios]
        eigenvalues = arclength_scaled_gulati_eigenvalues(points)
        rows[name] = {
            "n": n,
            "perimeter": perimeter(points),
            "zero_mode_abs": float(abs(eigenvalues[0])),
            "ratio_min": float(min(ratio_values)),
            "ratio_max": float(max(ratio_values)),
            "ratio_mean": float(np.mean(ratio_values)),
            "ratios": ratios,
        }
    return rows


def run_pressure(small_n: int, dense_n: int, coercivity_n: int) -> dict[str, Any]:
    conservation = conservation_psd_rows(small_n)
    coercivity = coercivity_rows(coercivity_n)
    weyl = weyl_rows(dense_n)

    failures: list[str] = []
    for name, row in conservation.items():
        if row["constant_residual_inf"] > 1e-9:
            failures.append(f"{name}: constant residual exceeded 1e-9")
        if row["min_eigenvalue"] < -1e-8:
            failures.append(f"{name}: minimum eigenvalue was negative")
        if row["cauchy_gram_error_inf"] > 1e-8:
            failures.append(f"{name}: Cauchy-Gram factorization error exceeded 1e-8")
    for name, rows in coercivity.items():
        final_ratio = rows[-1]["delta_times_coercivity_over_pi"]
        first_ratio = rows[0]["delta_times_coercivity_over_pi"]
        if abs(final_ratio - 1.0) > 0.05:
            failures.append(f"{name}: local pi/delta coercivity ratio missed by >5%")
        if abs(final_ratio - 1.0) >= abs(first_ratio - 1.0):
            failures.append(f"{name}: coercivity ratio did not improve as delta shrank")
    for name, row in weyl.items():
        if not (row["ratio_min"] > 0.90 and row["ratio_max"] < 1.03):
            failures.append(f"{name}: Weyl pair ratios outside expected range")

    return {
        "passed": not failures,
        "failures": failures,
        "conservation_psd_cauchy_gram": conservation,
        "near_boundary_coercivity": coercivity,
        "low_mode_weyl_pairs": weyl,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-n", type=int, default=64)
    parser.add_argument("--dense-n", type=int, default=256)
    parser.add_argument("--coercivity-n", type=int, default=4096)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.small_n < 16:
        raise SystemExit("--small-n must be at least 16")
    if args.dense_n < 64:
        raise SystemExit("--dense-n must be at least 64")
    if args.coercivity_n < 512:
        raise SystemExit("--coercivity-n must be at least 512")

    payload = run_pressure(args.small_n, args.dense_n, args.coercivity_n)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
