"""Pressure-test the finite Gulati quadrature primitives from the v41 note."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from inverse_shape.operators import gulati_laplacian
from inverse_shape.quadrature import (
    apply_cycle_gulati,
    circle_gulati_coercivity,
    cycle_gulati_condition_number,
    cycle_gulati_eigenvalues,
    near_singular_circle_table,
    regular_polygon_points,
    solve_cycle_gulati,
)


def _sparse_trig_rhs(n: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    return np.cos(3.0 * theta) + 0.25 * np.sin(5.0 * theta)


def _median_apply_time(n: int, repeats: int) -> float:
    theta = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n
    values = np.cos(7.0 * theta) + 0.125 * np.sin(11.0 * theta)
    timings: list[float] = []
    apply_cycle_gulati(values)
    for _ in range(repeats):
        start = time.perf_counter()
        apply_cycle_gulati(values)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def run_pressure(max_power: int, near_n: int, repeats: int) -> dict[str, Any]:
    dense_n = 32
    dense_gu = gulati_laplacian(regular_polygon_points(dense_n))
    dense_eig_error = float(
        np.max(
            np.abs(
                np.sort(np.linalg.eigvalsh(dense_gu))
                - np.sort(cycle_gulati_eigenvalues(dense_n))
            )
        )
    )

    powers = sorted(set([12, 14, max_power]))
    conservation = []
    solves = []
    timings = []
    for power in powers:
        n = 2**power
        ones = np.ones(n, dtype=np.float64)
        conserved = float(np.linalg.norm(apply_cycle_gulati(ones), ord=np.inf))
        rhs = _sparse_trig_rhs(n)
        solution = solve_cycle_gulati(rhs)
        residual = float(np.linalg.norm(apply_cycle_gulati(solution) - rhs, ord=np.inf))
        conservation.append({"n": n, "gulati_ones_inf": conserved})
        solves.append({"n": n, "residual_inf": residual})
        elapsed = _median_apply_time(n, repeats)
        timings.append(
            {
                "n": n,
                "wall_clock_ms": 1000.0 * elapsed,
                "ns_per_n_log2_n": 1e9 * elapsed / (n * np.log2(n)),
            }
        )

    near_singular = near_singular_circle_table(n=near_n)
    coercivity = []
    for delta in (1e-2, 1e-4, 1e-6):
        value = circle_gulati_coercivity(1.0 + delta)
        coercivity.append({"delta": delta, "delta_times_gulati_over_pi": value * delta / np.pi})

    failures: list[str] = []
    if dense_eig_error > 1e-10:
        failures.append(f"dense spectrum mismatch {dense_eig_error:.3e}")
    if max(row["gulati_ones_inf"] for row in conservation) != 0.0:
        failures.append("Gulati conservation on constants was not exact")
    if max(row["residual_inf"] for row in solves) > 1e-10:
        failures.append("boundary solve residual exceeded 1e-10")
    if max(row["spectral_relative_error"] for row in near_singular) > 1e-12:
        failures.append("spectral near-singular error exceeded 1e-12")
    if near_singular[-1]["trapezoid_relative_error"] < 1e-5:
        failures.append("trapezoid near-singular degradation did not appear")

    return {
        "passed": not failures,
        "failures": failures,
        "dense_spectrum": {"n": dense_n, "max_abs_error": dense_eig_error},
        "condition_numbers": {
            "n_64": cycle_gulati_condition_number(64),
            "n_512": cycle_gulati_condition_number(512),
        },
        "conservation": conservation,
        "boundary_solves": solves,
        "timings": timings,
        "coercivity": coercivity,
        "near_singular": near_singular,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-power", type=int, default=16)
    parser.add_argument("--near-n", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.max_power < 12:
        raise SystemExit("--max-power must be at least 12")
    if args.near_n < 64:
        raise SystemExit("--near-n must be at least 64")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    payload = run_pressure(args.max_power, args.near_n, args.repeats)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
