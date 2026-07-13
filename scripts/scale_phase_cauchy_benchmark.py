#!/usr/bin/env python3
"""Benchmark the fixed-rank nested scale-phase Cauchy compiler."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.quadrature import PI, _cos, _exp, _log, _sin  # noqa: E402
from inverse_shape.scale_phase_cauchy import (  # noqa: E402
    ScalePhaseCauchyQJet,
    StaticTriangularCauchyPlan,
)
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_scale_phase_mode,
    reference_scale_phase_spectral,
)


OUT = ROOT / "outputs" / "scale_phase_cauchy"
PLAN_SIZES = (64, 128, 256, 512, 1024)
QJET_SIZES = (16, 32, 64, 128, 256, 512, 1024)
N_THETA = 8


def nonuniform_rhos(count):
    return tuple(
        -1.4
        + 2.8 * index / (count - 1)
        + 0.045 * _sin(2.0 * PI * index / (count - 1))
        for index in range(count)
    )


def values(count):
    return tuple(
        _cos(0.17 * index) + 0.13j * _sin(0.29 * index)
        for index in range(count)
    )


def field(rhos):
    rows = []
    for index, rho in enumerate(rhos):
        rows.append(
            tuple(
                0.2
                + 0.3 * rho
                + 0.11 * _cos(2.0 * PI * 3 * phase / N_THETA)
                + 0.02j * (index + 1) * _sin(2.0 * PI * phase / N_THETA)
                for phase in range(N_THETA)
            )
        )
    return tuple(rows)


def relative_vector_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for value, reference in zip(left, right, strict=True)
    )
    denominator = max(1.0, *(abs(complex(value)) for value in right))
    return numerator / denominator


def relative_grid_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for left_row, right_row in zip(left, right, strict=True)
        for value, reference in zip(left_row, right_row, strict=True)
    )
    denominator = max(
        1.0,
        *(
            abs(complex(value))
            for reference_row in right
            for value in reference_row
        ),
    )
    return numerator / denominator


def median_seconds(operation, repeats=3):
    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings)


def fit_power(rows, size_key, time_key):
    points = [
        (_log(float(row[size_key])), _log(float(row[time_key])))
        for row in rows
        if row[time_key] > 0.0
    ]
    mean_x = sum(x for x, _y in points) / len(points)
    mean_y = sum(y for _x, y in points) / len(points)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    )
    denominator = sum((x - mean_x) ** 2 for x, _y in points)
    return numerator / denominator


def plan_campaign():
    rows = []
    for count in PLAN_SIZES:
        rhos = nonuniform_rhos(count)
        nodes = tuple(_exp(2.0 * rho) for rho in rhos)
        source = values(count)
        compile_start = time.perf_counter()
        plan = StaticTriangularCauchyPlan(nodes)
        compile_ms = 1000.0 * (time.perf_counter() - compile_start)
        fast, fast_seconds = median_seconds(
            lambda plan=plan, source=source, rhos=rhos: (
                plan.apply_exponential_mode(source, rhos, 7)
            ),
        )
        if count <= 512:
            direct, direct_seconds = median_seconds(
                lambda plan=plan, source=source, rhos=rhos: (
                    reference_scale_phase_mode(plan, source, rhos, 7)
                ),
            )
            error = relative_vector_error(fast, direct)
            direct_ms = 1000.0 * direct_seconds
        else:
            error = None
            direct_ms = None
        stats = plan.stats()
        rows.append(
            {
                "n_scale": count,
                "compile_ms": compile_ms,
                "nested_apply_ms": 1000.0 * fast_seconds,
                "direct_apply_ms": direct_ms,
                "relative_error": error,
                "compressed_blocks": stats["compressed_blocks"],
                "exact_blocks": stats["exact_blocks"],
                "compressed_pair_fraction": stats[
                    "compressed_pair_fraction"
                ],
                "cluster_records": stats["cluster_records"],
                "stored_block_records": stats["stored_block_records"],
                "stored_factor_entries": stats["stored_factor_entries"],
                "stored_interaction_matrices": stats[
                    "stored_interaction_matrices"
                ],
                "stored_dense_matrix": False,
            }
        )
    return rows


def qjet_campaign():
    rows = []
    for n_scale in QJET_SIZES:
        rhos = nonuniform_rhos(n_scale)
        weights = tuple(
            _exp(2.0 * rho) * (1.0 + 0.03 * _cos(0.2 * index))
            for index, rho in enumerate(rhos)
        )
        qjet = ScalePhaseCauchyQJet(rhos, N_THETA, weights)
        source = field(rhos)
        fast, seconds = median_seconds(
            lambda qjet=qjet, source=source: qjet.apply(source)
        )
        if n_scale <= 128:
            direct = reference_scale_phase_spectral(qjet, source)
            error = relative_grid_error(fast, direct)
        else:
            error = None
        rows.append(
            {
                "n_scale": n_scale,
                "n_theta": N_THETA,
                "n_nodes": n_scale * N_THETA,
                "apply_ms": 1000.0 * seconds,
                "relative_error": error,
                "constant_residual": qjet.constant_residual(),
                "cluster_records": qjet.plan.stats()["cluster_records"],
                "stored_block_records": qjet.plan.stats()[
                    "stored_block_records"
                ],
                "stored_dense_matrix": False,
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary):
    lines = [
        "# Nested scale-phase Cauchy benchmark",
        "",
        "The production path stores one scale tree and sparse block endpoint "
        "records. Fixed-size Chebyshev transfer and interaction tiles are "
        "compiled once and reused; no `N x N` table is formed.",
        "",
        "| n-scale | nested ms | direct ms | rel. error | compressed pairs | "
        "cluster records | block records |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["plan_rows"]:
        direct = (
            f"{row['direct_apply_ms']:.3e}"
            if row["direct_apply_ms"] is not None
            else "not run"
        )
        error = (
            f"{row['relative_error']:.3e}"
            if row["relative_error"] is not None
            else "not run"
        )
        lines.append(
            f"| {row['n_scale']} | {row['nested_apply_ms']:.3e} | "
            f"{direct} | {error} | "
            f"{row['compressed_pair_fraction']:.6f} | "
            f"{row['cluster_records']} | {row['stored_block_records']} |"
        )
    lines.extend(
        [
            "",
            "The fitted nested-mode exponent is "
            f"{summary['fits']['nested_mode_time_exponent']:.3f}; the direct "
            "reference exponent over its measured range is "
            f"{summary['fits']['direct_mode_time_exponent']:.3f}. Persistent "
            "pointwise pair factors remain zero. The retained interaction "
            "tiles have fixed `32 x 32` size, so their total storage is "
            "linear in the scale-node count.",
            "",
            "| total nodes | full QJet ms | rel. error | constant residual |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in summary["qjet_rows"]:
        error = (
            f"{row['relative_error']:.3e}"
            if row["relative_error"] is not None
            else "not run"
        )
        lines.append(
            f"| {row['n_nodes']} | {row['apply_ms']:.3e} | {error} | "
            f"{row['constant_residual']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The full QJet exponent is "
            f"{summary['fits']['full_qjet_time_exponent']:.3f}. These tests "
            "audit the exact continuum angular Fourier operator. A finite "
            "angular pair sum has additional alias rungs and is deliberately "
            "not used as the reference.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_rows = plan_campaign()
    qjet_rows = qjet_campaign()
    direct_rows = [
        row for row in plan_rows if row["direct_apply_ms"] is not None
    ]
    summary = {
        "method": "fixed_rank_nested_scale_phase_cauchy",
        "rank": 32,
        "stored_dense_matrix": False,
        "stored_pair_table": False,
        "stored_global_interaction_matrix": False,
        "persistent_storage": "O(N)",
        "apply_work": "O(p^2 N + N log n_theta), fixed p",
        "plan_rows": plan_rows,
        "qjet_rows": qjet_rows,
        "fits": {
            "nested_mode_time_exponent": fit_power(
                plan_rows,
                "n_scale",
                "nested_apply_ms",
            ),
            "direct_mode_time_exponent": fit_power(
                direct_rows,
                "n_scale",
                "direct_apply_ms",
            ),
            "full_qjet_time_exponent": fit_power(
                qjet_rows,
                "n_nodes",
                "apply_ms",
            ),
        },
    }
    write_csv(OUT / "plan_scaling.csv", plan_rows)
    write_csv(OUT / "qjet_scaling.csv", qjet_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
