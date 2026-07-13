#!/usr/bin/env python3
"""Audit sparse support and conditioning of the peeled resultant generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.multivariate_resultant import (  # noqa: E402
    MultivariateResultantPeeledJetQJet,
)
from inverse_shape.quadrature import TAU, _cos, _log, _sin  # noqa: E402


OUT = ROOT / "outputs" / "multivariate_resultant"
SIZES = (4, 6, 8, 10, 12, 14, 16)
STRICT_AUDIT_TOLERANCE = 5.0e-13


def geometry(count):
    points = tuple(
        (
            (1.0 + 0.1 * _cos(3.0 * TAU * index / count))
            * _cos(TAU * index / count),
            (1.0 + 0.1 * _cos(3.0 * TAU * index / count))
            * _sin(TAU * index / count),
            -0.7
            + 1.4 * (index + 0.5) / count
            + 0.07 * _sin(5.0 * TAU * index / count),
        )
        for index in range(count)
    )
    weights = tuple(0.8 + 0.05 * index for index in range(count))
    values = tuple(
        point[0] + 0.2 * point[1] - 0.1 * point[2] + 0.03j * index
        for index, point in enumerate(points)
    )
    return points, weights, values


def direct_q(points, weights, values):
    output = []
    for left, left_point in enumerate(points):
        total = 0.0 + 0.0j
        for right, right_point in enumerate(points):
            if left == right:
                continue
            distance_squared = sum(
                (left_point[axis] - right_point[axis]) ** 2
                for axis in range(3)
            )
            total += (
                weights[right]
                * (values[left] - values[right])
                / distance_squared
            )
        output.append(total)
    return tuple(output)


def relative_l2(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def median_seconds(operation, repeats=3):
    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings)


def fit_power(rows, key):
    transformed = [
        (_log(float(row["n"])), _log(float(row[key])))
        for row in rows
        if float(row[key]) > 0.0
    ]
    mean_x = sum(x for x, _y in transformed) / len(transformed)
    mean_y = sum(y for _x, y in transformed) / len(transformed)
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in transformed
    )
    denominator = sum((x - mean_x) ** 2 for x, _y in transformed)
    return numerator / denominator


def campaign():
    rows = []
    for count in SIZES:
        points, weights, values = geometry(count)
        qjet = MultivariateResultantPeeledJetQJet(
            points,
            weights,
            support_budget=100000,
            audit_mode="full",
            audit_tolerance=1.0,
            fallback=False,
        )
        start = time.perf_counter()
        candidate = qjet.apply(values)
        resultant_seconds = time.perf_counter() - start
        reference, direct_seconds = median_seconds(
            lambda: direct_q(points, weights, values),
            repeats=5,
        )
        stats = qjet.stats()
        theoretical_simplex_support = (
            (2 * count + 3) * (2 * count + 2) * (2 * count + 1) // 6
        )
        rows.append(
            {
                "n": count,
                "resultant_ms": 1000.0 * resultant_seconds,
                "streamed_direct_ms": 1000.0 * direct_seconds,
                "denominator_support": stats["denominator_support"],
                "theoretical_simplex_support": theoretical_simplex_support,
                "total_retained_support": (
                    stats["denominator_support"]
                    + sum(stats["numerator_supports"])
                ),
                "scalar_polynomial_multiplications": (
                    stats["scalar_polynomial_multiplications"]
                ),
                "peeled_sum_audit_error": stats["audit_relative_error"],
                "graph_relative_error": relative_l2(reference, candidate),
                "passes_strict_audit": (
                    stats["audit_relative_error"] <= STRICT_AUDIT_TOLERANCE
                ),
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
        "# Multivariate peeled-resultant benchmark",
        "",
        "The product tree carries `D`, `N_w`, and `N_wf` in a normalized "
        "sparse monomial basis. Every run uses a full independent peeled-sum "
        "audit and stores no pair matrix.",
        "",
        "| N | resultant ms | direct ms | support D | total support | "
        "peeled audit error | strict pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['n']} | {row['resultant_ms']:.3e} | "
            f"{row['streamed_direct_ms']:.3e} | "
            f"{row['denominator_support']} | "
            f"{row['total_retained_support']} | "
            f"{row['peeled_sum_audit_error']:.3e} | "
            f"{'yes' if row['passes_strict_audit'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The fitted denominator-support exponent is "
            f"{summary['fits']['denominator_support_exponent']:.3f}. The "
            "measured arithmetic exponent is "
            f"{summary['fits']['resultant_time_exponent']:.3f}, compared with "
            f"{summary['fits']['direct_time_exponent']:.3f} for the streamed "
            "reference over this small range.",
            "",
            "The algebraic finite-part identity is correct, but a generic "
            "three-variable product fills a cubic monomial simplex and the "
            "monomial evaluation becomes ill-conditioned. With the strict "
            f"`{STRICT_AUDIT_TOLERANCE:.1e}` audit, the tested resultant path "
            f"is retained only through N={summary['largest_strict_pass_n']}; "
            "larger cases repay by the exact "
            "stream in production mode.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = campaign()
    summary = {
        "method": "sparse_multivariate_peeled_resultant",
        "formula": "(Delta N_a-a_i Delta^2 D/20)/Delta D",
        "support_budget": 100000,
        "strict_audit_tolerance": STRICT_AUDIT_TOLERANCE,
        "stored_dense_matrix": False,
        "largest_strict_pass_n": max(
            (row["n"] for row in rows if row["passes_strict_audit"]),
            default=0,
        ),
        "rows": rows,
        "fits": {
            "denominator_support_exponent": fit_power(
                rows,
                "denominator_support",
            ),
            "resultant_time_exponent": fit_power(rows, "resultant_ms"),
            "direct_time_exponent": fit_power(rows, "streamed_direct_ms"),
        },
    }
    write_csv(OUT / "support_conditioning.csv", rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
