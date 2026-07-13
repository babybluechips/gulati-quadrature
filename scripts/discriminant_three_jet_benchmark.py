#!/usr/bin/env python3
"""Audit the generated discriminant three-jet QJet path.

The fast path uses the project's foundational radix-two QJet FFT.  The direct
reference streams pairs and is used only as an accuracy and scaling control;
neither path stores a dense matrix.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.discriminant_three_jet import (  # noqa: E402
    RootOfUnityDiscriminantQJet,
    root_of_unity_numerator_jets,
)
from inverse_shape.quadrature import TAU, _cos, _sin  # noqa: E402


OUT = ROOT / "outputs" / "discriminant_three_jet"
SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
NON_RADIX_SIZES = (150, 151, 300, 600)
DIRECT_LIMIT = 1024


def deterministic_data(count):
    values = tuple(
        _cos(3.0 * TAU * index / count)
        + 0.2 * _sin(7.0 * TAU * index / count)
        + 0.05j * _cos(11.0 * TAU * index / count)
        for index in range(count)
    )
    weights = tuple(
        1.0
        + 0.1 * _cos(2.0 * TAU * index / count)
        + 0.03 * _sin(5.0 * TAU * index / count)
        for index in range(count)
    )
    return values, weights


def direct_circle_q(values, weights, roots, radius):
    inverse_radius_squared = 1.0 / (radius * radius)
    output = []
    for left, left_root in enumerate(roots):
        total = 0.0 + 0.0j
        for right, right_root in enumerate(roots):
            if left == right:
                continue
            chord_squared = abs(left_root - right_root) ** 2
            total += (
                weights[right]
                * (values[left] - values[right])
                / chord_squared
            )
        output.append(total * inverse_radius_squared)
    return tuple(output)


def relative_l2(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def median_seconds(operation, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings)


def fit_power(rows, key):
    points = [
        (math.log(float(row["n"])), math.log(float(row[key])))
        for row in rows
        if row.get(key) is not None and float(row[key]) > 0.0
    ]
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    numerator = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    )
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    return numerator / denominator


def contraction_operation(qjet, values, signed_weights, weight_jet, field_jet):
    signed = qjet.generated.apply(
        values,
        signed_weights,
        weight_jet,
        field_jet,
    )
    return tuple(
        -qjet.roots[index] * complex(signed[index])
        for index in range(qjet.n)
    )


def scaling_campaign():
    rows = []
    radius = 1.0
    for count in SIZES:
        values, weights = deterministic_data(count)
        start = time.perf_counter()
        qjet = RootOfUnityDiscriminantQJet(count)
        first = qjet.apply_circle_euclidean(values, weights, radius)
        construct_first_seconds = time.perf_counter() - start
        signed_weights = tuple(
            weights[index] * qjet.roots[index] for index in range(count)
        )
        weighted_field = tuple(
            signed_weights[index] * values[index] for index in range(count)
        )
        weight_jet = root_of_unity_numerator_jets(signed_weights)
        field_jet = root_of_unity_numerator_jets(weighted_field)

        fast, fast_seconds = median_seconds(
            lambda: qjet.apply_circle_euclidean(values, weights, radius),
            repeats=5,
        )
        contracted, contraction_seconds = median_seconds(
            lambda: contraction_operation(
                qjet,
                values,
                signed_weights,
                weight_jet,
                field_jet,
            ),
            repeats=7,
        )
        contraction_error = relative_l2(fast, contracted)
        first_apply_consistency = relative_l2(first, fast)

        direct = None
        direct_seconds = None
        direct_error = None
        speedup = None
        if count <= DIRECT_LIMIT:
            direct, direct_seconds = median_seconds(
                lambda: direct_circle_q(
                    values,
                    weights,
                    qjet.roots,
                    radius,
                ),
                repeats=3,
            )
            direct_error = relative_l2(direct, fast)
            speedup = direct_seconds / fast_seconds

        constant = qjet.apply_circle_euclidean(
            (1.0,) * count,
            weights,
            radius,
        )
        rows.append(
            {
                "n": count,
                "construct_and_first_apply_ms": (
                    1000.0 * construct_first_seconds
                ),
                "generated_apply_ms": 1000.0 * fast_seconds,
                "three_jet_contraction_ms": 1000.0 * contraction_seconds,
                "streamed_direct_ms": (
                    None
                    if direct_seconds is None
                    else 1000.0 * direct_seconds
                ),
                "relative_l2_error_vs_streamed_direct": direct_error,
                "relative_l2_contraction_consistency": contraction_error,
                "relative_l2_first_apply_consistency": (
                    first_apply_consistency
                ),
                "constant_null_residual": max(
                    abs(complex(value)) for value in constant
                ),
                "direct_over_generated_speedup": speedup,
                "persistent_complex_entries": 4 * count,
                "adaptive_rank": 0,
            }
        )
    return rows


def non_radix_campaign():
    rows = []
    radius = 1.7
    for count in NON_RADIX_SIZES:
        values, weights = deterministic_data(count)
        start = time.perf_counter()
        qjet = RootOfUnityDiscriminantQJet(count)
        first = qjet.apply_circle_euclidean(values, weights, radius)
        construct_first_seconds = time.perf_counter() - start
        generated, generated_seconds = median_seconds(
            lambda: qjet.apply_circle_euclidean(values, weights, radius),
            repeats=3,
        )
        direct, direct_seconds = median_seconds(
            lambda: direct_circle_q(
                values,
                weights,
                qjet.roots,
                radius,
            ),
            repeats=3,
        )
        rows.append(
            {
                "n": count,
                "transform_strategy": qjet.stats()["fft_strategy"],
                "construct_and_first_apply_ms": (
                    1000.0 * construct_first_seconds
                ),
                "generated_apply_ms": 1000.0 * generated_seconds,
                "streamed_direct_ms": 1000.0 * direct_seconds,
                "relative_l2_error_vs_streamed_direct": relative_l2(
                    direct, generated
                ),
                "relative_l2_first_apply_consistency": relative_l2(
                    first, generated
                ),
                "adaptive_rank": 0,
            }
        )
    return rows


def best_scalar_closure(points):
    model = []
    target = []
    for left, left_point in enumerate(points):
        for right, right_point in enumerate(points):
            if left == right:
                continue
            difference = left_point - right_point
            model.append(
                -left_point * right_point / (difference * difference)
            )
            target.append(1.0 / (abs(difference) ** 2))
    denominator = sum(abs(value) ** 2 for value in model)
    scalar = sum(
        complex(left).conjugate() * right
        for left, right in zip(model, target, strict=True)
    ) / denominator
    fitted = tuple(scalar * value for value in model)
    return scalar, relative_l2(target, fitted)


def metric_closure_campaign():
    count = 16
    angles = tuple(TAU * index / count for index in range(count))
    shapes = {
        "common_circle": tuple(
            complex(_cos(angle), _sin(angle)) for angle in angles
        ),
        "ellipse": tuple(
            complex(1.8 * _cos(angle), 0.7 * _sin(angle))
            for angle in angles
        ),
        "scale_phase_spiral": tuple(
            (1.0 + 0.24 * _cos(3.0 * angle))
            * complex(_cos(angle), _sin(angle))
            for angle in angles
        ),
    }
    rows = []
    for name, points in shapes.items():
        scalar, residual = best_scalar_closure(points)
        rows.append(
            {
                "geometry": name,
                "best_scalar_real": scalar.real,
                "best_scalar_imag": scalar.imag,
                "relative_pair_kernel_residual": residual,
                "single_univariate_metric_closure": residual < 1.0e-12,
            }
        )
    return rows


def write_csv(path, rows):
    fieldnames = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_scientific(value):
    if value is None:
        return "-"
    return f"{float(value):.3e}"


def write_report(summary):
    lines = [
        "# Discriminant three-jet benchmark",
        "",
        "The generated root-of-unity path uses the foundational QJet FFT and "
        "the fixed-width weighted discriminant contraction. It stores no dense "
        "matrix and selects no numerical rank.",
        "",
        "## Scaling and accuracy",
        "",
        "| N | build + first ms | generated ms | contraction ms | "
        "direct ms | rel. error | direct / generated |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scaling"]:
        lines.append(
            "| {n} | {first} | {generated} | {contraction} | {direct} | "
            "{error} | {speedup} |".format(
                n=row["n"],
                first=format_scientific(
                    row["construct_and_first_apply_ms"]
                ),
                generated=format_scientific(row["generated_apply_ms"]),
                contraction=format_scientific(
                    row["three_jet_contraction_ms"]
                ),
                direct=format_scientific(row["streamed_direct_ms"]),
                error=format_scientific(
                    row["relative_l2_error_vs_streamed_direct"]
                ),
                speedup=(
                    "-"
                    if row["direct_over_generated_speedup"] is None
                    else f"{row['direct_over_generated_speedup']:.2f}x"
                ),
            )
        )
    lines.extend(
        [
            "",
            "The fitted exponents are "
            f"{summary['fits']['generated_apply_exponent']:.3f} for the full "
            "generator plus contraction, "
            f"{summary['fits']['three_jet_contraction_exponent']:.3f} for the "
            "pre-generated contraction, and "
            f"{summary['fits']['streamed_direct_exponent']:.3f} for the "
            "quadratic streamed reference.",
            "",
            "## Non-radix-two audit",
            "",
            "| N | transform | build + first ms | generated ms | direct ms | "
            "relative error |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["non_radix_two"]:
        lines.append(
            f"| {row['n']} | {row['transform_strategy']} | "
            f"{row['construct_and_first_apply_ms']:.3e} | "
            f"{row['generated_apply_ms']:.3e} | "
            f"{row['streamed_direct_ms']:.3e} | "
            f"{row['relative_l2_error_vs_streamed_direct']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Metric closure audit",
            "",
            "| geometry | best scalar | relative pair-kernel residual | "
            "univariate closure |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in summary["metric_closure"]:
        scalar = complex(row["best_scalar_real"], row["best_scalar_imag"])
        lines.append(
            f"| {row['geometry']} | {scalar.real:.6g}"
            f"{scalar.imag:+.2e}i | "
            f"{row['relative_pair_kernel_residual']:.3e} | "
            f"{'yes' if row['single_univariate_metric_closure'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The common-circle Euclidean kernel closes exactly through the "
            "holomorphic three-jet. A scalar copy of that closure fails on the "
            "ellipse and varying-radius curve. Those geometries require their "
            "Schwarz/Joukowski or bivariate resultant generator; the three-jet "
            "contraction by itself does not remove that requirement.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaling = scaling_campaign()
    non_radix = non_radix_campaign()
    metric_closure = metric_closure_campaign()
    summary = {
        "method": "generated_root_of_unity_discriminant_three_jet",
        "numerical_dependencies": "project QJet FFT only; no NumPy",
        "stored_dense_matrix": False,
        "adaptive_rank": 0,
        "complexity": {
            "numerator_jet_generation": "O(N log N)",
            "fixed_width_contraction": "O(N)",
            "total_apply": "O(N log N)",
            "persistent_storage": "O(N)",
            "streamed_direct_reference": "O(N^2) time, O(N) storage",
        },
        "timing_protocol": {
            "construct_and_first_apply": "one timed construction plus apply",
            "generated_apply": "median of 5 warm applies",
            "three_jet_contraction": "median of 7 warm contractions",
            "streamed_direct": "median of 3 streamed references",
            "input": "deterministic Fourier field and nonuniform weights",
        },
        "fits": {
            "generated_apply_exponent": fit_power(
                scaling, "generated_apply_ms"
            ),
            "three_jet_contraction_exponent": fit_power(
                scaling, "three_jet_contraction_ms"
            ),
            "streamed_direct_exponent": fit_power(
                scaling, "streamed_direct_ms"
            ),
        },
        "scaling": scaling,
        "non_radix_two": non_radix,
        "metric_closure": metric_closure,
    }
    write_csv(OUT / "scaling.csv", scaling)
    write_csv(OUT / "non_radix_two.csv", non_radix)
    write_csv(OUT / "metric_closure.csv", metric_closure)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
