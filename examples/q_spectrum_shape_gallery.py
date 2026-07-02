#!/usr/bin/env python3
"""Matrix-free Q Ritz spectra for the complex-plane shape gallery.

The boundary Q operator is applied as

    (Qv)_i = sum_{j != i} (v_i - v_j) / |z_i - z_j|^2.

The script never stores the dense Q matrix.  It stores boundary samples,
temporary vectors, and the small Lanczos tridiagonal used only for spectral
diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discriminant_curvature_shape_gallery import shape_specs
from inverse_shape.quadrature import build_pullback_metric_qjet


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "discriminant_curvature_shape_gallery"
SPECTRUM_SAMPLE_COUNT = 256
LANCZOS_STEPS = 124
RITZ_COUNT = 12
TAU = 2.0 * math.pi


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(values: list[float]) -> float:
    return math.sqrt(max(dot(values, values), 0.0))


def subtract_mean(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def scale(values: list[float], factor: float) -> list[float]:
    return [factor * value for value in values]


def deterministic_start(sample_count: int, phase: float) -> list[float]:
    values = [
        math.cos((0.371 + 0.019 * phase) * (i + 1))
        + 0.43 * math.sin((1.137 + 0.011 * phase) * (i + 1))
        + 0.17 * math.cos((2.071 + 0.007 * phase) * (i + 1))
        for i in range(sample_count)
    ]
    values = subtract_mean(values)
    size = max(norm(values), 1.0e-300)
    return scale(values, 1.0 / size)


def estimate_pullback_speeds(points: list[complex], *, closed: bool, force_unit_circle: bool) -> list[float]:
    """Estimate |dz/dtheta| from generated samples for the pullback QJet."""

    if force_unit_circle:
        return [1.0 for _ in points]
    n = len(points)
    if n < 3:
        raise ValueError("at least three samples are required")
    speeds: list[float] = []
    if closed:
        for index in range(n):
            left = points[(index - 1) % n]
            right = points[(index + 1) % n]
            speeds.append(abs(right - left) * n / (2.0 * TAU))
    else:
        step = TAU / n
        for index in range(n):
            if index == 0:
                speed = abs(points[1] - points[0]) / step
            elif index == n - 1:
                speed = abs(points[-1] - points[-2]) / step
            else:
                speed = abs(points[index + 1] - points[index - 1]) / (2.0 * step)
            speeds.append(speed)
    positive = sorted(value for value in speeds if value > 0.0 and math.isfinite(value))
    if not positive:
        raise ValueError("degenerate pullback speed samples")
    floor = max(positive[len(positive) // 2] * 1.0e-8, 1.0e-12)
    return [max(value, floor) if math.isfinite(value) else floor for value in speeds]


def tridiagonal_eigenvalues(alphas: list[float], betas: list[float]) -> list[float]:
    """Jacobi eigenvalues of the small Lanczos tridiagonal."""

    size = len(alphas)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, alpha in enumerate(alphas):
        matrix[i][i] = alpha
    for i, beta in enumerate(betas[: max(0, size - 1)]):
        matrix[i][i + 1] = beta
        matrix[i + 1][i] = beta

    scale_hint = max((abs(alpha) for alpha in alphas), default=1.0)
    scale_hint = max(scale_hint, max((abs(beta) for beta in betas), default=0.0), 1.0)
    tolerance = 1.0e-11 * scale_hint
    max_rotations = max(2000, 10 * size * size)

    for _ in range(max_rotations):
        pivot_i = 0
        pivot_j = 1 if size > 1 else 0
        largest = 0.0
        for i in range(size):
            row = matrix[i]
            for j in range(i + 1, size):
                value = abs(row[j])
                if value > largest:
                    largest = value
                    pivot_i = i
                    pivot_j = j
        if largest <= tolerance or size <= 1:
            break

        p = pivot_i
        q = pivot_j
        app = matrix[p][p]
        aqq = matrix[q][q]
        apq = matrix[p][q]
        if abs(apq) <= tolerance:
            matrix[p][q] = 0.0
            matrix[q][p] = 0.0
            continue

        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for k in range(size):
            if k == p or k == q:
                continue
            akp = matrix[k][p]
            akq = matrix[k][q]
            new_kp = cosine * akp - sine * akq
            new_kq = sine * akp + cosine * akq
            matrix[k][p] = new_kp
            matrix[p][k] = new_kp
            matrix[k][q] = new_kq
            matrix[q][k] = new_kq

        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        matrix[p][q] = 0.0
        matrix[q][p] = 0.0

    return sorted(matrix[i][i] for i in range(size))


def lanczos_extreme_ritz(
    qjet: object,
    *,
    steps: int,
    count: int,
    sign: float,
    phase: float,
) -> tuple[list[float], float, int]:
    """Return largest Ritz values of sign * Q from matrix-free Lanczos."""

    n = qjet.n
    q = deterministic_start(n, phase)
    previous = [0.0 for _ in range(n)]
    beta_previous = 0.0
    basis: list[list[float]] = []
    alphas: list[float] = []
    betas: list[float] = []
    start = perf_counter()

    for step in range(min(steps, n - 1)):
        basis.append(q)
        applied = qjet.apply(q)
        z = [sign * float(complex(value).real) for value in applied]
        if step > 0:
            z = [value - beta_previous * old for value, old in zip(z, previous, strict=True)]
        alpha = dot(q, z)
        z = [value - alpha * current for value, current in zip(z, q, strict=True)]

        for _ in range(2):
            z = subtract_mean(z)
            for old in basis:
                coefficient = dot(old, z)
                if abs(coefficient) > 0.0:
                    z = [value - coefficient * old_value for value, old_value in zip(z, old, strict=True)]

        beta = norm(z)
        alphas.append(alpha)
        if beta <= 1.0e-10 * max(1.0, abs(alpha)) or step == min(steps, n - 1) - 1:
            break
        betas.append(beta)
        previous = q
        q = scale(z, 1.0 / beta)
        beta_previous = beta

    values = tridiagonal_eigenvalues(alphas, betas)
    largest = list(reversed(values[-count:]))
    return largest, 1000.0 * (perf_counter() - start), len(alphas)


def q_ritz_spectrum(spec: dict[str, object]) -> dict[str, object]:
    points = spec["points"]
    assert isinstance(points, list)
    speeds = estimate_pullback_speeds(
        points,
        closed=bool(spec["closed"]),
        force_unit_circle=str(spec["name"]) == "circle",
    )
    qjet = build_pullback_metric_qjet(speeds)
    high, high_ms, high_steps = lanczos_extreme_ritz(
        qjet,
        steps=LANCZOS_STEPS,
        count=RITZ_COUNT,
        sign=1.0,
        phase=0.0,
    )
    negative_low, low_ms, low_steps = lanczos_extreme_ritz(
        qjet,
        steps=LANCZOS_STEPS,
        count=RITZ_COUNT,
        sign=-1.0,
        phase=5.0,
    )
    low = sorted(max(0.0, -value) for value in negative_low)
    fast_units = int(qjet.stats()["apply_work_units"])
    pairwise_units = len(points) * (len(points) - 1) // 2
    return {
        "low_nonzero_ritz": low,
        "high_ritz": high,
        "low_elapsed_ms": low_ms,
        "high_elapsed_ms": high_ms,
        "low_lanczos_steps": low_steps,
        "high_lanczos_steps": high_steps,
        "speed_min": min(speeds),
        "speed_median": median(speeds),
        "speed_max": max(speeds),
        "speed_anisotropy": max(speeds) / min(speeds),
        "fast_apply_work_units": fast_units,
        "pairwise_apply_work_units": pairwise_units,
        "estimated_pairwise_over_fast": pairwise_units / max(float(fast_units), 1.0),
        "uses_radix2_fft": bool(qjet.stats()["uses_radix2_fft"]),
    }


def circle_exact_low_with_multiplicity(sample_count: int, count: int) -> list[float]:
    values: list[float] = []
    mode = 1
    while len(values) < count:
        value = 0.5 * mode * (sample_count - mode)
        if mode == sample_count - mode:
            values.append(value)
        else:
            values.extend([value, value])
        mode += 1
    return values[:count]


def relative_l2(left: list[float], right: list[float]) -> float:
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    denominator = max(math.sqrt(sum(a * a for a in left)), 1.0e-300)
    return numerator / denominator


def fast_apply_units(sample_count: int) -> int:
    levels = int(math.log2(sample_count)) if sample_count > 0 and sample_count & (sample_count - 1) == 0 else sample_count
    if sample_count > 0 and sample_count & (sample_count - 1) == 0:
        return 2 * sample_count * max(levels, 1) + 6 * sample_count
    return sample_count * sample_count


def pairwise_units(sample_count: int) -> int:
    return sample_count * max(sample_count - 1, 0) // 2


def write_spectrum_plot(results: dict[str, dict[str, object]]) -> Path:
    figure_path = OUT / "q_spectrum_shape_gallery.png"
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Matrix-free Q Ritz spectra across complex-plane shape encodings", fontsize=13)
    flat_axes = [ax for row in axes for ax in row]

    for ax, (name, result) in zip(flat_axes, results.items()):
        low = result["low_nonzero_ritz"]
        high = result["high_ritz"]
        assert isinstance(low, list)
        assert isinstance(high, list)
        x = list(range(1, len(low) + 1))
        ax.plot(x, low, color="0.0", marker="o", markersize=2.7, linewidth=1.0, label="low")
        ax.plot(x, high, color="0.45", marker="s", markersize=2.2, linewidth=0.85, label="high")
        ax.set_yscale("log")
        ax.grid(color="0.90", linewidth=0.5)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.text(
            0.02,
            0.02,
            f"lambda1={low[0]:.2e}\nhigh1={high[0]:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9, "pad": 2},
        )
        if name == "circle":
            ax.legend(loc="upper left", fontsize=6, frameon=False)

    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def write_outputs(specs: list[dict[str, object]], results: dict[str, dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ritz_csv = OUT / "q_spectrum_shape_gallery_ritz.csv"
    summary_csv = OUT / "q_spectrum_shape_gallery_summary.csv"
    json_path = OUT / "q_spectrum_shape_gallery.json"
    report_path = OUT / "q_spectrum_shape_gallery.md"
    figure_path = write_spectrum_plot(results)

    ritz_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        name = str(spec["name"])
        result = results[name]
        low = result["low_nonzero_ritz"]
        high = result["high_ritz"]
        assert isinstance(low, list)
        assert isinstance(high, list)
        for branch, values in (("low_nonzero", low), ("high", high)):
            for rank, value in enumerate(values, start=1):
                ritz_rows.append(
                    {
                        "shape": name,
                        "branch": branch,
                        "rank": rank,
                        "eigenvalue": value,
                        "log10_eigenvalue": math.log10(max(value, 1.0e-300)),
                        "sample_count": SPECTRUM_SAMPLE_COUNT,
                        "requested_lanczos_steps": LANCZOS_STEPS,
                    }
                )
        summary_rows.append(
            {
                "shape": name,
                "family": spec["family"],
                "lambda_1": low[0],
                "lambda_2": low[1],
                "lambda_3": low[2],
                "lambda_4": low[3],
                "lambda_8": low[7],
                "lambda_12": low[11],
                "largest_ritz": high[0],
                "high_4": high[3],
                "high_12": high[11],
                "condition_proxy": high[0] / max(low[0], 1.0e-300),
                "speed_anisotropy": result["speed_anisotropy"],
                "fast_apply_work_units": result["fast_apply_work_units"],
                "pairwise_apply_work_units": result["pairwise_apply_work_units"],
                "estimated_pairwise_over_fast": result["estimated_pairwise_over_fast"],
                "uses_radix2_fft": result["uses_radix2_fft"],
                "low_elapsed_ms": result["low_elapsed_ms"],
                "high_elapsed_ms": result["high_elapsed_ms"],
                "low_lanczos_steps": result["low_lanczos_steps"],
                "high_lanczos_steps": result["high_lanczos_steps"],
            }
        )

    with ritz_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ritz_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ritz_rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    circle_low = results["circle"]["low_nonzero_ritz"]
    assert isinstance(circle_low, list)
    exact_circle = circle_exact_low_with_multiplicity(SPECTRUM_SAMPLE_COUNT, RITZ_COUNT)
    circle_rel_l2 = relative_l2(circle_low, exact_circle)

    payload = {
        "method": {
            "dense_q_matrix_stored": False,
            "operator_application": "a_i=1/|dz/dtheta|_i; (Q_pull v)_i=a_i*sum_j c_ij*a_j*(v_i-v_j)",
            "stored_state": "boundary samples, pullback speed QJets, temporary Krylov vectors, and small Lanczos tridiagonal only",
            "spectrum_type": "matrix-free pullback principal Ritz eigenvalue levels; multiplicities are not audited except circle calibration",
            "sample_count": SPECTRUM_SAMPLE_COUNT,
            "lanczos_steps": LANCZOS_STEPS,
            "ritz_count_per_branch": RITZ_COUNT,
            "units": "inverse normalized-radius squared",
            "fast_operator": "pullback metric FFT weighted-edge QJet",
        },
        "circle_calibration": {
            "exact_low_levels_with_multiplicity": exact_circle,
            "computed_low_ritz": circle_low,
            "relative_l2_error": circle_rel_l2,
        },
        "summary": {row["shape"]: row for row in summary_rows},
        "ritz": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Q Spectrum Shape Gallery",
        "",
        "This is the spectrum companion to the discriminant-curvature gallery.  It applies the pullback principal Q matrix-free from complex boundary samples and reports Ritz eigenvalue levels for each shape.",
        "",
        f"![Q spectrum gallery]({figure_path})",
        "",
        "## Operator",
        "",
        "```text",
        "a_i = 1 / |dz/dtheta|_i",
        "(Q_pull v)_i = a_i sum_{j != i} c_ij a_j (v_i - v_j)",
        "c_ij = |exp(i theta_i) - exp(i theta_j)|^(-2)",
        "```",
        "",
        "No dense Q matrix is stored.  The dense object is only the small Lanczos tridiagonal used for the spectral readout.  Q application is the fast pullback-metric path: circle FFT kernel, metric-speed borrow, local repay.",
        "",
        "The units are inverse normalized-radius squared.  All shapes are centered and scaled before Q is applied.",
        "",
        "## Circle Calibration",
        "",
        "For the unit circle with `n=256`, the low Q levels are `lambda_m=m(n-m)/2`, with the usual sine/cosine multiplicity away from the Nyquist mode.  The relative L2 error of the computed low Ritz levels against the first 12 exact multiplicity-aware levels is `{:.3e}`.".format(
            circle_rel_l2
        ),
        "",
        "## Cost Model",
        "",
        "The previous diagnostic applied Q by direct pairs, costing `n(n-1)/2` pair interactions per apply.  The new path stores only the pullback speed jet and uses the circle FFT kernel, so each apply costs about `O(n log n)` work units for radix-2 sample counts.",
        "",
        "| shape | fast apply units | pairwise units | pairwise/fast | speed anisotropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {shape} | `{fast}` | `{pair}` | `{ratio:.3e}` | `{aniso:.3e}` |".format(
                shape=row["shape"],
                fast=int(row["fast_apply_work_units"]),
                pair=int(row["pairwise_apply_work_units"]),
                ratio=float(row["estimated_pairwise_over_fast"]),
                aniso=float(row["speed_anisotropy"]),
            )
        )
    lines.extend(
        [
            "",
            "For larger radix-2 runs the ratio grows quickly:",
            "",
            "| n | fast apply units | pairwise units | pairwise/fast |",
            "|---:|---:|---:|---:|",
        ]
    )
    for sample_count in (256, 1024, 4096, 16384):
        fast = fast_apply_units(sample_count)
        pairwise = pairwise_units(sample_count)
        lines.append(
            f"| `{sample_count}` | `{fast}` | `{pairwise}` | `{pairwise / max(float(fast), 1.0):.3e}` |"
        )
    lines.extend(
        [
            "",
            "## Eigenvalue Summary",
            "",
            "| shape | lambda1 | lambda2 | lambda3 | lambda4 | lambda8 | lambda12 | largest Ritz | condition proxy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {shape} | `{l1:.6e}` | `{l2:.6e}` | `{l3:.6e}` | `{l4:.6e}` | `{l8:.6e}` | `{l12:.6e}` | `{hi:.6e}` | `{cond:.6e}` |".format(
                shape=row["shape"],
                l1=float(row["lambda_1"]),
                l2=float(row["lambda_2"]),
                l3=float(row["lambda_3"]),
                l4=float(row["lambda_4"]),
                l8=float(row["lambda_8"]),
                l12=float(row["lambda_12"]),
                hi=float(row["largest_ritz"]),
                cond=float(row["condition_proxy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- figure: `{figure_path}`",
            f"- Ritz CSV: `{ritz_csv}`",
            f"- summary CSV: `{summary_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figure": str(figure_path), "ritz_csv": str(ritz_csv)}, indent=2))


def main() -> None:
    specs = shape_specs(SPECTRUM_SAMPLE_COUNT)
    results: dict[str, dict[str, object]] = {}
    for spec in specs:
        results[str(spec["name"])] = q_ritz_spectrum(spec)
    write_outputs(specs, results)


if __name__ == "__main__":
    main()
