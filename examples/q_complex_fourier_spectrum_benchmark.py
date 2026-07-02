#!/usr/bin/env python3
"""Complex de Moivre/Fourier spectrum probe for pullback Q.

This follows the scale-phase description in the paper: use complex characters
``exp(i k theta)`` as the spectral basis, apply the pullback QJet by the FFT
kernel, and diagonalize only a small Hermitian Fourier block.  It is a fast
band-projected spectral diagnostic, not a full dense Q eigensolve.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discriminant_curvature_shape_gallery import shape_specs
from inverse_shape.quadrature import build_pullback_metric_qjet
from q_spectrum_shape_gallery import (
    RITZ_COUNT,
    SPECTRUM_SAMPLE_COUNT,
    TAU,
    estimate_pullback_speeds,
    relative_l2,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "discriminant_curvature_shape_gallery"
LOW_MODE_RADIUS = 24
HIGH_MODE_RADIUS = 24


def fourier_mode(sample_count: int, mode: int) -> list[complex]:
    scale = 1.0 / math.sqrt(sample_count)
    return [
        scale * complex(math.cos(TAU * mode * index / sample_count), math.sin(TAU * mode * index / sample_count))
        for index in range(sample_count)
    ]


def complex_inner(left: list[complex], right: list[complex]) -> complex:
    return sum(a.conjugate() * b for a, b in zip(left, right, strict=True))


def low_modes(radius: int) -> list[int]:
    modes: list[int] = []
    for mode in range(1, radius + 1):
        modes.extend([mode, -mode])
    return modes


def high_modes(sample_count: int, radius: int) -> list[int]:
    center = sample_count // 2
    modes = [center]
    for offset in range(1, radius + 1):
        modes.extend([center - offset, center + offset])
    return modes


def hermitian_block(qjet: object, modes: list[int]) -> tuple[list[list[complex]], float, int]:
    n = qjet.n
    basis = [fourier_mode(n, mode % n) for mode in modes]
    start = perf_counter()
    applied = [[complex(value) for value in qjet.apply(vector)] for vector in basis]
    matrix: list[list[complex]] = []
    for row_vector in basis:
        matrix.append([complex_inner(row_vector, column) for column in applied])
    for i in range(len(matrix)):
        matrix[i][i] = complex(matrix[i][i].real, 0.0)
        for j in range(i + 1, len(matrix)):
            average = 0.5 * (matrix[i][j] + matrix[j][i].conjugate())
            matrix[i][j] = average
            matrix[j][i] = average.conjugate()
    elapsed_ms = 1000.0 * (perf_counter() - start)
    return matrix, elapsed_ms, len(modes)


def symmetric_jacobi_eigenvalues(matrix: list[list[float]]) -> list[float]:
    size = len(matrix)
    if size == 0:
        return []
    scale_hint = max(max(abs(value) for value in row) for row in matrix)
    tolerance = 1.0e-11 * max(scale_hint, 1.0)
    max_rotations = max(4000, 16 * size * size)
    a = [row[:] for row in matrix]

    for _ in range(max_rotations):
        pivot_i = 0
        pivot_j = 1 if size > 1 else 0
        largest = 0.0
        for i in range(size):
            for j in range(i + 1, size):
                value = abs(a[i][j])
                if value > largest:
                    largest = value
                    pivot_i = i
                    pivot_j = j
        if largest <= tolerance or size <= 1:
            break

        p = pivot_i
        q = pivot_j
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        if abs(apq) <= tolerance:
            a[p][q] = 0.0
            a[q][p] = 0.0
            continue
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for k in range(size):
            if k == p or k == q:
                continue
            akp = a[k][p]
            akq = a[k][q]
            new_kp = cosine * akp - sine * akq
            new_kq = sine * akp + cosine * akq
            a[k][p] = new_kp
            a[p][k] = new_kp
            a[k][q] = new_kq
            a[q][k] = new_kq

        a[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        a[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0
    return sorted(a[i][i] for i in range(size))


def hermitian_eigenvalues(matrix: list[list[complex]]) -> list[float]:
    size = len(matrix)
    real_lift = [[0.0 for _ in range(2 * size)] for _ in range(2 * size)]
    for i in range(size):
        for j in range(size):
            value = matrix[i][j]
            real_lift[i][j] = value.real
            real_lift[i][j + size] = -value.imag
            real_lift[i + size][j] = value.imag
            real_lift[i + size][j + size] = value.real
    lifted = symmetric_jacobi_eigenvalues(real_lift)
    compressed = []
    for index in range(0, len(lifted), 2):
        if index + 1 < len(lifted):
            compressed.append(0.5 * (lifted[index] + lifted[index + 1]))
        else:
            compressed.append(lifted[index])
    return compressed


def complex_fourier_spectrum(spec: dict[str, object]) -> dict[str, object]:
    points = spec["points"]
    assert isinstance(points, list)
    speeds = estimate_pullback_speeds(
        points,
        closed=bool(spec["closed"]),
        force_unit_circle=str(spec["name"]) == "circle",
    )
    qjet = build_pullback_metric_qjet(speeds)

    low_block, low_apply_ms, low_applications = hermitian_block(qjet, low_modes(LOW_MODE_RADIUS))
    low_start = perf_counter()
    low_values = [value for value in hermitian_eigenvalues(low_block) if value > 1.0e-8]
    low_values = sorted(low_values)[:RITZ_COUNT]
    low_eig_ms = 1000.0 * (perf_counter() - low_start)

    high_block, high_apply_ms, high_applications = hermitian_block(
        qjet,
        high_modes(qjet.n, HIGH_MODE_RADIUS),
    )
    high_start = perf_counter()
    high_values = list(reversed(sorted(hermitian_eigenvalues(high_block))))[:RITZ_COUNT]
    high_eig_ms = 1000.0 * (perf_counter() - high_start)

    return {
        "low_nonzero_ritz": low_values,
        "high_ritz": high_values,
        "low_apply_ms": low_apply_ms,
        "high_apply_ms": high_apply_ms,
        "low_eig_ms": low_eig_ms,
        "high_eig_ms": high_eig_ms,
        "total_ms": low_apply_ms + high_apply_ms + low_eig_ms + high_eig_ms,
        "low_mode_count": low_applications,
        "high_mode_count": high_applications,
        "speed_anisotropy": max(speeds) / min(speeds),
    }


def load_lanczos_reference() -> dict[str, object]:
    path = OUT / "q_spectrum_shape_gallery.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compare_with_reference(name: str, result: dict[str, object], reference: dict[str, object]) -> dict[str, float]:
    ritz = reference.get("ritz", {}) if reference else {}
    ref = ritz.get(name, {}) if isinstance(ritz, dict) else {}
    ref_low = ref.get("low_nonzero_ritz", [])
    ref_high = ref.get("high_ritz", [])
    low = result["low_nonzero_ritz"]
    high = result["high_ritz"]
    assert isinstance(low, list)
    assert isinstance(high, list)
    out = {
        "low_relative_l2_vs_lanczos": float("nan"),
        "high_relative_l2_vs_lanczos": float("nan"),
        "lanczos_total_ms": float("nan"),
        "speedup_vs_lanczos": float("nan"),
    }
    if isinstance(ref_low, list) and len(ref_low) >= len(low):
        out["low_relative_l2_vs_lanczos"] = relative_l2(low, ref_low[: len(low)])
    if isinstance(ref_high, list) and len(ref_high) >= len(high):
        out["high_relative_l2_vs_lanczos"] = relative_l2(high, ref_high[: len(high)])
    if isinstance(ref, dict):
        low_ms = ref.get("low_elapsed_ms")
        high_ms = ref.get("high_elapsed_ms")
        if isinstance(low_ms, int | float) and isinstance(high_ms, int | float):
            out["lanczos_total_ms"] = float(low_ms + high_ms)
            out["speedup_vs_lanczos"] = out["lanczos_total_ms"] / max(float(result["total_ms"]), 1.0e-300)
    return out


def write_plot(results: dict[str, dict[str, object]], comparisons: dict[str, dict[str, float]]) -> Path:
    figure_path = OUT / "q_complex_fourier_spectrum_benchmark.png"
    names = list(results)
    speedups = [comparisons[name]["speedup_vs_lanczos"] for name in names]
    low_errors = [comparisons[name]["low_relative_l2_vs_lanczos"] for name in names]
    high_errors = [comparisons[name]["high_relative_l2_vs_lanczos"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    axes[0].bar(range(len(names)), speedups, color="0.25")
    axes[0].set_yscale("log")
    axes[0].set_title("complex Fourier block speedup vs real Lanczos")
    axes[0].set_ylabel("speedup")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    axes[0].grid(axis="y", color="0.9")

    axes[1].plot(range(len(names)), low_errors, color="0.0", marker="o", linewidth=1.0, label="low")
    axes[1].plot(range(len(names)), high_errors, color="0.5", marker="s", linewidth=1.0, label="high")
    axes[1].set_yscale("log")
    axes[1].set_title("relative L2 difference from Lanczos")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    axes[1].grid(color="0.9")
    axes[1].legend(frameon=False)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def write_outputs(results: dict[str, dict[str, object]], comparisons: dict[str, dict[str, float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "q_complex_fourier_spectrum_benchmark.csv"
    json_path = OUT / "q_complex_fourier_spectrum_benchmark.json"
    report_path = OUT / "q_complex_fourier_spectrum_benchmark.md"
    figure_path = write_plot(results, comparisons)

    rows: list[dict[str, object]] = []
    for name, result in results.items():
        comparison = comparisons[name]
        rows.append(
            {
                "shape": name,
                "complex_total_ms": result["total_ms"],
                "lanczos_total_ms": comparison["lanczos_total_ms"],
                "speedup_vs_lanczos": comparison["speedup_vs_lanczos"],
                "low_relative_l2_vs_lanczos": comparison["low_relative_l2_vs_lanczos"],
                "high_relative_l2_vs_lanczos": comparison["high_relative_l2_vs_lanczos"],
                "low_mode_count": result["low_mode_count"],
                "high_mode_count": result["high_mode_count"],
                "speed_anisotropy": result["speed_anisotropy"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "method": {
            "basis": "complex de Moivre/Fourier characters exp(i k theta)",
            "dense_q_matrix_stored": False,
            "dense_object": "small Hermitian Fourier-Galerkin block only",
            "low_mode_radius": LOW_MODE_RADIUS,
            "high_mode_radius": HIGH_MODE_RADIUS,
            "sample_count": SPECTRUM_SAMPLE_COUNT,
            "interpretation": "fast band-projected spectrum probe, not a full Krylov spectrum",
        },
        "results": results,
        "comparisons": comparisons,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Complex Fourier Q Spectrum Benchmark",
        "",
        "This tests the paper's de Moivre route: project the pullback Q operator onto complex characters `exp(i k theta)`, apply Q by the FFT QJet, and diagonalize only a small Hermitian Fourier block.",
        "",
        f"![complex benchmark]({figure_path})",
        "",
        "This is faster because it replaces two 124-step real Lanczos runs with direct probes of selected complex Fourier bands. It is exact on the circle when the requested modes are inside the block; on strongly singular shapes the high branch is a band probe and can differ from full Lanczos if the top mode is localized outside the selected Fourier window.",
        "",
        "| shape | complex ms | real Lanczos ms | speedup | low rel L2 | high rel L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {shape} | `{complex_ms:.3f}` | `{lanczos_ms:.3f}` | `{speedup:.3e}` | `{low_err:.3e}` | `{high_err:.3e}` |".format(
                shape=row["shape"],
                complex_ms=float(row["complex_total_ms"]),
                lanczos_ms=float(row["lanczos_total_ms"]),
                speedup=float(row["speedup_vs_lanczos"]),
                low_err=float(row["low_relative_l2_vs_lanczos"]),
                high_err=float(row["high_relative_l2_vs_lanczos"]),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- figure: `{figure_path}`",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figure": str(figure_path), "csv": str(csv_path)}, indent=2))


def main() -> None:
    specs = shape_specs(SPECTRUM_SAMPLE_COUNT)
    reference = load_lanczos_reference()
    results: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, float]] = {}
    for spec in specs:
        name = str(spec["name"])
        result = complex_fourier_spectrum(spec)
        results[name] = result
        comparisons[name] = compare_with_reference(name, result, reference)
    write_outputs(results, comparisons)


if __name__ == "__main__":
    main()
