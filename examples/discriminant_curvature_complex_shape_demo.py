#!/usr/bin/env python3
"""Discriminant curvature through z = exp(i theta).

The demo does not assemble a dense Q matrix.  It stores only generated complex
boundary samples and the row-sum curvature

    q_jj = sum_{k != j} |z_j - z_k|^{-2}.

For the circle z_j = exp(i theta_j), this row-sum comes from the Hessian of
the log-discriminant in the theta variables.  For polynomial and rational
complex shape charts, the same chord curvature highlights cusps, tangent
cones, and asymptotic residuals.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "discriminant_curvature_complex_shape"
TAU = 2.0 * math.pi


def cis(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def circle(theta: float) -> complex:
    return cis(theta)


def cardioid(theta: float) -> complex:
    w = cis(theta)
    return w - 0.5 * (w * w + 1.0)


def nephroid(theta: float) -> complex:
    w = cis(theta)
    return 3.0 * w - w * w * w


def rational_chart(t: float) -> complex:
    q = -0.02 + 0.18 * t
    residual = 0.025 * ((t - 0.45) ** 2) * ((t + 0.85) ** 3) / (1.0 + t**6)
    return complex(t, q + residual)


def rational_asymptote(t: float) -> complex:
    return complex(t, -0.02 + 0.18 * t)


def sample_periodic(curve, n: int) -> list[complex]:
    return [curve(TAU * (j + 0.5) / n) for j in range(n)]


def sample_interval(curve, n: int, left: float = -3.0, right: float = 3.0) -> list[complex]:
    return [curve(left + (right - left) * j / (n - 1)) for j in range(n)]


def discriminant_row_sums(points: list[complex]) -> list[float]:
    out: list[float] = []
    for i, zi in enumerate(points):
        total = 0.0
        for j, zj in enumerate(points):
            if i == j:
                continue
            distance = abs(zi - zj)
            total += 1.0 / max(distance * distance, 1.0e-300)
        out.append(total)
    return out


def log_values(values: list[float]) -> list[float]:
    return [math.log10(max(value, 1.0e-300)) for value in values]


def compact(value: float) -> str:
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def tangent_direction(curve, theta0: float, eps: float = 1.0e-4) -> complex:
    z0 = curve(theta0)
    dz = curve(theta0 + eps) - z0
    if abs(dz) < 1.0e-12:
        dz = curve(theta0 + 2.0 * eps) - z0
    return dz / abs(dz)


def draw_tangent(ax, point: complex, direction: complex, span: float) -> None:
    a = point - span * direction
    b = point + span * direction
    ax.plot([a.real, b.real], [a.imag, b.imag], color="0.55", linestyle="--", linewidth=1.0)


def plot_shape_curvature(ax, points: list[complex], values: list[float], title: str, cusp_thetas=(), curve=None) -> None:
    logs = log_values(values)
    norm = mcolors.Normalize(vmin=min(logs), vmax=max(logs))
    xs = [z.real for z in points] + [points[0].real]
    ys = [z.imag for z in points] + [points[0].imag]
    ax.plot(xs, ys, color="0.82", linewidth=0.8)
    scatter = ax.scatter([z.real for z in points], [z.imag for z in points], c=logs, cmap="gray_r", norm=norm, s=9)
    if curve is not None:
        for theta0 in cusp_thetas:
            z0 = curve(theta0)
            draw_tangent(ax, z0, tangent_direction(curve, theta0), 0.45 if len(cusp_thetas) == 1 else 0.7)
            ax.scatter([z0.real], [z0.imag], facecolors="white", edgecolors="0.05", s=42, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    return scatter


def circle_eigenvalues(n: int) -> list[tuple[int, float]]:
    return [(m, 0.5 * m * (n - m)) for m in range(n // 2 + 1)]


def make_figure(payload: dict[str, object], path: Path) -> None:
    circle_points = payload["circle_points"]
    cardioid_points = payload["cardioid_points"]
    nephroid_points = payload["nephroid_points"]
    rational_points = payload["rational_points"]
    circle_q = payload["circle_qdiag"]
    cardioid_q = payload["cardioid_qdiag"]
    nephroid_q = payload["nephroid_qdiag"]
    rational_q = payload["rational_qdiag"]
    assert isinstance(circle_points, list)
    assert isinstance(cardioid_points, list)
    assert isinstance(nephroid_points, list)
    assert isinstance(rational_points, list)
    assert isinstance(circle_q, list)
    assert isinstance(cardioid_q, list)
    assert isinstance(nephroid_q, list)
    assert isinstance(rational_q, list)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Discriminant curvature from z = exp(i theta) to hard shape charts", fontsize=13)

    ax = axes[0][0]
    ax.axis("off")
    text = "\n".join(
        [
            "complex normal form",
            "",
            "z_j = exp(i theta_j)",
            "D(z) = prod_{i<j}(z_i-z_j)",
            "L(theta)=log |D(z(theta))|",
            "",
            "d^2 L / dtheta_i dtheta_j",
            "  = 1 / |z_i-z_j|^2,  i != j",
            "",
            "Q = - Hess_theta L",
            "Q_ij = -|z_i-z_j|^{-2}",
            "Q_ii = sum_{j != i}|z_i-z_j|^{-2}",
        ]
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=10, family="monospace")

    ax = axes[0][1]
    eig = circle_eigenvalues(96)
    ax.plot([m for m, _ in eig], [value for _, value in eig], color="0.1", linewidth=1.4)
    ax.set_title("circle spectrum: lambda_m=m(n-m)/2")
    ax.set_xlabel("mode m")
    ax.set_ylabel("discriminant stiffness")
    ax.grid(color="0.9", linewidth=0.6)

    ax = axes[0][2]
    plot_shape_curvature(ax, circle_points, circle_q, "circle: constant curvature density")

    ax = axes[1][0]
    plot_shape_curvature(ax, cardioid_points, cardioid_q, "cardioid: one cusp spike", (0.0,), cardioid)

    ax = axes[1][1]
    plot_shape_curvature(ax, nephroid_points, nephroid_q, "nephroid: two cusp spikes", (0.0, math.pi), nephroid)

    ax = axes[1][2]
    t_values = [-3.0 + 6.0 * i / (len(rational_points) - 1) for i in range(len(rational_points))]
    ax.plot([z.real for z in rational_points], [z.imag for z in rational_points], color="0.1", linewidth=1.4, label="rational chart")
    asym = [rational_asymptote(t) for t in t_values]
    ax.plot([z.real for z in asym], [z.imag for z in asym], color="0.55", linestyle="--", linewidth=1.1, label="Q asymptote")
    logs = log_values(rational_q)
    ax.scatter([z.real for z in rational_points], [z.imag for z in rational_points], c=logs, cmap="gray_r", s=7)
    for root, label in ((0.45, "touch"), (-0.85, "cross")):
        z = rational_chart(root)
        ax.scatter([z.real], [z.imag], facecolors="white", edgecolors="0.05", s=38, zorder=5)
        ax.text(z.real, z.imag + 0.08, label, ha="center", fontsize=8)
    ax.set_title("rational chart: asymptote plus residual")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.grid(color="0.9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def complex_pairs(points: list[complex]) -> list[list[float]]:
    return [[float(z.real), float(z.imag)] for z in points]


def max_index(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def periodic_neighbor_pair(theta0: float, n: int) -> list[int]:
    center = theta0 * n / TAU - 0.5
    lower = math.floor(center) % n
    upper = (lower + 1) % n
    return [lower, upper]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(payload: dict[str, object], report_path: Path, figure_path: Path, csv_path: Path, json_path: Path) -> None:
    cardioid_q = payload["cardioid_qdiag"]
    nephroid_q = payload["nephroid_qdiag"]
    circle_q = payload["circle_qdiag"]
    assert isinstance(cardioid_q, list)
    assert isinstance(nephroid_q, list)
    assert isinstance(circle_q, list)
    circle_variation = max(circle_q) / min(circle_q)
    c_index = max_index(cardioid_q)
    nephroid_right = periodic_neighbor_pair(0.0, len(nephroid_q))
    nephroid_left = periodic_neighbor_pair(math.pi, len(nephroid_q))
    lines = [
        "# Discriminant Curvature Through `exp(i theta)`",
        "",
        "The complex-plane shape story is controlled by the log-discriminant.",
        "",
        f"![discriminant curvature]({figure_path})",
        "",
        "## Chain",
        "",
        "Start on the circle:",
        "",
        "```text",
        "z_j = exp(i theta_j) = cos(theta_j) + sqrt(-1) sin(theta_j)",
        "D(z) = prod_{i<j} (z_i - z_j)",
        "L(theta) = log |D(z(theta))|.",
        "```",
        "",
        "Differentiating through `exp(i theta)` gives",
        "",
        "```text",
        "partial_i partial_j L = 1 / |z_i - z_j|^2,  i != j.",
        "```",
        "",
        "With the positive graph-Laplacian sign convention used by Q,",
        "",
        "```text",
        "Q = - Hess_theta L",
        "Q_ij = -1 / |z_i - z_j|^2",
        "Q_ii = sum_{j != i} 1 / |z_i - z_j|^2.",
        "```",
        "",
        "So Q is discriminant curvature: it is the boundary stiffness of the log-discriminant after the shape is encoded as complex samples.",
        "",
        "## Why This Captures The Hard Parts",
        "",
        "For an analytic shape, write the boundary as a deformation of the circle:",
        "",
        "```text",
        "z(theta) = psi(exp(i theta)).",
        "```",
        "",
        "The same chain rule sends",
        "",
        "```text",
        "d/dtheta psi(exp(i theta)) = psi'(exp(i theta)) i exp(i theta).",
        "```",
        "",
        "When this tangent is nonzero, nearby chords scale like `|z_theta| |theta_i-theta_j|`. When the tangent vanishes, the first nonzero jet gives a cusp or tangent cone and the inverse-square chord curvature spikes. For rational charts, polynomial division `N/D = Q + R/D` separates the asymptotic carrier `Q` from the residual; the chord curvature still sees both because it is computed from the physical complex points.",
        "",
        "## Numeric Checks",
        "",
        f"- circle curvature row-sum variation: `{circle_variation:.6f}`; it is constant up to roundoff/sampling symmetry",
        f"- cardioid maximum curvature index: `{c_index}` near its single cusp",
        f"- nephroid cusp-neighbor index pairs: `{nephroid_right}` at `z=2`, `{nephroid_left}` at `z=-2`",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        f"- Figure: `{figure_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    periodic_n = 384
    interval_n = 384
    circle_points = sample_periodic(circle, periodic_n)
    cardioid_points = sample_periodic(cardioid, periodic_n)
    nephroid_points = sample_periodic(nephroid, periodic_n)
    rational_points = sample_interval(rational_chart, interval_n)

    circle_q = discriminant_row_sums(circle_points)
    cardioid_q = discriminant_row_sums(cardioid_points)
    nephroid_q = discriminant_row_sums(nephroid_points)
    rational_q = discriminant_row_sums(rational_points)

    payload = {
        "method": {
            "dense_matrix_stored": False,
            "stored": "boundary QJets plus Q diagonal row-sum curvature; no dense Q matrix",
            "discriminant": "D(z)=prod_{i<j}(z_i-z_j)",
            "curvature": "Q=-Hess_theta log|D(z(theta))| on circle; generalized by physical chords",
        },
        "circle_points": circle_points,
        "cardioid_points": cardioid_points,
        "nephroid_points": nephroid_points,
        "rational_points": rational_points,
        "circle_qdiag": circle_q,
        "cardioid_qdiag": cardioid_q,
        "nephroid_qdiag": nephroid_q,
        "rational_qdiag": rational_q,
    }

    figure_path = OUT / "discriminant_curvature_complex_shape.png"
    csv_path = OUT / "discriminant_curvature_samples.csv"
    json_path = OUT / "discriminant_curvature_complex_shape.json"
    report_path = OUT / "discriminant_curvature_complex_shape.md"

    make_figure(payload, figure_path)

    rows: list[dict[str, object]] = []
    for name, points, values in (
        ("circle", circle_points, circle_q),
        ("cardioid", cardioid_points, cardioid_q),
        ("nephroid", nephroid_points, nephroid_q),
        ("rational", rational_points, rational_q),
    ):
        for index, (point, value) in enumerate(zip(points, values)):
            rows.append(
                {
                    "shape": name,
                    "index": index,
                    "real": point.real,
                    "imag": point.imag,
                    "q_diag_row_sum": value,
                    "log10_q_diag": math.log10(max(value, 1.0e-300)),
                }
            )
    write_csv(csv_path, rows)

    json_payload = {
        "method": payload["method"],
        "summary": {
            "circle_qdiag_min": min(circle_q),
            "circle_qdiag_max": max(circle_q),
            "cardioid_max_index": max_index(cardioid_q),
            "cardioid_max_qdiag": max(cardioid_q),
            "nephroid_right_cusp_neighbor_indices": periodic_neighbor_pair(0.0, len(nephroid_q)),
            "nephroid_left_cusp_neighbor_indices": periodic_neighbor_pair(math.pi, len(nephroid_q)),
            "nephroid_max_qdiag": max(nephroid_q),
            "rational_max_qdiag": max(rational_q),
        },
        "circle_points": complex_pairs(circle_points),
        "cardioid_points": complex_pairs(cardioid_points),
        "nephroid_points": complex_pairs(nephroid_points),
        "rational_points": complex_pairs(rational_points),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_report(payload, report_path, figure_path, csv_path, json_path)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
