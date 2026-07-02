#!/usr/bin/env python3
"""Gallery of discriminant curvature on varied complex-plane shapes.

Each shape is represented by generated complex samples z_j.  The plotted
quantity is the row-sum discriminant curvature

    q_j = sum_{k != j} |z_j - z_k|^{-2}.

This is the diagonal of Q, but the script never stores the dense Q matrix.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "discriminant_curvature_shape_gallery"
TAU = 2.0 * math.pi


def cis(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def normalize(points: list[complex]) -> list[complex]:
    center = sum(points, 0j) / len(points)
    shifted = [z - center for z in points]
    scale = max(abs(z) for z in shifted)
    if scale <= 0.0:
        return shifted
    return [z / scale for z in shifted]


def sample_periodic(curve, n: int) -> list[complex]:
    return normalize([curve(TAU * (j + 0.5) / n) for j in range(n)])


def sample_open(curve, n: int, left: float, right: float) -> list[complex]:
    return normalize([curve(left + (right - left) * j / (n - 1)) for j in range(n)])


def polygon_perimeter(vertices: list[complex]) -> float:
    return sum(abs(vertices[(i + 1) % len(vertices)] - vertices[i]) for i in range(len(vertices)))


def sample_polygon(vertices: list[complex], n: int, phase: float = 0.5) -> list[complex]:
    lengths = [abs(vertices[(i + 1) % len(vertices)] - vertices[i]) for i in range(len(vertices))]
    total = sum(lengths)
    out: list[complex] = []
    edge = 0
    edge_start = 0.0
    for j in range(n):
        target = total * (j + phase) / n
        while edge + 1 < len(vertices) and edge_start + lengths[edge] < target:
            edge_start += lengths[edge]
            edge += 1
        span = lengths[edge]
        alpha = 0.0 if span <= 0.0 else (target - edge_start) / span
        a = vertices[edge]
        b = vertices[(edge + 1) % len(vertices)]
        out.append(a + alpha * (b - a))
    return normalize(out)


def circle(theta: float) -> complex:
    return cis(theta)


def golden_ellipse(theta: float) -> complex:
    return complex(3.0 * math.cos(theta), math.sqrt(5.0) * math.sin(theta))


def rounded_square(theta: float) -> complex:
    c = math.cos(theta)
    s = math.sin(theta)
    p = 4.0
    x = math.copysign(abs(c) ** (2.0 / p), c)
    y = math.copysign(abs(s) ** (2.0 / p), s)
    return complex(x, y)


def flower(theta: float) -> complex:
    r = 1.0 + 0.20 * math.cos(5.0 * theta) + 0.08 * math.sin(3.0 * theta)
    return r * cis(theta)


def cardioid(theta: float) -> complex:
    w = cis(theta)
    return w - 0.5 * (w * w + 1.0)


def nephroid(theta: float) -> complex:
    w = cis(theta)
    return 3.0 * w - w * w * w


def astroid(theta: float) -> complex:
    c = math.cos(theta)
    s = math.sin(theta)
    return complex(c * c * c, s * s * s)


def joukowski_airfoil(theta: float) -> complex:
    a = 1.0
    center = complex(-0.10, 0.08)
    radius = abs(complex(a, 0.0) - center)
    theta_c = math.atan2(-center.imag, a - center.real)
    zeta = center + radius * cis(theta + theta_c)
    return zeta + (a * a) / zeta


def square_vertices() -> list[complex]:
    return [complex(-1, -1), complex(1, -1), complex(1, 1), complex(-1, 1)]


def star_vertices() -> list[complex]:
    out: list[complex] = []
    for j in range(10):
        radius = 1.0 if j % 2 == 0 else 0.42
        theta = math.pi / 2.0 + TAU * j / 10.0
        out.append(radius * cis(theta))
    return out


def stealth_vertices() -> list[complex]:
    return [
        complex(1.35, 0.0),
        complex(0.36, 0.20),
        complex(0.04, 0.62),
        complex(-0.92, 0.31),
        complex(-1.28, 0.0),
        complex(-0.92, -0.31),
        complex(0.04, -0.62),
        complex(0.36, -0.20),
    ]


def rational_chart(t: float) -> complex:
    q = -0.02 + 0.18 * t
    residual = 0.025 * ((t - 0.45) ** 2) * ((t + 0.85) ** 3) / (1.0 + t**6)
    return complex(t, q + residual)


def discriminant_row_sums(points: list[complex]) -> list[float]:
    values: list[float] = []
    for i, zi in enumerate(points):
        total = 0.0
        for j, zj in enumerate(points):
            if i == j:
                continue
            distance = abs(zi - zj)
            total += 1.0 / max(distance * distance, 1.0e-300)
        values.append(total)
    return values


def shape_specs(n: int) -> list[dict[str, object]]:
    return [
        {
            "name": "circle",
            "family": "smooth normal form",
            "hard_part": "none; constant chord-curvature density",
            "closed": True,
            "points": sample_periodic(circle, n),
        },
        {
            "name": "golden_ellipse",
            "family": "smooth conic",
            "hard_part": "metric modulation from closed-form Laurent pullback",
            "closed": True,
            "points": sample_periodic(golden_ellipse, n),
        },
        {
            "name": "rounded_square",
            "family": "smooth high-curvature",
            "hard_part": "large but finite curvature bands",
            "closed": True,
            "points": sample_periodic(rounded_square, n),
        },
        {
            "name": "flower_nonconvex",
            "family": "smooth nonconvex",
            "hard_part": "oscillatory metric/chord modulation",
            "closed": True,
            "points": sample_periodic(flower, n),
        },
        {
            "name": "cardioid_one_cusp",
            "family": "cusp",
            "hard_part": "one vanishing tangent; one tangent cone",
            "closed": True,
            "points": sample_periodic(cardioid, n),
        },
        {
            "name": "nephroid_two_cusps",
            "family": "multi-cusp",
            "hard_part": "two vanishing tangents",
            "closed": True,
            "points": sample_periodic(nephroid, n),
        },
        {
            "name": "astroid_four_cusps",
            "family": "multi-cusp",
            "hard_part": "four tangent-cone endpoints",
            "closed": True,
            "points": sample_periodic(astroid, n),
        },
        {
            "name": "joukowski_airfoil",
            "family": "Joukowski cusp",
            "hard_part": "critical-point cusp from zeta+a^2/zeta",
            "closed": True,
            "points": sample_periodic(joukowski_airfoil, n),
        },
        {
            "name": "square_polygon",
            "family": "polygonal corners",
            "hard_part": "four tangent jumps / Kondrat'ev vertices",
            "closed": True,
            "points": sample_polygon(square_vertices(), n, phase=0.0),
        },
        {
            "name": "star_polygon",
            "family": "re-entrant polygon",
            "hard_part": "alternating convex/re-entrant corner channel",
            "closed": True,
            "points": sample_polygon(star_vertices(), n, phase=0.0),
        },
        {
            "name": "double_concave_stealth",
            "family": "concave polygon",
            "hard_part": "symmetric concave vertex scattering",
            "closed": True,
            "points": sample_polygon(stealth_vertices(), n, phase=0.0),
        },
        {
            "name": "rational_asymptote_chart",
            "family": "open rational chart",
            "hard_part": "Q asymptote plus R/D touch-cross residual",
            "closed": False,
            "points": sample_open(rational_chart, n, -3.0, 3.0),
        },
    ]


def log_values(values: list[float]) -> list[float]:
    return [math.log10(max(value, 1.0e-300)) for value in values]


def percentile(values: list[float], fraction: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def robust_ratio(values: list[float]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    low = sorted_values[max(0, int(0.10 * (n - 1)))]
    high = sorted_values[min(n - 1, int(0.90 * (n - 1)))]
    return high / max(low, 1.0e-300)


def top_peak_points(points: list[complex], values: list[float], count: int = 4) -> list[dict[str, float]]:
    picked: list[int] = []
    for index in sorted(range(len(values)), key=lambda i: values[i], reverse=True):
        if all(min(abs(index - j), len(values) - abs(index - j)) > 6 for j in picked):
            picked.append(index)
        if len(picked) >= count:
            break
    return [
        {
            "index": index,
            "real": points[index].real,
            "imag": points[index].imag,
            "q_diag": values[index],
        }
        for index in picked
    ]


def draw_panel(ax, spec: dict[str, object], values: list[float]) -> None:
    points = spec["points"]
    closed = bool(spec["closed"])
    assert isinstance(points, list)
    logs = log_values(values)
    color_low = percentile(logs, 0.05)
    color_high = percentile(logs, 0.95)
    if color_high <= color_low:
        color_low = min(logs)
        color_high = max(logs)
    if color_high <= color_low:
        color_high = color_low + 1.0
    norm = mcolors.Normalize(vmin=color_low, vmax=color_high, clip=True)
    segments = []
    edge_count = len(points) if closed else len(points) - 1
    outline_x = [z.real for z in points]
    outline_y = [z.imag for z in points]
    if closed:
        outline_x.append(points[0].real)
        outline_y.append(points[0].imag)
    ax.plot(outline_x, outline_y, color="0.74", linewidth=0.75, zorder=1)
    for i in range(edge_count):
        a = points[i]
        b = points[(i + 1) % len(points)]
        segments.append([(a.real, a.imag), (b.real, b.imag)])
    edge_colors = []
    for i in range(edge_count):
        j = (i + 1) % len(points)
        edge_colors.append(0.5 * (logs[i] + logs[j]))
    collection = LineCollection(segments, cmap="gray_r", norm=norm, linewidths=2.15)
    collection.set_array(edge_colors)
    collection.set_zorder(2)
    ax.add_collection(collection)
    peaks = top_peak_points(points, values, count=4)
    ax.scatter([p["real"] for p in peaks], [p["imag"] for p in peaks], facecolors="white", edgecolors="0.05", s=20, linewidths=0.8)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.91", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{spec['name']}\n{spec['family']}", fontsize=8)
    ratio = max(values) / max(median(values), 1.0e-300)
    ax.text(
        0.02,
        0.02,
        f"max/med={ratio:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9, "pad": 2},
    )


def write_outputs(specs: list[dict[str, object]], q_values: dict[str, list[float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_path = OUT / "discriminant_curvature_shape_gallery.png"
    summary_csv = OUT / "discriminant_curvature_shape_gallery_summary.csv"
    samples_csv = OUT / "discriminant_curvature_shape_gallery_samples.csv"
    json_path = OUT / "discriminant_curvature_shape_gallery.json"
    report_path = OUT / "discriminant_curvature_shape_gallery.md"

    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Discriminant curvature across complex-plane shape encodings", fontsize=13)
    flat_axes = [ax for row in axes for ax in row]
    for ax, spec in zip(flat_axes, specs):
        draw_panel(ax, spec, q_values[str(spec["name"])])
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    json_summary: dict[str, object] = {}
    for spec in specs:
        name = str(spec["name"])
        points = spec["points"]
        assert isinstance(points, list)
        values = q_values[name]
        q_min = min(values)
        q_med = median(values)
        q_max = max(values)
        peaks = top_peak_points(points, values)
        summary = {
            "shape": name,
            "family": spec["family"],
            "hard_part": spec["hard_part"],
            "sample_count": len(points),
            "q_min": q_min,
            "q_median": q_med,
            "q_max": q_max,
            "max_over_median": q_max / max(q_med, 1.0e-300),
            "q90_over_q10": robust_ratio(values),
            "peak_points": peaks,
        }
        json_summary[name] = summary
        summary_rows.append({key: value for key, value in summary.items() if key != "peak_points"})
        for index, (point, value) in enumerate(zip(points, values)):
            sample_rows.append(
                {
                    "shape": name,
                    "index": index,
                    "real": point.real,
                    "imag": point.imag,
                    "q_diag": value,
                    "log10_q_diag": math.log10(max(value, 1.0e-300)),
                }
            )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with samples_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    payload = {
        "method": {
            "dense_matrix_stored": False,
            "q_diag": "q_j=sum_{k!=j}|z_j-z_k|^{-2}",
            "normalization": "each generated complex point set is centered and scaled by max radius before q_diag is computed",
            "chain": "z=exp(i theta) -> log discriminant Hessian -> Q -> local hard-part curvature",
        },
        "summary": json_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_lines = sorted(summary_rows, key=lambda row: float(row["max_over_median"]), reverse=True)
    lines = [
        "# Discriminant Curvature Shape Gallery",
        "",
        "This gallery shows the same discriminant-curvature mechanism on smooth, conic, nonconvex, cusped, polygonal, airfoil, stealth-like, and rational/asymptotic shape charts.",
        "",
        f"![shape gallery]({figure_path})",
        "",
        "## Rule",
        "",
        "Every panel starts from generated complex boundary samples `z_j`. The plotted scalar is",
        "",
        "```text",
        "q_j = sum_{k != j} |z_j - z_k|^(-2).",
        "```",
        "",
        "This is the diagonal discriminant curvature of Q. It is computed by the borrow-compute-repay chord protocol without storing the dense Q matrix.",
        "",
        "## What To Read",
        "",
        "- Smooth shapes show distributed curvature modulation.",
        "- Cusp and airfoil charts show sharp spikes where the tangent from `psi(exp(i theta))` vanishes.",
        "- Polygons show vertex channels where the tangent jumps.",
        "- The rational chart shows the quotient/asymptote carrier with the residual touch/cross points still visible through chord curvature.",
        "",
        "## Strongest Hard-Part Contrasts",
        "",
        "| shape | family | max/median | q90/q10 | hard part |",
        "|---|---|---:|---:|---|",
    ]
    for row in top_lines[:12]:
        lines.append(
            "| {shape} | {family} | `{mom:.3e}` | `{rob:.3e}` | {hard} |".format(
                shape=row["shape"],
                family=row["family"],
                mom=float(row["max_over_median"]),
                rob=float(row["q90_over_q10"]),
                hard=row["hard_part"],
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary CSV: `{summary_csv}`",
            f"- samples CSV: `{samples_csv}`",
            f"- JSON: `{json_path}`",
            f"- figure: `{figure_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "figure": str(figure_path)}, indent=2))


def main() -> None:
    n = 260
    specs = shape_specs(n)
    q_values: dict[str, list[float]] = {}
    for spec in specs:
        points = spec["points"]
        assert isinstance(points, list)
        q_values[str(spec["name"])] = discriminant_row_sums(points)
    write_outputs(specs, q_values)


if __name__ == "__main__":
    main()
