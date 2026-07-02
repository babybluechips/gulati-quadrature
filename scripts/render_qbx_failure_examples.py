#!/usr/bin/env python3
"""Render concrete QBX source-free disk failure examples."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.quadrature import outward_unit_normals  # noqa: E402

TAU = 2.0 * math.pi


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def perimeter(points):
    return sum(dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))


def resample_closed_curve(points, n):
    lengths = [dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    total = sum(lengths)
    out = []
    edge = 0
    edge_start = 0.0
    for k in range(n):
        target = total * k / n
        while edge + 1 < len(points) and edge_start + lengths[edge] < target:
            edge_start += lengths[edge]
            edge += 1
        span = lengths[edge]
        alpha = 0.0 if span <= 0.0 else (target - edge_start) / span
        a = points[edge]
        b = points[(edge + 1) % len(points)]
        out.append((a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1])))
    return out


def cardioid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        radius = 1.0 - math.cos(theta)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def nephroid_dense(samples):
    out = []
    for i in range(samples):
        theta = TAU * i / samples
        out.append(
            (
                3.0 * math.cos(theta) - math.cos(3.0 * theta),
                3.0 * math.sin(theta) - math.sin(3.0 * theta),
            )
        )
    return out


def nearest_index(points, target):
    return min(range(len(points)), key=lambda index: dist(points[index], target))


def target_from_sample(points, sample_index, ratio):
    h = perimeter(points) / len(points)
    normal = tuple(outward_unit_normals(points)[sample_index])
    base = points[sample_index]
    return (base[0] + ratio * h * normal[0], base[1] + ratio * h * normal[1])


def qbx_failure_geometry(dense_points, ratio, *, n=512, qbx_n=4096, radius_factor=4.0):
    coarse = resample_closed_curve(dense_points, n)
    qbx_points = resample_closed_curve(dense_points, qbx_n)
    sample_index = 0
    target = target_from_sample(coarse, sample_index, ratio)
    qbx_index = nearest_index(qbx_points, coarse[sample_index])
    normal = tuple(outward_unit_normals(qbx_points)[qbx_index])
    base = qbx_points[qbx_index]
    offset = (target[0] - base[0], target[1] - base[1])
    signed_delta = offset[0] * normal[0] + offset[1] * normal[1]
    center = (
        base[0] + radius_factor * signed_delta * normal[0],
        base[1] + radius_factor * signed_delta * normal[1],
    )
    target_radius = dist(center, target)
    source_distances = [dist(center, point) for point in qbx_points]
    closest_index = min(range(len(qbx_points)), key=lambda index: source_distances[index])
    source_radius = source_distances[closest_index]
    return {
        "coarse": coarse,
        "qbx_points": qbx_points,
        "sample": coarse[sample_index],
        "base": base,
        "target": target,
        "center": center,
        "closest_source": qbx_points[closest_index],
        "target_radius": target_radius,
        "source_radius": source_radius,
        "margin": source_radius - target_radius,
        "h": perimeter(coarse) / n,
    }


def cusp_indices_for_shape(name, boundary):
    if name == "cardioid_single_cusp":
        return (0,)
    if name == "nephroid_two_cusps":
        return (0, len(boundary) // 2)
    return ()


def display_name(name):
    if name == "cardioid_single_cusp":
        return "cardioid_single_cusp (1 cusp)"
    if name == "nephroid_two_cusps":
        return "nephroid_two_cusps (2 cusps)"
    return name


def add_global_inset(ax, name, geometry):
    boundary = geometry["qbx_points"]
    xs = [point[0] for point in boundary] + [boundary[0][0]]
    ys = [point[1] for point in boundary] + [boundary[0][1]]
    inset = ax.inset_axes([0.57, 0.56, 0.38, 0.38])
    inset.plot(xs, ys, color="#374151", linewidth=0.9)
    inset.scatter(*geometry["base"], color="#111827", s=12, zorder=5)
    inset.scatter(*geometry["target"], color="#dc2626", s=14, zorder=6)
    for index in cusp_indices_for_shape(name, boundary):
        inset.scatter(*boundary[index], facecolors="none", edgecolors="#dc2626", s=34, linewidths=1.0, zorder=7)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("full shape", fontsize=7, pad=1)
    for spine in inset.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.7)


def render_case(ax, name, dense_points, ratio):
    geometry = qbx_failure_geometry(dense_points, ratio)
    boundary = geometry["qbx_points"]
    xs = [point[0] for point in boundary] + [boundary[0][0]]
    ys = [point[1] for point in boundary] + [boundary[0][1]]
    ax.plot(xs, ys, color="#1f2937", linewidth=1.2)
    disk = plt.Circle(
        geometry["center"],
        geometry["target_radius"],
        fill=False,
        color="#dc2626",
        linewidth=1.3,
        linestyle="--",
    )
    ax.add_patch(disk)
    ax.scatter(*geometry["target"], color="#dc2626", s=36, label="target", zorder=5)
    ax.scatter(*geometry["center"], color="#2563eb", s=32, label="QBX center", zorder=5)
    ax.scatter(*geometry["closest_source"], color="#f97316", s=28, label="closest source", zorder=5)
    ax.scatter(*geometry["base"], color="#111827", s=18, label="boundary base", zorder=5)
    points = [geometry["target"], geometry["center"], geometry["closest_source"], geometry["base"]]
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    span = max(12.0 * geometry["h"], 3.2 * geometry["target_radius"], 1.0e-3)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e5e7eb", linewidth=0.5)
    ax.set_title(f"{display_name(name)}\nΔ/h={ratio}, margin={geometry['margin']:.2e}", fontsize=10)
    ax.text(
        0.02,
        0.02,
        f"source r={geometry['source_radius']:.2e}\ntarget r={geometry['target_radius']:.2e}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "pad": 3},
    )
    add_global_inset(ax, name, geometry)


def save_shape_grid(output, name, dense_points, ratios):
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.35), constrained_layout=True)
    if len(ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, ratios, strict=True):
        render_case(ax, name, dense_points, ratio)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    output_dir = ROOT / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = 16384
    ratios = (0.2, 0.1, 0.05)
    shapes = (
        ("cardioid_single_cusp", cardioid_dense(samples)),
        ("nephroid_two_cusps", nephroid_dense(samples)),
    )
    combined, axes = plt.subplots(2, 3, figsize=(12.6, 8.65), constrained_layout=True)
    paths = []
    for row, (name, points) in enumerate(shapes):
        shape_path = output_dir / f"qbx_failure_{name}.png"
        save_shape_grid(shape_path, name, points, ratios)
        paths.append(shape_path)
        for col, ratio in enumerate(ratios):
            render_case(axes[row][col], name, points, ratio)
    handles, labels = axes[0][0].get_legend_handles_labels()
    combined.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=8)
    combined_path = output_dir / "qbx_failure_examples.png"
    combined.savefig(combined_path, dpi=220)
    plt.close(combined)
    paths.append(combined_path)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
