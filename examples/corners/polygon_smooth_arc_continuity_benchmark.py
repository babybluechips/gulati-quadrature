#!/usr/bin/env python3
"""Polygon quadrature through smooth arcs plus continuity corrections.

The benchmark represents each polygon as a C1 smooth-arc surrogate by replacing
every vertex with a quadratic tangent arc.  The principal quadrature is then
performed on smooth primitives.  A local continuity correction repays the
difference between the smooth arc and the two exact polygon edge stubs at each
corner.

This is the polygon analogue of the BGK ledger: compute on a smooth/continuous
model, then repay the discrete boundary crossing/endpoint/corner defect.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "polygon_smooth_arc_continuity"
TAU = 2.0 * math.pi
ZETA_HALF = -1.4603545088095868128894991525152980124672293310126
BGK_BETA = -ZETA_HALF / math.sqrt(2.0 * math.pi)
ERROR_FLOOR = 1.0e-15
GAUSS8_X = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
GAUSS8_W = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)

Point = tuple[float, float]


@dataclass(frozen=True)
class PolygonSpec:
    name: str
    family: str
    vertices: tuple[Point, ...]


@dataclass(frozen=True)
class RoundedPolygon:
    vertices: tuple[Point, ...]
    trims: tuple[float, ...]
    edge_lengths: tuple[float, ...]
    cumulative: tuple[float, ...]
    perimeter: float


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(scale: float, a: Point) -> Point:
    return (scale * a[0], scale * a[1])


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def unit(a: Point) -> Point:
    length = norm(a)
    if length <= 0.0:
        raise ValueError("zero-length vector")
    return (a[0] / length, a[1] / length)


def dist(a: Point, b: Point) -> float:
    return norm(sub(a, b))


def polygon_area(vertices: tuple[Point, ...]) -> float:
    return 0.5 * sum(cross(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))


def ensure_ccw(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
    return vertices if polygon_area(vertices) > 0.0 else tuple(reversed(vertices))


def edge_lengths(vertices: tuple[Point, ...]) -> tuple[float, ...]:
    return tuple(dist(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices)))


def cumulative_lengths(lengths: tuple[float, ...]) -> tuple[float, ...]:
    values = [0.0]
    total = 0.0
    for length in lengths[:-1]:
        total += length
        values.append(total)
    return tuple(values)


def edge_dir(vertices: tuple[Point, ...], index: int) -> Point:
    return unit(sub(vertices[(index + 1) % len(vertices)], vertices[index]))


def outward_normal(direction: Point) -> Point:
    return (direction[1], -direction[0])


def interior_angle(vertices: tuple[Point, ...], index: int) -> float:
    n = len(vertices)
    incoming = sub(vertices[index], vertices[(index - 1) % n])
    outgoing = sub(vertices[(index + 1) % n], vertices[index])
    turn = math.atan2(cross(incoming, outgoing), dot(incoming, outgoing))
    return max(0.05, min(TAU - 0.05, math.pi - turn))


def regularized_hurwitz_zeta(s: float, beta: float = 0.5, terms: int = 64) -> float:
    count = max(8, terms)
    partial = sum((k + beta) ** (-s) for k in range(count))
    tail = count + beta
    if abs(s - 1.0) < 1.0e-12:
        return partial
    return partial + tail ** (1.0 - s) / (s - 1.0) + 0.5 * tail ** (-s) + (s / 12.0) * tail ** (-s - 1.0)


def density(s: float, perimeter: float) -> float:
    theta = TAU * ((s / perimeter) % 1.0)
    return 1.0 + 0.23 * math.cos(3.0 * theta + 0.2) + 0.11 * math.sin(5.0 * theta - 0.4)


def log_kernel(point: Point, target: Point) -> float:
    return math.log(max(dist(point, target), 1.0e-300))


def line_point(a: Point, b: Point, t: float) -> Point:
    return add(mul(1.0 - t, a), mul(t, b))


def bezier_point(a: Point, c: Point, b: Point, t: float) -> Point:
    u = 1.0 - t
    return add(add(mul(u * u, a), mul(2.0 * u * t, c)), mul(t * t, b))


def bezier_derivative(a: Point, c: Point, b: Point, t: float) -> Point:
    return add(mul(2.0 * (1.0 - t), sub(c, a)), mul(2.0 * t, sub(b, c)))


def integrate_line(a: Point, b: Point, s0: float, s1: float, target: Point, perimeter: float, panels: int) -> float:
    length = dist(a, b)
    if length <= 0.0:
        return 0.0
    total = 0.0
    panel_count = max(1, panels)
    for panel in range(panel_count):
        left = panel / panel_count
        right = (panel + 1) / panel_count
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        for x, weight in zip(GAUSS8_X, GAUSS8_W, strict=True):
            t = mid + half * x
            s = s0 + t * (s1 - s0)
            point = line_point(a, b, t)
            total += weight * half * density(s, perimeter) * log_kernel(point, target) * length
    return total


def integrate_bezier(a: Point, c: Point, b: Point, s0: float, s1: float, target: Point, perimeter: float, panels: int) -> float:
    total = 0.0
    panel_count = max(1, panels)
    for panel in range(panel_count):
        left = panel / panel_count
        right = (panel + 1) / panel_count
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        for x, weight in zip(GAUSS8_X, GAUSS8_W, strict=True):
            t = mid + half * x
            s = s0 + t * (s1 - s0)
            point = bezier_point(a, c, b, t)
            speed = norm(bezier_derivative(a, c, b, t))
            total += weight * half * density(s, perimeter) * log_kernel(point, target) * speed
    return total


def allocate_panels(lengths: list[float], total_panels: int) -> list[int]:
    total_length = sum(max(length, 0.0) for length in lengths)
    if total_length <= 0.0:
        return [1 for _ in lengths]
    raw = [max(1.0, total_panels * length / total_length) for length in lengths]
    panels = [max(1, int(math.floor(value))) for value in raw]
    while sum(panels) < total_panels:
        index = max(range(len(panels)), key=lambda item: raw[item] - panels[item])
        panels[index] += 1
    while sum(panels) > total_panels:
        index = max(range(len(panels)), key=lambda item: panels[item] - raw[item] if panels[item] > 1 else -1.0)
        panels[index] -= 1
    return panels


def rounded_polygon(vertices: tuple[Point, ...], trim_fraction: float = 0.18) -> RoundedPolygon:
    vertices = ensure_ccw(vertices)
    lengths = edge_lengths(vertices)
    trims = [trim_fraction * min(lengths[i - 1], lengths[i]) for i in range(len(vertices))]
    for _ in range(6):
        for edge, length in enumerate(lengths):
            right = (edge + 1) % len(vertices)
            total_trim = trims[edge] + trims[right]
            maximum = 0.72 * length
            if total_trim > maximum:
                factor = maximum / total_trim
                trims[edge] *= factor
                trims[right] *= factor
    return RoundedPolygon(vertices, tuple(trims), lengths, cumulative_lengths(lengths), sum(lengths))


def smooth_primitives(model: RoundedPolygon) -> list[dict[str, object]]:
    primitives: list[dict[str, object]] = []
    vertices = model.vertices
    n = len(vertices)
    for i in range(n):
        d_prev = edge_dir(vertices, (i - 1) % n)
        d_next = edge_dir(vertices, i)
        p0 = sub(vertices[i], mul(model.trims[i], d_prev))
        p1 = add(vertices[i], mul(model.trims[i], d_next))
        primitives.append(
            {
                "kind": "arc",
                "a": p0,
                "c": vertices[i],
                "b": p1,
                "s0": model.cumulative[i] - model.trims[i],
                "s1": model.cumulative[i] + model.trims[i],
                "length_hint": dist(p0, vertices[i]) + dist(vertices[i], p1),
                "corner": i,
            }
        )
        edge_start = p1
        next_i = (i + 1) % n
        edge_end = sub(vertices[next_i], mul(model.trims[next_i], d_next))
        primitives.append(
            {
                "kind": "line",
                "a": edge_start,
                "b": edge_end,
                "s0": model.cumulative[i] + model.trims[i],
                "s1": model.cumulative[i] + model.edge_lengths[i] - model.trims[next_i],
                "length_hint": dist(edge_start, edge_end),
            }
        )
    return primitives


def integrate_smooth_model(model: RoundedPolygon, target: Point, total_panels: int) -> float:
    primitives = smooth_primitives(model)
    panels = allocate_panels([float(item["length_hint"]) for item in primitives], total_panels)
    total = 0.0
    for primitive, panel_count in zip(primitives, panels, strict=True):
        if primitive["kind"] == "line":
            total += integrate_line(
                primitive["a"], primitive["b"], primitive["s0"], primitive["s1"], target, model.perimeter, panel_count
            )
        else:
            total += integrate_bezier(
                primitive["a"],
                primitive["c"],
                primitive["b"],
                primitive["s0"],
                primitive["s1"],
                target,
                model.perimeter,
                panel_count,
            )
    return total


def integrate_polygon(vertices: tuple[Point, ...], target: Point, total_panels: int) -> float:
    vertices = ensure_ccw(vertices)
    lengths = edge_lengths(vertices)
    cumulative = cumulative_lengths(lengths)
    perimeter = sum(lengths)
    panels = allocate_panels(list(lengths), total_panels)
    total = 0.0
    for i, panel_count in enumerate(panels):
        total += integrate_line(
            vertices[i],
            vertices[(i + 1) % len(vertices)],
            cumulative[i],
            cumulative[i] + lengths[i],
            target,
            perimeter,
            panel_count,
        )
    return total


def continuity_correction(model: RoundedPolygon, target: Point, panels_per_corner: int) -> tuple[float, list[dict[str, float]]]:
    total = 0.0
    ledger: list[dict[str, float]] = []
    vertices = model.vertices
    n = len(vertices)
    h = model.perimeter / max(n, 1)
    for i in range(n):
        prev_edge = (i - 1) % n
        d_prev = edge_dir(vertices, prev_edge)
        d_next = edge_dir(vertices, i)
        p0 = sub(vertices[i], mul(model.trims[i], d_prev))
        p1 = add(vertices[i], mul(model.trims[i], d_next))
        s_corner = model.cumulative[i]
        exact_in = integrate_line(p0, vertices[i], s_corner - model.trims[i], s_corner, target, model.perimeter, panels_per_corner)
        exact_out = integrate_line(vertices[i], p1, s_corner, s_corner + model.trims[i], target, model.perimeter, panels_per_corner)
        smooth = integrate_bezier(
            p0,
            vertices[i],
            p1,
            s_corner - model.trims[i],
            s_corner + model.trims[i],
            target,
            model.perimeter,
            panels_per_corner,
        )
        delta = exact_in + exact_out - smooth
        omega = interior_angle(vertices, i)
        lam = math.pi / omega
        zeta_weight = regularized_hurwitz_zeta(1.0 - lam, beta=0.5)
        total += delta
        ledger.append(
            {
                "corner": float(i),
                "interior_angle": omega,
                "lambda": lam,
                "trim": model.trims[i],
                "local_delta": delta,
                "zeta_weight": zeta_weight,
                "h_lambda": (h / model.perimeter) ** lam,
            }
        )
    return total, ledger


def corner_outward_target(vertices: tuple[Point, ...], index: int) -> Point:
    vertices = ensure_ccw(vertices)
    prev_dir = edge_dir(vertices, (index - 1) % len(vertices))
    next_dir = edge_dir(vertices, index)
    normal = add(outward_normal(prev_dir), outward_normal(next_dir))
    if norm(normal) <= 1.0e-12:
        normal = outward_normal(next_dir)
    direction = unit(normal)
    scale = 0.16 * min(edge_lengths(vertices)[index - 1], edge_lengths(vertices)[index])
    return add(vertices[index], mul(scale, direction))


def edge_outward_target(vertices: tuple[Point, ...], edge: int) -> Point:
    vertices = ensure_ccw(vertices)
    direction = edge_dir(vertices, edge)
    midpoint = mul(0.5, add(vertices[edge], vertices[(edge + 1) % len(vertices)]))
    scale = 0.12 * edge_lengths(vertices)[edge]
    return add(midpoint, mul(scale, outward_normal(direction)))


def shape_specs() -> tuple[PolygonSpec, ...]:
    star_vertices = []
    for index in range(10):
        radius = 1.18 if index % 2 == 0 else 0.48
        angle = math.pi / 2.0 + TAU * index / 10.0
        star_vertices.append((radius * math.cos(angle), radius * math.sin(angle)))
    return (
        PolygonSpec(
            "square",
            "convex polygon",
            ensure_ccw(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))),
        ),
        PolygonSpec(
            "l_notch",
            "single reentrant corner",
            ensure_ccw(((-1.20, -1.00), (1.20, -1.00), (1.20, 0.22), (0.16, 0.22), (0.16, 1.05), (-1.20, 1.05))),
        ),
        PolygonSpec("star", "alternating acute/reentrant", ensure_ccw(tuple(star_vertices))),
        PolygonSpec(
            "stealth_double_concave",
            "symmetric concave polygon",
            ensure_ccw(((1.35, 0.0), (0.50, 0.34), (-1.10, 0.58), (-0.42, 0.16), (-0.42, -0.16), (-1.10, -0.58), (0.50, -0.34))),
        ),
    )


def hard_corner_index(vertices: tuple[Point, ...]) -> int:
    angles = [interior_angle(ensure_ccw(vertices), i) for i in range(len(vertices))]
    return max(range(len(angles)), key=lambda i: abs(angles[i] - math.pi))


def run_case(shape: PolygonSpec, target_kind: str, panels: int, reference_panels: int) -> dict[str, object]:
    vertices = ensure_ccw(shape.vertices)
    model = rounded_polygon(vertices)
    if target_kind == "corner":
        corner = hard_corner_index(vertices)
        target = corner_outward_target(vertices, corner)
    elif target_kind == "edge":
        corner = -1
        target = edge_outward_target(vertices, 0)
    else:
        raise ValueError(f"unknown target_kind: {target_kind}")
    reference = integrate_polygon(vertices, target, reference_panels)
    raw = integrate_polygon(vertices, target, panels)
    smooth = integrate_smooth_model(model, target, panels)
    correction, ledger = continuity_correction(model, target, panels_per_corner=max(4, panels // len(vertices)))
    corrected = smooth + correction
    return {
        "shape": shape.name,
        "family": shape.family,
        "target_kind": target_kind,
        "hard_corner_index": corner,
        "panels": panels,
        "reference_panels": reference_panels,
        "target_x": target[0],
        "target_y": target[1],
        "reference": reference,
        "raw_polygon": raw,
        "smooth_arc": smooth,
        "smooth_arc_plus_continuity": corrected,
        "raw_abs_error": abs(raw - reference),
        "smooth_abs_error": abs(smooth - reference),
        "corrected_abs_error": abs(corrected - reference),
        "raw_rel_error": abs(raw - reference) / max(abs(reference), 1.0e-300),
        "smooth_rel_error": abs(smooth - reference) / max(abs(reference), 1.0e-300),
        "corrected_rel_error": abs(corrected - reference) / max(abs(reference), 1.0e-300),
        "smooth_to_corrected_improvement": abs(smooth - reference) / max(abs(corrected - reference), 1.0e-300),
        "raw_to_corrected_improvement": abs(raw - reference) / max(abs(corrected - reference), 1.0e-300),
        "continuity_delta": correction,
        "corner_ledger": ledger,
        "vertex_count": len(vertices),
        "perimeter": model.perimeter,
        "bgk_beta": BGK_BETA,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "shape",
        "family",
        "target_kind",
        "hard_corner_index",
        "panels",
        "reference_panels",
        "reference",
        "raw_polygon",
        "smooth_arc",
        "smooth_arc_plus_continuity",
        "raw_rel_error",
        "smooth_rel_error",
        "corrected_rel_error",
        "smooth_to_corrected_improvement",
        "raw_to_corrected_improvement",
        "continuity_delta",
        "vertex_count",
        "perimeter",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_corner_ledger_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "shape",
        "target_kind",
        "panels",
        "corner",
        "interior_angle",
        "lambda",
        "trim",
        "local_delta",
        "zeta_weight",
        "h_lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for item in row["corner_ledger"]:
                writer.writerow(
                    {
                        "shape": row["shape"],
                        "target_kind": row["target_kind"],
                        "panels": row["panels"],
                        "corner": item["corner"],
                        "interior_angle": item["interior_angle"],
                        "lambda": item["lambda"],
                        "trim": item["trim"],
                        "local_delta": item["local_delta"],
                        "zeta_weight": item["zeta_weight"],
                        "h_lambda": item["h_lambda"],
                    }
                )


def plotted_error(value: float) -> float:
    return max(abs(value), ERROR_FLOOR)


def render_shape_panel(axis, shape: PolygonSpec) -> None:
    vertices = ensure_ccw(shape.vertices)
    model = rounded_polygon(vertices)
    polygon_x = [point[0] for point in vertices] + [vertices[0][0]]
    polygon_y = [point[1] for point in vertices] + [vertices[0][1]]
    axis.plot(polygon_x, polygon_y, color="0.0", linewidth=1.0, label="exact polygon")
    smooth_points: list[Point] = []
    for primitive in smooth_primitives(model):
        samples = 24 if primitive["kind"] == "arc" else 3
        for sample in range(samples):
            t = sample / max(samples - 1, 1)
            if primitive["kind"] == "line":
                smooth_points.append(line_point(primitive["a"], primitive["b"], t))
            else:
                smooth_points.append(bezier_point(primitive["a"], primitive["c"], primitive["b"], t))
    smooth_points.append(smooth_points[0])
    axis.plot([p[0] for p in smooth_points], [p[1] for p in smooth_points], color="0.55", linewidth=1.0, linestyle=(0, (3, 2)), label="smooth arcs")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(shape.name.replace("_", " "), fontsize=8)
    axis.grid(color="0.92", linewidth=0.5)


def make_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    shapes = shape_specs()
    for axis, shape in zip(axes[0], shapes, strict=True):
        render_shape_panel(axis, shape)
    axes[0][0].legend(frameon=False, fontsize=7)

    latest = max(int(row["panels"]) for row in rows)
    plot_rows = [row for row in rows if int(row["panels"]) == latest and row["target_kind"] == "corner"]
    labels = [str(row["shape"]).replace("_", "\n") for row in plot_rows]
    x = list(range(len(plot_rows)))
    width = 0.23
    for offset, key, color, label in (
        (-width, "raw_rel_error", "0.05", "raw polygon"),
        (0.0, "smooth_rel_error", "0.55", "smooth arcs"),
        (width, "corrected_rel_error", "0.25", "smooth + continuity"),
    ):
        axes[1][0].bar(
            [item + offset for item in x],
            [plotted_error(float(row[key])) for row in plot_rows],
            width=width,
            color=color,
            label=label,
        )
    axes[1][0].set_yscale("log")
    axes[1][0].set_ylim(ERROR_FLOOR * 0.35, 2.0)
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(labels, fontsize=7)
    axes[1][0].set_ylabel(f"relative error (floor {ERROR_FLOOR:.0e})")
    axes[1][0].set_title(f"near-corner target, panels={latest}")
    axes[1][0].grid(axis="y", color="0.90")
    axes[1][0].legend(frameon=False, fontsize=7)

    for axis, shape in zip(axes[1][1:], shapes[:3], strict=True):
        subset = [row for row in rows if row["shape"] == shape.name and row["target_kind"] == "corner"]
        panel_values = sorted({int(row["panels"]) for row in subset})
        for key, color, label in (
            ("smooth_rel_error", "0.55", "smooth arcs"),
            ("corrected_rel_error", "0.05", "smooth + continuity"),
        ):
            axis.plot(
                panel_values,
                [plotted_error(float(next(row for row in subset if int(row["panels"]) == p)[key])) for p in panel_values],
                marker="o",
                color=color,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_title(shape.name.replace("_", " "), fontsize=8)
        axis.set_xlabel("panels")
        axis.set_ylabel(f"rel. error (floor {ERROR_FLOOR:.0e})")
        axis.set_xticks(panel_values)
        axis.set_xticklabels([str(value) for value in panel_values])
        axis.set_ylim(ERROR_FLOOR * 0.35, 2.0)
        axis.grid(True, which="both", color="0.90")
    axes[1][1].legend(frameon=False, fontsize=7)
    fig.suptitle("Polygon quadrature as smooth arcs plus continuity correction", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def format_error(value: float) -> str:
    if abs(value) < ERROR_FLOOR:
        return f"<{ERROR_FLOOR:.0e}"
    return f"{value:.3e}"


def format_improvement(numerator: float, denominator: float) -> str:
    if abs(denominator) < ERROR_FLOOR:
        if abs(numerator) < ERROR_FLOOR:
            return "floor"
        return f">={abs(numerator) / ERROR_FLOOR:.2e}x"
    return f"{abs(numerator) / abs(denominator):.2f}x"


def write_report(path: Path, rows: list[dict[str, object]], figure_path: Path, csv_path: Path, json_path: Path, ledger_path: Path) -> None:
    latest = max(int(row["panels"]) for row in rows)
    latest_rows = [row for row in rows if int(row["panels"]) == latest and row["target_kind"] == "corner"]
    lines = [
        "# Polygon Smooth-Arc Continuity Benchmark",
        "",
        "This tests the polygonal quadrature ledger requested here: represent the polygon by smooth arcs for the principal calculation, then repay the continuity/corner defect locally.",
        "",
        f"![polygon smooth arc benchmark]({figure_path})",
        "",
        "## Protocol",
        "",
        "For each polygon vertex `v_i`, the benchmark cuts back adjacent edges and inserts a C1 quadratic tangent arc. The smooth model is then integrated as line-plus-arc primitives. The continuity correction adds back",
        "",
        "```text",
        "exact incoming edge stub + exact outgoing edge stub - smooth corner arc.",
        "```",
        "",
        "That is the polygon version of the BGK bookkeeping: compute on the continuous/smooth model, then repay the discrete boundary crossing/corner defect. The ledger also records the Kondrat'ev exponent `lambda = pi / omega` and the Hurwitz endpoint factor `zeta(1-lambda, 1/2)` at each corner.",
        "",
        "The calculation stores no dense Q matrix here; the boundary is carried by primitive generators plus a local corner ledger.",
        "",
        f"`beta_BGK = {BGK_BETA:.16f}` is the square-root endpoint special case.",
        "",
        "## Latest Near-Corner Results",
        "",
        "| Shape | Raw polygon err | Smooth arc err | Corrected err | Smooth/corrected | Raw/corrected | Continuity delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in latest_rows:
        lines.append(
            "| {shape} | `{raw}` | `{smooth}` | `{corr}` | `{si}` | `{ri}` | `{delta:.3e}` |".format(
                shape=row["shape"],
                raw=format_error(float(row["raw_rel_error"])),
                smooth=format_error(float(row["smooth_rel_error"])),
                corr=format_error(float(row["corrected_rel_error"])),
                si=format_improvement(float(row["smooth_rel_error"]), float(row["corrected_rel_error"])),
                ri=format_improvement(float(row["raw_rel_error"]), float(row["corrected_rel_error"])),
                delta=float(row["continuity_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Near-Corner Convergence",
            "",
            "| Shape | Panels | Smooth arc err | Corrected err |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in [item for item in rows if item["target_kind"] == "corner"]:
        lines.append(
            "| {shape} | `{panels}` | `{smooth}` | `{corr}` |".format(
                shape=row["shape"],
                panels=row["panels"],
                smooth=format_error(float(row["smooth_rel_error"])),
                corr=format_error(float(row["corrected_rel_error"])),
            )
        )
    lines.extend(
        [
            "",
            "## What This Shows",
            "",
            "- The smooth-arc representation is useful as a principal continuous model, but it changes the local corner ledger.",
            "- The continuity correction is local and geometry-derived; it replaces the rounded corner contribution with the exact two-edge corner contribution.",
            "- The same bookkeeping slot carries the BGK square-root endpoint constant for barriers and the Kondrat'ev/Hurwitz exponent for polygon corners.",
            "",
            "## Artifacts",
            "",
            f"- CSV: `{csv_path}`",
            f"- corner ledger CSV: `{ledger_path}`",
            f"- JSON: `{json_path}`",
            f"- figure: `{figure_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_values = (24, 48, 96, 192)
    reference_panels = 8192
    rows: list[dict[str, object]] = []
    for shape in shape_specs():
        for target_kind in ("corner", "edge"):
            for panels in panel_values:
                rows.append(run_case(shape, target_kind, panels, reference_panels))

    csv_path = OUT / "polygon_smooth_arc_continuity_benchmark.csv"
    ledger_path = OUT / "polygon_smooth_arc_corner_ledger.csv"
    json_path = OUT / "polygon_smooth_arc_continuity_benchmark.json"
    figure_path = OUT / "polygon_smooth_arc_continuity_benchmark.png"
    report_path = OUT / "polygon_smooth_arc_continuity_benchmark.md"
    write_csv(csv_path, rows)
    write_corner_ledger_csv(ledger_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "method": {
                    "dense_q_matrix_stored": False,
                    "quadrature": "8-point Gauss panels on exact polygon, smooth arc surrogate, and local continuity correction",
                    "continuity_correction": "exact local edge stubs minus smooth arc at each corner",
                    "bgk_beta": BGK_BETA,
                    "corner_factor": "lambda=pi/interior_angle, Hurwitz zeta(1-lambda, 1/2) recorded in corner ledger",
                    "reference_panels": reference_panels,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_figure(rows, figure_path)
    write_report(report_path, rows, figure_path, csv_path, json_path, ledger_path)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path), "csv": str(csv_path), "ledger": str(ledger_path)}, indent=2))


if __name__ == "__main__":
    main()
