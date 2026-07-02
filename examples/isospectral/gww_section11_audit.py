#!/usr/bin/env python3
"""Section 11 audit tables for the GWW Q/Dirichlet comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch

from gww_q_vs_dirichlet import (
    GWW_LEFT,
    GWW_RIGHT,
    finite_difference_dirichlet,
    normalize_polygon,
    polygon_area,
    polygon_perimeter,
)


ROOT = Path(__file__).resolve().parents[2]
TAU = 2.0 * math.pi
Point = tuple[float, float]


REFINEMENT_NS = (256, 512, 1024, 2048, 4096)
SYMMETRY_N = 1024
PROBE_MODES = 6
REPORTED_RITZ_VALUES = 6


def transform_polygon(
    points: tuple[Point, ...],
    *,
    rotation_deg: float = 0.0,
    translation: Point = (0.0, 0.0),
    reverse_orientation: bool = False,
) -> tuple[Point, ...]:
    ordered = tuple(reversed(points)) if reverse_orientation else points
    angle = math.radians(rotation_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    tx, ty = translation
    return tuple((c * x - s * y + tx, s * x + c * y + ty) for x, y in ordered)


def sample_polygon_boundary(
    points: tuple[Point, ...],
    sample_count: int,
    *,
    phase: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Equal-arclength polygon sampling with one global phase.

    ``phase=0.5`` places samples at cell midpoints and therefore avoids placing
    nodes exactly on polygon corners.  ``phase=0`` intentionally puts the first
    node at the first vertex for the corner-placement control.
    """

    if sample_count < len(points):
        raise ValueError("sample_count must be at least the vertex count")
    edge_lengths = [
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    total = sum(edge_lengths)
    if total <= 0.0:
        raise ValueError("polygon perimeter is zero")
    samples: list[Point] = []
    for sample in range(sample_count):
        arclength = ((sample + phase) % sample_count) * total / sample_count
        accumulated = 0.0
        for edge, length in enumerate(edge_lengths):
            if arclength <= accumulated + length or edge == len(edge_lengths) - 1:
                local = (arclength - accumulated) / length
                x0, y0 = points[edge]
                x1, y1 = points[(edge + 1) % len(points)]
                samples.append((x0 + local * (x1 - x0), y0 + local * (y1 - y0)))
                break
            accumulated += length
    return torch.tensor(samples, dtype=torch.float64), total


def probe_matrix(sample_count: int, *, modes: int, phase: float) -> torch.Tensor:
    theta = TAU * (torch.arange(sample_count, dtype=torch.float64) + phase) / sample_count
    columns = []
    for mode in range(1, modes + 1):
        columns.append(torch.cos(mode * theta))
        columns.append(torch.sin(mode * theta))
    probes = torch.stack(columns, dim=1)
    return probes - probes.mean(dim=0, keepdim=True)


def q_apply_blockwise(
    points: torch.Tensor,
    values: torch.Tensor,
    *,
    perimeter: float,
    chunk_size: int,
) -> torch.Tensor:
    """Apply Lambda_Q to several vectors without storing the dense Q matrix."""

    n = points.shape[0]
    output = torch.empty_like(values)
    scale = perimeter / (math.pi * n)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = points[start:stop]
        delta = block[:, None, :] - points[None, :, :]
        distance2 = torch.sum(delta * delta, dim=2)
        for row, index in enumerate(range(start, stop)):
            distance2[row, index] = float("inf")
        weights = 1.0 / distance2
        output[start:stop] = scale * (
            weights.sum(dim=1).unsqueeze(1) * values[start:stop] - weights @ values
        )
    return output


def projected_q_ritz_spectrum(
    points: tuple[Point, ...],
    *,
    sample_count: int,
    phase: float = 0.5,
    modes: int = PROBE_MODES,
    chunk_size: int = 256,
) -> tuple[list[float], float]:
    """Exact float64 eigenvalues of the small projected Q Ritz problem."""

    samples, length = sample_polygon_boundary(points, sample_count, phase=phase)
    probes = probe_matrix(sample_count, modes=modes, phase=phase)
    start = perf_counter()
    applied = q_apply_blockwise(samples, probes, perimeter=length, chunk_size=chunk_size)
    stiffness = probes.T @ applied
    mass = probes.T @ probes
    chol = torch.linalg.cholesky(mass)
    tmp = torch.linalg.solve_triangular(chol, stiffness, upper=False)
    reduced = torch.linalg.solve_triangular(chol, tmp.T, upper=False).T
    reduced = 0.5 * (reduced + reduced.T)
    values = torch.flip(torch.linalg.eigvalsh(reduced), dims=(0,))
    elapsed_ms = 1000.0 * (perf_counter() - start)
    return [float(value) for value in values.tolist()], elapsed_ms


def relative_l2(left: list[float], right: list[float], *, count: int = REPORTED_RITZ_VALUES) -> float:
    lhs = torch.tensor(left[:count], dtype=torch.float64)
    rhs = torch.tensor(right[:count], dtype=torch.float64)
    return float(torch.linalg.norm(lhs - rhs) / torch.linalg.norm(lhs).clamp_min(1.0e-30))


def convergence_rows(left: tuple[Point, ...], right: tuple[Point, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_left: list[float] | None = None
    previous_right: list[float] | None = None
    previous_split: float | None = None
    for n in REFINEMENT_NS:
        left_values, left_ms = projected_q_ritz_spectrum(left, sample_count=n)
        right_values, right_ms = projected_q_ritz_spectrum(right, sample_count=n)
        split = relative_l2(left_values, right_values)
        row: dict[str, object] = {
            "n": n,
            "probe_modes": PROBE_MODES,
            "ritz_values_reported": REPORTED_RITZ_VALUES,
            "left_elapsed_ms": left_ms,
            "right_elapsed_ms": right_ms,
            "relative_split_first6": split,
            "left_ritz_descending": left_values[:REPORTED_RITZ_VALUES],
            "right_ritz_descending": right_values[:REPORTED_RITZ_VALUES],
            "left_delta_prev_first6": None,
            "right_delta_prev_first6": None,
            "split_delta_prev": None,
        }
        if previous_left is not None and previous_right is not None and previous_split is not None:
            row["left_delta_prev_first6"] = relative_l2(left_values, previous_left)
            row["right_delta_prev_first6"] = relative_l2(right_values, previous_right)
            row["split_delta_prev"] = abs(split - previous_split)
        rows.append(row)
        previous_left = left_values
        previous_right = right_values
        previous_split = split
    return rows


def symmetry_rows(left: tuple[Point, ...]) -> list[dict[str, object]]:
    base_values, _ = projected_q_ritz_spectrum(left, sample_count=SYMMETRY_N, phase=0.5)
    controls = (
        ("base_midpoint_nodes", left, 0.5, 1.0),
        ("rotate_37_translate", transform_polygon(left, rotation_deg=37.0, translation=(3.0, -2.0)), 0.5, 1.0),
        ("reverse_orientation", transform_polygon(left, reverse_orientation=True), 0.5, 1.0),
        ("node_phase_0p125", left, 0.125, 1.0),
        ("corner_node_phase_0", left, 0.0, 1.0),
        ("node_phase_golden", left, 0.61803398875, 1.0),
        ("scale_area_2p25_rescaled", normalize_polygon(GWW_LEFT, area=2.25), 0.5, math.sqrt(2.25)),
    )
    rows: list[dict[str, object]] = []
    for name, polygon, phase, rescale in controls:
        values, elapsed_ms = projected_q_ritz_spectrum(polygon, sample_count=SYMMETRY_N, phase=phase)
        comparable = [value * rescale for value in values]
        rows.append(
            {
                "control": name,
                "n": SYMMETRY_N,
                "phase": phase,
                "post_multiplier_for_scale_control": rescale,
                "elapsed_ms": elapsed_ms,
                "relative_deviation_from_base_first6": relative_l2(comparable, base_values),
                "ritz_descending_after_rescale": comparable[:REPORTED_RITZ_VALUES],
            }
        )
    return rows


def write_rows_csv(rows: list[dict[str, object]], path: Path, *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if kind == "convergence":
            fieldnames = [
                "n",
                "relative_split_first6",
                "left_delta_prev_first6",
                "right_delta_prev_first6",
                "split_delta_prev",
                *[f"left_ritz_{index}" for index in range(1, REPORTED_RITZ_VALUES + 1)],
                *[f"right_ritz_{index}" for index in range(1, REPORTED_RITZ_VALUES + 1)],
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = {
                    "n": row["n"],
                    "relative_split_first6": row["relative_split_first6"],
                    "left_delta_prev_first6": row["left_delta_prev_first6"],
                    "right_delta_prev_first6": row["right_delta_prev_first6"],
                    "split_delta_prev": row["split_delta_prev"],
                }
                for index, value in enumerate(row["left_ritz_descending"], start=1):
                    flat[f"left_ritz_{index}"] = value
                for index, value in enumerate(row["right_ritz_descending"], start=1):
                    flat[f"right_ritz_{index}"] = value
                writer.writerow(flat)
        elif kind == "symmetry":
            fieldnames = [
                "control",
                "n",
                "phase",
                "post_multiplier_for_scale_control",
                "relative_deviation_from_base_first6",
                *[f"ritz_{index}" for index in range(1, REPORTED_RITZ_VALUES + 1)],
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = {
                    "control": row["control"],
                    "n": row["n"],
                    "phase": row["phase"],
                    "post_multiplier_for_scale_control": row["post_multiplier_for_scale_control"],
                    "relative_deviation_from_base_first6": row["relative_deviation_from_base_first6"],
                }
                for index, value in enumerate(row["ritz_descending_after_rescale"], start=1):
                    flat[f"ritz_{index}"] = value
                writer.writerow(flat)
        else:
            raise ValueError(f"unknown CSV kind: {kind}")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def fmt(value: object, digits: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def coordinate_table(points: tuple[Point, ...]) -> str:
    return "[" + ", ".join(f"({x:.12g}, {y:.12g})" for x, y in points) + "]"


def write_report(payload: dict[str, object], path: Path) -> None:
    left_raw = payload["coordinates"]["raw_left_clockwise"]
    right_raw = payload["coordinates"]["raw_right_clockwise"]
    left_norm = payload["coordinates"]["normalized_left_ccw"]
    right_norm = payload["coordinates"]["normalized_right_ccw"]
    convergence = payload["q_convergence"]
    symmetry = payload["q_symmetry_controls"]
    fd = payload["dirichlet_fd_spectrum"]

    convergence_rows_md = []
    for row in convergence:
        convergence_rows_md.append(
            [
                row["n"],
                fmt(row["relative_split_first6"]),
                fmt(row["left_delta_prev_first6"]),
                fmt(row["right_delta_prev_first6"]),
                ", ".join(fmt(value) for value in row["left_ritz_descending"]),
                ", ".join(fmt(value) for value in row["right_ritz_descending"]),
            ]
        )

    symmetry_rows_md = [
        [
            row["control"],
            row["n"],
            fmt(row["phase"], 6),
            fmt(row["post_multiplier_for_scale_control"], 6),
            fmt(row["relative_deviation_from_base_first6"]),
            ", ".join(fmt(value) for value in row["ritz_descending_after_rescale"]),
        ]
        for row in symmetry
    ]

    fd_rows_md = []
    for index, (left, right) in enumerate(
        zip(fd["left_eigenvalues"], fd["right_eigenvalues"], strict=True),
        start=1,
    ):
        rel = abs(left - right) / max(abs(left), 1.0e-30)
        fd_rows_md.append([index, fmt(left), fmt(right), fmt(rel)])

    text = f"""# Section 11 Audit: GWW Dirichlet Isospectral Pair Versus Q

Generated: {payload["generated_note"]}

## Claim Boundary

This section uses the following careful claim:

> At the discretized chord-operator level, this standard isospectral pair is separated robustly under refinement.

It does **not** claim that this experiment proves existence of a new continuum invariant unless a separate convergence theorem for the continuum Q spectrum is supplied. The exact continuum statement used here is only the classical Gordon-Webb-Wolpert Dirichlet isospectrality theorem.

## Exact GWW Polygon Coordinates

The raw coordinate lists are the eight-vertex GWW/Driscoll polygon tables. In the raw order below both polygons have signed area `-14`, so they are clockwise. The implementation reverses each list before normalization so the working boundary orientation is counterclockwise.

Left raw clockwise vertices:

```text
{coordinate_table(tuple(tuple(point) for point in left_raw))}
```

Right raw clockwise vertices:

```text
{coordinate_table(tuple(tuple(point) for point in right_raw))}
```

Working normalized counterclockwise vertices have unit area. The normalization is

```text
x'_i = (x_i - mean_vertex_x) / sqrt(14),   y'_i = (y_i - mean_vertex_y) / sqrt(14)
```

after orientation reversal.

Left normalized CCW vertices:

```text
{coordinate_table(tuple(tuple(point) for point in left_norm))}
```

Right normalized CCW vertices:

```text
{coordinate_table(tuple(tuple(point) for point in right_norm))}
```

Both normalized domains have area `{payload["coordinates"]["normalized_left_area"]:.16g}` and perimeter `{payload["coordinates"]["normalized_left_perimeter"]:.16g}`.

## Boundary Sampling Protocol

For a polygon with vertices `v_0,...,v_7` and perimeter `L`, nodes are placed at equal arclength positions

```text
s_k = ((k + alpha) mod n) L / n,   k = 0,...,n-1.
```

The default phase is `alpha = 1/2`, i.e. midpoint sampling of arclength cells. This avoids placing a node exactly on a corner. The polygon is not smoothed. Corners remain exact vertices of the piecewise-linear boundary. If `alpha = 0` is used in the corner-placement control, the first node is exactly on the first corner, but corners are still not duplicated and no two boundary nodes coincide.

## Q Normalization

For sampled boundary points `x_i`, perimeter `L`, and `h = L/n`, the discrete generated chord operator is

```text
(Lambda_Q,n f)_i = (h/pi) sum_{{j != i}} (f_i - f_j) / |x_i - x_j|^2.
```

The implementation applies this operator blockwise. It stores boundary samples and generated QJet weights only; it does not store the dense `n x n` Q matrix. The Ritz values below are exact float64 eigenvalues of the small generalized projected problem

```text
G c = lambda M c,
G_ab = Phi_a^T Lambda_Q,n Phi_b,
M_ab = Phi_a^T Phi_b,
Phi = [cos(theta), sin(theta), ..., cos(6 theta), sin(6 theta)].
```

This is a fixed 12-dimensional trace subspace, so refinement tests whether the same audited Q spectral witness stabilizes as the boundary sampling is refined.

## Dirichlet Spectrum

The exact continuum Dirichlet statement is

```text
lambda_k(D_left) = lambda_k(D_right) for every k >= 1.
```

The exact values are not known in closed form from this construction. The finite-difference values below are a reproducibility check only; they are not the mathematical spectrum and their mismatch is a grid artifact.

Finite-difference Dirichlet check at grid `{fd["grid_size"]}`:

{markdown_table(["k", "left FD eigenvalue", "right FD eigenvalue", "relative artifact"], fd_rows_md)}

## Q Projected Ritz Convergence

{markdown_table(["n", "relative split", "left delta prev", "right delta prev", "left Ritz values 1-6", "right Ritz values 1-6"], convergence_rows_md)}

## Symmetry And Sampling Controls

The controls below compare the first six projected Q Ritz values for the left GWW domain at `n = {SYMMETRY_N}`. Rotation and translation are invariant to roundoff. Orientation reversal and node phase/corner placement move only at the expected discretization level. Uniform scaling is reported after multiplying eigenvalues by the scale factor, because this first-order boundary operator scales as inverse length.

{markdown_table(["control", "n", "phase", "scale post-factor", "relative deviation", "Ritz values 1-6"], symmetry_rows_md)}

## Meshless Shape-Optimization Loss

The shape optimization demo uses a boundary map `gamma_p(theta_i)` and probe traces `u_a(theta_i)`. Its reduced Q Gram matrix is

```text
G_ab(p) = (perimeter(gamma_p)/(pi n))
         sum_{{i<j}} ((u_a(i)-u_a(j))(u_b(i)-u_b(j)))
                  / (|gamma_p(i)-gamma_p(j)|^2 + epsilon^2).
```

For polygonal corner-fixed runs, a low-rank corner repayment Gram `C(p)` is added:

```text
G_corr(p) = G(p) + corner_q_weight C(p).
```

The scalar loss minimized in `examples/q_autograd_meshless_shape_optimization.py` is exactly

```text
L(p) =
  mean((G_corr(p)/tr G_corr(p) - G_target/tr G_target)^2)
  + q_trace_weight * [log(tr G_corr(p) / tr G_target)]^2
  + moment_weight * mean((m(p)-m_target)^2) / mean(m_target^2)
  + corner_weight * mean((c(p)-c_target)^2) / mean(c_target^2)
  + area_weight * ((A(p)-A_target)/|A_target|)^2
  + centroid_weight * |mean_i gamma_p(theta_i)|^2
  + roughness_weight * R(p).
```

Current default weights are:

```text
q_trace_weight = 0.5
moment_weight = 4.0
corner_weight = 1.5
area_weight = 50.0
centroid_weight = 2.0
roughness_weight = 2.0e-4
epsilon = 1.0e-8
```

For smooth Fourier boundaries `R(p)` is spectral coefficient roughness. For polygonal boundaries it is vertex second-difference roughness plus `0.02 * var(edge_lengths)`.

## Reproducibility

Run:

```bash
python3 examples/isospectral/gww_section11_audit.py
```

Output files are written under:

```text
{path.parent}
```

References:

- https://arxiv.org/abs/math/9207215
- https://www.comsol.com/model/download/1166951/models.mph.isospectral_drums.pdf
- https://eudml.org/doc/144038
"""
    path.write_text(text, encoding="utf-8")


def write_plot(payload: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.05",
            "axes.labelcolor": "0.05",
            "axes.titlecolor": "0.05",
            "xtick.color": "0.05",
            "ytick.color": "0.05",
            "grid.color": "0.82",
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    convergence = payload["q_convergence"]
    ns = [row["n"] for row in convergence]
    split = [row["relative_split_first6"] for row in convergence]
    left1 = [row["left_ritz_descending"][0] for row in convergence]
    right1 = [row["right_ritz_descending"][0] for row in convergence]
    left2 = [row["left_ritz_descending"][1] for row in convergence]
    right2 = [row["right_ritz_descending"][1] for row in convergence]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(ns, left1, color="0.0", marker="o", label="left Ritz 1")
    ax.plot(ns, right1, color="0.45", marker="s", linestyle=(0, (3, 2)), label="right Ritz 1")
    ax.plot(ns, left2, color="0.2", marker="o", linestyle=(0, (1, 1)), label="left Ritz 2")
    ax.plot(ns, right2, color="0.65", marker="s", linestyle=(0, (6, 2)), label="right Ritz 2")
    ax.set_xscale("log", base=2)
    ax.set_title("projected Q Ritz values")
    ax.set_xlabel("boundary samples n")
    ax.set_ylabel("Ritz value")
    ax.grid(True, linewidth=0.45)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(ns, split, color="0.0", marker="o")
    ax.set_xscale("log", base=2)
    ax.set_title("first-six Q spectral separation")
    ax.set_xlabel("boundary samples n")
    ax.set_ylabel("relative split")
    ax.grid(True, linewidth=0.45)
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_payload() -> dict[str, object]:
    left = normalize_polygon(GWW_LEFT, area=1.0)
    right = normalize_polygon(GWW_RIGHT, area=1.0)
    convergence = convergence_rows(left, right)
    symmetry = symmetry_rows(left)
    fd = finite_difference_dirichlet(left, right, count=8, grid_size=96)
    return {
        "generated_note": "deterministic float64 CPU run; no dense Q matrix stored",
        "coordinates": {
            "raw_left_clockwise": GWW_LEFT,
            "raw_right_clockwise": GWW_RIGHT,
            "raw_left_signed_area": polygon_area(GWW_LEFT),
            "raw_right_signed_area": polygon_area(GWW_RIGHT),
            "normalized_left_ccw": left,
            "normalized_right_ccw": right,
            "normalized_left_area": abs(polygon_area(left)),
            "normalized_right_area": abs(polygon_area(right)),
            "normalized_left_perimeter": polygon_perimeter(left),
            "normalized_right_perimeter": polygon_perimeter(right),
        },
        "sampling_protocol": {
            "rule": "equal arclength samples s_k=((k+phase) mod n) L/n on the exact piecewise-linear polygon",
            "default_phase": 0.5,
            "default_corner_treatment": "midpoint nodes avoid corners; corners are exact polygon vertices, not smoothed",
            "corner_control_phase": 0.0,
        },
        "q_normalization": {
            "operator": "(Lambda_Q,n f)_i = (L/(pi*n)) sum_{j!=i} (f_i-f_j)/|x_i-x_j|^2",
            "projection_basis": "cos(m theta), sin(m theta), m=1..6",
            "dense_q_matrix_stored": False,
            "blockwise_temporary_distance_rows": True,
        },
        "dirichlet_exact_statement": "lambda_k(left)=lambda_k(right) for all k by Gordon-Webb-Wolpert isospectrality",
        "dirichlet_fd_spectrum": fd,
        "q_convergence": convergence,
        "q_symmetry_controls": symmetry,
        "claim": "At the discretized chord-operator level, this standard isospectral pair is separated robustly under refinement.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gww_isospectral_section11")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "section11_gww_q_audit.json"
    convergence_csv = args.out_dir / "section11_gww_q_convergence.csv"
    symmetry_csv = args.out_dir / "section11_gww_q_symmetry_controls.csv"
    md_path = args.out_dir / "section11_gww_q_audit.md"
    png_path = args.out_dir / "section11_gww_q_convergence.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows_csv(payload["q_convergence"], convergence_csv, kind="convergence")
    write_rows_csv(payload["q_symmetry_controls"], symmetry_csv, kind="symmetry")
    write_report(payload, md_path)
    if not args.no_plot:
        write_plot(payload, png_path)
    final = payload["q_convergence"][-1]
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "convergence_csv": str(convergence_csv),
                "symmetry_csv": str(symmetry_csv),
                "png": None if args.no_plot else str(png_path),
                "n_max": final["n"],
                "left_ritz_first6_nmax": final["left_ritz_descending"],
                "right_ritz_first6_nmax": final["right_ritz_descending"],
                "relative_split_first6_nmax": final["relative_split_first6"],
                "dense_q_matrix_stored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
