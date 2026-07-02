#!/usr/bin/env python3
"""Corner correction benchmark for Q spectra.

This script tests whether the existing corner repayment ideas help on polygonal
corners:

* raw chord Q, applied matrix-free;
* Kondrat'ev/Mellin/Hurwitz low-rank corner repayment;
* Joukowsky/Chebyshev endpoint prewarping of panel samples;
* the combined prewarp plus low-rank repayment.

The benchmark compares small projected Ritz spectra against a fine matrix-free
reference.  It never stores the dense boundary Q matrix; the largest temporary
kernel block has shape ``chunk_size x n``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
TAU = 2.0 * math.pi
Point = tuple[float, float]


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    family: str
    vertices: tuple[Point, ...]


@dataclass(frozen=True)
class SampledBoundary:
    points: torch.Tensor
    theta: torch.Tensor
    weights: torch.Tensor
    corner_indices: tuple[int, ...]
    perimeter: float
    edge_counts: tuple[int, ...]
    rule: str


@dataclass(frozen=True)
class Evaluation:
    values: torch.Tensor
    normalized_values: torch.Tensor
    stiffness: torch.Tensor
    mass: torch.Tensor
    correction: torch.Tensor
    elapsed_ms: float
    corner_lambdas: tuple[float, ...]
    edge_counts: tuple[int, ...]


def signed_area(vertices: tuple[Point, ...]) -> float:
    return 0.5 * sum(
        x0 * y1 - y0 * x1
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1])
    )


def ensure_ccw(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
    return vertices if signed_area(vertices) > 0.0 else tuple(reversed(vertices))


def polygon_perimeter(vertices: tuple[Point, ...]) -> float:
    return sum(math.dist(vertices[index], vertices[(index + 1) % len(vertices)]) for index in range(len(vertices)))


def edge_counts(vertices: tuple[Point, ...], sample_count: int) -> tuple[int, ...]:
    if sample_count < len(vertices):
        raise ValueError("sample_count must be at least the vertex count")
    lengths = [math.dist(vertices[index], vertices[(index + 1) % len(vertices)]) for index in range(len(vertices))]
    total = sum(lengths)
    raw = [length * sample_count / total for length in lengths]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) < sample_count:
        deficits = [raw[index] - counts[index] for index in range(len(counts))]
        index = max(range(len(counts)), key=lambda item: deficits[item])
        counts[index] += 1
    while sum(counts) > sample_count:
        surpluses = [counts[index] - raw[index] if counts[index] > 1 else -1.0 for index in range(len(counts))]
        index = max(range(len(counts)), key=lambda item: surpluses[item])
        counts[index] -= 1
    return tuple(counts)


def periodic_weights(arclength: torch.Tensor, perimeter: float) -> torch.Tensor:
    previous_arclength = torch.roll(arclength, shifts=1)
    next_arclength = torch.roll(arclength, shifts=-1)
    previous_arclength[0] -= perimeter
    next_arclength[-1] += perimeter
    weights = 0.5 * (next_arclength - previous_arclength)
    return weights * (perimeter / weights.sum())


def sample_polygon(vertices: tuple[Point, ...], sample_count: int, *, rule: str) -> SampledBoundary:
    vertices = ensure_ccw(vertices)
    counts = edge_counts(vertices, sample_count)
    perimeter = polygon_perimeter(vertices)
    points: list[Point] = []
    arclength: list[float] = []
    corner_indices: list[int] = []
    accumulated = 0.0
    for edge, count in enumerate(counts):
        corner_indices.append(len(points))
        start = vertices[edge]
        stop = vertices[(edge + 1) % len(vertices)]
        length = math.dist(start, stop)
        for local_index in range(count):
            raw_t = local_index / count
            if rule == "uniform":
                t = raw_t
            elif rule == "joukowsky":
                t = 0.5 * (1.0 - math.cos(math.pi * raw_t))
            else:
                raise ValueError(f"unknown sampling rule: {rule}")
            x = start[0] + t * (stop[0] - start[0])
            y = start[1] + t * (stop[1] - start[1])
            points.append((x, y))
            arclength.append(accumulated + t * length)
        accumulated += length
    point_tensor = torch.tensor(points, dtype=torch.float64)
    arclength_tensor = torch.tensor(arclength, dtype=torch.float64)
    weights = periodic_weights(arclength_tensor, perimeter)
    theta = TAU * arclength_tensor / perimeter
    return SampledBoundary(
        points=point_tensor,
        theta=theta,
        weights=weights,
        corner_indices=tuple(corner_indices),
        perimeter=perimeter,
        edge_counts=counts,
        rule=rule,
    )


def weighted_orthonormalize(columns: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    basis: list[torch.Tensor] = []
    total_weight = weights.sum()
    for column in columns:
        vector = column - (weights * column).sum() / total_weight
        for existing in basis:
            coefficient = (weights * vector * existing).sum() / (weights * existing * existing).sum().clamp_min(1.0e-30)
            vector = vector - coefficient * existing
        norm = torch.sqrt((weights * vector * vector).sum()).clamp_min(0.0)
        if float(norm) > 1.0e-10:
            basis.append(vector / norm)
    if not basis:
        raise ValueError("probe basis is empty")
    return torch.stack(basis, dim=1)


def probe_matrix(boundary: SampledBoundary, modes: int, *, basis_kind: str) -> torch.Tensor:
    columns = []
    theta = boundary.theta
    weights = boundary.weights
    for mode in range(1, modes + 1):
        columns.append(torch.cos(mode * theta))
        columns.append(torch.sin(mode * theta))
    if basis_kind == "corner_augmented":
        angles = corner_interior_angles(boundary)
        lambdas = math.pi / angles
        arclength = boundary.theta * (boundary.perimeter / TAU)
        for slot, corner_index in enumerate(boundary.corner_indices):
            lam = lambdas[slot]
            offset = torch.abs(arclength - arclength[corner_index])
            cyclic_offset = torch.minimum(offset, torch.tensor(boundary.perimeter, dtype=weights.dtype) - offset)
            radius = (cyclic_offset + 0.5 * weights.mean()).clamp_min(torch.finfo(weights.dtype).eps) / boundary.perimeter
            flux_profile = radius.pow(lam - 1.0)
            trace_profile = radius.pow(lam)
            columns.append(flux_profile)
            columns.append(trace_profile)
    elif basis_kind != "global_fourier":
        raise ValueError(f"unknown basis kind: {basis_kind}")
    return weighted_orthonormalize(columns, weights)


def q_apply_blockwise(
    points: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size: int,
    epsilon: float,
) -> torch.Tensor:
    """Apply the chord Q generator without forming the dense matrix."""

    n = points.shape[0]
    output = torch.empty_like(values)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = points[start:stop]
        delta = block[:, None, :] - points[None, :, :]
        distance2 = torch.sum(delta * delta, dim=2) + epsilon * epsilon
        for row, index in enumerate(range(start, stop)):
            distance2[row, index] = float("inf")
        kernel = weights.unsqueeze(0) / distance2
        output[start:stop] = (kernel.sum(dim=1).unsqueeze(1) * values[start:stop] - kernel @ values) / math.pi
    return output


def generalized_ritz(stiffness: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(mass)
    tmp = torch.linalg.solve_triangular(chol, stiffness, upper=False)
    reduced = torch.linalg.solve_triangular(chol, tmp.T, upper=False).T
    reduced = 0.5 * (reduced + reduced.T)
    return torch.flip(torch.linalg.eigvalsh(reduced), dims=(0,))


def corner_interior_angles(boundary: SampledBoundary) -> torch.Tensor:
    angles = []
    points = boundary.points
    n = points.shape[0]
    for index in boundary.corner_indices:
        previous_point = points[(index - 1) % n]
        point = points[index]
        next_point = points[(index + 1) % n]
        incoming = point - previous_point
        outgoing = next_point - point
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        dot = torch.sum(incoming * outgoing)
        turn = torch.atan2(cross, dot)
        angle = math.pi - turn
        angles.append(torch.clamp(angle, min=0.08, max=TAU - 0.08))
    return torch.stack(angles)


def regularized_hurwitz_zeta(s: torch.Tensor, beta: float, terms: int) -> torch.Tensor:
    """Euler-Maclaurin regularized Hurwitz zeta for the corner channel."""

    count = max(8, int(terms))
    beta_tensor = torch.tensor(beta, dtype=s.dtype, device=s.device)
    k = torch.arange(count, dtype=s.dtype, device=s.device)
    q = k + beta_tensor
    partial = torch.sum(q.pow(-s))
    tail = torch.tensor(float(count), dtype=s.dtype, device=s.device) + beta_tensor
    return partial + tail.pow(1.0 - s) / (s - 1.0) + 0.5 * tail.pow(-s) + (s / 12.0) * tail.pow(-s - 1.0)


def corner_repayment_stiffness(
    boundary: SampledBoundary,
    probes: torch.Tensor,
    *,
    zeta_terms: int,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Low-rank Kondrat'ev/Mellin corner repayment in probe space."""

    angles = corner_interior_angles(boundary)
    lambdas = math.pi / angles
    gram = torch.zeros((probes.shape[1], probes.shape[1]), dtype=probes.dtype)
    arclength = boundary.theta * (boundary.perimeter / TAU)
    h_ratio = torch.tensor(1.0 / boundary.points.shape[0], dtype=probes.dtype)
    weights = boundary.weights
    total_weight = weights.sum()
    for slot, corner_index in enumerate(boundary.corner_indices):
        angle = angles[slot]
        lam = lambdas[slot]
        s = 1.0 - lam
        defect = (angle - math.pi) / math.pi
        offset = torch.abs(arclength - arclength[corner_index])
        cyclic_offset = torch.minimum(offset, torch.tensor(boundary.perimeter, dtype=probes.dtype) - offset)
        radius = (cyclic_offset + 0.5 * weights.mean()).clamp_min(torch.finfo(probes.dtype).eps) / boundary.perimeter
        profile = radius.pow(lam - 1.0)
        profile = profile - (weights * profile).sum() / total_weight
        profile_norm = torch.sqrt((weights * profile * profile).sum() / total_weight).clamp_min(1.0e-14)
        profile = profile / profile_norm
        centered_probes = probes - probes[corner_index].unsqueeze(0)
        response = (weights.unsqueeze(1) * profile.unsqueeze(1) * centered_probes).sum(dim=0) / total_weight
        zeta_weight = torch.abs(regularized_hurwitz_zeta(s, beta=0.5, terms=zeta_terms))
        amplitude = torch.abs(defect) * zeta_weight * h_ratio.pow(lam)
        gram = gram + amplitude * torch.outer(response, response)
    return gram, tuple(float(value) for value in lambdas.tolist())


def evaluate_boundary(
    shape: ShapeSpec,
    *,
    sample_count: int,
    modes: int,
    basis_kind: str,
    rule: str,
    chunk_size: int,
    epsilon: float,
    zeta_terms: int,
) -> Evaluation:
    boundary = sample_polygon(shape.vertices, sample_count, rule=rule)
    probes = probe_matrix(boundary, modes, basis_kind=basis_kind)
    start = perf_counter()
    applied = q_apply_blockwise(
        boundary.points,
        probes,
        boundary.weights,
        chunk_size=chunk_size,
        epsilon=epsilon,
    )
    stiffness = probes.T @ (boundary.weights.unsqueeze(1) * applied)
    mass = probes.T @ (boundary.weights.unsqueeze(1) * probes)
    correction, lambdas = corner_repayment_stiffness(boundary, probes, zeta_terms=zeta_terms)
    values = generalized_ritz(stiffness, mass)
    normalized = values / values.sum().clamp_min(1.0e-30)
    elapsed_ms = 1000.0 * (perf_counter() - start)
    return Evaluation(
        values=values,
        normalized_values=normalized,
        stiffness=stiffness,
        mass=mass,
        correction=correction,
        elapsed_ms=elapsed_ms,
        corner_lambdas=lambdas,
        edge_counts=boundary.edge_counts,
    )


def corrected_values(evaluation: Evaluation, weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    stiffness = evaluation.stiffness + weight * evaluation.correction
    values = generalized_ritz(stiffness, evaluation.mass)
    return values, values / values.sum().clamp_min(1.0e-30)


def relative_spectrum_error(values: torch.Tensor, reference: torch.Tensor, count: int) -> float:
    lhs = values[:count]
    rhs = reference[:count]
    return float(torch.linalg.norm(lhs - rhs) / torch.linalg.norm(rhs).clamp_min(1.0e-30))


def shape_specs() -> tuple[ShapeSpec, ...]:
    star_vertices = []
    for index in range(10):
        radius = 1.18 if index % 2 == 0 else 0.48
        angle = math.pi / 2.0 + TAU * index / 10.0
        star_vertices.append((radius * math.cos(angle), radius * math.sin(angle)))

    return (
        ShapeSpec(
            name="square_convex_corners",
            family="convex right-angle polygon",
            vertices=ensure_ccw(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))),
        ),
        ShapeSpec(
            name="l_notch_reentrant",
            family="single 270-degree reentrant corner",
            vertices=ensure_ccw(
                (
                    (-1.20, -1.00),
                    (1.20, -1.00),
                    (1.20, 0.22),
                    (0.16, 0.22),
                    (0.16, 1.05),
                    (-1.20, 1.05),
                )
            ),
        ),
        ShapeSpec(
            name="star_alternating_corners",
            family="alternating acute and reentrant corners",
            vertices=ensure_ccw(tuple(star_vertices)),
        ),
        ShapeSpec(
            name="stealth_double_concave",
            family="symmetric double-concave aircraft-like polygon",
            vertices=ensure_ccw(
                (
                    (1.35, 0.00),
                    (0.50, 0.34),
                    (-1.10, 0.58),
                    (-0.42, 0.16),
                    (-0.42, -0.16),
                    (-1.10, -0.58),
                    (0.50, -0.34),
                )
            ),
        ),
    )


def best_weighted_error(
    evaluation: Evaluation,
    reference: torch.Tensor,
    *,
    top_count: int,
    weight_grid: tuple[float, ...],
) -> tuple[float, float, list[float]]:
    best_error = float("inf")
    best_weight = 0.0
    best_values: torch.Tensor | None = None
    for weight in weight_grid:
        _, normalized = corrected_values(evaluation, weight)
        error = relative_spectrum_error(normalized, reference, top_count)
        if error < best_error:
            best_error = error
            best_weight = weight
            best_values = normalized
    assert best_values is not None
    return best_error, best_weight, [float(value) for value in best_values[:top_count].tolist()]


def run_benchmark(
    *,
    output_dir: Path,
    n_values: tuple[int, ...],
    reference_n: int,
    modes: int,
    top_count: int,
    default_corner_weight: float,
    zeta_terms: int,
    chunk_size: int,
    epsilon: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_grid = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50, 4.00)
    rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []

    basis_kinds = ("global_fourier", "corner_augmented")
    for shape in shape_specs():
        for basis_kind in basis_kinds:
            reference_eval = evaluate_boundary(
                shape,
                sample_count=reference_n,
                modes=modes,
                basis_kind=basis_kind,
                rule="uniform",
                chunk_size=chunk_size,
                epsilon=epsilon,
                zeta_terms=zeta_terms,
            )
            reference = reference_eval.normalized_values
            reference_rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "basis_kind": basis_kind,
                    "probe_dimension": int(reference_eval.values.numel()),
                    "reference_n": reference_n,
                    "reference_rule": "uniform",
                    "reference_elapsed_ms": reference_eval.elapsed_ms,
                    "reference_ritz_normalized_top": [float(value) for value in reference[:top_count].tolist()],
                    "reference_ritz_raw_top": [float(value) for value in reference_eval.values[:top_count].tolist()],
                    "corner_lambdas": list(reference_eval.corner_lambdas),
                }
            )

            for sample_count in n_values:
                raw_eval = evaluate_boundary(
                    shape,
                    sample_count=sample_count,
                    modes=modes,
                    basis_kind=basis_kind,
                    rule="uniform",
                    chunk_size=chunk_size,
                    epsilon=epsilon,
                    zeta_terms=zeta_terms,
                )
                jouk_eval = evaluate_boundary(
                    shape,
                    sample_count=sample_count,
                    modes=modes,
                    basis_kind=basis_kind,
                    rule="joukowsky",
                    chunk_size=chunk_size,
                    epsilon=epsilon,
                    zeta_terms=zeta_terms,
                )

                raw_error = relative_spectrum_error(raw_eval.normalized_values, reference, top_count)
                raw_default_values, raw_default_normalized = corrected_values(raw_eval, default_corner_weight)
                km_default_error = relative_spectrum_error(raw_default_normalized, reference, top_count)
                km_best_error, km_best_weight, km_best_values = best_weighted_error(
                    raw_eval,
                    reference,
                    top_count=top_count,
                    weight_grid=weight_grid,
                )

                jouk_error = relative_spectrum_error(jouk_eval.normalized_values, reference, top_count)
                jouk_default_values, jouk_default_normalized = corrected_values(jouk_eval, default_corner_weight)
                jouk_km_default_error = relative_spectrum_error(jouk_default_normalized, reference, top_count)
                jouk_km_best_error, jouk_km_best_weight, jouk_km_best_values = best_weighted_error(
                    jouk_eval,
                    reference,
                    top_count=top_count,
                    weight_grid=weight_grid,
                )

                methods = {
                    "raw": raw_error,
                    "kondratiev_mellin_default": km_default_error,
                    "kondratiev_mellin_best_grid": km_best_error,
                    "joukowsky_prewarp": jouk_error,
                    "joukowsky_plus_km_default": jouk_km_default_error,
                    "joukowsky_plus_km_best_grid": jouk_km_best_error,
                }
                best_method = min(methods, key=methods.__getitem__)
                rows.append(
                    {
                        "shape": shape.name,
                        "family": shape.family,
                        "basis_kind": basis_kind,
                        "probe_dimension": int(raw_eval.values.numel()),
                        "n": sample_count,
                        "reference_n": reference_n,
                        "modes": modes,
                        "top_count": top_count,
                        "raw_error": raw_error,
                        "km_default_error": km_default_error,
                        "km_best_error": km_best_error,
                        "km_best_weight": km_best_weight,
                        "joukowsky_error": jouk_error,
                        "joukowsky_km_default_error": jouk_km_default_error,
                        "joukowsky_km_best_error": jouk_km_best_error,
                        "joukowsky_km_best_weight": jouk_km_best_weight,
                        "best_method": best_method,
                        "best_error": methods[best_method],
                        "raw_to_best_improvement": raw_error / max(methods[best_method], 1.0e-30),
                        "km_default_improvement": raw_error / max(km_default_error, 1.0e-30),
                        "joukowsky_improvement": raw_error / max(jouk_error, 1.0e-30),
                        "combined_default_improvement": raw_error / max(jouk_km_default_error, 1.0e-30),
                        "raw_elapsed_ms": raw_eval.elapsed_ms,
                        "joukowsky_elapsed_ms": jouk_eval.elapsed_ms,
                        "edge_counts_uniform": list(raw_eval.edge_counts),
                        "edge_counts_joukowsky": list(jouk_eval.edge_counts),
                        "corner_lambdas": list(raw_eval.corner_lambdas),
                        "raw_ritz_normalized_top": [
                            float(value) for value in raw_eval.normalized_values[:top_count].tolist()
                        ],
                        "km_default_ritz_normalized_top": [
                            float(value) for value in raw_default_normalized[:top_count].tolist()
                        ],
                        "km_best_ritz_normalized_top": km_best_values,
                        "joukowsky_ritz_normalized_top": [
                            float(value) for value in jouk_eval.normalized_values[:top_count].tolist()
                        ],
                        "joukowsky_km_default_ritz_normalized_top": [
                            float(value) for value in jouk_default_normalized[:top_count].tolist()
                        ],
                        "joukowsky_km_best_ritz_normalized_top": jouk_km_best_values,
                        "raw_ritz_top": [float(value) for value in raw_eval.values[:top_count].tolist()],
                        "km_default_ritz_top": [float(value) for value in raw_default_values[:top_count].tolist()],
                        "joukowsky_ritz_top": [float(value) for value in jouk_eval.values[:top_count].tolist()],
                        "joukowsky_km_default_ritz_top": [
                            float(value) for value in jouk_default_values[:top_count].tolist()
                        ],
                    }
                )

    csv_path = output_dir / "corner_correction_benchmark.csv"
    csv_columns = [
        "shape",
        "family",
        "basis_kind",
        "probe_dimension",
        "n",
        "reference_n",
        "raw_error",
        "km_default_error",
        "km_best_error",
        "km_best_weight",
        "joukowsky_error",
        "joukowsky_km_default_error",
        "joukowsky_km_best_error",
        "joukowsky_km_best_weight",
        "best_method",
        "best_error",
        "raw_to_best_improvement",
        "km_default_improvement",
        "joukowsky_improvement",
        "combined_default_improvement",
        "raw_elapsed_ms",
        "joukowsky_elapsed_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in csv_columns})

    payload = {
        "method": {
            "dense_matrix_stored": False,
            "q_application": "blockwise borrow-compute-repay chord Q apply",
            "corner_repayment": "Kondratiev/Mellin profiles with Euler-Maclaurin Hurwitz zeta weights",
            "joukowsky_prewarp": "per-edge Chebyshev/Joukowsky endpoint clustering t=(1-cos(pi k/m))/2",
            "default_corner_weight": default_corner_weight,
            "weight_grid": list(weight_grid),
            "zeta_terms": zeta_terms,
            "modes": modes,
            "top_count": top_count,
            "reference_n": reference_n,
            "basis_kinds": list(basis_kinds),
        },
        "references": reference_rows,
        "rows": rows,
    }
    json_path = output_dir / "corner_correction_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure_path = output_dir / "corner_correction_benchmark.png"
    render_figure(rows, reference_rows, figure_path)

    report_path = output_dir / "corner_correction_benchmark.md"
    write_report(payload, report_path, figure_path, csv_path, json_path)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(figure_path),
        "report": str(report_path),
        "rows": rows,
        "references": reference_rows,
    }


def render_figure(rows: list[dict[str, object]], reference_rows: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.15",
            "axes.labelcolor": "0.1",
            "xtick.color": "0.1",
            "ytick.color": "0.1",
            "text.color": "0.1",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.3,
        }
    )
    plot_basis = "corner_augmented"
    plot_references = [row for row in reference_rows if row["basis_kind"] == plot_basis]
    shapes = [row["shape"] for row in plot_references]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    axes_flat = list(axes.ravel())
    markers = {
        "raw_error": ("o", "0.05", "raw Q"),
        "km_default_error": ("s", "0.35", "Kondrat'ev/Mellin"),
        "joukowsky_error": ("^", "0.60", "Joukowsky"),
        "joukowsky_km_default_error": ("D", "0.15", "combined"),
    }
    for axis, shape in zip(axes_flat, shapes):
        subset = [row for row in rows if row["shape"] == shape and row["basis_kind"] == plot_basis]
        ns = sorted({int(row["n"]) for row in subset})
        for key, (marker, color, label) in markers.items():
            values = []
            for n_value in ns:
                row = next(item for item in subset if int(item["n"]) == n_value)
                values.append(float(row[key]))
            axis.plot(ns, values, marker=marker, color=color, label=label)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(True, which="both", color="0.88", linewidth=0.6)
        axis.set_title(str(shape).replace("_", " "))
        axis.set_xlabel("boundary nodes n")
        axis.set_ylabel("relative top-spectrum error")
    axes_flat[0].legend(frameon=False, loc="best")
    fig.suptitle("Corner Q Corrections: Corner-Augmented Spectrum Error vs Fine Reference", fontsize=10)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compact_float(value: object) -> str:
    return f"{float(value):.3e}"


def write_report(payload: dict[str, object], report_path: Path, figure_path: Path, csv_path: Path, json_path: Path) -> None:
    rows = payload["rows"]
    references = payload["references"]
    assert isinstance(rows, list)
    assert isinstance(references, list)
    latest_n = max(int(row["n"]) for row in rows)
    latest_rows = [row for row in rows if int(row["n"]) == latest_n]
    basis_kinds = tuple(dict.fromkeys(str(row["basis_kind"]) for row in latest_rows))

    lines = [
        "# Corner Correction Benchmark",
        "",
        "This run tests corner corrections on projected Q spectra, not on a stored dense boundary matrix.",
        "",
        f"![Corner correction benchmark]({figure_path})",
        "",
        "## Protocol",
        "",
        "- Raw Q: blockwise chord-kernel application with weighted boundary samples.",
        "- Kondrat'ev/Mellin repayment: each vertex contributes a stored low-rank profile with exponent `lambda = pi / omega`, where `omega` is the interior angle.",
        "- Hurwitz/zeta scaling: the corner amplitude uses an Euler-Maclaurin regularized `zeta(1-lambda, 1/2)` factor.",
        "- Joukowsky prewarp: each polygon edge is sampled with `t_k = (1 - cos(pi k / m)) / 2`, clustering near panel endpoints.",
        "- Reference: the same projected Q spectrum on a fine uniform boundary sample.",
        "",
        "The correction is therefore judged by convergence of a small generalized Ritz spectrum.",
        "",
        "## Latest Refinement",
        "",
    ]

    for basis_kind in basis_kinds:
        lines.extend(
            [
                f"### `{basis_kind}`",
                "",
                "| Shape | Probe dim | Raw err | KM default err | Joukowsky err | Combined default err | Best method | Best err | Raw/best | Corner lambdas |",
                "|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in [item for item in latest_rows if item["basis_kind"] == basis_kind]:
            lambdas = ", ".join(f"{float(value):.3f}" for value in row["corner_lambdas"])
            lines.append(
                "| {shape} | {dim} | `{raw}` | `{km}` | `{j}` | `{c}` | `{best}` | `{best_err}` | `{improve:.2f}x` | `{lambdas}` |".format(
                    shape=row["shape"],
                    dim=row["probe_dimension"],
                    raw=compact_float(row["raw_error"]),
                    km=compact_float(row["km_default_error"]),
                    j=compact_float(row["joukowsky_error"]),
                    c=compact_float(row["joukowsky_km_default_error"]),
                    best=row["best_method"],
                    best_err=compact_float(row["best_error"]),
                    improve=float(row["raw_to_best_improvement"]),
                    lambdas=lambdas,
                )
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Reference Spectra",
            "",
            "| Shape | Basis | Probe dim | Reference n | Normalized Ritz values, first modes | Raw Ritz values, first modes |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in references:
        normalized = ", ".join(f"{float(value):.8f}" for value in row["reference_ritz_normalized_top"][:6])
        raw = ", ".join(f"{float(value):.8f}" for value in row["reference_ritz_raw_top"][:6])
        lines.append(
            f"| {row['shape']} | `{row['basis_kind']}` | {row['probe_dimension']} | {row['reference_n']} | `{normalized}` | `{raw}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Verdict from the latest refinement:",
            "",
            "- Smooth global probes: raw weighted Q is already the best or tied-best result; corner corrections do not improve the smooth projected spectrum.",
            "- Corner-augmented probes: the Joukowsky/Chebyshev endpoint prewarp is the useful correction in this run.",
            "- The current positive low-rank Kondrat'ev/Mellin/Hurwitz repayment is effectively neutral. The best positive weight grid usually selects zero, so this layer should not yet be called load-bearing for corner spectra.",
            "",
            "The default Kondrat'ev/Mellin repayment is deliberately fixed rather than fitted. The `*_best_grid` columns in the JSON/CSV show how much calibration headroom exists if the corner amplitude is tuned against a reference.",
            "",
            "When `lambda < 1`, the domain has a reentrant corner and the singular corner channel is genuinely strong. Convex right-angle corners have `lambda > 1`; in those cases the low-rank repayment is expected to be weaker and endpoint placement usually matters more than the zeta amplitude.",
            "",
            "The Joukowsky row here is a sampling/preconditioning test, not a full exterior Riemann-map solve. It answers whether endpoint clustering helps the same Q spectrum before installing a heavier conformal pullback.",
            "",
            "## Artifacts",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
            f"- Figure: `{figure_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "corner_correction_benchmark")
    parser.add_argument("--n-values", type=int, nargs="+", default=(96, 192, 384, 768))
    parser.add_argument("--reference-n", type=int, default=3072)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--top-count", type=int, default=8)
    parser.add_argument("--default-corner-weight", type=float, default=0.35)
    parser.add_argument("--zeta-terms", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--epsilon", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        output_dir=args.output_dir,
        n_values=tuple(args.n_values),
        reference_n=args.reference_n,
        modes=args.modes,
        top_count=args.top_count,
        default_corner_weight=args.default_corner_weight,
        zeta_terms=args.zeta_terms,
        chunk_size=args.chunk_size,
        epsilon=args.epsilon,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows" and key != "references"}, indent=2))


if __name__ == "__main__":
    main()
