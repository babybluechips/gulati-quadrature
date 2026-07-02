#!/usr/bin/env python3
"""Meshless shape optimization through an autodiff Q objective.

This demo follows the Fourier boundary parameterization used in the attached
meshless-shape paper, but replaces the paper's hand-derived kernel gradient
with PyTorch autograd through a matrix-free Q functional.

The optimized shape is represented only by Fourier coefficients.  The objective
matches a target shape's reduced Q Gram matrix on smooth probe traces:

    G_ab(gamma) = (h/pi) sum_{i<j}
        (u_a(theta_i)-u_a(theta_j))(u_b(theta_i)-u_b(theta_j))
        / |gamma_i-gamma_j|^2.

The demo never forms the dense boundary Q matrix.  It stores only the boundary
QJets, a pair list, and a small probe-space Gram matrix.

Use ``--case polygon`` for a vertex-parameterized polygonal optimization and
``--case polygon_corner_fixed`` to activate the low-rank corner repayment
channel based on stored corner QJets, interior angles, and a Hurwitz-style
Mellin weight.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class DemoConfig:
    n: int = 128
    modes: int = 5
    probe_modes: int = 6
    case: str = "smooth_fourier"
    steps: int = 1200
    adam_lr: float = 1.5e-2
    lbfgs_steps: int = 80
    q_epsilon: float = 1.0e-8
    q_trace_weight: float = 0.5
    moment_order: int = 4
    moment_weight: float = 4.0
    corner_fix: bool = False
    corner_weight: float = 1.5
    corner_q_weight: float = 0.35
    corner_zeta_terms: int = 64
    field_grid_x: int = 180
    field_grid_y: int = 96
    field_solve_iterations: int = 96
    field_ridge: float = 1.0e-6
    area_weight: float = 50.0
    centroid_weight: float = 2.0
    roughness_weight: float = 2.0e-4
    dtype: str = "float64"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    points: torch.Tensor
    area: torch.Tensor
    corner_indices: tuple[int, ...]
    edge_counts: tuple[int, ...] = ()
    vertices: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def theta_grid(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.arange(n, dtype=dtype, device=device) * (TAU / n)


def coefficient_count(modes: int) -> int:
    return 2 + 4 * modes


def unpack_mode(params: torch.Tensor, mode: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = 2 + 4 * (mode - 1)
    return params[offset], params[offset + 1], params[offset + 2], params[offset + 3]


def circle_params(
    modes: int,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    params = torch.zeros(coefficient_count(modes), dtype=dtype, device=device)
    params[0] = center[0]
    params[1] = center[1]
    params[2] = radius
    params[5] = radius
    return params


def target_params(modes: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    params = circle_params(modes, radius=1.0, dtype=dtype, device=device)
    # x = a0 + sum a_k cos(k theta) + b_k sin(k theta)
    # y = c0 + sum c_k cos(k theta) + d_k sin(k theta)
    entries = {
        1: (1.18, 0.02, 0.03, 0.76),
        2: (0.10, -0.04, 0.04, 0.03),
        3: (-0.06, 0.03, 0.05, -0.02),
        4: (0.025, -0.015, -0.02, 0.018),
        5: (-0.018, 0.010, 0.012, -0.010),
    }
    for mode, values in entries.items():
        if mode <= modes:
            offset = 2 + 4 * (mode - 1)
            params[offset : offset + 4] = torch.tensor(values, dtype=dtype, device=device)
    return params


def stealth_polygon_vertices(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A seven-vertex double-concave stealth planform."""

    vertices = torch.tensor(
        [
            (1.30, 0.00),
            (0.42, 0.34),
            (-1.08, 0.62),
            (-0.38, 0.12),
            (-0.38, -0.12),
            (-1.08, -0.62),
            (0.42, -0.34),
        ],
        dtype=dtype,
        device=device,
    )
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if polygon_area_points(vertices) < 0.0:
        vertices = torch.flip(vertices, dims=(0,))
    return vertices


def polygon_area_points(points: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(points, shifts=-1, dims=0)
    cross = points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0]
    return 0.5 * cross.sum()


def polygon_sample_counts(vertices: torch.Tensor, n: int) -> list[int]:
    edge_lengths = torch.linalg.norm(torch.roll(vertices, shifts=-1, dims=0) - vertices, dim=1)
    lengths = [float(value.detach().cpu()) for value in edge_lengths]
    if n < 2 * len(lengths):
        raise ValueError("polygon case requires at least two samples per edge")
    total = sum(lengths)
    raw = [n * length / total for length in lengths]
    counts = [max(2, int(math.floor(value))) for value in raw]
    while sum(counts) < n:
        index = max(range(len(counts)), key=lambda idx: raw[idx] - counts[idx])
        counts[index] += 1
    while sum(counts) > n:
        index = max(range(len(counts)), key=lambda idx: counts[idx] - raw[idx])
        if counts[index] <= 2:
            break
        counts[index] -= 1
    if sum(counts) != n:
        raise ValueError("could not allocate polygon samples")
    return counts


def sample_polygon_boundary_with_counts(vertices: torch.Tensor, counts: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    samples = []
    corner_indices = []
    offset = 0
    for index, count in enumerate(counts):
        start = vertices[index]
        end = vertices[(index + 1) % vertices.shape[0]]
        corner_indices.append(offset)
        t = torch.arange(count, dtype=vertices.dtype, device=vertices.device) / count
        samples.append((1.0 - t).unsqueeze(1) * start + t.unsqueeze(1) * end)
        offset += count
    return torch.cat(samples, dim=0), tuple(corner_indices)


def sample_polygon_boundary(vertices: torch.Tensor, n: int) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    counts = tuple(polygon_sample_counts(vertices, n))
    points, corner_indices = sample_polygon_boundary_with_counts(vertices, counts)
    return points, corner_indices, counts


def make_target(config: DemoConfig, theta: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> TargetSpec:
    if config.case == "smooth_fourier":
        coefficients = target_params(config.modes, dtype=dtype, device=device)
        points = fourier_boundary(coefficients, theta, config.modes).detach()
        area = fourier_area(coefficients, config.modes).detach().abs()
        return TargetSpec(
            name=config.case,
            points=points,
            area=area,
            corner_indices=(),
            coefficients=coefficients.detach(),
        )
    if config.case in {"polygon", "polygon_corner_fixed"}:
        vertices = stealth_polygon_vertices(dtype=dtype, device=device)
        points, corner_indices, edge_counts = sample_polygon_boundary(vertices, config.n)
        area = polygon_area_points(vertices).detach().abs()
        return TargetSpec(
            name=config.case,
            points=points.detach(),
            area=area,
            corner_indices=corner_indices,
            edge_counts=edge_counts,
            vertices=vertices.detach(),
        )
    raise ValueError(f"unknown demo case: {config.case}")


def fourier_boundary(params: torch.Tensor, theta: torch.Tensor, modes: int) -> torch.Tensor:
    x = params[0] + torch.zeros_like(theta)
    y = params[1] + torch.zeros_like(theta)
    for mode in range(1, modes + 1):
        a, b, c, d = unpack_mode(params, mode)
        mt = mode * theta
        x = x + a * torch.cos(mt) + b * torch.sin(mt)
        y = y + c * torch.cos(mt) + d * torch.sin(mt)
    return torch.stack((x, y), dim=1)


def fourier_area(params: torch.Tensor, modes: int) -> torch.Tensor:
    area = torch.zeros((), dtype=params.dtype, device=params.device)
    for mode in range(1, modes + 1):
        a, b, c, d = unpack_mode(params, mode)
        area = area + mode * (a * d - b * c)
    return math.pi * area


def spectral_roughness(params: torch.Tensor, modes: int, power: float = 4.0) -> torch.Tensor:
    total = torch.zeros((), dtype=params.dtype, device=params.device)
    for mode in range(2, modes + 1):
        a, b, c, d = unpack_mode(params, mode)
        total = total + (mode**power) * (a * a + b * b + c * c + d * d)
    return total


def is_polygon_case(config: DemoConfig) -> bool:
    return config.case in {"polygon", "polygon_corner_fixed"}


def polygon_vertex_params(vertices: torch.Tensor) -> torch.Tensor:
    return vertices.reshape(-1)


def unpack_polygon_vertices(params: torch.Tensor) -> torch.Tensor:
    if params.numel() % 2 != 0:
        raise ValueError("polygon vertex parameter vector must have even length")
    return params.reshape((-1, 2))


def initial_polygon_params(vertices: torch.Tensor) -> torch.Tensor:
    index = torch.arange(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
    perturb = torch.stack(
        (
            0.10 * torch.sin(1.7 * index + 0.3),
            0.08 * torch.cos(2.1 * index + 0.6),
        ),
        dim=1,
    )
    scaled = vertices * torch.tensor((0.82, 0.88), dtype=vertices.dtype, device=vertices.device)
    return polygon_vertex_params(scaled + perturb)


def polygon_vertex_roughness(vertices: torch.Tensor) -> torch.Tensor:
    prev_vertices = torch.roll(vertices, shifts=1, dims=0)
    next_vertices = torch.roll(vertices, shifts=-1, dims=0)
    second = next_vertices - 2.0 * vertices + prev_vertices
    edge = torch.roll(vertices, shifts=-1, dims=0) - vertices
    edge_lengths = torch.linalg.norm(edge, dim=1)
    return torch.mean(torch.sum(second * second, dim=1)) + 0.02 * torch.var(edge_lengths)


def shape_boundary(
    params: torch.Tensor,
    theta: torch.Tensor,
    config: DemoConfig,
    edge_counts: tuple[int, ...],
) -> torch.Tensor:
    if is_polygon_case(config):
        vertices = unpack_polygon_vertices(params)
        points, _ = sample_polygon_boundary_with_counts(vertices, edge_counts)
        return points
    return fourier_boundary(params, theta, config.modes)


def shape_area(params: torch.Tensor, config: DemoConfig) -> torch.Tensor:
    if is_polygon_case(config):
        return polygon_area_points(unpack_polygon_vertices(params)).abs()
    return fourier_area(params, config.modes)


def shape_roughness(params: torch.Tensor, config: DemoConfig) -> torch.Tensor:
    if is_polygon_case(config):
        return polygon_vertex_roughness(unpack_polygon_vertices(params))
    return spectral_roughness(params, config.modes)


def pair_indices(n: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(n, n, offset=1, device=device)


def perimeter(points: torch.Tensor) -> torch.Tensor:
    edges = torch.roll(points, shifts=-1, dims=0) - points
    return torch.linalg.norm(edges, dim=1).sum()


def boundary_moments(points: torch.Tensor, order: int) -> torch.Tensor:
    """Weighted boundary moments used to pin the Q spectrum's null directions."""

    next_points = torch.roll(points, shifts=-1, dims=0)
    edges = next_points - points
    lengths = torch.linalg.norm(edges, dim=1)
    weights = lengths / lengths.sum().clamp_min(torch.finfo(points.dtype).eps)
    mids = 0.5 * (points + next_points)
    x = mids[:, 0]
    y = mids[:, 1]
    moments = []
    for degree in range(1, order + 1):
        for x_power in range(degree, -1, -1):
            y_power = degree - x_power
            moments.append((weights * (x**x_power) * (y**y_power)).sum())
    return torch.stack(moments)


def probe_matrix(theta: torch.Tensor, probe_modes: int) -> torch.Tensor:
    probes = []
    for mode in range(1, probe_modes + 1):
        probes.append(torch.cos(mode * theta))
        probes.append(torch.sin(mode * theta))
    return torch.stack(probes, dim=1)


def reduced_q_gram(
    points: torch.Tensor,
    probes: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return the small probe-space Q Gram matrix without a dense Q matrix."""

    deltas = points[pair_i] - points[pair_j]
    dist2 = (deltas * deltas).sum(dim=1) + epsilon * epsilon
    probe_deltas = probes[pair_i] - probes[pair_j]
    weighted = probe_deltas / dist2.sqrt().unsqueeze(1)
    scale = perimeter(points) / (math.pi * points.shape[0])
    return scale * (weighted.T @ weighted)


def normalized_gram(gram: torch.Tensor) -> torch.Tensor:
    trace = torch.trace(gram).clamp_min(torch.finfo(gram.dtype).eps)
    return gram / trace


def corner_interior_angles(points: torch.Tensor, corner_indices: tuple[int, ...]) -> torch.Tensor:
    if not corner_indices:
        return torch.empty(0, dtype=points.dtype, device=points.device)
    angles = []
    n = points.shape[0]
    for index in corner_indices:
        prev_point = points[(index - 1) % n]
        point = points[index]
        next_point = points[(index + 1) % n]
        incoming = point - prev_point
        outgoing = next_point - point
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        dot = torch.sum(incoming * outgoing)
        turn = torch.atan2(cross, dot)
        angle = math.pi - turn
        angles.append(torch.clamp(angle, min=0.08, max=TAU - 0.08))
    return torch.stack(angles)


def corner_features(points: torch.Tensor, corner_indices: tuple[int, ...]) -> torch.Tensor:
    angles = corner_interior_angles(points, corner_indices)
    if angles.numel() == 0:
        return angles
    lambdas = math.pi / angles
    defects = (angles - math.pi) / math.pi
    return torch.cat((defects, lambdas))


def regularized_hurwitz_zeta(s: torch.Tensor, beta: float, terms: int) -> torch.Tensor:
    """Euler-Maclaurin regularized Hurwitz zeta for the corner channel."""

    count = max(8, int(terms))
    beta_tensor = torch.tensor(beta, dtype=s.dtype, device=s.device)
    k = torch.arange(count, dtype=s.dtype, device=s.device)
    q = k + beta_tensor
    partial = torch.sum(q.pow(-s))
    tail = torch.tensor(float(count), dtype=s.dtype, device=s.device) + beta_tensor
    return partial + tail.pow(1.0 - s) / (s - 1.0) + 0.5 * tail.pow(-s) + (s / 12.0) * tail.pow(-s - 1.0)


def corner_repayment_gram(
    points: torch.Tensor,
    probes: torch.Tensor,
    corner_indices: tuple[int, ...],
    *,
    terms: int,
) -> torch.Tensor:
    """Low-rank Hurwitz/Mellin corner channel added without a dense Q matrix."""

    if not corner_indices:
        return torch.zeros((probes.shape[1], probes.shape[1]), dtype=points.dtype, device=points.device)
    n = points.shape[0]
    grid = torch.arange(n, dtype=points.dtype, device=points.device)
    angles = corner_interior_angles(points, corner_indices)
    gram = torch.zeros((probes.shape[1], probes.shape[1]), dtype=points.dtype, device=points.device)
    h = torch.tensor(1.0 / n, dtype=points.dtype, device=points.device)
    for slot, corner_index in enumerate(corner_indices):
        angle = angles[slot]
        lam = math.pi / angle
        s = 1.0 - lam
        defect = (angle - math.pi) / math.pi
        offset = torch.abs(grid - float(corner_index))
        cyclic_offset = torch.minimum(offset, torch.tensor(float(n), dtype=points.dtype, device=points.device) - offset)
        radius = (cyclic_offset + 0.5) * h
        profile = radius.pow(lam - 1.0)
        profile = profile - profile.mean()
        profile = profile / torch.sqrt(torch.mean(profile * profile)).clamp_min(1.0e-12)
        centered_probes = probes - probes[corner_index].unsqueeze(0)
        response = torch.sum(profile.unsqueeze(1) * centered_probes, dim=0) / n
        zeta_weight = torch.abs(regularized_hurwitz_zeta(s, beta=0.5, terms=terms))
        amplitude = torch.abs(defect) * zeta_weight * h.pow(lam)
        gram = gram + amplitude * torch.outer(response, response)
    return gram


def corrected_q_gram(
    points: torch.Tensor,
    probes: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    corner_indices: tuple[int, ...],
    config: DemoConfig,
) -> torch.Tensor:
    gram = reduced_q_gram(points, probes, pair_i, pair_j, epsilon=config.q_epsilon)
    if config.corner_fix and corner_indices:
        gram = gram + config.corner_q_weight * corner_repayment_gram(
            points,
            probes,
            corner_indices,
            terms=config.corner_zeta_terms,
        )
    return gram


def objective(
    params: torch.Tensor,
    target_gram: torch.Tensor,
    target_area: torch.Tensor,
    target_moments: torch.Tensor,
    target_corner_features: torch.Tensor,
    corner_indices: tuple[int, ...],
    edge_counts: tuple[int, ...],
    theta: torch.Tensor,
    probes: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    config: DemoConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = shape_boundary(params, theta, config, edge_counts)
    gram = corrected_q_gram(points, probes, pair_i, pair_j, corner_indices, config)
    gram_loss = torch.mean((normalized_gram(gram) - normalized_gram(target_gram)) ** 2)
    gram_trace = torch.trace(gram)
    target_trace = torch.trace(target_gram)
    trace_loss = torch.log(gram_trace.clamp_min(1.0e-30) / target_trace.clamp_min(1.0e-30)).square()
    moments = boundary_moments(points, config.moment_order)
    moment_loss = torch.mean((moments - target_moments) ** 2) / torch.mean(target_moments**2).clamp_min(1.0e-12)
    if config.corner_fix and corner_indices:
        corners = corner_features(points, corner_indices)
        corner_loss = torch.mean((corners - target_corner_features) ** 2) / torch.mean(
            target_corner_features**2
        ).clamp_min(1.0e-12)
    else:
        corner_loss = torch.zeros((), dtype=params.dtype, device=params.device)
    area = shape_area(params, config)
    area_loss = ((area - target_area) / target_area.abs().clamp_min(1.0e-12)) ** 2
    centroid_loss = torch.mean(points, dim=0).square().sum()
    roughness = shape_roughness(params, config)
    loss = (
        gram_loss
        + config.q_trace_weight * trace_loss
        + config.moment_weight * moment_loss
        + config.corner_weight * corner_loss
        + config.area_weight * area_loss
        + config.centroid_weight * centroid_loss
        + config.roughness_weight * roughness
    )
    return loss, {
        "gram_loss": gram_loss,
        "trace_loss": trace_loss,
        "moment_loss": moment_loss,
        "corner_loss": corner_loss,
        "area_loss": area_loss,
        "centroid_loss": centroid_loss,
        "roughness": roughness,
        "area": area,
        "gram_trace": gram_trace,
    }


def relative_boundary_rms(points: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sqrt(torch.mean(torch.sum((points - target) ** 2, dim=1)))
    denominator = torch.sqrt(torch.mean(torch.sum(target**2, dim=1))).clamp_min(1.0e-12)
    return numerator / denominator


def nearest_squared_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    best = torch.full(
        (source.shape[0],),
        float("inf"),
        dtype=source.dtype,
        device=source.device,
    )
    for target_point in target:
        dist2 = torch.sum((source - target_point) ** 2, dim=1)
        best = torch.minimum(best, dist2)
    return best


def relative_symmetric_chamfer(points: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = nearest_squared_distances(points, target)
    backward = nearest_squared_distances(target, points)
    rms = torch.sqrt(0.5 * (forward.mean() + backward.mean()))
    scale = torch.sqrt(torch.mean(torch.sum(target**2, dim=1))).clamp_min(1.0e-12)
    return rms / scale


def relative_hausdorff(points: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = nearest_squared_distances(points, target)
    backward = nearest_squared_distances(target, points)
    distance = torch.sqrt(torch.maximum(forward.max(), backward.max()))
    scale = torch.sqrt(torch.mean(torch.sum(target**2, dim=1))).clamp_min(1.0e-12)
    return distance / scale


def boundary_panel_weights(points: torch.Tensor) -> torch.Tensor:
    previous_points = torch.roll(points, shifts=1, dims=0)
    next_points = torch.roll(points, shifts=-1, dims=0)
    left = torch.linalg.norm(points - previous_points, dim=1)
    right = torch.linalg.norm(next_points - points, dim=1)
    return 0.5 * (left + right)


def single_layer_apply(points: torch.Tensor, weights: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """Apply the log single-layer operator without materializing its matrix."""

    out = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    panel = weights.clamp_min(1.0e-12)
    self_log = torch.log(0.5 * panel) - 1.0
    for source_index in range(points.shape[0]):
        delta = points - points[source_index]
        radius2 = torch.sum(delta * delta, dim=1).clamp_min(1.0e-30)
        kernel = 0.5 * torch.log(radius2)
        kernel = kernel.clone()
        kernel[source_index] = self_log[source_index]
        out = out + weights[source_index] * density[source_index] * kernel
    return out


def conjugate_gradient_matrix_free(
    apply_operator,
    rhs: torch.Tensor,
    *,
    iterations: int,
    tolerance: float = 1.0e-10,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    x = torch.zeros_like(rhs)
    residual = rhs - apply_operator(x)
    direction = residual.clone()
    residual_energy = torch.sum(residual * residual)
    rhs_norm = torch.sqrt(torch.sum(rhs * rhs)).clamp_min(1.0e-14)
    used = 0
    for used in range(1, max(1, iterations) + 1):
        applied = apply_operator(direction)
        denom = torch.sum(direction * applied).clamp_min(1.0e-30)
        alpha = residual_energy / denom
        x = x + alpha * direction
        residual = residual - alpha * applied
        next_energy = torch.sum(residual * residual)
        if torch.sqrt(next_energy) / rhs_norm <= tolerance:
            residual_energy = next_energy
            break
        beta = next_energy / residual_energy.clamp_min(1.0e-30)
        direction = residual + beta * direction
        residual_energy = next_energy
    return x, used, torch.sqrt(residual_energy) / rhs_norm


def solve_grounded_uniform_field_density(
    points: torch.Tensor,
    *,
    iterations: int,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Solve S sigma = -x for a grounded conductor in a uniform field."""

    weights = boundary_panel_weights(points)
    rhs = -(points[:, 0] - torch.mean(points[:, 0]))

    def apply_single(values: torch.Tensor) -> torch.Tensor:
        return single_layer_apply(points, weights, values)

    normal_rhs = apply_single(rhs)
    ridge_tensor = torch.tensor(ridge, dtype=points.dtype, device=points.device)

    def apply_normal(values: torch.Tensor) -> torch.Tensor:
        return apply_single(apply_single(values)) + ridge_tensor * values

    density, used, residual = conjugate_gradient_matrix_free(
        apply_normal,
        normal_rhs,
        iterations=iterations,
    )
    return density, {
        "field_model": "matrix_free_laplace_single_layer_grounded_uniform_field",
        "field_boundary_condition": "phi_total = 0 on sampled polygon boundary",
        "field_solve_iterations": used,
        "field_relative_residual": float(residual.detach().cpu()),
        "field_dense_matrix_stored": False,
    }


def points_inside_polygon(query: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    x = query[:, 0]
    y = query[:, 1]
    inside = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
    vertices = polygon
    count = vertices.shape[0]
    for index in range(count):
        start = vertices[index]
        end = vertices[(index + 1) % count]
        yi = start[1]
        yj = end[1]
        crosses = (yi > y) != (yj > y)
        x_at_y = (end[0] - start[0]) * (y - yi) / (yj - yi + 1.0e-30) + start[0]
        inside = torch.logical_xor(inside, crosses & (x < x_at_y))
    return inside


def numerical_field_heatmap(points: torch.Tensor, config: DemoConfig) -> tuple[dict[str, object], dict[str, object]]:
    density, stats = solve_grounded_uniform_field_density(
        points,
        iterations=config.field_solve_iterations,
        ridge=config.field_ridge,
    )
    weights = boundary_panel_weights(points)
    minimum = torch.min(points, dim=0).values
    maximum = torch.max(points, dim=0).values
    span = (maximum - minimum).clamp_min(1.0e-6)
    pad = torch.tensor((0.22, 0.26), dtype=points.dtype, device=points.device) * torch.max(span)
    xs = torch.linspace(minimum[0] - pad[0], maximum[0] + pad[0], config.field_grid_x, dtype=points.dtype, device=points.device)
    ys = torch.linspace(minimum[1] - pad[1], maximum[1] + pad[1], config.field_grid_y, dtype=points.dtype, device=points.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    query = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    potential = query[:, 0] - torch.mean(points[:, 0])
    grad_x = torch.ones(query.shape[0], dtype=points.dtype, device=points.device)
    grad_y = torch.zeros(query.shape[0], dtype=points.dtype, device=points.device)
    for source_index in range(points.shape[0]):
        delta = query - points[source_index]
        radius2 = torch.sum(delta * delta, dim=1).clamp_min(1.0e-10)
        charge = weights[source_index] * density[source_index]
        potential = potential + charge * 0.5 * torch.log(radius2)
        grad_x = grad_x + charge * delta[:, 0] / radius2
        grad_y = grad_y + charge * delta[:, 1] / radius2
    strength = torch.sqrt(grad_x * grad_x + grad_y * grad_y).reshape(config.field_grid_y, config.field_grid_x)
    inside = points_inside_polygon(query, points).reshape(config.field_grid_y, config.field_grid_x)
    stats.update(
        {
            "field_grid_x": config.field_grid_x,
            "field_grid_y": config.field_grid_y,
            "field_quantity": "|grad(x + S sigma)|",
            "field_density_l2": float(torch.sqrt(torch.mean(density * density)).detach().cpu()),
        }
    )
    arrays = {
        "field_x": xs.detach().cpu().tolist(),
        "field_y": ys.detach().cpu().tolist(),
        "field_strength": strength.detach().cpu().tolist(),
        "field_inside_mask": inside.detach().cpu().to(torch.int32).tolist(),
        "field_density": density.detach().cpu().tolist(),
    }
    return arrays, stats


def finite_difference_gradient_check(
    params: torch.Tensor,
    target_gram: torch.Tensor,
    target_area: torch.Tensor,
    target_moments: torch.Tensor,
    target_corner_features: torch.Tensor,
    corner_indices: tuple[int, ...],
    edge_counts: tuple[int, ...],
    theta: torch.Tensor,
    probes: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    config: DemoConfig,
) -> dict[str, float]:
    checked = 2 + 4 * min(2, config.modes)
    params_var = params.detach().clone().requires_grad_(True)
    loss, _ = objective(
        params_var,
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        corner_indices,
        edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )
    loss.backward()
    index = checked
    scalar_step = torch.tensor(1.0e-5, dtype=params.dtype, device=params.device)
    plus = params.detach().clone()
    minus = params.detach().clone()
    plus[index] = plus[index] + scalar_step
    minus[index] = minus[index] - scalar_step
    loss_plus, _ = objective(
        plus,
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        corner_indices,
        edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )
    loss_minus, _ = objective(
        minus,
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        corner_indices,
        edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )
    scalar_finite_diff = (loss_plus - loss_minus) / (2.0 * scalar_step)
    scalar_autograd = params_var.grad[index]
    scalar_rel = torch.abs(scalar_autograd - scalar_finite_diff) / torch.clamp(
        torch.abs(scalar_finite_diff),
        min=1.0e-12,
    )
    scalar_abs = torch.abs(scalar_autograd - scalar_finite_diff)

    direction_grid = torch.arange(params.numel(), dtype=params.dtype, device=params.device)
    direction = torch.sin(12.9898 * direction_grid + 78.233)
    direction = direction / torch.linalg.norm(direction).clamp_min(1.0e-12)
    directional_autograd = torch.sum(params_var.grad * direction)
    direction_step = torch.tensor(1.0e-6, dtype=params.dtype, device=params.device)
    loss_plus, _ = objective(
        params.detach() + direction_step * direction,
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        corner_indices,
        edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )
    loss_minus, _ = objective(
        params.detach() - direction_step * direction,
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        corner_indices,
        edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )
    directional_finite_diff = (loss_plus - loss_minus) / (2.0 * direction_step)
    directional_rel = torch.abs(directional_autograd - directional_finite_diff) / torch.clamp(
        torch.abs(directional_finite_diff),
        min=1.0e-12,
    )
    directional_abs = torch.abs(directional_autograd - directional_finite_diff)
    return {
        "coefficient_index": int(index),
        "scalar_autograd": float(scalar_autograd.detach().cpu()),
        "scalar_finite_difference": float(scalar_finite_diff.detach().cpu()),
        "scalar_absolute_error": float(scalar_abs.detach().cpu()),
        "scalar_relative_error": float(scalar_rel.detach().cpu()),
        "directional_autograd": float(directional_autograd.detach().cpu()),
        "directional_finite_difference": float(directional_finite_diff.detach().cpu()),
        "directional_absolute_error": float(directional_abs.detach().cpu()),
        "directional_relative_error": float(directional_rel.detach().cpu()),
        "relative_error": float(directional_rel.detach().cpu()),
    }


def optimize(config: DemoConfig, *, device: torch.device) -> dict[str, object]:
    dtype = dtype_from_name(config.dtype)
    theta = theta_grid(config.n, dtype=dtype, device=device)
    probes = probe_matrix(theta, config.probe_modes)
    pair_i, pair_j = pair_indices(config.n, device=device)

    target = make_target(config, theta, dtype=dtype, device=device)
    target_points = target.points
    target_area = target.area
    target_gram = corrected_q_gram(
        target_points,
        probes,
        pair_i,
        pair_j,
        target.corner_indices,
        config,
    ).detach()
    target_moments = boundary_moments(target_points, config.moment_order).detach()
    target_corner_features = corner_features(target_points, target.corner_indices).detach()

    if is_polygon_case(config):
        if target.vertices is None:
            raise ValueError("polygon target is missing vertices")
        initial = initial_polygon_params(target.vertices)
    else:
        initial = circle_params(
            config.modes,
            radius=float(torch.sqrt(target_area.abs() / math.pi).detach().cpu()),
            dtype=dtype,
            device=device,
        )
    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=config.adam_lr)
    history = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(
            params,
            target_gram,
            target_area,
            target_moments,
            target_corner_features,
            target.corner_indices,
            target.edge_counts,
            theta,
            probes,
            pair_i,
            pair_j,
            config,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=10.0)
        optimizer.step()
        if step % max(1, config.steps // 20) == 0 or step == config.steps - 1:
            with torch.no_grad():
                points = shape_boundary(params, theta, config, target.edge_counts)
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "gram_loss": float(metrics["gram_loss"].detach().cpu()),
                        "trace_loss": float(metrics["trace_loss"].detach().cpu()),
                        "moment_loss": float(metrics["moment_loss"].detach().cpu()),
                        "corner_loss": float(metrics["corner_loss"].detach().cpu()),
                        "area_relative_error": float(
                            ((metrics["area"] - target_area).abs() / target_area.abs()).detach().cpu()
                        ),
                        "boundary_relative_rms": float(relative_boundary_rms(points, target_points).detach().cpu()),
                        "symmetric_chamfer": float(relative_symmetric_chamfer(points, target_points).detach().cpu()),
                    }
                )

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            [params],
            max_iter=config.lbfgs_steps,
            line_search_fn="strong_wolfe",
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss_value, _ = objective(
                params,
                target_gram,
                target_area,
                target_moments,
                target_corner_features,
                target.corner_indices,
                target.edge_counts,
                theta,
                probes,
                pair_i,
                pair_j,
                config,
            )
            loss_value.backward()
            return loss_value

        lbfgs.step(closure)

    with torch.no_grad():
        final_loss, final_metrics = objective(
            params,
            target_gram,
            target_area,
            target_moments,
            target_corner_features,
            target.corner_indices,
            target.edge_counts,
            theta,
            probes,
            pair_i,
            pair_j,
            config,
        )
        final_points = shape_boundary(params, theta, config, target.edge_counts)
        initial_points = shape_boundary(initial, theta, config, target.edge_counts)
        final_gram = corrected_q_gram(final_points, probes, pair_i, pair_j, target.corner_indices, config)
        raw_final_gram = reduced_q_gram(final_points, probes, pair_i, pair_j, epsilon=config.q_epsilon)
        raw_target_gram = reduced_q_gram(target_points, probes, pair_i, pair_j, epsilon=config.q_epsilon)
        gram_relative_error = torch.linalg.norm(
            normalized_gram(final_gram) - normalized_gram(target_gram)
        ) / torch.linalg.norm(normalized_gram(target_gram)).clamp_min(1.0e-12)
        raw_gram_relative_error = torch.linalg.norm(
            normalized_gram(raw_final_gram) - normalized_gram(raw_target_gram)
        ) / torch.linalg.norm(normalized_gram(raw_target_gram)).clamp_min(1.0e-12)
        final_corner_features = corner_features(final_points, target.corner_indices)
        field_arrays, field_stats = numerical_field_heatmap(target_points, config)

    grad_check = finite_difference_gradient_check(
        params.detach(),
        target_gram,
        target_area,
        target_moments,
        target_corner_features,
        target.corner_indices,
        target.edge_counts,
        theta,
        probes,
        pair_i,
        pair_j,
        config,
    )

    return {
        "config": asdict(config),
        "pair_count": int(pair_i.numel()),
        "probe_count": int(probes.shape[1]),
        "dense_q_matrix_stored": False,
        "corner_fix_used": bool(config.corner_fix and target.corner_indices),
        "corner_channel_rank": len(target.corner_indices) if config.corner_fix else 0,
        "protocol": "meshless Fourier QJet -> pairwise Q energy -> autograd -> coefficient update",
        "field": field_stats,
        "history": history,
        "gradient_check": grad_check,
        "initial": {
            "area": float(shape_area(initial, config).detach().cpu()),
            "parameterization": "polygon_vertices" if is_polygon_case(config) else "fourier",
            "boundary_relative_rms": float(relative_boundary_rms(initial_points, target_points).detach().cpu()),
            "symmetric_chamfer": float(relative_symmetric_chamfer(initial_points, target_points).detach().cpu()),
            "hausdorff": float(relative_hausdorff(initial_points, target_points).detach().cpu()),
        },
        "target": {
            "case": target.name,
            "area": float(target_area.detach().cpu()),
            "q_gram_trace": float(torch.trace(target_gram).detach().cpu()),
            "corner_count": len(target.corner_indices),
            "corner_indices": list(target.corner_indices),
            "corner_features": target_corner_features.detach().cpu().tolist(),
        },
        "final": {
            "loss": float(final_loss.detach().cpu()),
            "gram_loss": float(final_metrics["gram_loss"].detach().cpu()),
            "trace_loss": float(final_metrics["trace_loss"].detach().cpu()),
            "moment_loss": float(final_metrics["moment_loss"].detach().cpu()),
            "corner_loss": float(final_metrics["corner_loss"].detach().cpu()),
            "q_gram_relative_error": float(gram_relative_error.detach().cpu()),
            "raw_q_gram_relative_error": float(raw_gram_relative_error.detach().cpu()),
            "area": float(final_metrics["area"].detach().cpu()),
            "area_relative_error": float(
                ((final_metrics["area"] - target_area).abs() / target_area.abs()).detach().cpu()
            ),
            "boundary_relative_rms": float(relative_boundary_rms(final_points, target_points).detach().cpu()),
            "symmetric_chamfer": float(relative_symmetric_chamfer(final_points, target_points).detach().cpu()),
            "hausdorff": float(relative_hausdorff(final_points, target_points).detach().cpu()),
            "roughness": float(final_metrics["roughness"].detach().cpu()),
            "corner_features": final_corner_features.detach().cpu().tolist(),
        },
        "arrays": {
            "theta": theta.detach().cpu().tolist(),
            "initial_points": initial_points.detach().cpu().tolist(),
            "target_points": target_points.detach().cpu().tolist(),
            "optimized_points": final_points.detach().cpu().tolist(),
            "initial_coefficients": initial.detach().cpu().tolist(),
            "target_coefficients": None if target.coefficients is None else target.coefficients.detach().cpu().tolist(),
            "optimized_coefficients": params.detach().cpu().tolist(),
            "optimized_vertices": None
            if not is_polygon_case(config)
            else unpack_polygon_vertices(params.detach()).cpu().tolist(),
            "target_vertices": None if target.vertices is None else target.vertices.detach().cpu().tolist(),
            **field_arrays,
        },
    }


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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    arrays = payload["arrays"]
    initial = torch.tensor(arrays["initial_points"])
    target = torch.tensor(arrays["target_points"])
    optimized = torch.tensor(arrays["optimized_points"])
    target_vertices = None if arrays["target_vertices"] is None else torch.tensor(arrays["target_vertices"])
    field_x = arrays.get("field_x")
    field_y = arrays.get("field_y")
    field_strength = arrays.get("field_strength")
    field_inside_mask = arrays.get("field_inside_mask")
    history = payload["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25))
    ax = axes[0]
    if field_x is not None and field_y is not None and field_strength is not None and field_inside_mask is not None:
        field = torch.log1p(torch.tensor(field_strength))
        mask = torch.tensor(field_inside_mask, dtype=torch.bool)
        outside = field[~mask]
        if outside.numel() > 0:
            lo = torch.quantile(outside, 0.02)
            hi = torch.quantile(outside, 0.98)
            field = torch.clamp(field, min=lo, max=hi)
            level_min = lo
            level_max = hi
        else:
            level_min = torch.min(field)
            level_max = torch.max(field)
        field[mask] = float("nan")
        image = ax.imshow(
            field.tolist(),
            extent=(field_x[0], field_x[-1], field_y[0], field_y[-1]),
            origin="lower",
            cmap="Greys",
            alpha=0.78,
            interpolation="bilinear",
            aspect="auto",
        )
        if float(level_max - level_min) > 1.0e-12:
            levels = torch.linspace(level_min, level_max, 9)
            ax.contour(
                field_x,
                field_y,
                field.tolist(),
                levels=[float(level) for level in levels[1:-1]],
                colors="0.18",
                linewidths=0.32,
                alpha=0.38,
            )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.033, pad=0.012)
        colorbar.set_label("log(1 + field)", labelpad=6)
        colorbar.outline.set_linewidth(0.45)
    for data, label, style, color, linewidth in (
        (initial, "initial", (0, (4, 2)), "0.62", 1.35),
        (target, "target", "-", "0.00", 1.65),
        (optimized, "optimized", (0, (1, 1)), "0.20", 1.65),
    ):
        closed = torch.cat((data, data[:1]), dim=0)
        ax.plot(closed[:, 0], closed[:, 1], linestyle=style, color=color, linewidth=linewidth, label=label)
    if target_vertices is not None:
        ax.scatter(target_vertices[:, 0], target_vertices[:, 1], s=12, marker="o", color="0.05", linewidths=0.0)
    ax.set_aspect("equal", adjustable="box")
    title = str(payload["target"]["case"]).replace("_", " ")
    ax.set_title(f"meshless Q shape optimization: {title}")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linewidth=0.45, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax = axes[1]
    ax.semilogy(
        [row["step"] for row in history],
        [row["loss"] for row in history],
        color="0.00",
        linestyle="-",
        linewidth=1.45,
        label="loss",
    )
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["gram_loss"], 1.0e-18) for row in history],
        color="0.30",
        linestyle=(0, (4, 2)),
        linewidth=1.35,
        label="Q Gram loss",
    )
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["moment_loss"], 1.0e-18) for row in history],
        color="0.50",
        linestyle=(0, (1, 1)),
        linewidth=1.35,
        label="moment loss",
    )
    if payload["corner_fix_used"]:
        ax.semilogy(
            [row["step"] for row in history],
            [max(row["corner_loss"], 1.0e-18) for row in history],
            color="0.60",
            linestyle=(0, (2, 1, 1, 1)),
            linewidth=1.25,
            label="corner loss",
        )
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["symmetric_chamfer"], 1.0e-18) for row in history],
        color="0.72",
        linestyle=(0, (6, 2, 1, 2)),
        linewidth=1.2,
        label="symmetric Chamfer",
    )
    ax.set_xlabel("Adam step")
    ax.set_title("autograd optimization trace")
    ax.grid(True, linewidth=0.45, alpha=0.9)
    ax.legend(loc="best", frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/q_autograd_shape_optimization")
    parser.add_argument(
        "--case",
        choices=("smooth_fourier", "polygon", "polygon_corner_fixed"),
        default=DemoConfig.case,
    )
    parser.add_argument("--n", type=int, default=DemoConfig.n)
    parser.add_argument("--modes", type=int, default=DemoConfig.modes)
    parser.add_argument("--probe-modes", type=int, default=DemoConfig.probe_modes)
    parser.add_argument("--steps", type=int, default=DemoConfig.steps)
    parser.add_argument("--adam-lr", type=float, default=DemoConfig.adam_lr)
    parser.add_argument("--lbfgs-steps", type=int, default=DemoConfig.lbfgs_steps)
    parser.add_argument("--q-trace-weight", type=float, default=DemoConfig.q_trace_weight)
    parser.add_argument("--moment-order", type=int, default=DemoConfig.moment_order)
    parser.add_argument("--moment-weight", type=float, default=DemoConfig.moment_weight)
    parser.add_argument("--corner-fix", action="store_true")
    parser.add_argument("--corner-weight", type=float, default=DemoConfig.corner_weight)
    parser.add_argument("--corner-q-weight", type=float, default=DemoConfig.corner_q_weight)
    parser.add_argument("--corner-zeta-terms", type=int, default=DemoConfig.corner_zeta_terms)
    parser.add_argument("--area-weight", type=float, default=DemoConfig.area_weight)
    parser.add_argument("--centroid-weight", type=float, default=DemoConfig.centroid_weight)
    parser.add_argument("--roughness-weight", type=float, default=DemoConfig.roughness_weight)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=DemoConfig.dtype)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DemoConfig(
        n=args.n,
        modes=args.modes,
        probe_modes=args.probe_modes,
        case=args.case,
        steps=args.steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
        q_trace_weight=args.q_trace_weight,
        moment_order=args.moment_order,
        moment_weight=args.moment_weight,
        corner_fix=args.corner_fix or args.case == "polygon_corner_fixed",
        corner_weight=args.corner_weight,
        corner_q_weight=args.corner_q_weight,
        corner_zeta_terms=args.corner_zeta_terms,
        area_weight=args.area_weight,
        centroid_weight=args.centroid_weight,
        roughness_weight=args.roughness_weight,
        dtype=args.dtype,
    )
    device = torch.device("cpu")
    payload = optimize(config, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "q_autograd_meshless_shape_optimization"
        if config.case == "smooth_fourier"
        else f"q_autograd_meshless_shape_optimization_{config.case}"
    )
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_path = out_dir / f"{stem}.png"
    if not args.no_plot:
        write_plot(payload, plot_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "png": None if args.no_plot else str(plot_path),
                "case": payload["target"]["case"],
                "dense_q_matrix_stored": payload["dense_q_matrix_stored"],
                "corner_fix_used": payload["corner_fix_used"],
                "corner_channel_rank": payload["corner_channel_rank"],
                "pair_count": payload["pair_count"],
                "probe_count": payload["probe_count"],
                "gradient_check_relative_error": payload["gradient_check"]["relative_error"],
                "gradient_check_absolute_error": payload["gradient_check"]["directional_absolute_error"],
                "initial_boundary_relative_rms": payload["initial"]["boundary_relative_rms"],
                "initial_symmetric_chamfer": payload["initial"]["symmetric_chamfer"],
                "final_boundary_relative_rms": payload["final"]["boundary_relative_rms"],
                "final_symmetric_chamfer": payload["final"]["symmetric_chamfer"],
                "final_hausdorff": payload["final"]["hausdorff"],
                "final_q_gram_relative_error": payload["final"]["q_gram_relative_error"],
                "final_raw_q_gram_relative_error": payload["final"]["raw_q_gram_relative_error"],
                "final_q_trace_loss": payload["final"]["trace_loss"],
                "final_moment_loss": payload["final"]["moment_loss"],
                "final_corner_loss": payload["final"]["corner_loss"],
                "final_area_relative_error": payload["final"]["area_relative_error"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
