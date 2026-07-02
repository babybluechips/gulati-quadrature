#!/usr/bin/env python3
"""Synthetic inverse shape reconstruction from radar/RCS scattering.

This is a controlled radar inverse problem rather than a claim of full Maxwell
RCS inversion.  The forward map is a high-frequency scalar physical-optics
surrogate for coherent monostatic backscatter,

    A(k, phi) = int_{illuminated boundary}
        max(0, -n dot d_phi) exp(2 i k d_phi dot x) ds,
    RCS(k, phi) = |A(k, phi)|^2.

The demo generates multi-frequency coherent scattering observations from a
hidden symmetric cranked-delta aircraft planform, displays the derived log-RCS,
then reconstructs the fourteen planform parameters by PyTorch autograd through the
matrix-free scattering map.  Set ``--coherent-weight 0`` to run the much more
ambiguous magnitude-only RCS ablation.
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
class RadarConfig:
    samples_per_edge: int = 36
    angle_count: int = 120
    frequencies: tuple[float, ...] = (6.0, 8.5, 11.0, 13.5)
    steps: int = 900
    adam_lr: float = 1.8e-2
    lbfgs_steps: int = 50
    rcs_floor: float = 1.0e-10
    coherent_weight: float = 1.0
    noise_level: float = 0.0
    order_weight: float = 0.4
    area_barrier_weight: float = 0.02
    dtype: str = "float64"


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def positive(value: torch.Tensor, minimum: float = 1.0e-4) -> torch.Tensor:
    return torch.sqrt(value * value + minimum * minimum)


def symmetric_aircraft_vertices(params: torch.Tensor) -> torch.Tensor:
    """Return a mirrored cranked-delta aircraft planform with a tail notch."""

    (
        nose_x,
        body_x,
        body_y,
        kink_x,
        kink_y,
        tip_x,
        tip_y,
        trailing_x,
        trailing_y,
        tailplane_x,
        tailplane_y,
        exhaust_x,
        exhaust_y,
        tail_x,
    ) = params
    body_y = positive(body_y)
    kink_y = positive(kink_y)
    tip_y = positive(tip_y)
    trailing_y = positive(trailing_y)
    tailplane_y = positive(tailplane_y)
    exhaust_y = positive(exhaust_y)
    vertices = torch.stack(
        (
            torch.stack((nose_x, torch.zeros_like(nose_x))),
            torch.stack((body_x, body_y)),
            torch.stack((kink_x, kink_y)),
            torch.stack((tip_x, tip_y)),
            torch.stack((trailing_x, trailing_y)),
            torch.stack((tailplane_x, tailplane_y)),
            torch.stack((exhaust_x, exhaust_y)),
            torch.stack((tail_x, torch.zeros_like(tail_x))),
            torch.stack((exhaust_x, -exhaust_y)),
            torch.stack((tailplane_x, -tailplane_y)),
            torch.stack((trailing_x, -trailing_y)),
            torch.stack((tip_x, -tip_y)),
            torch.stack((kink_x, -kink_y)),
            torch.stack((body_x, -body_y)),
        )
    )
    return vertices - vertices.mean(dim=0, keepdim=True)


def target_params(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (
            1.48,
            0.92,
            0.13,
            0.35,
            0.40,
            -0.34,
            0.78,
            -0.88,
            0.34,
            -1.30,
            0.50,
            -1.13,
            0.16,
            -1.52,
        ),
        dtype=dtype,
        device=device,
    )


def initial_params(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (
            1.24,
            0.78,
            0.10,
            0.18,
            0.31,
            -0.18,
            0.61,
            -0.70,
            0.27,
            -1.08,
            0.40,
            -0.94,
            0.11,
            -1.30,
        ),
        dtype=dtype,
        device=device,
    )


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(vertices, shifts=-1, dims=0)
    cross = vertices[:, 0] * shifted[:, 1] - vertices[:, 1] * shifted[:, 0]
    return 0.5 * cross.sum()


def sampled_boundary(
    vertices: torch.Tensor,
    samples_per_edge: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boundary quadrature points, outward normals, and panel weights."""

    if polygon_area(vertices) < 0.0:
        vertices = torch.flip(vertices, dims=(0,))
    points = []
    normals = []
    weights = []
    t = (
        torch.arange(samples_per_edge, dtype=vertices.dtype, device=vertices.device) + 0.5
    ) / samples_per_edge
    for index in range(vertices.shape[0]):
        start = vertices[index]
        end = vertices[(index + 1) % vertices.shape[0]]
        edge = end - start
        length = torch.linalg.norm(edge).clamp_min(1.0e-12)
        points.append((1.0 - t).unsqueeze(1) * start + t.unsqueeze(1) * end)
        normal = torch.stack((edge[1], -edge[0])) / length
        normals.append(normal.expand(samples_per_edge, 2))
        weights.append(
            torch.ones(samples_per_edge, dtype=vertices.dtype, device=vertices.device)
            * (length / samples_per_edge)
        )
    return torch.cat(points, dim=0), torch.cat(normals, dim=0), torch.cat(weights, dim=0)


def radar_directions(count: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta = torch.arange(count, dtype=dtype, device=device) * (TAU / count)
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)


def scattering_amplitude(
    params: torch.Tensor,
    config: RadarConfig,
    directions: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Compute multi-frequency coherent monostatic scattering amplitude."""

    vertices = symmetric_aircraft_vertices(params)
    points, normals, weights = sampled_boundary(vertices, config.samples_per_edge)
    illumination = torch.clamp(-(directions @ normals.T), min=0.0)
    projected = directions @ points.T
    rows = []
    for frequency in frequencies:
        phase = 2.0 * frequency * projected
        carrier = torch.complex(torch.cos(phase), torch.sin(phase))
        weighted = weights.unsqueeze(0) * illumination * carrier
        rows.append(torch.sum(weighted, dim=1))
    return torch.stack(rows, dim=0)


def rcs_log_from_amplitude(amplitude: torch.Tensor, floor: float) -> torch.Tensor:
    rcs = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    return torch.log10(rcs + floor)


def rcs_log_observations(
    params: torch.Tensor,
    config: RadarConfig,
    directions: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    """Compute multi-frequency monostatic log-RCS from polygon parameters."""

    return rcs_log_from_amplitude(scattering_amplitude(params, config, directions, frequencies), config.rcs_floor)


def observation_with_noise(observation: torch.Tensor, noise_level: float) -> torch.Tensor:
    if noise_level <= 0.0:
        return observation.detach()
    grid = torch.arange(observation.numel(), dtype=observation.dtype, device=observation.device)
    deterministic = torch.sin(12.9898 * grid + 78.233).reshape_as(observation)
    return (observation + noise_level * deterministic).detach()


def complex_observation_with_noise(observation: torch.Tensor, noise_level: float) -> torch.Tensor:
    if noise_level <= 0.0:
        return observation.detach()
    grid = torch.arange(observation.numel(), dtype=observation.real.dtype, device=observation.device)
    real_noise = torch.sin(12.9898 * grid + 78.233).reshape_as(observation.real)
    imag_noise = torch.cos(7.137 * grid + 19.19).reshape_as(observation.real)
    scale = torch.sqrt(torch.mean(observation.real * observation.real + observation.imag * observation.imag)).clamp_min(1.0e-12)
    return (observation + noise_level * scale * torch.complex(real_noise, imag_noise)).detach()


def realism_penalty(params: torch.Tensor) -> torch.Tensor:
    (
        nose_x,
        body_x,
        body_y,
        kink_x,
        kink_y,
        tip_x,
        tip_y,
        trailing_x,
        trailing_y,
        tailplane_x,
        tailplane_y,
        exhaust_x,
        exhaust_y,
        tail_x,
    ) = params
    body_y = positive(body_y)
    kink_y = positive(kink_y)
    tip_y = positive(tip_y)
    trailing_y = positive(trailing_y)
    tailplane_y = positive(tailplane_y)
    exhaust_y = positive(exhaust_y)
    penalties = (
        torch.relu(body_x - nose_x + 0.22).square()
        + torch.relu(kink_x - body_x + 0.18).square()
        + torch.relu(tip_x - kink_x + 0.22).square()
        + torch.relu(trailing_x - tip_x + 0.18).square()
        + torch.relu(tailplane_x - trailing_x + 0.12).square()
        + torch.relu(tailplane_x - exhaust_x + 0.05).square()
        + torch.relu(tail_x - tailplane_x + 0.08).square()
        + torch.relu(body_y - kink_y + 0.04).square()
        + torch.relu(kink_y - tip_y + 0.12).square()
        + torch.relu(trailing_y - tip_y + 0.12).square()
        + torch.relu(trailing_y - tailplane_y + 0.04).square()
        + torch.relu(exhaust_y - trailing_y + 0.04).square()
    )
    return penalties


def objective(
    params: torch.Tensor,
    observed_log_rcs: torch.Tensor,
    observed_amplitude: torch.Tensor,
    observation_scale: torch.Tensor,
    amplitude_scale: torch.Tensor,
    config: RadarConfig,
    directions: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_amplitude = scattering_amplitude(params, config, directions, frequencies)
    predicted = rcs_log_from_amplitude(predicted_amplitude, config.rcs_floor)
    residual = (predicted - observed_log_rcs) / observation_scale
    rcs_loss = torch.mean(residual * residual)
    complex_residual = (predicted_amplitude - observed_amplitude) / amplitude_scale
    coherent_loss = torch.mean(complex_residual.real * complex_residual.real + complex_residual.imag * complex_residual.imag)
    vertices = symmetric_aircraft_vertices(params)
    area = torch.abs(polygon_area(vertices))
    order_loss = realism_penalty(params)
    area_barrier = torch.relu(0.55 - area).square()
    loss = (
        rcs_loss
        + config.coherent_weight * coherent_loss
        + config.order_weight * order_loss
        + config.area_barrier_weight * area_barrier
    )
    return loss, {
        "rcs_loss": rcs_loss,
        "coherent_loss": coherent_loss,
        "order_loss": order_loss,
        "area_barrier": area_barrier,
        "area": area,
    }


def closed(points: torch.Tensor) -> torch.Tensor:
    return torch.cat((points, points[:1]), dim=0)


def vertex_rms(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = source - target
    scale = torch.sqrt(torch.mean(torch.sum(target * target, dim=1))).clamp_min(1.0e-12)
    return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=1))) / scale


def nearest_squared_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    best = torch.full((source.shape[0],), float("inf"), dtype=source.dtype, device=source.device)
    for point in target:
        dist2 = torch.sum((source - point) ** 2, dim=1)
        best = torch.minimum(best, dist2)
    return best


def symmetric_chamfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = nearest_squared_distances(source, target)
    backward = nearest_squared_distances(target, source)
    scale = torch.sqrt(torch.mean(torch.sum(target * target, dim=1))).clamp_min(1.0e-12)
    return torch.sqrt(0.5 * (forward.mean() + backward.mean())) / scale


def finite_difference_check(
    params: torch.Tensor,
    observed_log_rcs: torch.Tensor,
    observed_amplitude: torch.Tensor,
    observation_scale: torch.Tensor,
    amplitude_scale: torch.Tensor,
    config: RadarConfig,
    directions: torch.Tensor,
    frequencies: torch.Tensor,
) -> dict[str, float]:
    variable = params.detach().clone().requires_grad_(True)
    loss, _ = objective(
        variable,
        observed_log_rcs,
        observed_amplitude,
        observation_scale,
        amplitude_scale,
        config,
        directions,
        frequencies,
    )
    loss.backward()
    direction_grid = torch.arange(params.numel(), dtype=params.dtype, device=params.device)
    direction = torch.sin(4.7 * direction_grid + 0.31)
    direction = direction / torch.linalg.norm(direction).clamp_min(1.0e-12)
    autograd = torch.sum(variable.grad * direction)
    step = torch.tensor(1.0e-6, dtype=params.dtype, device=params.device)
    plus, _ = objective(
        params + step * direction,
        observed_log_rcs,
        observed_amplitude,
        observation_scale,
        amplitude_scale,
        config,
        directions,
        frequencies,
    )
    minus, _ = objective(
        params - step * direction,
        observed_log_rcs,
        observed_amplitude,
        observation_scale,
        amplitude_scale,
        config,
        directions,
        frequencies,
    )
    finite_diff = (plus - minus) / (2.0 * step)
    absolute = torch.abs(autograd - finite_diff)
    relative = absolute / torch.abs(finite_diff).clamp_min(1.0e-12)
    scaled = absolute / torch.maximum(
        torch.maximum(torch.abs(finite_diff), torch.abs(autograd)),
        torch.tensor(1.0e-8, dtype=params.dtype, device=params.device),
    )
    return {
        "directional_autograd": float(autograd.detach().cpu()),
        "directional_finite_difference": float(finite_diff.detach().cpu()),
        "directional_absolute_error": float(absolute.detach().cpu()),
        "directional_relative_error": float(relative.detach().cpu()),
        "directional_scaled_error": float(scaled.detach().cpu()),
        "note": "relative error is ill-conditioned when the final directional derivative is near zero",
    }


def reconstruct(config: RadarConfig, *, device: torch.device) -> dict[str, object]:
    dtype = dtype_from_name(config.dtype)
    directions = radar_directions(config.angle_count, dtype=dtype, device=device)
    frequencies = torch.tensor(config.frequencies, dtype=dtype, device=device)
    target = target_params(dtype=dtype, device=device)
    initial = initial_params(dtype=dtype, device=device)
    target_amplitude = scattering_amplitude(target, config, directions, frequencies).detach()
    target_log_rcs = rcs_log_from_amplitude(target_amplitude, config.rcs_floor).detach()
    observed_amplitude = complex_observation_with_noise(target_amplitude, config.noise_level)
    observed_log_rcs = observation_with_noise(target_log_rcs, config.noise_level)
    observation_scale = torch.std(observed_log_rcs).clamp_min(1.0e-6)
    amplitude_scale = torch.sqrt(
        torch.mean(observed_amplitude.real * observed_amplitude.real + observed_amplitude.imag * observed_amplitude.imag)
    ).clamp_min(1.0e-6)
    params = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=config.adam_lr)

    target_vertices = symmetric_aircraft_vertices(target).detach()
    target_samples, _, _ = sampled_boundary(target_vertices, config.samples_per_edge)
    history = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = objective(
            params,
            observed_log_rcs,
            observed_amplitude,
            observation_scale,
            amplitude_scale,
            config,
            directions,
            frequencies,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=5.0)
        optimizer.step()
        if step % max(1, config.steps // 24) == 0 or step == config.steps - 1:
            with torch.no_grad():
                vertices = symmetric_aircraft_vertices(params)
                samples, _, _ = sampled_boundary(vertices, config.samples_per_edge)
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "rcs_loss": float(metrics["rcs_loss"].detach().cpu()),
                        "coherent_loss": float(metrics["coherent_loss"].detach().cpu()),
                        "vertex_rms": float(vertex_rms(vertices, target_vertices).detach().cpu()),
                        "chamfer": float(symmetric_chamfer(samples, target_samples).detach().cpu()),
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
                observed_log_rcs,
                observed_amplitude,
                observation_scale,
                amplitude_scale,
                config,
                directions,
                frequencies,
            )
            loss_value.backward()
            return loss_value

        lbfgs.step(closure)

    with torch.no_grad():
        final_loss, final_metrics = objective(
            params,
            observed_log_rcs,
            observed_amplitude,
            observation_scale,
            amplitude_scale,
            config,
            directions,
            frequencies,
        )
        initial_vertices = symmetric_aircraft_vertices(initial)
        final_vertices = symmetric_aircraft_vertices(params)
        initial_samples, _, _ = sampled_boundary(initial_vertices, config.samples_per_edge)
        final_samples, _, _ = sampled_boundary(final_vertices, config.samples_per_edge)
        final_log_rcs = rcs_log_observations(params, config, directions, frequencies)
        initial_log_rcs = rcs_log_observations(initial, config, directions, frequencies)
        final_amplitude = scattering_amplitude(params, config, directions, frequencies)
        initial_amplitude = scattering_amplitude(initial, config, directions, frequencies)
        rcs_relative = torch.linalg.norm(final_log_rcs - observed_log_rcs) / torch.linalg.norm(
            observed_log_rcs
        ).clamp_min(1.0e-12)
        initial_rcs_relative = torch.linalg.norm(initial_log_rcs - observed_log_rcs) / torch.linalg.norm(
            observed_log_rcs
        ).clamp_min(1.0e-12)
        amplitude_relative = torch.linalg.norm(final_amplitude - observed_amplitude) / torch.linalg.norm(
            observed_amplitude
        ).clamp_min(1.0e-12)
        initial_amplitude_relative = torch.linalg.norm(initial_amplitude - observed_amplitude) / torch.linalg.norm(
            observed_amplitude
        ).clamp_min(1.0e-12)
        final_vertex_rms = vertex_rms(final_vertices, target_vertices)
        initial_vertex_rms = vertex_rms(initial_vertices, target_vertices)
        final_chamfer = symmetric_chamfer(final_samples, target_samples)
        initial_chamfer = symmetric_chamfer(initial_samples, target_samples)

    gradient_check = finite_difference_check(
        params.detach(),
        observed_log_rcs,
        observed_amplitude,
        observation_scale,
        amplitude_scale,
        config,
        directions,
        frequencies,
    )
    return {
        "config": asdict(config),
        "method": "synthetic_coherent_radar_scattering_autograd_inverse_shape_with_rcs_display",
        "aircraft_planform": "symmetric cranked-delta planform with fuselage shoulders, swept wing tips, tailplane points, and exhaust notch",
        "forward_model": "matrix_free_physical_optics_scalar_rcs",
        "inverse_data": "coherent monostatic scattering amplitude; derived log-RCS stored for display",
        "dense_scattering_matrix_stored": False,
        "unknown_parameter_count": int(params.numel()),
        "observation_count": int(observed_log_rcs.numel()),
        "gradient_check": gradient_check,
        "history": history,
        "metrics": {
            "initial_log_rcs_relative_error": float(initial_rcs_relative.detach().cpu()),
            "final_log_rcs_relative_error": float(rcs_relative.detach().cpu()),
            "initial_coherent_relative_error": float(initial_amplitude_relative.detach().cpu()),
            "final_coherent_relative_error": float(amplitude_relative.detach().cpu()),
            "initial_vertex_rms": float(initial_vertex_rms.detach().cpu()),
            "final_vertex_rms": float(final_vertex_rms.detach().cpu()),
            "initial_chamfer": float(initial_chamfer.detach().cpu()),
            "final_chamfer": float(final_chamfer.detach().cpu()),
            "final_loss": float(final_loss.detach().cpu()),
            "final_rcs_loss": float(final_metrics["rcs_loss"].detach().cpu()),
            "final_coherent_loss": float(final_metrics["coherent_loss"].detach().cpu()),
            "final_area": float(final_metrics["area"].detach().cpu()),
        },
        "arrays": {
            "angles": (torch.arange(config.angle_count, dtype=dtype) * (TAU / config.angle_count)).tolist(),
            "frequencies": frequencies.detach().cpu().tolist(),
            "target_vertices": target_vertices.detach().cpu().tolist(),
            "initial_vertices": initial_vertices.detach().cpu().tolist(),
            "reconstructed_vertices": final_vertices.detach().cpu().tolist(),
            "target_boundary_samples": target_samples.detach().cpu().tolist(),
            "initial_boundary_samples": initial_samples.detach().cpu().tolist(),
            "reconstructed_boundary_samples": final_samples.detach().cpu().tolist(),
            "observed_log_rcs": observed_log_rcs.detach().cpu().tolist(),
            "initial_log_rcs": initial_log_rcs.detach().cpu().tolist(),
            "reconstructed_log_rcs": final_log_rcs.detach().cpu().tolist(),
            "observed_amplitude_real": observed_amplitude.real.detach().cpu().tolist(),
            "observed_amplitude_imag": observed_amplitude.imag.detach().cpu().tolist(),
            "reconstructed_amplitude_real": final_amplitude.real.detach().cpu().tolist(),
            "reconstructed_amplitude_imag": final_amplitude.imag.detach().cpu().tolist(),
            "target_params": target.detach().cpu().tolist(),
            "initial_params": initial.detach().cpu().tolist(),
            "reconstructed_params": params.detach().cpu().tolist(),
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    arrays = payload["arrays"]
    target_vertices = torch.tensor(arrays["target_vertices"])
    initial_vertices = torch.tensor(arrays["initial_vertices"])
    recovered_vertices = torch.tensor(arrays["reconstructed_vertices"])
    observed = torch.tensor(arrays["observed_log_rcs"])
    recovered = torch.tensor(arrays["reconstructed_log_rcs"])
    initial = torch.tensor(arrays["initial_log_rcs"])
    angles = torch.tensor(arrays["angles"])
    frequencies = torch.tensor(arrays["frequencies"])
    history = payload["history"]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    for vertices, label, style, color, linewidth in (
        (initial_vertices, "initial", (0, (4, 2)), "0.64", 1.25),
        (target_vertices, "target", "-", "0.00", 1.9),
        (recovered_vertices, "reconstructed", (0, (1, 1)), "0.18", 1.8),
    ):
        curve = closed(vertices)
        ax.plot(curve[:, 0], curve[:, 1], linestyle=style, color=color, linewidth=linewidth, label=label)
        ax.scatter(vertices[:, 0], vertices[:, 1], color=color, s=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("cranked-delta aircraft reconstructed from radar scattering")
    ax.grid(True, linewidth=0.45, alpha=0.9)
    ax.legend(loc="best", frameon=False)

    ax = axes[0, 1]
    image = ax.imshow(
        observed.tolist(),
        extent=(0.0, 360.0, float(frequencies[0]), float(frequencies[-1])),
        origin="lower",
        cmap="Greys",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("observed log-RCS matrix")
    ax.set_xlabel("look angle (deg)")
    ax.set_ylabel("wavenumber")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("log10 RCS")
    colorbar.outline.set_linewidth(0.45)

    ax = axes[1, 0]
    middle = len(frequencies) // 2
    degrees = angles * (180.0 / math.pi)
    ax.plot(degrees, observed[middle], color="0.00", linewidth=1.5, label="observed")
    ax.plot(degrees, recovered[middle], color="0.25", linestyle=(0, (1, 1)), linewidth=1.35, label="reconstructed")
    ax.plot(degrees, initial[middle], color="0.68", linestyle=(0, (4, 2)), linewidth=1.15, label="initial")
    ax.set_title(f"RCS cut at k={float(frequencies[middle]):.1f}")
    ax.set_xlabel("look angle (deg)")
    ax.set_ylabel("log10 RCS")
    ax.grid(True, linewidth=0.45, alpha=0.9)
    ax.legend(loc="best", frameon=False)

    ax = axes[1, 1]
    ax.semilogy([row["step"] for row in history], [row["loss"] for row in history], color="0.0", label="loss")
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["rcs_loss"], 1.0e-18) for row in history],
        color="0.35",
        linestyle=(0, (4, 2)),
        label="RCS loss",
    )
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["coherent_loss"], 1.0e-18) for row in history],
        color="0.50",
        linestyle=(0, (6, 2, 1, 2)),
        label="coherent loss",
    )
    ax.semilogy(
        [row["step"] for row in history],
        [max(row["chamfer"], 1.0e-18) for row in history],
        color="0.62",
        linestyle=(0, (1, 1)),
        label="shape Chamfer",
    )
    ax.set_title("autograd inverse reconstruction")
    ax.set_xlabel("Adam step")
    ax.grid(True, linewidth=0.45, alpha=0.9)
    ax.legend(loc="best", frameon=False)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/radar_rcs_inverse_shape"))
    parser.add_argument("--samples-per-edge", type=int, default=RadarConfig.samples_per_edge)
    parser.add_argument("--angle-count", type=int, default=RadarConfig.angle_count)
    parser.add_argument("--frequencies", default=",".join(str(value) for value in RadarConfig.frequencies))
    parser.add_argument("--steps", type=int, default=RadarConfig.steps)
    parser.add_argument("--adam-lr", type=float, default=RadarConfig.adam_lr)
    parser.add_argument("--lbfgs-steps", type=int, default=RadarConfig.lbfgs_steps)
    parser.add_argument("--noise-level", type=float, default=RadarConfig.noise_level)
    parser.add_argument("--coherent-weight", type=float, default=RadarConfig.coherent_weight)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=RadarConfig.dtype)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frequencies = tuple(float(value) for value in args.frequencies.split(",") if value.strip())
    config = RadarConfig(
        samples_per_edge=args.samples_per_edge,
        angle_count=args.angle_count,
        frequencies=frequencies,
        steps=args.steps,
        adam_lr=args.adam_lr,
        lbfgs_steps=args.lbfgs_steps,
        noise_level=args.noise_level,
        coherent_weight=args.coherent_weight,
        dtype=args.dtype,
    )
    payload = reconstruct(config, device=torch.device("cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "radar_rcs_inverse_shape.json"
    png_path = args.out_dir / "radar_rcs_inverse_shape.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_plot(payload, png_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "png": None if args.no_plot else str(png_path),
                "dense_scattering_matrix_stored": payload["dense_scattering_matrix_stored"],
                "unknown_parameter_count": payload["unknown_parameter_count"],
                "observation_count": payload["observation_count"],
                "initial_log_rcs_relative_error": payload["metrics"]["initial_log_rcs_relative_error"],
                "final_log_rcs_relative_error": payload["metrics"]["final_log_rcs_relative_error"],
                "initial_coherent_relative_error": payload["metrics"]["initial_coherent_relative_error"],
                "final_coherent_relative_error": payload["metrics"]["final_coherent_relative_error"],
                "initial_chamfer": payload["metrics"]["initial_chamfer"],
                "final_chamfer": payload["metrics"]["final_chamfer"],
                "initial_vertex_rms": payload["metrics"]["initial_vertex_rms"],
                "final_vertex_rms": payload["metrics"]["final_vertex_rms"],
                "gradient_check_absolute_error": payload["gradient_check"]["directional_absolute_error"],
                "gradient_check_scaled_error": payload["gradient_check"]["directional_scaled_error"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
