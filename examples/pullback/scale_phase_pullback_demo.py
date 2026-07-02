#!/usr/bin/env python3
"""Visual scale-phase pullback demo for the Cauchy/Q calculus.

This is a didactic demo, not a production Riemann-map solver.  The curve is a
funky closed piecewise-cubic boundary.  The report separates the real production
stage, where the exterior map Phi is computed once per geometry, from the visual
collar used here to show the scale-phase mechanism.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter

import torch


ROOT = Path(__file__).resolve().parents[2]
TAU = 2.0 * math.pi


@dataclass(frozen=True)
class DemoConfig:
    n_boundary: int = 256
    dense_curve_samples: int = 4096
    modes: int = 48
    target_count: int = 100_000
    target_chunk: int = 4096
    dtype: str = "float64"


CONTROL_POINTS: tuple[tuple[float, float], ...] = (
    (1.62, 0.05),
    (1.08, 0.50),
    (0.58, 1.18),
    (-0.22, 0.88),
    (-0.92, 1.08),
    (-1.52, 0.32),
    (-1.07, -0.24),
    (-1.36, -0.96),
    (-0.35, -1.23),
    (0.18, -0.58),
    (0.93, -1.02),
    (1.37, -0.35),
)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def polygon_area(points: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(points, shifts=-1, dims=0)
    return 0.5 * torch.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])


def catmull_rom_closed(control: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
    pieces = []
    count = control.shape[0]
    t = (
        torch.arange(samples_per_segment, dtype=control.dtype, device=control.device)
        / samples_per_segment
    )
    t2 = t * t
    t3 = t2 * t
    for index in range(count):
        p0 = control[(index - 1) % count]
        p1 = control[index]
        p2 = control[(index + 1) % count]
        p3 = control[(index + 2) % count]
        segment = 0.5 * (
            2.0 * p1
            + (-p0 + p2).unsqueeze(0) * t.unsqueeze(1)
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3).unsqueeze(0) * t2.unsqueeze(1)
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3).unsqueeze(0) * t3.unsqueeze(1)
        )
        pieces.append(segment)
    points = torch.cat(pieces, dim=0)
    if polygon_area(points) < 0.0:
        points = torch.flip(points, dims=(0,))
    return points - torch.mean(points, dim=0, keepdim=True)


def arclength_resample(points: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    closed = torch.cat((points, points[:1]), dim=0)
    edges = closed[1:] - closed[:-1]
    lengths = torch.linalg.norm(edges, dim=1)
    cumulative = torch.cat((torch.zeros(1, dtype=points.dtype, device=points.device), torch.cumsum(lengths, dim=0)))
    total = float(cumulative[-1].detach().cpu())
    targets = torch.arange(n, dtype=points.dtype, device=points.device) * (total / n)
    out = []
    edge_index = 0
    for target in targets:
        while edge_index + 1 < len(cumulative) - 1 and target > cumulative[edge_index + 1]:
            edge_index += 1
        start_s = cumulative[edge_index]
        length = lengths[edge_index].clamp_min(torch.finfo(points.dtype).eps)
        local = (target - start_s) / length
        out.append(closed[edge_index] + local * edges[edge_index])
    samples = torch.stack(out)
    theta = torch.arange(n, dtype=points.dtype, device=points.device) * (TAU / n)
    return samples, theta, total


def tangents_and_normals(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = torch.roll(points, shifts=-1, dims=0) - torch.roll(points, shifts=1, dims=0)
    tangent = tangent / torch.linalg.norm(tangent, dim=1).clamp_min(1.0e-14).unsqueeze(1)
    normal = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=1)
    if polygon_area(points) < 0.0:
        normal = -normal
    return tangent, normal


def density(theta: torch.Tensor) -> torch.Tensor:
    return (
        0.72 * torch.cos(2.0 * theta - 0.4)
        - 0.38 * torch.sin(5.0 * theta + 0.2)
        + 0.18 * torch.cos(11.0 * theta - 0.9)
    )


def modal_coefficients(values: torch.Tensor, modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    coeffs = torch.fft.fft(values.to(torch.complex128)) / values.shape[0]
    k_values = torch.arange(-modes, modes + 1, dtype=torch.int64, device=values.device)
    selected = torch.empty(2 * modes + 1, dtype=torch.complex128, device=values.device)
    n = values.shape[0]
    for slot, mode in enumerate(k_values.tolist()):
        selected[slot] = coeffs[mode % n]
    return k_values, selected


def evaluate_modal_targets(
    k_values: torch.Tensor,
    coeffs: torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
    *,
    chunk: int,
) -> torch.Tensor:
    out = torch.empty(theta.shape[0], dtype=torch.complex128, device=theta.device)
    k_float = k_values.to(torch.float64)
    nonzero = k_values != 0
    modal_weights = torch.zeros_like(coeffs)
    modal_weights[nonzero] = -math.pi * coeffs[nonzero] / torch.abs(k_float[nonzero])
    sigma0 = coeffs[k_values == 0][0] if torch.any(k_values == 0) else torch.tensor(0.0, dtype=torch.complex128)
    for start in range(0, theta.shape[0], chunk):
        stop = min(start + chunk, theta.shape[0])
        rho_block = rho[start:stop].unsqueeze(1)
        theta_block = theta[start:stop].unsqueeze(1)
        damping = torch.exp(-torch.abs(k_float).unsqueeze(0) * rho_block)
        phase = torch.exp(1j * k_float.unsqueeze(0) * theta_block)
        out[start:stop] = torch.sum(modal_weights.unsqueeze(0) * damping * phase, dim=1) + 2.0 * math.pi * sigma0 * rho[start:stop]
    return out


def benchmark_costs(config: DemoConfig, theta: torch.Tensor, values: torch.Tensor) -> dict[str, float | int | str]:
    start = perf_counter()
    k_values, coeffs = modal_coefficients(values, config.modes)
    fft_ms = 1000.0 * (perf_counter() - start)

    target_theta = torch.arange(config.target_count, dtype=theta.dtype, device=theta.device)
    target_theta = TAU * ((0.61803398875 * target_theta) % 1.0)
    target_rho = 0.015 + 0.85 * ((0.754877666 * torch.arange(config.target_count, dtype=theta.dtype, device=theta.device)) % 1.0)
    start = perf_counter()
    result = evaluate_modal_targets(k_values, coeffs, target_rho, target_theta, chunk=config.target_chunk)
    eval_ms = 1000.0 * (perf_counter() - start)
    return {
        "density_fft_ms": fft_ms,
        "target_count": config.target_count,
        "modal_eval_ms_for_targets": eval_ms,
        "modal_eval_us_per_target": 1000.0 * eval_ms / config.target_count,
        "modes_each_side": config.modes,
        "checksum_real": float(torch.mean(result.real).detach().cpu()),
        "checksum_imag": float(torch.mean(result.imag).detach().cpu()),
        "note": "timing covers only fixed-basis scale/phase modal target evaluation, not Riemann-map construction",
    }


def build_demo(config: DemoConfig) -> dict[str, object]:
    dtype = dtype_from_name(config.dtype)
    control = torch.tensor(CONTROL_POINTS, dtype=dtype)
    curve = catmull_rom_closed(control, max(8, config.dense_curve_samples // len(CONTROL_POINTS)))
    start = perf_counter()
    boundary, theta, perimeter = arclength_resample(curve, config.n_boundary)
    resample_ms = 1000.0 * (perf_counter() - start)
    tangent, normal = tangents_and_normals(boundary)
    sigma = density(theta)
    k_values, coeffs = modal_coefficients(sigma, config.modes)
    costs = benchmark_costs(config, theta, sigma)

    high_curvature_index = int(torch.argmax(torch.linalg.norm(torch.roll(tangent, shifts=-1, dims=0) - tangent, dim=1)).detach().cpu())
    theta0 = float(theta[high_curvature_index].detach().cpu())
    h = perimeter / config.n_boundary
    rho_levels = torch.tensor((0.04, 0.16, 0.48, 0.95), dtype=dtype)
    # This is a visual collar along the outward normal, exaggerated for
    # readability.  In a production run, physical points are
    # Phi^{-1}(exp(rho + i theta0)).
    collar_points = boundary[high_curvature_index].unsqueeze(0) + (rho_levels * 0.42).unsqueeze(1) * normal[high_curvature_index].unsqueeze(0)

    mode_abs = torch.arange(0, 18, dtype=dtype)
    damping_rows = {
        f"rho_{float(rho):.2f}": torch.exp(-mode_abs * rho).detach().cpu().tolist()
        for rho in (torch.tensor(0.04, dtype=dtype), torch.tensor(0.16, dtype=dtype), torch.tensor(0.48, dtype=dtype))
    }
    return {
        "config": asdict(config),
        "curve": {
            "type": "closed piecewise-cubic Catmull-Rom funky curve",
            "control_points": CONTROL_POINTS,
            "boundary_points": boundary.detach().cpu().tolist(),
            "theta": theta.detach().cpu().tolist(),
            "perimeter": perimeter,
            "arclength_resample_ms": resample_ms,
            "orientation": "counterclockwise",
            "note": "visual curve is piecewise smooth; analytic-class error claims require an analytic Riemann-map computation",
        },
        "scale_phase": {
            "formula": "exp(rho + i theta) = exp(rho) exp(i theta)",
            "demoivre": "(exp(i theta))^k = exp(i k theta)",
            "mode_extension": "exp(i k theta) -> exp(-abs(k) rho) exp(i k theta)",
            "target_phase_theta": theta0,
            "target_index": high_curvature_index,
            "rho_levels": rho_levels.detach().cpu().tolist(),
            "visual_collar_points": collar_points.detach().cpu().tolist(),
            "damping_by_abs_mode": damping_rows,
        },
        "modal": {
            "k_values": k_values.detach().cpu().tolist(),
            "density_hat_real": coeffs.real.detach().cpu().tolist(),
            "density_hat_imag": coeffs.imag.detach().cpu().tolist(),
        },
        "costs": costs,
        "cost_ledger": [
            {"stage": "map theta<->s", "one_time_per_geometry": "iter x O(n log n)", "per_target": "-", "scales": "once"},
            {"stage": "phase resample", "one_time_per_geometry": "O(n log n)", "per_target": "-", "scales": "once"},
            {"stage": "density FFT", "one_time_per_geometry": "O(n log n) / density", "per_target": "-", "scales": "amortized across targets"},
            {"stage": "scale flow", "one_time_per_geometry": "-", "per_target": "O(n) diagonal", "scales": "target only touches rho"},
            {"stage": "phase synthesis", "one_time_per_geometry": "-", "per_target": "O(n) modal sum", "scales": "target only touches theta"},
        ],
        "pseudocode": [
            "precompute_geometry(Gamma):",
            "  z_j = arclength_sample(Gamma, n)",
            "  theta_j = FourierNewtonExteriorMap(Gamma)  # one-time, not per target",
            "  R = resample_density_to_phase_grid(theta_j)",
            "  return PullbackQJet(z_j, theta_j, R)",
            "",
            "precompute_density(sigma):",
            "  sigma_hat = FFT(R sigma)",
            "  return sigma_hat",
            "",
            "evaluate_target(x, sigma_hat):",
            "  w = Phi(x)",
            "  rho = log(abs(w)); theta = arg(w)",
            "  return sum_{k != 0} -pi*sigma_hat[k]*exp(-abs(k)*rho)*exp(i*k*theta)/abs(k)",
        ],
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
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    boundary = torch.tensor(payload["curve"]["boundary_points"], dtype=torch.float64)
    collar = torch.tensor(payload["scale_phase"]["visual_collar_points"], dtype=torch.float64)
    theta0 = float(payload["scale_phase"]["target_phase_theta"])
    rho_levels = payload["scale_phase"]["rho_levels"]
    damping = payload["scale_phase"]["damping_by_abs_mode"]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    closed = torch.cat((boundary, boundary[:1]), dim=0)
    ax.plot(closed[:, 0], closed[:, 1], color="0.0", linewidth=1.7)
    ax.scatter(boundary[::8, 0], boundary[::8, 1], color="0.25", s=7)
    ax.plot(collar[:, 0], collar[:, 1], color="0.45", linestyle=(0, (3, 2)), marker="o", markersize=3)
    for idx, rho in enumerate(rho_levels):
        ax.annotate(
            f"rho={rho:.2f}",
            xy=(float(collar[idx, 0]), float(collar[idx, 1])),
            xytext=(8, -9 + 11 * idx),
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.45},
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("physical curve Γ in complex plane")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.grid(True, linewidth=0.45)

    ax = axes[0, 1]
    theta_grid = torch.linspace(0.0, TAU, 400)
    ax.plot(theta_grid, torch.zeros_like(theta_grid), color="0.0", linewidth=1.2, label="boundary rho=0")
    for rho in (0.04, 0.16, 0.48, 0.95):
        ax.plot(theta_grid, torch.full_like(theta_grid, rho), color="0.65", linewidth=0.8)
    ax.axvline(theta0, color="0.15", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.scatter([theta0] * len(rho_levels), rho_levels, color="0.0", s=15)
    ax.set_xlim(0.0, TAU)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("conformal log cylinder")
    ax.set_xlabel("phase theta")
    ax.set_ylabel("scale rho = log |w|")
    ax.grid(True, linewidth=0.45)

    ax = axes[0, 2]
    mode = torch.arange(18)
    for slot, (label, values) in enumerate(damping.items()):
        style = ("-", (0, (4, 2)), (0, (1, 1)))[slot]
        color = ("0.0", "0.38", "0.62")[slot]
        ax.semilogy(mode, values, marker="o", markersize=3, linewidth=1.1, linestyle=style, color=color, label=label.replace("_", "="))
    ax.set_title("real scale flow exp(-|k| rho)")
    ax.set_xlabel("|k|")
    ax.set_ylabel("multiplier")
    ax.grid(True, linewidth=0.45)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    unit = torch.stack((torch.cos(theta_grid), torch.sin(theta_grid)), dim=1)
    ax.plot(unit[:, 0], unit[:, 1], color="0.0", linewidth=1.2)
    for angle, label in ((0.72, r"$e^{i\theta}$"), (2.16, r"$(e^{i\theta})^3=e^{i3\theta}$")):
        ax.arrow(0, 0, math.cos(angle), math.sin(angle), width=0.006, color="0.18", length_includes_head=True)
        ax.text(1.05 * math.cos(angle), 1.05 * math.sin(angle), label, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("De Moivre phase characters")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, linewidth=0.45)

    ax = axes[1, 1]
    ax.axis("off")
    rows = payload["cost_ledger"]
    table = ax.table(
        cellText=[
            [row["stage"], row["one_time_per_geometry"], row["per_target"]]
            for row in rows
        ],
        colLabels=("stage", "one-time", "per target"),
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=(0.34, 0.31, 0.35),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    table.scale(1.0, 1.35)
    for cell in table.get_celld().values():
        cell.set_edgecolor("0.72")
        cell.set_linewidth(0.4)
    ax.set_title("borrow-compute-repay cost ledger", pad=8)

    ax = axes[1, 2]
    ax.axis("off")
    pseudo = "\n".join(payload["pseudocode"])
    formula = (
        "exp(rho+i theta)=e^rho e^{i theta}\n"
        "(e^{i theta})^k=e^{i k theta}\n"
        "e^{i k theta} -> e^{-|k|rho} e^{i k theta}"
    )
    ax.text(0.0, 1.0, formula + "\n\n" + pseudo, va="top", ha="left", family="monospace", fontsize=6.4)
    ax.set_title("pseudocode and modal rule", pad=8)

    for ax in axes.ravel()[:4]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(payload: dict[str, object], path: Path) -> None:
    costs = payload["costs"]
    ledger = payload["cost_ledger"]
    ledger_lines = [
        "| Stage | One-time per geometry | Per target | Scaling note |",
        "|---|---|---|---|",
    ]
    for row in ledger:
        ledger_lines.append(
            f"| {row['stage']} | {row['one_time_per_geometry']} | {row['per_target']} | {row['scales']} |"
        )
    pseudo = "\n".join(payload["pseudocode"])
    text = f"""# Scale-Phase Pullback Demo

This is a visual demonstration of the scale-phase pullback used in the Cauchy/Q calculus.  It uses a funky closed piecewise-cubic boundary to make the geometry visibly non-conic.

Important limitation: this demo does not solve the exterior Riemann map.  It shows the mechanism and cost ledger.  In production, the map `Phi: exterior(Gamma) -> exterior(D)` is computed once per geometry, then reused for all targets.

## Complex Coordinates

The exterior disk coordinate is

```text
w = Phi(z) = exp(rho + i theta) = exp(rho) exp(i theta).
```

The two coordinates have separate jobs:

```text
rho   = log |w|      scale / distance from boundary in conformal units
theta = arg w        phase / boundary correspondence
```

De Moivre supplies the Fourier characters:

```text
(exp(i theta))^k = exp(i k theta).
```

The exterior harmonic continuation of a boundary mode is the same phase with real damping:

```text
exp(i k theta) -> exp(-|k| rho) exp(i k theta).
```

So the target enters through only two scalars, `rho` and `theta`, after the one-time pullback map has been built.

## Modal Evaluation

For a single-layer density with Fourier coefficients `sigma_hat[k]`, the circle-model exterior evaluation is

```text
I(w) = -pi sum_{{k != 0}} sigma_hat[k] exp(-|k| rho) exp(i k theta) / |k|
       + 2 pi sigma_hat[0] rho.
```

This is the diagonal scale flow `T_rho = exp(-rho |D|)` plus a phase character.  Scale does not rotate phase; phase does not change scale.

## Borrow-Compute-Repay Cost Ledger

{chr(10).join(ledger_lines)}

Measured toy timing on this demo:

```text
boundary samples              {payload["config"]["n_boundary"]}
modes each side               {payload["config"]["modes"]}
target count                  {costs["target_count"]}
arclength resample            {payload["curve"]["arclength_resample_ms"]:.3f} ms
density FFT                   {costs["density_fft_ms"]:.3f} ms
modal target evaluation       {costs["modal_eval_ms_for_targets"]:.3f} ms
microseconds per target       {costs["modal_eval_us_per_target"]:.6f}
```

The Riemann-map solve is intentionally not hidden in this timing.  It belongs in the one-time geometry column.

## Pseudocode

```text
{pseudo}
```

## Files

- Figure: `scale_phase_pullback_demo.png`
- JSON payload: `scale_phase_pullback_demo.json`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "scale_phase_pullback_demo")
    parser.add_argument("--n-boundary", type=int, default=DemoConfig.n_boundary)
    parser.add_argument("--modes", type=int, default=DemoConfig.modes)
    parser.add_argument("--target-count", type=int, default=DemoConfig.target_count)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DemoConfig(n_boundary=args.n_boundary, modes=args.modes, target_count=args.target_count)
    payload = build_demo(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "scale_phase_pullback_demo.json"
    md_path = args.out_dir / "scale_phase_pullback_demo.md"
    png_path = args.out_dir / "scale_phase_pullback_demo.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, md_path)
    if not args.no_plot:
        write_plot(payload, png_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "png": None if args.no_plot else str(png_path),
                "boundary_samples": payload["config"]["n_boundary"],
                "modes_each_side": payload["config"]["modes"],
                "target_count": payload["costs"]["target_count"],
                "density_fft_ms": payload["costs"]["density_fft_ms"],
                "modal_eval_ms_for_targets": payload["costs"]["modal_eval_ms_for_targets"],
                "modal_eval_us_per_target": payload["costs"]["modal_eval_us_per_target"],
                "riemann_map_cost_included": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
