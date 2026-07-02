#!/usr/bin/env python3
"""GWW isospectral drums: Dirichlet spectrum versus boundary Q witness.

The Gordon-Webb-Wolpert pair is the standard counterexample to hearing the
shape of a planar drum.  The two eight-sided polygonal domains below are
Dirichlet-isospectral by construction, so the Dirichlet Laplacian spectrum is
not allowed to distinguish them.

This example compares that exact Dirichlet fact against a matrix-free boundary
Q/DtN witness.  The Q side stores only the sampled boundary generator and uses
pairwise inverse-square chord weights through ``build_planar_domain_qjet``; it
does not materialize the dense Q matrix.  A small Lanczos tridiagonal is formed
only as a Ritz diagnostic for the generated operator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from inverse_shape.q_dtn import build_planar_domain_qjet


Point = tuple[float, float]
TAU = 2.0 * math.pi


GWW_LEFT: tuple[Point, ...] = (
    (-3.0, -3.0),
    (-3.0, -1.0),
    (1.0, 3.0),
    (1.0, 1.0),
    (3.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -3.0),
)

GWW_RIGHT: tuple[Point, ...] = (
    (-3.0, 1.0),
    (1.0, 1.0),
    (1.0, 3.0),
    (3.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (-1.0, -3.0),
    (-3.0, -1.0),
)


@dataclass(frozen=True)
class GWWConfig:
    boundary_samples: int = 96
    lanczos_steps: int = 56
    q_ritz_count: int = 12
    dirichlet_eigen_count: int = 8
    dirichlet_grid_size: int = 64
    area_normalization: float = 1.0
    run_finite_difference_dirichlet: bool = True


def polygon_area(points: tuple[Point, ...]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[index][1] * points[(index + 1) % len(points)][0]
        for index in range(len(points))
    )


def polygon_perimeter(points: tuple[Point, ...]) -> float:
    return sum(
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def arithmetic_center(points: tuple[Point, ...]) -> Point:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def normalize_polygon(points: tuple[Point, ...], *, area: float) -> tuple[Point, ...]:
    oriented = tuple(reversed(points)) if polygon_area(points) < 0.0 else points
    current = abs(polygon_area(oriented))
    if current <= 0.0:
        raise ValueError("cannot normalize a zero-area polygon")
    scale = math.sqrt(area / current)
    cx, cy = arithmetic_center(oriented)
    return tuple((scale * (x - cx), scale * (y - cy)) for x, y in oriented)


def sample_polygon_boundary(points: tuple[Point, ...], sample_count: int) -> tuple[Point, ...]:
    if sample_count < len(points):
        raise ValueError("sample_count must be at least the vertex count")
    edge_lengths = [
        math.dist(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    total = sum(edge_lengths)
    if total <= 0.0:
        raise ValueError("polygon perimeter is zero")
    out: list[Point] = []
    edge = 0
    accumulated = 0.0
    for sample in range(sample_count):
        arclength = (sample + 0.5) * total / sample_count
        while edge + 1 < len(points) and arclength > accumulated + edge_lengths[edge]:
            accumulated += edge_lengths[edge]
            edge += 1
        local = (arclength - accumulated) / edge_lengths[edge]
        x0, y0 = points[edge]
        x1, y1 = points[(edge + 1) % len(points)]
        out.append((x0 + local * (x1 - x0), y0 + local * (y1 - y0)))
    return tuple(out)


def closed(points: tuple[Point, ...]) -> tuple[Point, ...]:
    return points + (points[0],)


def vector_norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(values).clamp_min(1.0e-30)


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.norm(left - right) / vector_norm(left))


def q_apply_real(qjet: object, values: torch.Tensor) -> torch.Tensor:
    """Apply the generated Q/DtN operator without forming its matrix."""

    raw = qjet.qjet.apply(float(value) for value in values.tolist())
    return torch.tensor(
        [qjet.scale * float(complex(value).real) for value in raw],
        dtype=torch.float64,
    )


def q_lanczos_ritz(qjet: object, *, steps: int, count: int) -> tuple[list[float], float]:
    """Return largest Ritz values of the generated Q operator.

    Full reorthogonalization is used for numerical stability.  The dense object
    formed here is the small Lanczos tridiagonal, not the boundary Q matrix.
    """

    n = qjet.n
    if steps < count + 2:
        raise ValueError("lanczos_steps must exceed q_ritz_count by at least two")
    if steps >= n:
        steps = n - 2
    index = torch.arange(n, dtype=torch.float64)
    q = torch.cos(0.37 * index) + 0.31 * torch.sin(1.13 * index)
    q = q - torch.mean(q)
    q = q / vector_norm(q)
    q_previous = torch.zeros_like(q)
    beta = torch.tensor(0.0, dtype=torch.float64)
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    basis: list[torch.Tensor] = []

    start = perf_counter()
    for step in range(steps):
        basis.append(q)
        z = q_apply_real(qjet, q) - beta * q_previous
        alpha = torch.dot(q, z)
        z = z - alpha * q
        for old in basis:
            z = z - torch.dot(old, z) * old
        beta_next = vector_norm(z)
        alphas.append(alpha)
        if step < steps - 1:
            betas.append(beta_next)
        if float(beta_next) < 1.0e-12:
            break
        q_previous = q
        q = z / beta_next
        beta = beta_next

    tridiagonal = torch.diag(torch.stack(alphas))
    if len(alphas) > 1:
        offdiag = torch.stack(betas[: len(alphas) - 1])
        tridiagonal = tridiagonal + torch.diag(offdiag, diagonal=1) + torch.diag(offdiag, diagonal=-1)
    values = torch.linalg.eigvalsh(tridiagonal)
    largest = torch.flip(values[-count:], dims=(0,))
    return [float(value) for value in largest], 1000.0 * (perf_counter() - start)


def boundary_probe(sample_count: int) -> torch.Tensor:
    index = torch.arange(sample_count, dtype=torch.float64)
    theta = TAU * index / sample_count
    probe = torch.cos(3.0 * theta + 0.2) + 0.35 * torch.sin(7.0 * theta - 0.4)
    return probe - torch.mean(probe)


def finite_difference_dirichlet(
    left: tuple[Point, ...],
    right: tuple[Point, ...],
    *,
    count: int,
    grid_size: int,
) -> dict[str, object]:
    from inverse_shape.dirichlet import dirichlet_eigenvalues

    start = perf_counter()
    left_values = dirichlet_eigenvalues(left, k=count, grid_size=grid_size)
    right_values = dirichlet_eigenvalues(right, k=count, grid_size=grid_size)
    elapsed_ms = 1000.0 * (perf_counter() - start)
    left_tensor = torch.tensor(left_values.tolist(), dtype=torch.float64)
    right_tensor = torch.tensor(right_values.tolist(), dtype=torch.float64)
    per_mode = [
        abs(float(a - b)) / max(abs(float(a)), 1.0e-30)
        for a, b in zip(left_values.tolist(), right_values.tolist(), strict=True)
    ]
    return {
        "grid_size": grid_size,
        "eigen_count": count,
        "left_eigenvalues": [float(value) for value in left_values.tolist()],
        "right_eigenvalues": [float(value) for value in right_values.tolist()],
        "relative_l2_difference": relative_l2(left_tensor, right_tensor),
        "per_mode_relative_differences": per_mode,
        "elapsed_ms": elapsed_ms,
        "warning": (
            "This raster finite-difference check is not the mathematical truth "
            "for the GWW pair; the exact Dirichlet spectra are equal.  Grid "
            "misalignment creates small artificial differences."
        ),
    }


def run(config: GWWConfig) -> dict[str, object]:
    left = normalize_polygon(GWW_LEFT, area=config.area_normalization)
    right = normalize_polygon(GWW_RIGHT, area=config.area_normalization)
    left_samples = sample_polygon_boundary(left, config.boundary_samples)
    right_samples = sample_polygon_boundary(right, config.boundary_samples)
    left_q = build_planar_domain_qjet(left_samples)
    right_q = build_planar_domain_qjet(right_samples)

    left_ritz, left_q_ms = q_lanczos_ritz(
        left_q,
        steps=config.lanczos_steps,
        count=config.q_ritz_count,
    )
    right_ritz, right_q_ms = q_lanczos_ritz(
        right_q,
        steps=config.lanczos_steps,
        count=config.q_ritz_count,
    )
    left_ritz_tensor = torch.tensor(left_ritz, dtype=torch.float64)
    right_ritz_tensor = torch.tensor(right_ritz, dtype=torch.float64)

    probe = boundary_probe(config.boundary_samples)
    left_response = q_apply_real(left_q, probe)
    right_response = q_apply_real(right_q, probe)
    response_scale = max(float(vector_norm(left_response)), float(vector_norm(right_response)), 1.0e-30)
    response_difference = float(torch.linalg.norm(left_response - right_response) / response_scale)

    left_evaluation = left_q.apply_dtn(probe.tolist())
    right_evaluation = right_q.apply_dtn(probe.tolist())

    payload: dict[str, object] = {
        "config": asdict(config),
        "source_vertices": {
            "left": GWW_LEFT,
            "right": GWW_RIGHT,
            "coordinate_reference": (
                "COMSOL Application Library isospectral_drums geometry tables; "
                "Gordon-Webb-Wolpert/Driscoll eight-sided GWW drum pair."
            ),
            "references": (
                "https://arxiv.org/abs/math/9207215",
                "https://www.comsol.com/model/download/1166951/models.mph.isospectral_drums.pdf",
                "https://eudml.org/doc/144038",
            ),
        },
        "normalized_domains": {
            "left_vertices": left,
            "right_vertices": right,
            "left_boundary_samples": left_samples,
            "right_boundary_samples": right_samples,
            "left_area": abs(polygon_area(left)),
            "right_area": abs(polygon_area(right)),
            "left_perimeter": polygon_perimeter(left),
            "right_perimeter": polygon_perimeter(right),
        },
        "dirichlet_laplacian": {
            "mathematical_status": "GWW Dirichlet-isospectral pair",
            "distinguishes_domains": False,
            "exact_spectral_distance": 0.0,
            "reason": (
                "The two domains have identical Dirichlet Laplacian spectra by "
                "the Gordon-Webb-Wolpert transplantation construction."
            ),
        },
        "q_boundary_witness": {
            "operator": "PlanarDomainQJet inverse-square chord DtN generator",
            "dense_q_matrix_stored": False,
            "stored_state": "boundary samples plus generated QJet weights only",
            "lanczos_dense_object": "small tridiagonal Ritz diagnostic",
            "left_top_ritz": left_ritz,
            "right_top_ritz": right_ritz,
            "top_ritz_relative_l2_difference": relative_l2(left_ritz_tensor, right_ritz_tensor),
            "probe_response_relative_difference": response_difference,
            "left_lanczos_elapsed_ms": left_q_ms,
            "right_lanczos_elapsed_ms": right_q_ms,
            "left_ledger_status": left_evaluation.ledger.status,
            "right_ledger_status": right_evaluation.ledger.status,
            "left_protocol": left_evaluation.stats["protocol"],
            "right_protocol": right_evaluation.stats["protocol"],
        },
        "probe": {
            "boundary_trace": [float(value) for value in probe.tolist()],
            "left_q_response": [float(value) for value in left_response.tolist()],
            "right_q_response": [float(value) for value in right_response.tolist()],
        },
        "interpretation": (
            "The Dirichlet Laplacian spectrum is intentionally blind on this "
            "GWW pair.  The boundary Q/DtN generator is a different spectral "
            "object tied to boundary chord geometry.  At the discretized "
            "chord-operator level, its Ritz/probe signatures separate the two "
            "nonisometric domains; this experiment is not a standalone proof "
            "of a continuum Q-spectrum invariant."
        ),
    }
    if config.run_finite_difference_dirichlet:
        payload["dirichlet_laplacian"]["finite_difference_check"] = finite_difference_dirichlet(
            left,
            right,
            count=config.dirichlet_eigen_count,
            grid_size=config.dirichlet_grid_size,
        )
    return payload


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
    domains = payload["normalized_domains"]
    q = payload["q_boundary_witness"]
    d = payload["dirichlet_laplacian"]
    probe = payload["probe"]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    left = tuple(tuple(point) for point in domains["left_vertices"])
    right = tuple(tuple(point) for point in domains["right_vertices"])
    for offset, points, label, style, color in (
        (-1.9, left, "GWW left", "-", "0.00"),
        (1.9, right, "GWW right", (0, (3, 2)), "0.20"),
    ):
        shifted = tuple((x + offset, y) for x, y in points)
        curve = closed(shifted)
        ax.plot(
            [point[0] for point in curve],
            [point[1] for point in curve],
            linestyle=style,
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax.scatter(
            [point[0] for point in shifted],
            [point[1] for point in shifted],
            color=color,
            s=9,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GWW isospectral polygon pair")
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    ax = axes[0, 1]
    exact = [0.0 for _ in range(payload["config"]["dirichlet_eigen_count"])]
    modes = list(range(1, len(exact) + 1))
    ax.plot(modes, exact, color="0.0", linewidth=1.5, label="exact GWW difference")
    fd = d.get("finite_difference_check")
    if isinstance(fd, dict):
        ax.plot(
            modes,
            fd["per_mode_relative_differences"],
            color="0.55",
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            label="raster FD artifact",
        )
    ax.set_title("Dirichlet Laplacian spectrum")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("relative difference")
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    ax = axes[1, 0]
    q_modes = list(range(1, len(q["left_top_ritz"]) + 1))
    ax.plot(q_modes, q["left_top_ritz"], color="0.0", linewidth=1.55, label="GWW left Q")
    ax.plot(
        q_modes,
        q["right_top_ritz"],
        color="0.45",
        linestyle=(0, (3, 2)),
        linewidth=1.45,
        label="GWW right Q",
    )
    ax.set_title("matrix-free Q Ritz spectrum")
    ax.set_xlabel("largest Ritz index")
    ax.set_ylabel("Q/DtN Ritz value")
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    ax = axes[1, 1]
    indices = list(range(len(probe["left_q_response"])))
    scale = max(
        max(abs(float(value)) for value in probe["left_q_response"]),
        max(abs(float(value)) for value in probe["right_q_response"]),
        1.0e-30,
    )
    ax.plot(
        indices,
        [float(value) / scale for value in probe["left_q_response"]],
        color="0.0",
        linewidth=1.35,
        label="GWW left",
    )
    ax.plot(
        indices,
        [float(value) / scale for value in probe["right_q_response"]],
        color="0.50",
        linestyle=(0, (3, 2)),
        linewidth=1.25,
        label="GWW right",
    )
    ax.set_title("Q response to same boundary trace")
    ax.set_xlabel("arclength sample")
    ax.set_ylabel("normalized Qf")
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gww_isospectral_q_vs_dirichlet")
    parser.add_argument("--boundary-samples", type=int, default=GWWConfig.boundary_samples)
    parser.add_argument("--lanczos-steps", type=int, default=GWWConfig.lanczos_steps)
    parser.add_argument("--q-ritz-count", type=int, default=GWWConfig.q_ritz_count)
    parser.add_argument("--dirichlet-eigen-count", type=int, default=GWWConfig.dirichlet_eigen_count)
    parser.add_argument("--dirichlet-grid-size", type=int, default=GWWConfig.dirichlet_grid_size)
    parser.add_argument("--skip-fd-dirichlet", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GWWConfig(
        boundary_samples=args.boundary_samples,
        lanczos_steps=args.lanczos_steps,
        q_ritz_count=args.q_ritz_count,
        dirichlet_eigen_count=args.dirichlet_eigen_count,
        dirichlet_grid_size=args.dirichlet_grid_size,
        run_finite_difference_dirichlet=not args.skip_fd_dirichlet,
    )
    payload = run(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gww_q_vs_dirichlet.json"
    png_path = args.out_dir / "gww_q_vs_dirichlet.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_plot(payload, png_path)
    fd = payload["dirichlet_laplacian"].get("finite_difference_check")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "png": None if args.no_plot else str(png_path),
                "dirichlet_exact_spectral_distance": payload["dirichlet_laplacian"]["exact_spectral_distance"],
                "dirichlet_distinguishes_domains": payload["dirichlet_laplacian"]["distinguishes_domains"],
                "fd_dirichlet_relative_l2_difference": None
                if not isinstance(fd, dict)
                else fd["relative_l2_difference"],
                "q_dense_matrix_stored": payload["q_boundary_witness"]["dense_q_matrix_stored"],
                "q_top_ritz_relative_l2_difference": payload["q_boundary_witness"]["top_ritz_relative_l2_difference"],
                "q_probe_response_relative_difference": payload["q_boundary_witness"]["probe_response_relative_difference"],
                "q_left_ledger_status": payload["q_boundary_witness"]["left_ledger_status"],
                "q_right_ledger_status": payload["q_boundary_witness"]["right_ledger_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
