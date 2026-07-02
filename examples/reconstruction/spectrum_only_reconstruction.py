"""Demonstrate constrained reconstruction from Dirichlet eigenvalues only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inverse_shape.dirichlet import dirichlet_eigenvalues
from inverse_shape.metrics import rigid_rotation_hausdorff_align
from inverse_shape.spectrum_inverse import (
    reconstruct_star_shape_from_spectrum,
    star_boundary_from_spectral_coefficients,
)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[0]])


def build_spectrum_demo(out_dir: Path) -> dict[str, float | int | list[float] | str]:
    """Run the spectrum-only inverse demo and write a figure plus metrics."""

    plt = _load_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = 192
    grid_size = 44
    eigenvalue_count = 7
    true_coefficients = np.array([0.12, 0.05], dtype=np.float64)
    initial_coefficients = np.array([0.04, 0.02], dtype=np.float64)

    target = star_boundary_from_spectral_coefficients(true_coefficients, samples=samples)
    target_eigenvalues = dirichlet_eigenvalues(target, k=eigenvalue_count, grid_size=grid_size)
    result = reconstruct_star_shape_from_spectrum(
        target_eigenvalues,
        modes=1,
        initial=initial_coefficients,
        samples=samples,
        grid_size=grid_size,
        max_nfev=140,
        regularization=0.0,
    )
    aligned = rigid_rotation_hausdorff_align(result.boundary, target, rotations=720)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.plot(*_closed(target).T, color="#202124", lw=2.3, label="target")
    ax.plot(
        *_closed(aligned.points).T,
        "--",
        color="#d93025",
        lw=2.0,
        label="spectrum-only reconstruction",
    )
    ax.set_title(f"Boundary after rigid alignment\nHausdorff {aligned.hausdorff_error:.3e}")
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(alpha=0.22)

    ax = axes[1]
    index = np.arange(1, eigenvalue_count + 1)
    ax.plot(index, target_eigenvalues, "o-", color="#202124", lw=2.0, label="target spectrum")
    ax.plot(index, result.eigenvalues, "s--", color="#188038", lw=1.9, label="recovered spectrum")
    ax.set_title(f"First {eigenvalue_count} Dirichlet eigenvalues")
    ax.set_xlabel("eigenvalue index")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    residuals = np.array([result.initial_relative_residual, result.relative_residual])
    ax.bar(["initial", "recovered"], residuals, color=["#5f6368", "#1a73e8"])
    ax.set_title("Relative spectral residual")
    ax.set_ylabel(r"$||\lambda-\hat{\lambda}|| / ||\lambda||$")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(residuals):
        ax.text(idx, value, f"{value:.3e}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.set_facecolor("#fbfbfb")

    figure_path = out_dir / "spectrum_only_reconstruction.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    metrics: dict[str, float | int | list[float] | str] = {
        "eigenvalue_count": eigenvalue_count,
        "grid_size": grid_size,
        "true_coefficients": true_coefficients.tolist(),
        "initial_coefficients": initial_coefficients.tolist(),
        "recovered_coefficients": result.coefficients.tolist(),
        "initial_relative_residual": result.initial_relative_residual,
        "spectrum_relative_residual": result.relative_residual,
        "shape_hausdorff_after_rotation": aligned.hausdorff_error,
        "alignment_rotation_radians": aligned.angle,
        "optimizer_nfev": result.nfev,
        "target_eigenvalues": target_eigenvalues.tolist(),
        "recovered_eigenvalues": result.eigenvalues.tolist(),
        "figure": str(figure_path),
    }
    with (out_dir / "spectrum_only_reconstruction_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    metrics = build_spectrum_demo(args.out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
