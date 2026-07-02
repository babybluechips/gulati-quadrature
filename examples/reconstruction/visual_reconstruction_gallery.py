"""Generate visual reconstruction examples for the docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inverse_shape.datasets import complicated_polygon, piecewise_curved_boundary
from inverse_shape.geometry import BoundaryCurve, hausdorff_distance
from inverse_shape.metrics import procrustes_align
from inverse_shape.operators import (
    dressed_gulati_hessian,
    extract_flux_from_hessian,
    gulati_laplacian,
)
from inverse_shape.reconstruction import fit_star_shape_model, reconstruct_polygon_from_gulati


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[0]])


def build_gallery(out_dir: Path) -> dict[str, float | str]:
    plt = _load_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    polygon = BoundaryCurve(complicated_polygon(18)).normalized()
    gu_poly = gulati_laplacian(polygon.points)
    poly_recon = reconstruct_polygon_from_gulati(gu_poly)
    poly_aligned = procrustes_align(poly_recon.points, polygon.points)

    curved = BoundaryCurve(piecewise_curved_boundary(28)).normalized()
    gu_curved = gulati_laplacian(curved.points)
    curved_recon = reconstruct_polygon_from_gulati(gu_curved)
    curved_aligned = procrustes_align(curved_recon.points, curved.points)

    theta = np.linspace(0.0, 2.0 * np.pi, curved.n, endpoint=False)
    flux = 1.0 + 0.22 * np.cos(2.0 * theta - 0.25) + 0.08 * np.sin(5.0 * theta)
    h_res = dressed_gulati_hessian(curved.points, flux)
    flux_hat = extract_flux_from_hessian(curved.points, h_res, neighbor_window=5)

    star_fit = fit_star_shape_model(curved.points, modes=8)
    star_points = star_fit.boundary_points(curved.n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(*_closed(polygon.points).T, color="#202124", lw=2.2, label="original polygon")
    ax.plot(
        *_closed(poly_aligned.points).T,
        "--",
        color="#d93025",
        lw=1.8,
        label="Gulati reconstruction",
    )
    ax.scatter(polygon.points[:, 0], polygon.points[:, 1], s=18, color="#202124")
    ax.set_title(f"Complicated polygon from Gulati matrix (RMS {poly_aligned.rms_error:.2e})")
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(*_closed(curved.points).T, color="#202124", lw=2.2, label="piecewise curved boundary")
    ax.plot(
        *_closed(curved_aligned.points).T,
        "--",
        color="#188038",
        lw=1.8,
        label="sampled Gulati reconstruction",
    )
    ax.set_title(
        f"Piecewise curved shape from Gulati matrix (RMS {curved_aligned.rms_error:.2e})"
    )
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(theta, flux, color="#202124", lw=2.0, label="true flux")
    ax.plot(theta, flux_hat, "--", color="#1a73e8", lw=1.8, label="Laurent extracted")
    ax.set_title("Hadamard finite-part flux extraction")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\partial_\nu u_1$")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(*_closed(curved.points).T, color="#202124", lw=2.2, label="original")
    ax.plot(*_closed(star_points).T, "--", color="#f9ab00", lw=1.8, label="8-mode star fit")
    ax.set_title(
        "Low-mode star-shaped approximation "
        f"(Hausdorff {hausdorff_distance(curved.points, star_points):.2e})"
    )
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(alpha=0.2)

    for ax in axes.ravel():
        ax.set_facecolor("#fbfbfb")

    gallery_path = out_dir / "reconstruction_gallery.png"
    fig.savefig(gallery_path, dpi=180)
    plt.close(fig)

    metrics = {
        "polygon_gulati_rms": poly_aligned.rms_error,
        "polygon_gulati_hausdorff": poly_aligned.hausdorff_error,
        "curved_gulati_rms": curved_aligned.rms_error,
        "curved_gulati_hausdorff": curved_aligned.hausdorff_error,
        "flux_relative_error": float(np.linalg.norm(flux_hat - flux) / np.linalg.norm(flux)),
        "star_fit_hausdorff": hausdorff_distance(curved.points, star_points),
        "gallery": str(gallery_path),
    }
    with (out_dir / "reconstruction_gallery_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    metrics = build_gallery(args.out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
