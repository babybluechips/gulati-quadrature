"""Run a synthetic inverse-shape reconstruction demo."""

from __future__ import annotations

import numpy as np

from inverse_shape.geometry import BoundaryCurve, StarShapeModel, hausdorff_distance
from inverse_shape.operators import (
    dressed_gulati_hessian,
    extract_flux_from_hessian,
    gulati_laplacian,
)
from inverse_shape.reconstruction import fit_star_shape_model, reconstruct_polygon_from_gulati


def main() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.10, -0.03, 0.02]),
        sin=np.array([0.04, 0.05, -0.02]),
    )
    curve = BoundaryCurve(model.boundary_points(192)).normalized()
    gu = gulati_laplacian(curve.points)
    gulati_recon = reconstruct_polygon_from_gulati(gu)

    theta = np.linspace(0.0, 2.0 * np.pi, curve.n, endpoint=False)
    flux = 1.0 + 0.15 * np.cos(2.0 * theta)
    h_res = dressed_gulati_hessian(curve.points, flux)
    flux_hat = extract_flux_from_hessian(curve.points, h_res)

    star_fit = fit_star_shape_model(curve.points, modes=3)
    fit_points = star_fit.boundary_points(curve.n)

    print("Gulati reconstruction residual:", gulati_recon.residual_norm)
    print("Flux relative error:", np.linalg.norm(flux_hat - flux) / np.linalg.norm(flux))
    print("Star model Hausdorff:", hausdorff_distance(curve.points, fit_points))


if __name__ == "__main__":
    main()
