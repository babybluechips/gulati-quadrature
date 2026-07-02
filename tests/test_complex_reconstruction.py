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


def test_complicated_polygon_reconstructs_from_gulati_matrix() -> None:
    polygon = BoundaryCurve(complicated_polygon(18)).normalized()
    result = reconstruct_polygon_from_gulati(gulati_laplacian(polygon.points))
    aligned = procrustes_align(result.points, polygon.points)
    assert aligned.rms_error < 1e-10
    assert aligned.hausdorff_error < 1e-10


def test_piecewise_curved_sampled_boundary_reconstructs_from_gulati_matrix() -> None:
    curved = BoundaryCurve(piecewise_curved_boundary(20)).normalized()
    result = reconstruct_polygon_from_gulati(gulati_laplacian(curved.points))
    aligned = procrustes_align(result.points, curved.points)
    assert aligned.rms_error < 1e-9
    assert aligned.hausdorff_error < 1e-9


def test_piecewise_curved_flux_and_star_fit_are_stable() -> None:
    curved = BoundaryCurve(piecewise_curved_boundary(18)).normalized()
    theta = np.linspace(0.0, 2.0 * np.pi, curved.n, endpoint=False)
    flux = 1.0 + 0.18 * np.cos(2.0 * theta) + 0.07 * np.sin(5.0 * theta)
    h_res = dressed_gulati_hessian(curved.points, flux)
    recovered = extract_flux_from_hessian(curved.points, h_res, neighbor_window=5)
    assert np.linalg.norm(recovered - flux) / np.linalg.norm(flux) < 1e-10

    model = fit_star_shape_model(curved.points, modes=10)
    assert hausdorff_distance(curved.points, model.boundary_points(curved.n)) < 0.08
