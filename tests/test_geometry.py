import numpy as np

from inverse_shape.geometry import BoundaryCurve, StarShapeModel, hausdorff_distance
from inverse_shape.reconstruction import fit_star_shape_model


def test_boundary_normalization_sets_area_to_one() -> None:
    square = BoundaryCurve(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float))
    normalized = square.normalized()
    assert np.isclose(abs(normalized.area), 1.0)
    assert np.allclose(normalized.centroid, [0.0, 0.0], atol=1e-12)


def test_star_shape_fit_recovers_low_modes() -> None:
    model = StarShapeModel(
        center=np.array([0.2, -0.1]),
        base_radius=1.3,
        cos=np.array([0.08, -0.03]),
        sin=np.array([0.02, 0.04]),
    )
    points = model.boundary_points(256)
    fitted = fit_star_shape_model(points, modes=2, center=model.center)
    assert np.isclose(fitted.base_radius, model.base_radius, atol=1e-10)
    assert np.allclose(fitted.cos, model.cos, atol=1e-10)
    assert np.allclose(fitted.sin, model.sin, atol=1e-10)
    assert hausdorff_distance(points, fitted.boundary_points(256)) < 1e-10
